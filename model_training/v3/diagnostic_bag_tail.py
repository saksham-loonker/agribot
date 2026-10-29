"""Copied diagnostic suffix and identity-checked reuse of its frozen tokens."""

from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from app_eval import canon, classifier_letterbox_input
from leaf_feature_expert import LeafFeatureHead
from opened_context_data import read
from partial_dino_cache import CanonicalInputs, input_rows, original_dino
from partial_dino_healthy import prefix_cut, prefix_features, prefix_identity
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key
from population_siglip_tail import freeze


class DiagnosticBagTail(nn.Module):
    def __init__(self, backbone, warm_head):
        super().__init__()
        self.blocks = copy.deepcopy(backbone.blocks[prefix_cut(backbone):])
        self.norm = copy.deepcopy(backbone.norm)
        self.head = LeafFeatureHead("mlp", 1408)
        self.head.load_state_dict(warm_head)
        self.requires_grad_(True)

    def features(self, tokens, cnn):
        # DINOv2-S in the deployed graph pools its class token.
        dino = self.norm(self.blocks(tokens))[:, 0]
        return torch.cat([F.normalize(dino, dim=1) * math.sqrt(384), cnn], 1)

    def forward(self, tokens, cnn):
        return self.head(self.features(tokens, cnn))


def prepare_tokens(campaign, values, bags, fitting):
    p = Path(campaign)
    out = p / "phase89_diagnostic_bag_sft/prefix"
    out.mkdir(parents=True, exist_ok=True)
    donor = p / "phase51_dino_adaptation/prefix"
    completed = read(donor / "complete.json")
    assert completed["protocol_sha256"] == digest(donor / "protocol.json")
    assert completed["features_sha256"] == digest(donor / "features.npy")
    assert completed["tables_sha256"] == digest(donor / "tables.npy")
    backbone = original_dino(p).cuda().eval()
    assert completed["prefix_state_sha256"] == prefix_identity(backbone)
    active_paths = {r["path"] for r in fitting if r["active"]}
    tables = sorted({i for path in active_paths if path in bags
                     for indices in bags[path].values() for i in indices})
    donor_lookup = {int(table): i for i, table in enumerate(np.load(donor / "tables.npy"))}
    missing = [i for i in tables if i not in donor_lookup]
    old_descriptors = input_rows(p)
    base_count = len(read(p / "phase29_compact/inputs.json"))
    new_descriptors = read(p / "phase87_diagnostic_bag_features/new_inputs.json")
    protocol = dict(
        donor_complete_sha256=digest(donor / "complete.json"),
        prefix_state_sha256=completed["prefix_state_sha256"],
        feature_complete_sha256=digest(p / "phase87_diagnostic_bag_features/complete.json"),
        source_sha256=digest(__file__),
        fitting_rows_sha256=digest(p / "phase84_population_head/fitting_rows.json"),
        selected_tables=tables, missing_tables=missing,
        shape=[len(missing), 257, 384], dtype="float32",
        old_inputs_reused=len(tables)-len(missing),
        newly_computed_inputs=len(missing), tf32=False, fused_attention=False,
        ad_independent=False, olid_independent=False,
    )
    freeze(out / "protocol.json", protocol)
    progress = out / "progress.json"
    saved = read(progress) if progress.exists() else dict(completed=0, maximum_error=0.0)
    if progress.exists():
        assert saved["protocol_sha256"] == digest(out / "protocol.json")
    target = out / "missing_tokens.npy"
    assert target.exists() or shutil.disk_usage(p).free > len(missing)*257*384*4 + (1<<30)
    tokens = np.lib.format.open_memmap(target, mode="r+" if target.exists() else "w+",
                                     dtype=np.float32, shape=(len(missing),257,384))
    pixels = CanonicalInputs(p)
    try:
        with torch.inference_mode():
            for start in range(saved["completed"], len(missing), 64):
                indices = missing[start:start+64]
                batch = []
                for index in indices:
                    if index < base_count:
                        tensor = pixels(old_descriptors[index])
                    else:
                        row = new_descriptors[index-base_count]
                        with Image.open(row["path"]) as image:
                            rgb = np.asarray(image.convert("RGB"))
                        tensor = (canon.classifier_input if row["view"] == "center"
                                  else classifier_letterbox_input)(rgb,224,row["box"])
                        assert pixel_key(tensor) == row["input_sha256"]
                    batch.append(tensor)
                prefix = prefix_features(backbone, torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2))
                restored = backbone.forward_head(backbone.norm(backbone.blocks[prefix_cut(backbone):](prefix)), pre_logits=True)
                restored = (F.normalize(restored, dim=1)*math.sqrt(384)).cpu().numpy()
                saved["maximum_error"] = max(saved["maximum_error"], float(np.abs(restored-values[indices,:384]).max()))
                assert saved["maximum_error"] < 0.002, "Copied prefix does not reconstruct original diagnostic features"
                tokens[start:start+len(indices)] = prefix.cpu().numpy()
                tokens.flush()
                saved.update(completed=start+len(indices), protocol_sha256=digest(out / "protocol.json"))
                temporary = progress.with_suffix(".tmp")
                temporary.write_text(json.dumps(saved)); temporary.replace(progress)
                print(json.dumps(dict(stage="diagnostic_prefix", **saved, total=len(missing))), flush=True)
    finally:
        pixels.close()
    donor_values = np.load(donor / "features.npy", mmap_mode="r")
    missing_lookup = {table:i for i,table in enumerate(missing)}
    gpu = torch.empty((len(tables),257,384),dtype=torch.float32,device="cuda")
    for start in range(0,len(tables),64):
        arrays = [donor_values[donor_lookup[i]] if i in donor_lookup else tokens[missing_lookup[i]]
                  for i in tables[start:start+64]]
        gpu[start:start+len(arrays)] = torch.from_numpy(np.stack(arrays)).cuda()
    report = dict(saved, missing_tokens_sha256=digest(target))
    (out/"complete.json").write_text(json.dumps(report,indent=2))
    return backbone, gpu, {table:i for i,table in enumerate(tables)}, report
