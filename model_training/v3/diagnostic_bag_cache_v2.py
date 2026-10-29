"""Corrected input-hash joins and exhaustive reused-token reconstruction audit."""

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
from input_identity_join import input_identity_join


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
    out = p / "phase94_corrected_diagnostic_sft/prefix"
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
    old_descriptors = input_rows(p)
    base_count = len(read(p / "phase29_compact/inputs.json"))
    new_descriptors = read(p / "phase87_diagnostic_bag_features/new_inputs.json")
    target_descriptors = old_descriptors[:base_count] + new_descriptors
    donor_tables = np.load(donor / "tables.npy")
    joined = input_identity_join([target_descriptors[i] for i in tables],old_descriptors,donor_tables)
    donor_values = np.load(donor / "features.npy", mmap_mode="r")
    lookup = {table:(donor_values,offset) for table,offset in zip(tables,joined) if offset is not None}
    # Only the actually computed missing-token rows from the invalid run are
    # reusable. Its positional joins are forbidden; their identities are not.
    previous = p / "phase89_diagnostic_bag_sft/prefix"
    previous_protocol = read(previous / "protocol.json")
    previous_complete = read(previous / "complete.json")
    assert previous_complete["protocol_sha256"] == digest(previous / "protocol.json")
    assert previous_complete["missing_tokens_sha256"] == digest(previous / "missing_tokens.npy")
    assert previous_protocol["prefix_state_sha256"] == completed["prefix_state_sha256"]
    previous_tables = previous_protocol["missing_tables"]
    assert previous_complete["completed"] == len(previous_tables)
    previous_values = np.load(previous / "missing_tokens.npy",mmap_mode="r")
    joined = input_identity_join([target_descriptors[i] for i in tables],target_descriptors,previous_tables)
    for table,offset in zip(tables,joined):
        if offset is not None and table not in lookup:
            lookup[table]=(previous_values,offset)
    missing = [i for i in tables if i not in lookup]
    protocol = dict(
        donor_complete_sha256=digest(donor / "complete.json"),
        prefix_state_sha256=completed["prefix_state_sha256"],
        feature_complete_sha256=digest(p / "phase87_diagnostic_bag_features/complete.json"),
        source_sha256=digest(__file__),
        fitting_rows_sha256=digest(p / "phase84_population_head/fitting_rows.json"),
        selected_tables=tables, missing_tables=missing,
        join_source_sha256=digest(Path(__file__).with_name("input_identity_join.py")),
        target_input_identities=[target_descriptors[i]["input_sha256"] for i in tables],
        donor_input_descriptors_sha256=hashlib.sha256(json.dumps(old_descriptors).encode()).hexdigest(),
        previous_computed_cache_sha256=digest(previous/"complete.json"),
        positional_index_reuse_forbidden=True,
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
    missing_lookup = {table:i for i,table in enumerate(missing)}
    gpu = torch.empty((len(tables),257,384),dtype=torch.float32,device="cuda")
    maximum = 0.0
    with torch.inference_mode():
        for start in range(0,len(tables),64):
            part=tables[start:start+64]
            arrays = [lookup[i][0][lookup[i][1]] if i in lookup else tokens[missing_lookup[i]] for i in part]
            batch=torch.from_numpy(np.stack(arrays)).cuda()
            gpu[start:start+len(arrays)] = batch
            # Every reused AND new input must reconstruct its expected original
            # embedding; a sampled old-row-only audit misses namespace collisions.
            restored = backbone.forward_head(backbone.norm(backbone.blocks[prefix_cut(backbone):](batch)),pre_logits=True)
            expected=values[part,:384]
            error=float(np.abs((F.normalize(restored,dim=1)*math.sqrt(384)).cpu().numpy()-expected).max())
            maximum=max(maximum,error)
            assert maximum < 0.002, ("Canonical input/token identity mismatch",part,error)
    saved.update(all_selected_inputs_audited=len(tables),all_selected_maximum_error=maximum)
    report = dict(saved, missing_tokens_sha256=digest(target))
    (out/"complete.json").write_text(json.dumps(report,indent=2))
    return backbone, gpu, {table:i for i,table in enumerate(tables)}, report
