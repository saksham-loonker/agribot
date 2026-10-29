"""Cache only the new spatial representation; never recompute DINO features."""

from pathlib import Path
import argparse
import concurrent.futures
import json
import time

import numpy as np
from PIL import Image
import torch

from app_eval import canon, classifier_letterbox_input
from campaign_pixels import open_pixels
from compact_healthy_model import primary
from opened_context_data import read
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    args = ap.parse_args()
    p = Path(args.campaign)
    source = p / "phase29_compact"
    out = p / "phase37_spatial"
    out.mkdir(exist_ok=True)
    complete = read(source / "complete.json")
    protocol = dict(
        primary_bundle_sha256=digest(p / "export_cnn_safe/pytorch_bundle.pt"),
        input_descriptors_sha256=digest(source / "inputs.json"),
        fitting_rows_sha256=digest(source / "fitting_rows.json"),
        source_features_sha256=complete["features_sha256"],
        representation="Unchanged installed MobileNetV3 Small forward_features, FP32 NCHW 576x7x7 before global pooling",
        encoder_weights_frozen=True,
        new_dino_inference=0,
        all_inputs_previous_identity_checked=True,
        cold_olid_used=False,
        script_sha256=digest(__file__),
        model_source_sha256=digest(
            Path(__file__).with_name("compact_healthy_model.py")
        ),
    )
    identity = out / "cache_protocol.json"
    if identity.exists():
        assert read(identity) == protocol
    else:
        identity.write_text(json.dumps(protocol, indent=2))
    if (out / "complete.json").exists():
        done = read(out / "complete.json")
        assert done["protocol_sha256"] == digest(identity)
        assert done["maps_sha256"] == digest(out / "maps.npy")
        print(json.dumps(done), flush=True)
        return
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec, holder = primary(p)
    model = holder.primary.cuda().eval().requires_grad_(False)
    rows = read(source / "inputs.json")
    pixels = open_pixels(p / "phase10_leaf/cnn_inputs")
    maps = np.lib.format.open_memmap(
        out / "maps.partial.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 576, 7, 7),
    )

    def prepare(row):
        if row["kind"] == "old":
            value = np.array(pixels[row["index"]], copy=True)[None]
        else:
            with Image.open(row["path"]) as im:
                rgb = np.asarray(im.convert("RGB"))
            value = (
                classifier_letterbox_input
                if row["policy"] == "full"
                else canon.classifier_input
            )(rgb, 224, row["box"])
        assert pixel_key(value) == row["input_sha256"]
        return value

    with torch.inference_mode(), concurrent.futures.ThreadPoolExecutor(8) as pool:
        for start in range(0, len(rows), 256):
            part = rows[start : start + 256]
            x = (
                torch.from_numpy(np.concatenate(list(pool.map(prepare, part)), 0))
                .cuda()
                .permute(0, 3, 1, 2)
            )
            rgb = (
                x * model.canonical_std + model.canonical_mean - model.cnn_mean
            ) / model.cnn_std
            local = model.cnn.forward_features(rgb)
            assert local.shape[1:] == (576, 7, 7) and torch.isfinite(local).all()
            maps[start : start + len(part)] = local.cpu().numpy()
            if start % 4096 == 0:
                print(
                    json.dumps(dict(cached=start + len(part), total=len(rows))),
                    flush=True,
                )
    maps.flush()
    maps._mmap.close()
    (out / "maps.partial.npy").replace(out / "maps.npy")
    if hasattr(pixels, "close"):
        pixels.close()
    report = dict(
        protocol_sha256=digest(identity),
        maps_sha256=digest(out / "maps.npy"),
        inputs=len(rows),
        shape=[len(rows), 576, 7, 7],
        dtype="float32",
        elapsed_seconds=time.monotonic() - started,
        new_dino_inference=0,
        original_encoder_weights_trained=False,
        cold_olid_used=False,
    )
    (out / "complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
