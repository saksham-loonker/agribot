"""Acquire the pinned public image encoder and persist its verified metadata."""

from pathlib import Path
import argparse
import hashlib
import json

from huggingface_hub import hf_hub_download
import timm

MODEL = "fastvit_mci0.apple_mclip2_dfndr2b"
REPOSITORY = "timm/" + MODEL
REVISION = "577a2e992da2c7ba7cadc2a490bcfadfb2d69bce"
SHA256 = "70a995b7e7385126d0db94ee3d395d7fd3c251ce6d8dcd4f4c8d8fc7c29e3d04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(REPOSITORY, "model.safetensors", revision=REVISION))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
    config = timm.create_model(MODEL, pretrained=False).pretrained_cfg
    assert tuple(config["mean"]) == (0.0, 0.0, 0.0) and tuple(config["std"]) == (
        1.0,
        1.0,
        1.0,
    )
    assert tuple(config["input_size"]) == (3, 256, 256) and config["num_classes"] == 512
    source = dict(
        repository=REPOSITORY,
        revision=REVISION,
        file="model.safetensors",
        file_sha256=SHA256,
        local_file=str(path),
        model_name=MODEL,
        pretrained_cfg=config,
    )
    output = root / "mobileclip_source.json"
    if output.exists():
        old = json.loads(output.read_text())
        for key in ["repository", "revision", "file", "file_sha256", "model_name"]:
            assert old[key] == source[key]
    output.write_text(json.dumps(source, indent=2))
    print(
        json.dumps(dict(metadata=str(output), revision=REVISION, sha256=SHA256)),
        flush=True,
    )


if __name__ == "__main__":
    main()
