"""Freeze executable research source while preserving canonical import layout."""

from pathlib import Path
import argparse
import hashlib
import json


def snapshot(repo, destination):
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    files = []
    for directory in [
        "model_training/v3",
        "model_training/v2/final",
        "tests/model_evaluation",
    ]:
        files.extend(sorted((repo / directory).glob("*.py")))
    manifest = {}
    for source in files:
        relative = source.relative_to(repo)
        content = source.read_text()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_text() == content, (
                "Frozen code changed; version the snapshot",
                str(relative),
            )
        else:
            target.write_text(content)
        manifest[relative.as_posix()] = hashlib.sha256(content.encode()).hexdigest()
    for required in [
        "model_training/v3/app_eval.py",
        "model_training/v2/final/canon.py",
        "tests/model_evaluation/canonical_runtime.py",
    ]:
        assert required in manifest, (
            "Missing executable canonical dependency",
            required,
        )
    path = destination / "source_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    return destination / "model_training/v3"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(snapshot(args.repo, args.out))
