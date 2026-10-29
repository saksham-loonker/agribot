"""Acquire pinned Abu Dhabi greenhouse source for a pre-inference scope audit.

No model is constructed, inferred or fitted here. This source remains unopened
evaluation material until a later explicitly frozen protocol. Image labels and
capture families require a separate audit before using them as ground truth.
"""

from pathlib import Path
import argparse
import collections
import hashlib
import json
import zipfile

import requests


def hash_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    root = Path(a.campaign) / "phase9_ad_source"
    root.mkdir(exist_ok=True)
    url = "https://api.figshare.com/v2/articles/31145998/versions/2"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    assert metadata["doi"] == "10.6084/m9.figshare.31145998.v2"
    assert metadata["license"]["name"] == "CC BY 4.0"
    file = next(f for f in metadata["files"] if f["name"] == "AD_Dataset.zip")
    assert file["id"] == 61331008 and file["size"] == 287295367
    assert file["computed_md5"] == "d876444b1f495bab4ceda9ff0e272d86"
    snapshot = root / "publisher_metadata.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(metadata, indent=2))
    else:
        old = json.loads(snapshot.read_text())
        assert old["doi"] == metadata["doi"] and old["files"] == metadata["files"]
    archive = root / file["name"]
    partial = archive.with_suffix(".partial")
    if not archive.exists():
        start = partial.stat().st_size if partial.exists() else 0
        with requests.get(
            file["download_url"],
            headers={"Range": f"bytes={start}-"} if start else {},
            stream=True,
            timeout=(30, 180),
        ) as stream:
            stream.raise_for_status()
            resume = start > 0 and stream.status_code == 206
            if resume:
                assert stream.headers["Content-Range"].startswith(f"bytes {start}-")
            with partial.open("ab" if resume else "wb") as output:
                for block in stream.iter_content(1 << 20):
                    output.write(block)
        assert partial.stat().st_size == file["size"]
        assert hash_file(partial, "md5") == file["computed_md5"]
        partial.replace(archive)
    assert (
        archive.stat().st_size == file["size"]
        and hash_file(archive, "md5") == file["computed_md5"]
    )
    destination = root / "files"
    destination.mkdir(exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        assert sum(i.file_size for i in bundle.infolist()) < 5 * 2**30
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            assert target.is_relative_to(base), "Unsafe archive path"
            assert (
                item.external_attr >> 16
            ) & 0o170000 != 0o120000, "Archive symlink forbidden"
            if not item.is_dir() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        output.write(block)
    images = sorted(
        f
        for f in destination.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    report = dict(
        source_doi=metadata["doi"],
        source_url=metadata["url_public_html"],
        license=metadata["license"],
        authors=metadata["authors"],
        archive_sha256=hash_file(archive),
        publisher_md5=file["computed_md5"],
        images=len(images),
        folders=dict(
            collections.Counter(str(f.parent.relative_to(destination)) for f in images)
        ),
        sample_paths=[str(f.relative_to(destination)) for f in images[:12]],
        predictions_opened=False,
        training_used=False,
        status="Downloaded and checksum verified only; label/scope, duplicate and capture-family audits still required.",
    )
    (root / "download_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
