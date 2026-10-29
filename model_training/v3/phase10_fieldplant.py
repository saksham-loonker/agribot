"""Acquire the original FieldPlant tomato subset from its public export mirror.

The ZIP's embedded README identifies the original CC BY 4.0 Roboflow project.
Only catalog/CSV reads happen without --download. Selected ZIP members are read
with HTTP Range and checked against their ZIP CRC before atomic storage. No
diagnostic model is loaded or inferred, and no annotations are remapped here.
"""

from pathlib import Path
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import struct
import threading
import zlib
import zipfile

import requests

SOURCE_API = "https://www.kaggle.com/api/v1/datasets/download/bloox2/fieldplant?datasetVersionNumber=1"


def create_catalog(root):
    session = requests.Session()
    with session.get(
        SOURCE_API, headers={"Range": "bytes=-1"}, stream=True, timeout=(30, 90)
    ) as response:
        response.raise_for_status()
        assert response.status_code == 206
        length = int(response.headers["Content-Range"].split("/")[-1])
        location = response.url
    assert length == 5912207630, "Pinned export archive identity changed"

    class Remote(io.RawIOBase):
        def __init__(self):
            self.position = 0

        def seekable(self):
            return True

        def tell(self):
            return self.position

        def seek(self, offset, whence=0):
            self.position = (
                offset
                if whence == 0
                else self.position + offset if whence == 1 else length + offset
            )
            assert 0 <= self.position <= length
            return self.position

        def read(self, n=-1):
            n = length - self.position if n < 0 else min(n, length - self.position)
            if not n:
                return b""
            assert 0 < n < 20 * 2**20, "Only bounded ZIP metadata reads permitted"
            start = self.position
            end = start + n - 1
            with session.get(
                location,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 90),
            ) as response:
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["Content-Range"].startswith(
                    f"bytes {start}-{end}/"
                )
                data = response.content
            assert len(data) == n
            self.position += n
            return data

    with zipfile.ZipFile(Remote()) as bundle:
        entries = [
            dict(
                name=i.filename,
                size=i.file_size,
                compressed=i.compress_size,
                offset=i.header_offset,
                crc=i.CRC,
            )
            for i in bundle.infolist()
        ]
    assert len(entries) == 5159
    (root / "archive_catalog.json").write_text(
        json.dumps(
            dict(source_api=SOURCE_API, archive_bytes=length, entries=entries), indent=2
        )
    )


class Archive:
    def __init__(self, root):
        self.catalog = json.loads((root / "archive_catalog.json").read_text())
        self.local = threading.local()
        with requests.get(
            self.catalog["source_api"],
            headers={"Range": "bytes=-1"},
            stream=True,
            timeout=(30, 90),
        ) as response:
            response.raise_for_status()
            assert response.status_code == 206
            assert (
                int(response.headers["Content-Range"].split("/")[-1])
                == self.catalog["archive_bytes"]
            )
            self.location = response.url
            self.etag = response.headers.get("ETag")

    def read(self, entry):
        start = entry["offset"]
        end = min(self.catalog["archive_bytes"] - 1, start + entry["compressed"] + 8191)
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        with self.local.session.get(
            self.location,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=(30, 120),
        ) as response:
            response.raise_for_status()
            assert response.status_code == 206
            assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
            data = response.content
        header = struct.unpack("<4s5H3I2H", data[:30])
        assert header[0] == b"PK\x03\x04"
        method = header[3]
        name_length, extra_length = header[-2:]
        assert data[30 : 30 + name_length].decode("utf8") == entry["name"]
        position = 30 + name_length + extra_length
        assert position + entry["compressed"] <= len(data)
        raw = data[position : position + entry["compressed"]]
        assert method in (0, 8)
        result = zlib.decompress(raw, -15) if method == 8 else raw
        assert len(result) == entry["size"] and zlib.crc32(result) == entry["crc"]
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    root = Path(a.campaign) / "phase10_fieldplant"
    root.mkdir(exist_ok=True)
    if not (root / "archive_catalog.json").exists():
        create_catalog(root)
    archive = Archive(root)
    entries = {e["name"]: e for e in archive.catalog["entries"]}
    for name in ["README.dataset.txt", "README.roboflow.txt"]:
        raw = archive.read(entries[name])
        (root / name).write_bytes(raw)
    readme = (root / "README.dataset.txt").read_text()
    assert (
        "https://universe.roboflow.com/plant-disease-detection/fieldplant" in readme
        and "License: CC BY 4.0" in readme
    )
    table_name = next(name for name in entries if name.endswith(".csv"))
    raw = archive.read(entries[table_name])
    (root / "original_annotations.csv").write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert rows
    print("columns", list(rows[0]), flush=True)
    print("classes", dict(collections.Counter(r["class"] for r in rows)), flush=True)
    filenames = {r["filename"] for r in rows if r["class"].lower().startswith("tomato")}
    subset = [r for r in rows if r["filename"] in filenames]
    selected = []
    for filename in sorted(filenames):
        name = "train/" + filename
        assert name in entries and Path(filename).name == filename
        selected.append(entries[name])
    report = dict(
        source_api=archive.catalog["source_api"],
        original_source="https://universe.roboflow.com/plant-disease-detection/fieldplant",
        paper_doi="10.1109/ACCESS.2023.3263042",
        license="CC BY 4.0",
        archive_bytes=archive.catalog["archive_bytes"],
        archive_etag=archive.etag,
        archive_catalog_sha256=hashlib.sha256(
            (root / "archive_catalog.json").read_bytes()
        ).hexdigest(),
        annotation_csv_sha256=hashlib.sha256(raw).hexdigest(),
        tomato_photos=len(filenames),
        tomato_annotations=len(subset),
        annotation_classes=dict(collections.Counter(r["class"] for r in subset)),
        compressed_subset_bytes=sum(e["compressed"] for e in selected),
        predictions_opened=False,
        training_used=False,
        scope="Original export, auto-oriented pixels and EXIF stripped, no image augmentation according to embedded publisher export README. Annotation-role, overlap and capture-family audits required before training.",
    )
    (root / "subset_annotations.json").write_text(json.dumps(subset, indent=2))
    (root / "acquisition_protocol.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not a.download:
        return
    destination = root / "images"
    destination.mkdir(exist_ok=True)

    def download(entry):
        target = destination / Path(entry["name"]).name
        if target.exists():
            data = target.read_bytes()
            if len(data) == entry["size"] and zlib.crc32(data) == entry["crc"]:
                return target.name
        data = archive.read(entry)
        temp = target.with_suffix(".partial")
        temp.write_bytes(data)
        temp.replace(target)
        return target.name

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i, name in enumerate(pool.map(download, selected)):
            if i % 50 == 0:
                print(
                    json.dumps(dict(downloaded=i + 1, total=len(selected))), flush=True
                )
    report["downloaded"] = len(selected)
    report["all_member_crcs_verified"] = True
    report["full_archive_sha256_computed"] = False
    (root / "download_complete.json").write_text(json.dumps(report, indent=2))
    print("Subset download complete", flush=True)


if __name__ == "__main__":
    main()
