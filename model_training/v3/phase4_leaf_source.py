"""Seal leaf-only acceptance photos from publisher ZIP ranges, never training.

DOI 10.17632/swtt5wy2pr.1. Capture metadata describes leaves on A4 paper;
this is a separate capture domain, not evidence of unrestricted field accuracy.
Fruit classes are excluded. No source prediction is computed here.
"""

from pathlib import Path
import argparse, collections, csv, hashlib, io, json, re, time, zipfile
import requests
from campaign_data import profile

MAPPING = {
    "Healthy Leaf": "Healthy",
    "Early Blight Leaf": "Early_blight",
    "Late Blight leaf": "Late_blight",
    "Bushy Stunt Virus": "Other",
    "Leaf Curl Virus": "Other",
}


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=524288):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0
        self.block_size = block_size
        self.cache_start = 0
        self.cache = b""
        self.session = requests.Session()
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else (self.position + offset if whence == 1 else self.size + offset)
        )
        if self.position < 0:
            raise ValueError("Negative archive position")
        return self.position

    def read(self, count=-1):
        count = max(
            0,
            min(
                self.size - self.position,
                self.size - self.position if count < 0 else count,
            ),
        )
        if not count:
            return b""
        if not (
            self.cache_start <= self.position
            and self.position + count <= self.cache_start + len(self.cache)
        ):
            start = self.position
            end = min(self.size, start + max(count, self.block_size)) - 1
            for attempt in range(4):
                try:
                    response = self.session.get(
                        self.url,
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=(20, 60),
                        stream=True,
                    )
                    try:
                        if response.status_code != 206:
                            raise ValueError(
                                f"Server did not honor byte range: {response.status_code}"
                            )
                        assert (
                            response.headers.get("Content-Range")
                            == f"bytes {start}-{end}/{self.size}"
                        )
                        payload = response.raw.read(end - start + 2)
                        assert len(payload) == end - start + 1
                        self.url = response.url
                        self.cache_start = start
                        self.cache = payload
                        self.bytes_fetched += len(payload)
                    finally:
                        response.close()
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 + attempt)
        offset = self.position - self.cache_start
        result = self.cache[offset : offset + count]
        self.position += len(result)
        return result

    def close(self):
        self.session.close()
        super().close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    a = ap.parse_args()
    p = Path(a.campaign)
    out = p / "phase4_paper_leaf"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Acceptance cohort already sealed")
    metadata = p / "new_field_metadata.csv"
    rows = list(csv.DictReader(metadata.open(encoding="utf-8-sig")))
    assert len(rows) == 3500
    items = json.loads((p / "swtt5wy2pr_root_metadata.json").read_text())
    archive = next(x for x in items if x["filename"] == "Resize Dataset.zip")
    selected = []
    for category, label in MAPPING.items():
        groups = collections.defaultdict(list)
        for row in rows:
            if row["Class_Label"] != category:
                continue
            assert row["Augmented"].lower() == "no"
            number = int(re.search(r"(\d+)$", row["Object_ID"]).group())
            groups[number // 5].append(row)
        ordered = sorted(
            groups.items(),
            key=lambda x: hashlib.sha256(
                ("paper79:" + category + ":" + str(x[0])).encode()
            ).hexdigest(),
        )
        cap = 100 if label == "Healthy" else 80 if label != "Other" else 60
        for group, group_rows in ordered[:cap]:
            selected.append(
                (
                    category,
                    label,
                    group,
                    min(
                        group_rows,
                        key=lambda r: hashlib.sha256(
                            r["File_Name"].encode()
                        ).hexdigest(),
                    ),
                )
            )
    photos = []
    started = time.time()
    with HTTPRangeReader(
        archive["content_details"]["download_url"], archive["content_details"]["size"]
    ) as reader, zipfile.ZipFile(reader) as z:
        assert sum(not f.is_dir() for f in z.infolist()) == 3500
        for i, (category, label, group, meta) in enumerate(selected):
            name = "Resize Dataset/" + category + "/" + meta["File_Name"]
            info = z.getinfo(name)
            assert info.file_size < 10_000_000
            path = p / "sources/swtt5wy2pr_v1" / category / meta["File_Name"]
            assert path.resolve().is_relative_to(
                (p / "sources/swtt5wy2pr_v1").resolve()
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != info.file_size:
                path.write_bytes(z.read(info))
            # Recheck CRC even when resuming a partially downloaded acquisition.
            import binascii

            assert binascii.crc32(path.read_bytes()) & 0xFFFFFFFF == info.CRC
            row = dict(
                id="swtt5wy2pr:" + category + ":" + meta["File_Name"],
                path=str(path),
                source="swtt5wy2pr",
                species="tomato",
                label=label,
                raw_label=category,
                box=None,
                group="swtt5wy2pr:" + category + ":block" + str(group),
                split="external_acceptance",
                source_version="10.17632/swtt5wy2pr.1",
                source_url="https://data.mendeley.com/datasets/swtt5wy2pr/1",
                licence="CC BY 4.0",
                publisher_object_id=meta["Object_ID"],
                publisher_capture_date=meta["Capture_Date"],
                publisher_metadata=meta,
                archive_member=name,
                archive_crc32=info.CRC,
            )
            quality, _ = profile(row)
            row.update(quality)
            photos.append(row)
            if i % 25 == 0:
                print(
                    json.dumps(
                        dict(
                            photos=i,
                            total=len(selected),
                            network_bytes=reader.bytes_fetched,
                            seconds=time.time() - started,
                        )
                    ),
                    flush=True,
                )
        network_bytes = reader.bytes_fetched
    prior = []
    for name in [
        "original_quality_profile.json",
        "new_development_manifest.json",
        "locked_new_source_manifest.json",
        "locked_binary_manifest.json",
        "phase2_source_manifest.json",
        "phase2/slif_acceptance_manifest.json",
        "phase3_detector/source_rows.json",
        "phase4_tom2024/manifest.json",
    ]:
        prior += json.loads((p / name).read_text())
    sha = {r.get("file_sha256") for r in prior}
    inputs = {r.get("model_input_sha256") for r in prior}
    phash = {int(r["phash"], 16) for r in prior if r.get("phash")}
    eligible = []
    excluded = []
    for row in photos:
        reason = None
        if row["file_sha256"] in sha or row["model_input_sha256"] in inputs:
            reason = "exact_prior_overlap"
        elif any((int(row["phash"], 16) ^ h).bit_count() <= 2 for h in phash):
            reason = "close_prior_overlap"
        elif any(
            (int(row["phash"], 16) ^ int(r["phash"], 16)).bit_count() <= 2
            for r in eligible
        ):
            reason = "close_selected_duplicate"
        if reason:
            excluded.append(dict(id=row["id"], reason=reason))
        else:
            eligible.append(row)
    (out / "manifest.json").write_text(json.dumps(eligible, indent=2) + "\n")
    audit = dict(
        doi="10.17632/swtt5wy2pr.1",
        licence="CC BY 4.0",
        metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        archive_publisher_sha256=archive["content_details"]["sha256_hash"],
        whole_archive_sha256_verified=False,
        verification="Publisher CSV SHA verified at acquisition. Selected ZIP members checked by CRC32; local file/input SHA-256 recorded. Whole archive not downloaded.",
        network_bytes=network_bytes,
        manifest_sha256=hashlib.sha256(
            (out / "manifest.json").read_bytes()
        ).hexdigest(),
        selected=len(photos),
        eligible=len(eligible),
        counts=dict(collections.Counter(r["label"] for r in eligible)),
        excluded=excluded,
        predictions_opened=False,
        training_use=False,
        scope="Plucked leaves photographed on A4 paper according to capture metadata; fruit classes excluded",
        metadata_version_note="DOI release v1; CSV Dataset_Version field says V2",
        grouping="One image per block of five adjacent publisher Object_ID numbers; inferred capture protection, not verified plant IDs",
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
