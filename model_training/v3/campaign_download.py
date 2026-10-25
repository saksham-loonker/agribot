"""Acquire three newly selected public sources, verifying publisher SHA-256.

Archives, photos and extraction products remain outside git. The Bangladesh
binary source was initially reserved in full. Its later prospective author-train
admission and sealed author-validation subset are recorded by campaign_cnn_data.
"""

from pathlib import Path
import concurrent.futures, hashlib, json, shutil, time, zipfile
import requests

ROOT = Path("/home/genesis/agribot-data/recognition_campaign_20261024")


def download(job):
    source, item = job
    folder = ROOT / "sources" / source
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / item["filename"]
    expected = item["content_details"]["sha256_hash"]
    if (
        not target.exists()
        or hashlib.sha256(target.read_bytes()).hexdigest() != expected
    ):
        tmp = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with requests.get(
                    item["content_details"]["download_url"],
                    stream=True,
                    timeout=(20, 120),
                ) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                assert (
                    hashlib.sha256(tmp.read_bytes()).hexdigest() == expected
                ), "publisher digest mismatch"
                tmp.replace(target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    dest = folder / "images"
    with zipfile.ZipFile(target) as archive:
        assert sum(f.file_size for f in archive.infolist()) < 20_000_000_000
        for item in archive.infolist():
            path = Path(item.filename)
            if (
                item.is_dir()
                or "__MACOSX" in path.parts
                or path.suffix.lower()
                not in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".txt",
                    ".xml",
                    ".csv",
                    ".json",
                )
            ):
                continue
            output = (dest / path).resolve()
            if not output.is_relative_to(dest.resolve()):
                raise ValueError("unsafe archive path")
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists() or output.stat().st_size != item.file_size:
                with archive.open(item) as src, output.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(
        json.dumps(
            dict(
                source=source,
                file=target.name,
                bytes=target.stat().st_size,
                sha256=expected,
                verified=True,
            )
        ),
        flush=True,
    )


def main():
    assert (
        shutil.disk_usage(ROOT).free > 12_000_000_000
    ), "insufficient free space for isolated source acquisition"
    jobs = []
    for name in ["tlid", "pakistan", "binary"]:
        jobs += [
            (name, item)
            for item in json.loads((ROOT / (name + "_files.json")).read_text())
        ]
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        list(executor.map(download, jobs))
    print("All source archives verified and safely extracted", flush=True)


if __name__ == "__main__":
    main()
