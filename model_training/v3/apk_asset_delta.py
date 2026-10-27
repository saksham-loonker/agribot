"""Transfer an APK byte-for-byte while reusing identical local uncompressed model assets.

The recipe omits only verified ZIP_STORED payload ranges. ZIP metadata, alignment,
DEX, resources and APK signing blocks are transferred unchanged. Reconstruction
must match the complete source APK SHA-256 before it can replace any output.
"""

from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import struct
import zipfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_exact(source, destination, count, digest=None):
    while count:
        block = source.read(min(count, 1 << 20))
        if not block:
            raise ValueError("Truncated transfer input")
        destination.write(block)
        if digest is not None:
            digest.update(block)
        count -= len(block)


def prepare(apk, manifest, output):
    apk, output = Path(apk), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(manifest).read_text())
    contracts = metadata["classifier"]["members"] + [metadata["detector"]]
    if metadata.get("primary_recovery"):
        contracts += [metadata["primary_recovery"]["model"]]
    ranges = []
    with apk.open("rb") as stream, zipfile.ZipFile(apk) as archive:
        expected = {"assets/models/" + contract["file"] for contract in contracts}
        packaged = [info.filename for info in archive.infolist()
                    if info.filename.startswith("assets/models/") and not info.is_dir()]
        if set(packaged) != expected or len(packaged) != len(expected):
            raise ValueError("APK contains undeclared, duplicate or missing model assets")
        for contract in contracts:
            name = contract["file"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Invalid model asset name")
            info = archive.getinfo("assets/models/" + name)
            if (
                info.compress_type != zipfile.ZIP_STORED
                or info.compress_size != info.file_size
            ):
                raise ValueError("Model payload is not stored verbatim")
            with archive.open(info) as member:
                if (
                    hashlib.file_digest(member, "sha256").hexdigest()
                    != contract["sha256"]
                ):
                    raise ValueError("APK model does not match manifest")
            stream.seek(info.header_offset)
            header = stream.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            name_length, extra_length = struct.unpack_from("<HH", header, 26)
            offset = info.header_offset + 30 + name_length + extra_length
            ranges.append((offset, info.file_size, name, contract["sha256"]))
    segments, position = [], 0
    with apk.open("rb") as source, gzip.open(
        output / "literal.gz", "wb", compresslevel=3
    ) as literal:
        for offset, size, name, sha in sorted(ranges):
            if offset < position or offset + size > apk.stat().st_size:
                raise ValueError("Overlapping or invalid model ranges")
            count = offset - position
            segments.append(dict(literal=count))
            copy_exact(source, literal, count)
            source.seek(size, os.SEEK_CUR)
            segments.append(dict(model=name, size=size, sha256=sha))
            position = offset + size
        count = apk.stat().st_size - position
        segments.append(dict(literal=count))
        copy_exact(source, literal, count)
    recipe = dict(
        schema=1,
        apk_name=apk.name,
        apk_sha256=sha256(apk),
        apk_size=apk.stat().st_size,
        literal_sha256=sha256(output / "literal.gz"),
        segments=segments,
    )
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    return dict(
        apk_bytes=recipe["apk_size"],
        transfer_bytes=(output / "literal.gz").stat().st_size,
        apk_sha256=recipe["apk_sha256"],
    )


def reconstruct(recipe_directory, assets, target):
    directory, assets, target = Path(recipe_directory), Path(assets), Path(target)
    recipe = json.loads((directory / "recipe.json").read_text())
    if (
        recipe["schema"] != 1
        or sha256(directory / "literal.gz") != recipe["literal_sha256"]
    ):
        raise ValueError("Invalid transfer recipe or literal payload")
    for segment in recipe["segments"]:
        if "model" not in segment:
            if not isinstance(segment["literal"], int) or segment["literal"] < 0:
                raise ValueError("Invalid literal length")
            continue
        name = segment["model"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid model asset path")
        source = assets / name
        if (
            source.stat().st_size != segment["size"]
            or sha256(source) != segment["sha256"]
        ):
            raise ValueError("Local model identity mismatch: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".assembling")
    try:
        checksum = hashlib.sha256()
        with gzip.open(directory / "literal.gz", "rb") as literal, temporary.open(
            "wb"
        ) as output:
            for segment in recipe["segments"]:
                if "model" in segment:
                    with (assets / segment["model"]).open("rb") as source:
                        copy_exact(source, output, segment["size"], checksum)
                else:
                    copy_exact(literal, output, segment["literal"], checksum)
            if literal.read(1):
                raise ValueError("Unconsumed literal bytes")
        if (
            temporary.stat().st_size != recipe["apk_size"]
            or checksum.hexdigest() != recipe["apk_sha256"]
        ):
            raise ValueError("Reconstructed APK identity mismatch")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(path=str(target), apk_sha256=recipe["apk_sha256"], byte_identical=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="operation", required=True)
    pack = sub.add_parser("prepare")
    pack.add_argument("--apk", required=True)
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--output", required=True)
    restore = sub.add_parser("reconstruct")
    restore.add_argument("--recipe-directory", required=True)
    restore.add_argument("--assets", required=True)
    restore.add_argument("--target", required=True)
    args = ap.parse_args()
    result = (
        prepare(args.apk, args.manifest, args.output)
        if args.operation == "prepare"
        else reconstruct(args.recipe_directory, args.assets, args.target)
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
