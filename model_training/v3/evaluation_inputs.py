"""Pin evaluation source bytes before admitting cached predictions."""

from pathlib import Path
import hashlib


def input_identity(records):
    identifiers = [r["id"] for r in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evaluation manifest contains duplicate input IDs")
    inputs, checked = [], {}
    for record in records:
        path = Path(record["path"]).resolve()
        if path not in checked:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            checked[path] = digest.hexdigest()
        actual = checked[path]
        if record.get("file_sha256") and actual != record["file_sha256"]:
            raise ValueError(f"Evaluation source image changed: {record['id']}")
        inputs.append(dict(id=record["id"], path=str(path), sha256=actual))
    return inputs
