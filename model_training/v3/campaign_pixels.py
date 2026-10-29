"""Read immutable canonical pixel shards without duplicating the full cache."""

from pathlib import Path
import json
import numpy as np


class CompositePixels:
    def __init__(self, paths):
        self.parts = []
        self.closed = False
        try:
            self.parts = [np.load(path, mmap_mode="r") for path in paths]
            self._validate()
        except Exception:
            self.close()
            raise

    def _validate(self):
        if not self.parts or any(
            p.shape[1:] != self.parts[0].shape[1:] for p in self.parts
        ):
            raise ValueError("Pixel shard shapes differ")
        if any(p.dtype != np.float32 for p in self.parts):
            raise ValueError("Canonical shards must remain float32")
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])
        self.shape = (int(self.offsets[-1]),) + self.parts[0].shape[1:]
        self.dtype = np.dtype("float32")

    def close(self):
        for part in self.parts:
            mapping = getattr(part, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.parts = []
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise ValueError("Pixel cache is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, index):
        if self.closed:
            raise ValueError("Pixel cache is closed")
        scalar = np.isscalar(index)
        ids = (
            np.arange(self.shape[0])[index]
            if isinstance(index, slice)
            else np.asarray([index] if scalar else index, dtype=np.int64)
        )
        if ids.ndim != 1:
            raise IndexError(
                "Only scalar and one-dimensional pixel indexing is supported"
            )
        ids = np.where(ids < 0, ids + self.shape[0], ids)
        if ((ids < 0) | (ids >= self.shape[0])).any():
            raise IndexError("Pixel cache index out of range")
        output = np.empty((len(ids),) + self.shape[1:], dtype=np.float32)
        for part, start, end in zip(self.parts, self.offsets[:-1], self.offsets[1:]):
            mask = (ids >= start) & (ids < end)
            if mask.any():
                output[mask] = part[ids[mask] - start]
        return output[0] if scalar else output


def open_pixels(work, policy="center"):
    work = Path(work)
    metadata = work / "pixels.json"
    if metadata.exists():
        return CompositePixels(json.loads(metadata.read_text())[policy])
    return np.load(work / (policy + ".npy"), mmap_mode="r")
