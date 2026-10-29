"""Equal binary class, publisher source and inferred family training mass."""

from collections import Counter, defaultdict
import numpy as np


def source_class_group_weights(rows, indices, target="y"):
    keys = [(rows[i][target], rows[i]["source"], rows[i]["group"]) for i in indices]
    if not keys:
        raise ValueError("Cannot balance an empty partition")
    sizes = Counter(keys)
    groups, sources = defaultdict(set), defaultdict(set)
    for label, source, group in sizes:
        groups[label, source].add(group)
        sources[label].add(source)
    values = np.array(
        [
            1.0
            / (
                len(sources)
                * len(sources[label])
                * len(groups[label, source])
                * sizes[label, source, group]
            )
            for label, source, group in keys
        ],
        np.float64,
    )
    return values / values.sum()
