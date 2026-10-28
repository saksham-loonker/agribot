"""Preserve binary/source mass while balancing clinical conditions and families."""
from collections import Counter,defaultdict
import numpy as np


def clinical_replay_weights(rows,indices):
    keys=[(rows[i]['y'],rows[i]['source'],rows[i]['label'],rows[i]['group']) for i in indices]
    if not keys:raise ValueError('Cannot balance an empty partition')
    sizes=Counter(keys);sources=defaultdict(set);conditions=defaultdict(set);families=defaultdict(set)
    for y,source,label,family in sizes:
        if y not in (0,1) or bool(y)!=(label=='Healthy'):raise ValueError('Binary and clinical supervision disagree')
        sources[y].add(source);conditions[y,source].add(label);families[y,source,label].add(family)
    values=np.array([1/(len(sources)*len(sources[y])*len(conditions[y,source])*len(families[y,source,label])*sizes[y,source,label,family]) for y,source,label,family in keys],dtype=np.float64)
    return values/values.sum()
