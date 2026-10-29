"""Leaf decision keys shared by host evaluation and view-guard ablations."""

import numpy as np


def decision_key(probabilities, healthy=1, other=8):
    p = np.asarray(probabilities, dtype=np.float32)
    order = np.argsort(-p, kind="stable")
    top, second = map(int, order[:2])
    confidence = p[top]
    margin = confidence - p[second]
    if top == healthy:
        return (
            ("HEALTHY", top)
            if confidence >= np.float32(0.6) and margin >= np.float32(0.2)
            else ("UNSURE", top)
        )
    if top == other:
        return (
            ("UNKNOWN_CONDITION", top)
            if confidence >= np.float32(0.5)
            else ("UNSURE", top)
        )
    return (
        ("DISEASE", top)
        if confidence >= np.float32(0.5) and margin >= np.float32(0.15)
        else ("UNSURE", top)
    )


def needs_confirmation(reference, candidate, healthy=1, other=8, confirm_unknown=False):
    old = decision_key(reference, healthy, other)
    new = decision_key(candidate, healthy, other)
    kinds = (
        ("HEALTHY", "DISEASE", "UNKNOWN_CONDITION")
        if confirm_unknown
        else ("HEALTHY", "DISEASE")
    )
    return new[0] in kinds and old != new
