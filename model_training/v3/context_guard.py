"""Separate image evidence can reject Healthy without changing disease logits."""

from context_abstention import route_context


def route_guard(
    row,
    manifest,
    positive_probability,
    negative_healthy_probability,
    genus_probability,
    policy,
):
    # A named reference disease always keeps its original verdict and evidence.
    if row["app_replay"]["kind"] == "DISEASE":
        return row
    if (
        policy.get("preserve_original_healthy", False)
        and row["app_replay"]["kind"] == "HEALTHY"
    ):
        return row
    if row["contrast"] < 0.015 and row["sharpness"] < 0.012:
        return row
    cutoff = policy["negative_healthy_max"]
    rejected = cutoff > 0 and negative_healthy_probability <= cutoff
    positive = (
        positive_probability >= policy["healthy_min"]
        and genus_probability >= policy["genus_min"]
        and not rejected
    )
    if (
        policy.get("protect_suspected_disease", False)
        and row["app_replay"]["kind"] == "UNSURE"
        and row["app_replay"].get("label") not in (None, "", "Healthy", "Other")
    ):
        positive = False
    if (
        policy.get("preserve_all_uncertainty", False)
        and row["app_replay"]["kind"] == "UNSURE"
    ):
        positive = False
    revised = row
    if "in_domain_other_max" in policy:
        reference = row["app_replay"].get("probabilities")
        if reference is None:
            reference = row.get("context_whole_reference_probabilities")
        if reference is None:
            positive = False
        elif (
            reference[8] <= policy["in_domain_other_max"]
            and sum(reference[i] for i in (0, 2, 3, 4, 5, 6, 7))
            >= policy["in_domain_disease_mass_min"]
        ):
            positive = False
    if rejected and row["app_replay"]["kind"] == "HEALTHY":
        revised = dict(row)
        # Keep the classifier probabilities and actual observations. This is
        # an evidence conflict, not a new disease name or fabricated leaf vote.
        revised["app_replay"] = dict(row["app_replay"], kind="UNSURE")
        revised["healthy_context_rejection"] = dict(
            healthy_probability=float(negative_healthy_probability), scope="whole_frame"
        )
    return route_context(
        revised,
        row,
        manifest,
        bool(positive),
        False,
        float(positive_probability),
        False,
        False,
    )
