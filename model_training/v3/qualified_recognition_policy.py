"""Data-source-independent scalar routing for the frozen phase-69 candidate."""

from context_guard import route_guard


def route(row, manifest, whole, leaf_views, whole_dino=None, observation_kind=None):
    """Each real leaf supplies center/full [positive H, negative H, adapted H].

    This function never reads labels, publisher names or file identities. Image
    evidence is primary-view evidence; live apps must retain actual frame and
    leaf counts rather than reuse research replay counts as observations.
    """
    hp, nh, genus, context = map(float, whole)
    original_kind = row["app_replay"]["kind"]
    canonical_kind = original_kind if observation_kind is None else observation_kind
    minimum_positive = min((v[0] for leaf in leaf_views for v in leaf), default=1.0)
    minimum_negative = min(
        (max(v[1] for v in leaf) for leaf in leaf_views), default=1.0
    )
    leaf_veto = minimum_negative <= 0.001
    scoped = context >= 0.5 and genus >= 0.5
    eligible = len(leaf_views) >= 2 if context >= 0.5 else len(leaf_views) >= 1
    eligible = (
        eligible
        and canonical_kind not in ("DISEASE", "HEALTHY")
        and genus >= 0.5
        and hp >= 0.85
        and minimum_positive >= 0.1
    )
    conflict = False
    isolated = False
    if eligible:
        negative = [max(v[2] for v in leaf) for leaf in leaf_views]
        conflict = (
            any(v <= 0.2 for v in negative) or sum(v <= 0.3 for v in negative) >= 2
        )
        if context < 0.5:
            isolated = min(v[2] for leaf in leaf_views for v in leaf) >= 0.95
    positive_allowed = (
        (scoped or isolated)
        and minimum_positive >= 0.1
        and not conflict
        and not leaf_veto
    )
    candidate = route_guard(
        row,
        manifest,
        hp if positive_allowed else 0.0,
        0.0 if leaf_veto else nh if scoped else 1.0,
        genus,
        dict(
            healthy_min=0.85,
            negative_healthy_max=0.001,
            genus_min=0.5,
            protect_suspected_disease=True,
            in_domain_other_max=0.1,
            in_domain_disease_mass_min=0.6,
            preserve_original_healthy=not (scoped or leaf_veto),
        ),
    )
    if (
        original_kind != "HEALTHY"
        and candidate["app_replay"]["kind"] == "HEALTHY"
        and sum(leaf[0][2] < 0.5 for leaf in leaf_views) >= 2
    ):
        assert (
            whole_dino is not None
        ), "Whole-primary corroboration is required for local conflicts"
        if whole_dino < 0.5:
            return row
    return candidate
