"""Inactive proposal: audit label-supported abstention corrections explicitly.

This module is deliberately NOT imported by OpenedGuardScreen. Its existing
exact-retention check and promotion decision remain unchanged unless the user
explicitly authorizes the documented alternative.
"""


NONHEALTHY_LABELS=frozenset(('Early_blight','Late_blight','Leaf Miner','Magnesium Deficiency','Nitrogen Deficiency','Pottassium Deficiency','Spotted Wilt Virus','Other','Diseased_unspecified'))


def evaluate_clinical_retention(before,after):
    if len(before)!=len(after):raise ValueError('Retention cohorts differ')
    strict=True;named=True;healthy=True;limited=True;corrections=0
    for original,revised in zip(before,after):
        if original.get('path')!=revised.get('path') or original.get('label')!=revised.get('label'):raise ValueError('Retention case identity or label changed')
        a=original['app_replay'];b=revised['app_replay'];kind=a['kind'];label=original.get('label')
        if kind not in ('HEALTHY','DISEASE'):continue
        unchanged=a==b;strict &= unchanged
        if kind=='DISEASE':named &= unchanged
        elif not isinstance(label,str) or label not in NONHEALTHY_LABELS:healthy &= unchanged
        elif not unchanged:
            allowed=b['kind']=='UNSURE' and b.get('label') in (None,'')
            limited &= allowed;corrections += int(allowed)
    return dict(original_disease_and_healthy_exact=strict,original_named_disease_exact=named,original_correct_or_unlabeled_healthy_exact=healthy,changed_false_healthy_only_abstains=limited,label_supported_abstention_corrections=corrections,
        proposed_alternative_passed=named and healthy and limited,active_promotion_gate_changed=False)
