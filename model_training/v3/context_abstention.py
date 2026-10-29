"""Neutral Healthy evidence on strong contradiction; never invent a disease."""

import copy
import numpy as np
from app_eval import posterior,replay


def abstain_healthy(row,manifest):
    revised=dict(row);revised['leaves']=copy.deepcopy(row['leaves'])
    healthy=manifest['classifier']['labels'].index('Healthy')
    neutral=np.zeros(len(manifest['classifier']['labels']),np.float32)
    neutral[manifest['classifier']['other_index']]=-manifest['classifier'].get('other_logit_bias',0.)
    for leaf in revised['leaves']:
        if int(posterior(leaf['member_logits'],manifest).argmax())!=healthy:continue
        leaf['member_logits']=[neutral.tolist()]
        leaf['probabilities']=posterior(leaf['member_logits'],manifest).tolist()
        leaf['healthy_abstention']=True
    revised['app_replay']=replay(revised['leaves'],manifest,revised['quality_weight'])
    revised['healthy_context_abstention']=True
    return revised


def route_context(row, reference, manifest, positive, negative, probability,
                  blocked, reference_blocked):
    """Research image-evidence routing; thresholds/models are selected elsewhere.

    Image evidence never supplies a disease name or fictional detected leaf.
    A negative whole-image check neutralizes Healthy evidence, not disease mass.
    """
    revised=dict(row)
    if revised['contrast']<.015 and revised['sharpness']<.012:return revised
    if negative and revised['app_replay']['kind']=='HEALTHY':
        revised=abstain_healthy(revised,manifest)
    if (revised['app_replay']['kind']=='HEALTHY' and
        reference['app_replay']['kind']!='HEALTHY' and not positive):
        revised['leaves']=reference['leaves']
        revised['app_replay']=reference['app_replay']
        blocked=reference_blocked
    if revised['app_replay']['kind'] not in ('HEALTHY','DISEASE') and not blocked and positive:
        assert 0.<=probability<=1.
        probabilities=np.full(9,(1-probability)/8,np.float32)
        probabilities[1]=probability
        revised['whole_frame_evidence']=dict(probabilities=probabilities.tolist(),scope='whole_frame')
        revised['app_replay']=dict(kind='HEALTHY',label='Healthy',confidence=probability,
            probabilities=probabilities.tolist(),margin=float(probability-(1-probability)/8),
            frames=7,leaves=0,partial=False,scope='whole_frame')
    return revised
