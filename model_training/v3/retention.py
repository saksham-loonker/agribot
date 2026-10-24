"""Retain reliable disease knowledge while allowing supervised Healthy corrections."""
import torch
import torch.nn.functional as F


def reliable_retention(student, teacher, labels, healthy_index, temperature, balance='sample'):
    if temperature <= 0:
        raise ValueError('Retention temperature must be positive')
    mask = (labels != healthy_index) & (teacher.argmax(1) == labels)
    if not mask.any():
        return student.sum() * 0.0
    terms = F.kl_div(F.log_softmax(student[mask].float()/temperature, 1),
                     F.softmax(teacher[mask].detach().float()/temperature, 1),
                     reduction='none').sum(1) * temperature**2
    if balance == 'class':
        present = labels[mask].unique()
        return torch.stack([terms[labels[mask] == k].mean() for k in present]).mean()
    return terms.mean()
