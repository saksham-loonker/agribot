"""Training invariants shared by the runner and recovery regression tests."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_weights(state, labels, model_name=None, image_size=None, metadata=None):
    """Require semantic compatibility, including for a legacy bare state_dict."""
    meta = state if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict) else metadata
    if meta is None:
        raise ValueError('Bare weights need a sibling meta.json with classes/model/img')
    classes = meta.get('labels', meta.get('classes'))
    if classes != labels:
        raise ValueError('Checkpoint label order differs; implicit positional head remapping is forbidden')
    config = meta.get('args', meta)
    architecture = config.get('model', meta.get('model'))
    resolution = config.get('img', meta.get('img'))
    if model_name is not None and architecture != model_name:
        raise ValueError('Checkpoint architecture mismatch')
    if image_size is not None and resolution != image_size:
        raise ValueError('Checkpoint image size mismatch')
    weights = state['model'] if meta is state else state
    return {k.removeprefix('_orig_mod.').removeprefix('module.'): v for k, v in weights.items()}


def load_initial_weights(model, path, labels, model_name, image_size):
    path = Path(path)
    # Only owner-provided/local trusted checkpoints should be supplied here.
    state = torch.load(path, map_location='cpu', weights_only=True)
    meta_path = path.with_name('meta.json')
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
    weights = checkpoint_weights(state, labels, model_name, image_size, metadata)
    model.load_state_dict(weights, strict=True)
    return dict(kind='fine-tuning existing classifier', checkpoint=str(path.resolve()),
                sha256=sha256(path), metadata_sha256=sha256(meta_path) if metadata else None)


def validate_cache(cache, records, labels, size):
    if cache.dtype != np.uint8 or cache.shape != (len(records), size, size, 3):
        raise ValueError(f'Cache must be uint8 NHWC {(len(records), size, size, 3)}, got {cache.dtype} {cache.shape}')
    groups = {}
    for index, row in enumerate(records):
        if row.get('cache_index') != index or row['label'] not in labels:
            raise ValueError(f'Cache/label alignment failure at {index}')
        previous = groups.setdefault(row['group'], row['split'])
        if previous != row['split']:
            raise ValueError(f'Group crosses development splits: {row["group"]}')
    for split in ('train', 'val'):
        if not any(row['split'] == split for row in records):
            raise ValueError(f'Empty required split: {split}')


def set_trainable(model, warmup, unfreeze_blocks):
    depth = len(getattr(model, 'blocks', []))
    for name, param in model.named_parameters():
        enabled = name.startswith('head.')
        if not warmup and unfreeze_blocks != 0:
            if unfreeze_blocks < 0:
                enabled = True
            elif name.startswith('blocks.'):
                enabled = int(name.split('.')[1]) >= max(0, depth - unfreeze_blocks)
            elif name.startswith('norm.'):
                enabled = True
        param.requires_grad_(enabled)


def parameter_groups(model, head_lr, backbone_lr, decay, weight_decay):
    """Include every parameter once so later unfreezing does not reset AdamW."""
    if not 0 < decay <= 1 or head_lr <= 0 or backbone_lr <= 0:
        raise ValueError('Learning rates must be positive and layer decay in (0,1]')
    n = len(getattr(model, 'blocks', []))
    buckets = {}
    for name, param in model.named_parameters():
        if name.startswith('head.'):
            lr = head_lr
        else:
            depth = int(name.split('.')[1]) + 1 if name.startswith('blocks.') else (n + 1 if name.startswith('norm.') else 0)
            lr = backbone_lr * decay ** max(0, n + 1 - depth)
        wd = 0.0 if param.ndim <= 1 or name.endswith('.bias') else weight_decay
        buckets.setdefault((lr, wd), []).append(param)
    return [dict(params=params, lr=lr, weight_decay=wd) for (lr, wd), params in buckets.items()]


def capture_rng():
    numpy_state = np.random.get_state()
    # Plain primitives keep new checkpoints compatible with weights_only=True.
    numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    return dict(python=random.getstate(), numpy=numpy_state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def load_training_checkpoint(path):
    """Read local resume state, including legacy NumPy MT19937 keys, safely."""
    numpy_core = np._core if hasattr(np, '_core') else np.core
    allowed = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def assert_resume_compatible(state, config, manifest_sha, cache_sha):
    if state.get('schema') != 2:
        raise ValueError('Legacy last.pt lacks exact resume state; use --init-checkpoint for an explicit new run')
    if state['data_manifest_sha256'] != manifest_sha or state['cache_sha256'] != cache_sha:
        raise ValueError('Resume data/cache identity changed')
    mutable = {'resume', 'runs', 'work', 'run_name', 'stop_after_epoch'}
    differences = [k for k, value in config.items() if k not in mutable and state['args'].get(k) != value]
    if differences:
        raise ValueError(f'Resume configuration changed: {differences}')


def retention_pass(candidate, reference, labels, disease_drop, false_healthy_increase):
    disease = [x for x in labels if x not in ('Healthy', 'Other') and reference.get('class_counts', {}).get(x, 1)]
    failures = [x for x in disease if candidate['per_class_recall'][x] < reference['per_class_recall'][x] - disease_drop]
    if candidate['false_healthy_rate'] > reference['false_healthy_rate'] + false_healthy_increase:
        failures.append('false_healthy_rate')
    if candidate['per_class_recall']['Other'] < reference['per_class_recall']['Other'] - disease_drop:
        failures.append('Other')
    return not failures, failures


def atomic_save(state, path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)
