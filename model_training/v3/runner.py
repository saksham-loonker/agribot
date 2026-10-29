"""Persistent-optimizer, resumable runner for conservative classifier adaptation."""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F

from train import LABELS, MEAN, STD, sampling_weights, gpu_augment, predict, metrics_from_logits, set_seed
from training_support import (sha256, load_initial_weights, validate_cache, set_trainable,
    parameter_groups, capture_rng, restore_rng, assert_resume_compatible, retention_pass, atomic_save)
from retention import reliable_retention


def run(args):
    if args.bs < 1 or args.epochs < 1 or args.img < 14 or args.unfreeze_blocks < -1:
        raise ValueError('Invalid batch/epoch/image/unfreeze configuration')
    if args.init_checkpoint and args.no_pretrained:
        raise ValueError('--init-checkpoint and --no-pretrained are mutually exclusive')
    if args.healthy_weight <= 0 or args.retention_alpha < 0 or args.retention_temperature <= 0:
        raise ValueError('Invalid supervised weighting/retention configuration')
    if args.retention_alpha and not args.init_checkpoint:
        raise ValueError('Retention requires a verified original --init-checkpoint')
    set_seed(args.seed)
    torch.set_num_threads(min(8, __import__('os').cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    work = Path(args.work)
    run_dir = Path(args.runs) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(run_dir.iterdir()):
        raise ValueError('Run directory is not empty; select a new run name or --resume')
    manifest_path = work / 'data_manifest.json'
    manifest_sha = sha256(manifest_path)
    records = [x for x in json.loads(manifest_path.read_text()) if x['split'] in ('train', 'val', 'calib')]
    cache_path = work / f'cache_{args.cache_size}.npy'
    cache = np.load(cache_path, mmap_mode='r')
    validate_cache(cache, records, LABELS, args.cache_size)
    cache_sha = sha256(cache_path)
    meta = {k: [r[k] for r in records] for k in ('label', 'split', 'source', 'group')}
    device = torch.device('cuda')
    images = torch.from_numpy(np.array(cache)).to(device)
    labels = torch.tensor([LABELS.index(x['label']) for x in records], device=device)
    train, weights, prior = sampling_weights(meta, args)
    train, weights = train.to(device), weights.to(device)
    val_idx = torch.tensor([i for i, x in enumerate(records) if x['split'] == 'val'], device=device)
    steps = math.ceil(len(train) / args.bs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    model = timm.create_model(args.model, pretrained=not args.no_pretrained and not args.init_checkpoint and not args.resume,
                              num_classes=len(LABELS), img_size=args.img, drop_path_rate=args.drop_path).to(device)
    initialization = dict(kind='pretrained-backbone retraining' if not args.no_pretrained else 'random initialization')
    if args.init_checkpoint and not args.resume:
        initialization = load_initial_weights(model, args.init_checkpoint, LABELS, args.model, args.img)
    teacher = None
    if args.retention_alpha:
        teacher = timm.create_model(args.model, pretrained=False, num_classes=len(LABELS), img_size=args.img).to(device).eval()
        load_initial_weights(teacher, args.init_checkpoint, LABELS, args.model, args.img)
        teacher.requires_grad_(False)
    groups = parameter_groups(model, args.lr_head, args.lr_backbone, args.layer_decay, args.wd)
    optimizer = torch.optim.AdamW(groups, fused=True)
    total = args.epochs * steps
    warm_steps = args.warmup_epochs * steps
    def lr_scale(step):
        if warm_steps and step < warm_steps:
            return max(0.05, (step + 1) / warm_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1., (step - warm_steps) / max(1, total - warm_steps))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    ema = timm.utils.ModelEmaV3(model, decay=args.ema_decay, use_warmup=True)
    start, global_step, best_score = 0, 0, -float('inf')
    baseline = None
    if args.resume:
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        assert_resume_compatible(state, vars(args), manifest_sha, cache_sha)
        model.load_state_dict(state['model'], strict=True)
        ema.module.load_state_dict(state['ema'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start, global_step, best_score = state['epoch'] + 1, state['global_step'], state['best_score']
        initialization, baseline = state['initialization'], state['baseline_metrics']
        restore_rng(state['rng'])
    if baseline is None:
        baseline, _ = metrics_from_logits(predict(model, images, val_idx, args.img, precision=args.precision), labels[val_idx])
    config = dict(vars(args))
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    identity = dict(args=config, labels=LABELS, data_manifest_sha256=manifest_sha,
                    cache_sha256=cache_sha, effective_prior=prior.tolist(), initialization=initialization)
    if not args.resume:
        atomic_save(dict(model=model.state_dict(), epoch=-1, score=baseline['macro_recall'], **identity), run_dir / 'baseline.pt')
        # The unchanged classifier is an eligible fallback, never an accidental random-head replacement.
        if args.init_checkpoint:
            best_score = baseline['macro_recall'] + .5 * baseline['healthy_recall'] - .5 * baseline['false_healthy_rate']
            atomic_save(dict(model=model.state_dict(), epoch=-1, score=best_score, **identity), run_dir / 'best.pt')
    (run_dir / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))
    print(json.dumps(dict(event='baseline', initialization=initialization, metrics=baseline)), flush=True)
    counts = np.bincount(labels[train].cpu().numpy(), minlength=len(LABELS))
    cw = np.sqrt(len(train) / (len(LABELS) * np.maximum(counts, 1)))
    cw = np.clip(cw / cw.mean(), 1 / args.max_class_weight, args.max_class_weight)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if args.loss == 'weighted' else None
    if args.healthy_weight != 1:
        class_weights = class_weights if class_weights is not None else torch.ones(len(LABELS), device=device)
        class_weights[LABELS.index('Healthy')] *= args.healthy_weight
    log_prior = torch.tensor(np.log(np.clip(prior, 1e-6, None)), dtype=torch.float32, device=device)
    forward = torch.compile(model) if args.compile else model
    meta_out = dict(model=args.model, img=args.img, num_classes=len(LABELS), classes=LABELS,
                    **{k:v for k,v in identity.items() if k != 'labels'},
                    environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                                     gpu=torch.cuda.get_device_name(), timm=timm.__version__, platform=platform.platform()))
    for epoch in range(start, args.epochs):
        set_trainable(model, epoch < args.warmup_epochs, args.unfreeze_blocks)
        # Frozen features must stay deterministic during head-only adaptation.
        model.eval()
        model.head.train()
        if epoch >= args.warmup_epochs and args.unfreeze_blocks != 0:
            for block in model.blocks:
                if any(p.requires_grad for p in block.parameters()):
                    block.train()
        order = (torch.randperm(len(train), device=device) if args.sampling == 'uniform' else
                 torch.multinomial(weights, len(train), replacement=True))
        running, ce_running, kd_running, seen, t0 = 0., 0., 0., 0, time.time()
        sampled = torch.zeros(len(LABELS), dtype=torch.int64, device=device)
        for step in range(steps):
            idx = train[order[step * args.bs:(step + 1) * args.bs]]
            if not len(idx):
                break
            xb = gpu_augment(images[idx], args.img) if not args.no_augment else normalized_cache_input(images[idx], args.img)
            y = labels[idx]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                logits = forward(xb).float()
                adjusted = logits + args.logit_tau * log_prior if args.loss == 'logit_adjusted' else logits
                supervised = F.cross_entropy(adjusted, y, weight=class_weights, label_smoothing=args.label_smoothing)
                kd = logits.sum() * 0.0
                if teacher is not None:
                    with torch.no_grad(): teacher_logits = teacher(xb).float()
                    kd = reliable_retention(logits, teacher_logits, y, LABELS.index('Healthy'), args.retention_temperature, args.retention_balance)
                loss = supervised + args.retention_alpha * kd
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            global_step += 1
            ema.update(model, step=global_step)
            running += float(loss.detach()) * len(idx)
            ce_running += float(supervised.detach()) * len(idx)
            kd_running += float(kd.detach()) * len(idx)
            seen += len(idx)
            sampled += torch.bincount(y, minlength=len(LABELS))
        online, _ = metrics_from_logits(predict(model, images, val_idx, args.img, precision=args.precision), labels[val_idx])
        metrics, _ = metrics_from_logits(predict(ema.module, images, val_idx, args.img, precision=args.precision), labels[val_idx])
        eligible, failures = retention_pass(metrics, baseline, LABELS, args.max_disease_drop, args.max_false_healthy_increase)
        if not args.init_checkpoint:
            eligible = True  # Fresh-head runs have no competent classifier to preserve; deployment gates remain separate.
        score = metrics['macro_recall'] + .5 * metrics['healthy_recall'] - .5 * metrics['false_healthy_rate']
        row = dict(epoch=epoch, phase='warmup' if epoch < args.warmup_epochs else 'finetune',
                   seconds=round(time.time()-t0, 2), loss=running / max(1, seen), val_score=score,
                   supervised_loss=ce_running/max(1,seen), retention_loss=kd_running/max(1,seen),
                   eligible=eligible, retention_failures=failures, online_metrics=online,
                   global_step=global_step, last_gradient_norm=float(grad), samples_seen=seen,
                   sampled_class_counts=dict(zip(LABELS, sampled.cpu().tolist())),
                   learning_rates=[g['lr'] for g in optimizer.param_groups],
                   peak_vram_gb=round(torch.cuda.max_memory_allocated()/1e9, 2), **metrics)
        with (run_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if eligible and score > best_score:
            best_score = score
            atomic_save(dict(model=ema.module.state_dict(), epoch=epoch, score=score, **identity), run_dir / 'best.pt')
        atomic_save(dict(schema=2, model=model.state_dict(), ema=ema.module.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=capture_rng(),
                         epoch=epoch, global_step=global_step, best_score=best_score,
                         baseline_metrics=baseline, **identity), run_dir / 'last.pt')
        meta_out['best_epoch'] = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=False)['epoch']
        (run_dir/'meta.json').write_text(json.dumps(meta_out, indent=2))
        print(json.dumps({k:row[k] for k in ('epoch','seconds','loss','accuracy','healthy_recall','false_healthy_rate','eligible','retention_failures')}), flush=True)
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            break
    print('done', run_dir, 'best', best_score, flush=True)


def normalized_cache_input(images, size):
    x = images.permute(0,3,1,2).float() / 255.
    x = F.interpolate(x, size=(size,size), mode='bilinear', antialias=True, align_corners=False)
    return (x-MEAN.to(x.device))/STD.to(x.device)
