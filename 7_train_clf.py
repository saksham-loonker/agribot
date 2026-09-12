"""
7_train_clf.py
==============
Fine-tunes YOLO11 on Tomato-Village for 8-class
tomato disease classification (Variant-a: Multiclass).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY YOLO11 — NOT EfficientNet, MobileNet, or ResNet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Recent research (2025-2026) on the Tomato-Village dataset
  shows YOLO11 consistently outperforms traditional CNNs for
  plant disease classification:

  1. C3k2 blocks (Cross-Stage Partial, 2 conv kernels):
     Better gradient flow than standard residual blocks.
     Each stage splits channels, processes half through
     bottleneck convolutions, then concatenates — giving
     richer multi-scale features with fewer parameters.

  2. C2PSA (Partial Spatial Attention):
     Built-in channel + spatial attention mechanism that
     automatically focuses on disease-relevant regions
     (spots, lesions, discoloration) and suppresses
     background noise (stems, soil, shadows).
     This is analogous to the Attention-Guided Multi-Scale
     Feature Fusion (AGMS-FF) enhancer from recent
     literature on tomato disease classification.

  3. SPPF (Spatial Pyramid Pooling – Fast):
     Aggregates features at multiple receptive-field scales
     in a single pass, capturing both small lesion spots
     and large blight regions simultaneously.

  4. Ultralytics training pipeline:
     Battle-tested training loop with automatic EMA,
     cosine LR, AMP, augmentation, and checkpoint saving.
     Eliminates manual bugs (like our EMA-per-epoch issue).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL ARCHITECTURE — YOLO11m-cls (10.1M params)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Input (256×256×3)
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Stem: Conv(3×3, s=2) → BN → SiLU  →  128×128        │
  │        Conv(3×3, s=2) → BN → SiLU  →   64×64         │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Backbone (CSPDarknet with C3k2 + attention):          │
  │                                                        │
  │  Stage 1: C3k2(c=128)  ×2  →  64×64                  │
  │  Stage 2: C3k2(c=256)  ×2  →  32×32                  │
  │  Stage 3: C3k2(c=512)  ×2  →  16×16                  │
  │  Stage 4: C3k2(c=512)  ×2  →   8×8                   │
  │           + SPPF(k=5)       →   8×8                   │
  │           + C2PSA (attn)    →   8×8                   │
  │                                                        │
  │  C3k2  : split→bottleneck×2→concat→conv               │
  │  C2PSA : channel squeeze → spatial attn → excite      │
  │  SPPF  : MaxPool(5)→MaxPool(5)→MaxPool(5)→concat     │
  └─────┬──────────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────────┐
  │  Classify Head:                                        │
  │  Conv(1×1) → Adaptive AvgPool → 1280                  │
  │  Dropout(p=0.3)                                        │
  │  Linear(1280 → 8)   ← disease classes                 │
  └─────┬──────────────────────────────────────────────────┘
        │
  Class probabilities (softmax during inference)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING STRATEGY (automatic via Ultralytics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ▸ Transfer learning from ImageNet-pretrained YOLO11m-cls
  ▸ AdamW optimizer with cosine LR decay
  ▸ 10-epoch linear warmup
  ▸ EMA (Exponential Moving Average) — automatic per-batch
  ▸ AMP (Automatic Mixed Precision) for fast training
  ▸ Early stopping with patience=100 epochs
  ▸ Best model saved on top-1 val accuracy
  ▸ Augmentation: RandAugment + HSV jitter + flips +
    random erasing — all tuned for leaf disease images

RUN:
  python 7_train_clf.py
  python 7_train_clf.py --epochs 500 --batch 64
  python 7_train_clf.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT      = "./clf_dataset"
PROJECT_DIR    = "./clf_runs"
RUN_NAME       = "yolo11_tomatovillage_v2"
MODEL_NAME     = "yolo11l-cls.pt"    # l=large (14.1M params, more capacity)
                                     # Options: yolo11{n,s,m,l,x}-cls.pt

IMG_SIZE       = 384        # higher res → more spatial detail for disease texture
EPOCHS         = 100
BATCH          = 32         # larger model + 384px → lower batch
PATIENCE       = 150        # give more room to explore late improvements
SEED           = 42
WORKERS        = 4
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Train YOLO11 classifier on Tomato-Village")
    p.add_argument("--data",    default=DATA_ROOT,
                   help="Path to clf_dataset/")
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    p.add_argument("--batch",   type=int, default=BATCH)
    p.add_argument("--model",   default=MODEL_NAME,
                   help="YOLO11 cls variant: yolo11{n,s,m,l,x}-cls.pt")
    p.add_argument("--imgsz",   type=int, default=IMG_SIZE)
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint")
    p.add_argument("--name",    default=RUN_NAME)
    return p.parse_args()


def print_dataset_stats(data_root: Path):
    """Print per-class image counts for each split."""
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        counts = {c: len(list((split_dir / c).glob("*"))) for c in classes}
        total = sum(counts.values())
        print(f"  {split:5s}: {total:4d} images  ({len(classes)} classes)")
        for c in sorted(counts, key=lambda x: counts[x]):
            print(f"         {c}: {counts[c]}")
    print()


def main():
    args = parse_args()
    data_root = Path(args.data)

    if not data_root.exists():
        raise SystemExit(
            f"[ERROR] Dataset not found: {data_root}\n"
            f"        Run python 6_prepare_clf.py first.")

    try:
        from ultralytics import YOLO, __version__ as ul_ver
    except ImportError:
        raise SystemExit(
            "[ERROR] ultralytics not installed.\n"
            "        pip install ultralytics>=8.3.0")

    print(f"\n{'='*60}")
    print(f"  Tomato Disease Classifier — YOLO11 Training")
    print(f"{'='*60}")
    print(f"  Ultralytics : v{ul_ver}")
    print(f"  Model       : {args.model}")
    print(f"  Dataset     : Tomato-Village (Variant-a Multiclass)")
    print(f"  Input       : {args.imgsz}×{args.imgsz}px")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch       : {args.batch}")
    print(f"  Resume      : {args.resume}")
    print(f"  Output      : {PROJECT_DIR}/{args.name}/")
    print(f"{'='*60}\n")

    print_dataset_stats(data_root)

    # ── Load model ────────────────────────────────────────────────────────────
    weights_dir = Path(PROJECT_DIR) / args.name / "weights"

    if args.resume:
        last_pt = weights_dir / "last.pt"
        if not last_pt.exists():
            raise SystemExit(f"[ERROR] No checkpoint to resume: {last_pt}")
        print(f"Resuming from {last_pt}\n")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=str(data_root),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=PATIENCE,

            # Optimizer
            optimizer="AdamW",
            lr0=5e-4,           # lower LR for larger model + higher res
            lrf=0.005,          # final LR = lr0 × 0.005 = 2.5e-6
            weight_decay=1e-3,  # stronger weight decay for regularization
            warmup_epochs=15,   # longer warmup for larger model
            cos_lr=True,

            # Regularization
            dropout=0.4,        # higher dropout for smaller dataset
            label_smoothing=0.1,# stronger label smoothing

            # Augmentation (aggressive for leaf disease images)
            hsv_h=0.02,         # hue jitter
            hsv_s=0.5,          # saturation jitter (lighting variation)
            hsv_v=0.5,          # brightness jitter
            fliplr=0.5,         # horizontal flip
            flipud=0.5,         # vertical flip (crops have no preferred orientation)
            erasing=0.3,        # stronger random erasing (simulates occlusion)
            auto_augment="randaugment",  # Automated augmentation policy
            scale=0.3,          # random scale ±30% (size variation)
            translate=0.1,      # random translate ±10%

            # Output
            project=PROJECT_DIR,
            name=args.name,
            exist_ok=True,
            pretrained=True,
            seed=SEED,
            workers=WORKERS,
            amp=True,
            verbose=True,
            plots=True,         # save training curves + confusion matrix
        )

    # ── Find actual save dir (YOLO may prepend runs/classify/) ────────────────
    expected_dir = Path(PROJECT_DIR) / args.name
    # Check common locations
    candidates = [
        expected_dir,
        Path("runs") / "classify" / PROJECT_DIR / args.name,
        Path("runs") / "classify" / args.name,
    ]
    run_dir = expected_dir  # fallback
    for cand in candidates:
        if (cand / "weights" / "best.pt").exists():
            run_dir = cand
            break

    # If YOLO saved elsewhere, move it to expected location
    if run_dir != expected_dir and run_dir.exists():
        expected_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for item in run_dir.iterdir():
            dest = expected_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        run_dir = expected_dir

    # Save class mapping for 9_infer_clf.py compatibility
    if hasattr(model, "names") and model.names:
        class_map = {str(k): v for k, v in model.names.items()}
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "class_map.json", "w") as f:
            json.dump(class_map, f, indent=2)

    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights  : {best_pt}")
    print(f"  Last checkpoint: {last_pt}")
    print(f"\n  Next steps:")
    print(f"    python 8_eval_clf.py           # test-set evaluation")
    print(f"    python 9_infer_clf.py --folder ./output/<stem>/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
