"""
3_train.py
==========
Fine-tunes YOLOv8m on the merged leaf dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL ARCHITECTURE — YOLOv8m
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Input Image (640×640×3)
                │
    ┌─────▼──────────────────────────────────────────────────┐
    │  BACKBONE — CSPDarknet53                               │
    │  • 53 conv layers with Cross-Stage Partial (CSP)       │
    │    connections — reuses gradients, reduces compute      │
    │  • C2f blocks (bottleneck with 2 parallel paths)        │
    │  • SPPF (Spatial Pyramid Pooling Fast) at the end      │
    │    → captures context at 5×5, 9×9, 13×13 receptive     │
    │      fields simultaneously — critical for small leaves  │
    │  • Outputs 3 feature maps at strides 8, 16, 32         │
    │    (80×80, 40×40, 20×20 for 640px input)               │
    └─────┬──────────────┬──────────────┬────────────────────┘
                │  P3 (small)  │  P4 (med)    │  P5 (large)
                │  80×80       │  40×40       │  20×20
    ┌─────▼──────────────▼──────────────▼────────────────────┐
    │  NECK — PANet (Path Aggregation Network)                │
    │  • Top-down pathway: merges deep semantics downward     │
    │  • Bottom-up pathway: merges spatial detail upward      │
    │  • C2f blocks throughout                                │
    │  → Each output scale sees BOTH high-level semantics     │
    │    AND fine spatial detail — needed for overlapping     │
    │    leaves of varying sizes                              │
    └─────┬──────────────┬──────────────┬────────────────────┘
                │              │              │
    ┌─────▼──────────────▼──────────────▼────────────────────┐
    │  HEAD — Decoupled Detection (anchor-free)               │
    │  • Separate branches for classification + regression     │
    │  • Distribution Focal Loss (DFL) for box regression     │
    │  • Task-Aligned Assigner (TAL) for label assignment     │
    │  → No hand-tuned anchors needed; adapts to leaf         │
    │    aspect ratios automatically                          │
    └────────────────────────────────────────────────────────┘
                │
    NMS (Non-Maximum Suppression, IoU threshold 0.7)
                │
    Detected leaf bounding boxes

WHY YOLOv8x SPECIFICALLY:
    n  (3.2M params)  — too shallow; misses partially occluded leaves
    s  (11.2M)        — borderline on dense multi-leaf canopies
    m  (25.9M)        — right depth + VRAM fit (8GB @ batch 16, 640px)
    l  (43.7M)        — needs 12–16GB VRAM; marginal gain for this task
    x  (68.2M) ← USE  — best accuracy; fits in 32GB VRAM @ batch 32, 1280px

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUGMENTATION RATIONALE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    mosaic=1.0    Tiles 4 training images into one composite.
                                PlantDoc images often have 2–6 leaves; mosaic
                                creates 8–24 leaf instances per training step.
                                Most important augmentation for this task.

    copy_paste=0.1  Pastes leaf instances across images.
                                    Simulates dense canopies and heavy occlusion.

    degrees=45    Leaves grow at every angle; heavy rotation.

    scale=0.5     Handles close-up vs. distant leaves (tiny to large).

    hsv_s=0.7     Saturation jitter: overcast vs. sunny days.
    hsv_v=0.4     Brightness jitter: leaf shadows, backlit foliage.

    flipud=0.2    Vertical flip: top-down vs. upward-facing shots.

    iou=0.7       During NMS, keeps boxes with up to 70% overlap
                                because leaves genuinely overlap each other.

RUN:
    python 3_train.py
    python 3_train.py --epochs 100 --batch 8   # for weaker GPU
    python 3_train.py --device cpu              # force CPU
"""

import argparse
from pathlib import Path
import sys
import torch


# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML   = "./dataset/data.yaml"
MODEL_BASE     = "yolov8n.pt"       # smaller detector for Pi-oriented deployment
EPOCHS         = 120
IMG_SIZE       = 512
BATCH          = 16                 # 32×1280px fits comfortably in 32GB VRAM
PATIENCE       = 20                 # early stop epochs without improvement
WORKERS        = 4                 # match CPU cores; prevents GPU starvation
PROJECT        = "./runs/detect"
NAME           = "crop_detector_pi_v1"
DEVICE         = 0                  # GPU device id (0 = first GPU); use "cpu" to force CPU
LOG_FILE       = "./runs/detect/train.log"
# ─────────────────────────────────────────────────────────────────────────────


class _Tee:
        """Mirror all stdout/stderr writes to a file."""
        def __init__(self, log_path: str):
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                self._file = open(log_path, "a", buffering=1)  # line-buffered
                self._stdout = sys.stdout
                self._stderr = sys.stderr
                sys.stdout = self
                sys.stderr = self

        def write(self, data):
                self._stdout.write(data)
                self._file.write(data)

        def flush(self):
                self._stdout.flush()
                self._file.flush()

        def fileno(self):          # needed by some C-level libs
                return self._stdout.fileno()

        def close(self):
                sys.stdout = self._stdout
                sys.stderr = self._stderr
                self._file.close()


def detect_device(requested):
        """Return the best available device string for Ultralytics."""
        if str(requested).lower() == "cpu":
                return "cpu"
        if torch.cuda.is_available():
                gpu_id = int(requested) if str(requested).isdigit() else 0
                gpu_name = torch.cuda.get_device_name(gpu_id)
                vram = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)
                print(f"  ✓ CUDA GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
                return gpu_id
        else:
                print("  ⚠ No CUDA GPU found — falling back to CPU.")
                print("    Make sure you have the CUDA-enabled PyTorch installed:")
                print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                return "cpu"


def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--data",    default=DATASET_YAML)
        p.add_argument("--model",   default=MODEL_BASE)
        p.add_argument("--epochs",  type=int, default=EPOCHS)
        p.add_argument("--batch",   type=int, default=BATCH)
        p.add_argument("--imgsz",   type=int, default=IMG_SIZE)
        p.add_argument("--name",    default=NAME)
        p.add_argument("--device",  default=DEVICE,
                                     help="GPU device id (0, 1, …) or 'cpu'. Default: 0")
        p.add_argument("--resume",  action="store_true",
                                     help="Resume from last checkpoint if training was interrupted")
        p.add_argument("--log",     default=LOG_FILE,
                                     help="Path to log file (appended). Default: " + LOG_FILE)
        return p.parse_args()


def main():
        args = parse_args()

        tee = _Tee(args.log)
        print(f"Logging to: {args.log}\n")

        if not Path(args.data).exists():
                raise SystemExit(
                        f"[ERROR] Dataset not found: {args.data}\n"
                        f"        Run python 2_prepare.py first."
                )

        try:
                from ultralytics import YOLO
        except ImportError:
                raise SystemExit("[ERROR] Run: pip install ultralytics")

        def load_training_model(model_ref: str):
                try:
                        return YOLO(model_ref)
                except Exception as exc:
                        model_path = Path(model_ref)
                        if model_path.suffix == ".pt" and not model_path.exists():
                                fallback = model_path.with_suffix(".yaml").name
                                print(
                                        f"[WARN] Could not load pretrained weights '{model_ref}'.\n"
                                        f"       Falling back to scratch config '{fallback}'.\n"
                                        f"       Original error: {exc}"
                                )
                                try:
                                        return YOLO(fallback)
                                except Exception as fallback_exc:
                                        raise SystemExit(
                                                f"[ERROR] Failed to load '{model_ref}' and fallback '{fallback}'.\n"
                                                f"        Original error: {exc}\n"
                                                f"        Fallback error: {fallback_exc}"
                                        )
                        raise

        # ── Detect GPU / CPU ──────────────────────────────────────────────────────
        device = detect_device(args.device)

        print(f"\n{'='*60}")
        print(f"  Leaf Detector — Training")
        print(f"{'='*60}")
        print(f"  Device : {device}  ({'GPU' if device != 'cpu' else 'CPU'})")
        print(f"  Model  : {args.model}  (COCO pretrained → fine-tune)")
        print(f"  Data   : {args.data}")
        print(f"  Epochs : {args.epochs}  (early stop patience={PATIENCE})")
        print(f"  ImgSz  : {args.imgsz}px")
        print(f"  Batch  : {args.batch}")
        print(f"  Output : {PROJECT}/{args.name}/")
        print(f"{'='*60}\n")

        if args.resume:
                # Resume from last checkpoint
                last = Path(PROJECT) / args.name / "weights" / "last.pt"
                if not last.exists():
                        raise SystemExit(f"[ERROR] No checkpoint to resume from: {last}")
                model = YOLO(str(last))
                print(f"Resuming from: {last}")
        else:
                model = load_training_model(args.model)

        results = model.train(
                data         = args.data,
                epochs       = args.epochs,
                imgsz        = args.imgsz,
                batch        = args.batch,
                patience     = PATIENCE,
                workers      = WORKERS,
                cache        = False,   # disable dataset caching
                device       = device,       # ← GPU device (0) or "cpu"
                project      = PROJECT,
                name         = args.name,
                exist_ok     = True,

                # ── Augmentation ────────────────────────────────────────────────────
                mosaic       = 1.0,    # composite 4 images → more leaves per step
                copy_paste   = 0.1,    # paste leaf instances across images
                degrees      = 45.0,   # heavy rotation — leaves grow at all angles
                scale        = 0.5,    # zoom in/out — small vs. large leaves
                fliplr       = 0.5,    # horizontal flip
                flipud       = 0.2,    # vertical flip
                hsv_h        = 0.015,  # hue jitter — species colour variation
                hsv_s        = 0.7,    # saturation — cloudy vs sunny
                hsv_v        = 0.4,    # brightness — shadows and backlit leaves
                translate    = 0.1,    # small translation jitter
                perspective  = 0.0005, # slight perspective warp

                # ── Optimiser ───────────────────────────────────────────────────────
                optimizer    = "AdamW",
                lr0          = 0.0008,
                lrf          = 0.01,   # cosine decay: final lr = lr0 * lrf
                weight_decay = 0.0005,
                warmup_epochs= 3.0,
                momentum     = 0.937,

                # ── Loss ────────────────────────────────────────────────────────────
                box          = 7.5,    # box regression loss weight
                cls          = 0.5,    # classification loss weight (low: 1 class)
                dfl          = 1.5,    # distribution focal loss weight

                # ── NMS ─────────────────────────────────────────────────────────────
                iou          = 0.7,    # keep overlapping boxes (leaves overlap)

                # ── Performance (GPU) ───────────────────────────────────────────────
                amp          = True,   # mixed-precision (FP16) — faster on RTX 5090

                # ── Misc ────────────────────────────────────────────────────────────
                plots        = True,   # saves training curves + confusion matrix
                save_period  = 10,     # checkpoint every N epochs
                resume       = args.resume,
                verbose      = True,
        )

        best = Path(PROJECT) / args.name / "weights" / "best.pt"

        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best weights : {best}")
        try:
                print(f"  mAP@0.5      : {results.results_dict['metrics/mAP50(B)']:.4f}")
                print(f"  mAP@0.5:0.95 : {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        except Exception:
                pass
        print(f"\n  Next step: python 4_infer.py --image your_photo.jpg")
        print(f"{'='*60}\n")

        tee.close()


if __name__ == "__main__":
        main()
