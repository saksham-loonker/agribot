"""
8_eval_clf.py
=============
Evaluates the trained YOLO11 classifier on the held-out test split
of the Tomato-Village dataset (Variant-a: Multiclass, 8 classes).

OUTPUTS:
  clf_runs/yolo11_tomatovillage/
    eval_report.txt       — overall accuracy + per-class precision/recall/F1
    confusion_matrix.png  — 8×8 heatmap (colour-coded by error rate)

WHY A DEDICATED EVAL SCRIPT:
  Training monitors val_acc for checkpoint selection.
  Test split is NEVER seen during training or val.
  A separate eval pass on test gives an unbiased accuracy estimate.

  YOLO11's built-in val() gives top-1/top-5 accuracy but not
  per-class P/R/F1 or a detailed confusion matrix report.
  This script fills that gap.

RUN:
  python 8_eval_clf.py
  python 8_eval_clf.py --weights ./clf_runs/yolo11_tomatovillage/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = "./clf_dataset"
WEIGHTS     = "./clf_runs/yolo11_tomatovillage_v2/weights/best.pt"
REPORT_DIR  = "./clf_runs/yolo11_tomatovillage_v2"
TTA         = True         # manual TTA via flips (YOLO cls doesn't support augment=True)
IMG_SIZE    = 384
BATCH       = 32
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default=DATA_ROOT)
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--out",     default=REPORT_DIR)
    p.add_argument("--batch",   type=int, default=BATCH)
    p.add_argument("--no-tta",  action="store_true",
                   help="Disable test-time augmentation")
    return p.parse_args()


def compute_metrics(cm: np.ndarray, class_names: list):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    n = cm.shape[0]
    metrics = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        support   = int(cm[i, :].sum())
        metrics.append({
            "class":     class_names[i],
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   support,
        })
    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path):
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed — skipping confusion matrix plot.")
        return

    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    n = len(class_names)
    fig_size = max(16, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [c.replace("_", " ") for c in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix (row-normalised recall)", fontsize=12)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def write_report(
    overall_acc: float,
    metrics: list,
    worst_pairs: list,
    out_path: Path,
):
    lines = []
    lines.append("=" * 70)
    lines.append("  Tomato Disease Classifier — Evaluation Report (YOLO11)")
    lines.append("=" * 70)
    lines.append(f"\n  Overall Test Accuracy: {overall_acc*100:.2f}%\n")

    lines.append(f"  {'Class':<55} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    lines.append(f"  {'-'*83}")
    for m in sorted(metrics, key=lambda x: x["f1"]):
        lines.append(
            f"  {m['class']:<55} "
            f"{m['precision']*100:>5.1f}% "
            f"{m['recall']*100:>5.1f}% "
            f"{m['f1']*100:>5.1f}% "
            f"{m['support']:>6}"
        )

    macro_p  = sum(m["precision"] for m in metrics) / len(metrics)
    macro_r  = sum(m["recall"]    for m in metrics) / len(metrics)
    macro_f1 = sum(m["f1"]        for m in metrics) / len(metrics)
    lines.append(f"  {'-'*83}")
    lines.append(f"  {'MACRO AVG':<55} "
                 f"{macro_p*100:>5.1f}% {macro_r*100:>5.1f}% {macro_f1*100:>5.1f}%")

    lines.append(f"\n  Top-10 Most Confused Pairs:")
    lines.append(f"  {'True Class':<40} {'Predicted As':<40} {'Count':>6}")
    lines.append(f"  {'-'*90}")
    for true_cls, pred_cls, count in worst_pairs[:10]:
        lines.append(f"  {true_cls:<40} {pred_cls:<40} {count:>6}")

    report = "\n".join(lines)
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Report saved: {out_path}")


def main():
    args   = parse_args()
    use_tta = TTA and not args.no_tta
    out    = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.weights)
    if not ckpt_path.exists():
        raise SystemExit(
            f"[ERROR] Weights not found: {ckpt_path}\n"
            f"        Run python 7_train_clf.py first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] pip install ultralytics>=8.3.0")

    model = YOLO(str(ckpt_path))
    class_names = [model.names[i] for i in range(len(model.names))]
    n_classes   = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"\nLoaded: {ckpt_path}")
    print(f"Classes ({n_classes}): {', '.join(class_names)}")

    # ── Test set ──────────────────────────────────────────────────────────────
    test_dir = Path(args.data) / "test"
    if not test_dir.exists():
        raise SystemExit(f"[ERROR] Test dir not found: {test_dir}")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    tta_label = " + TTA" if use_tta else ""
    print(f"Evaluating on test set{tta_label}...\n")

    for class_name in class_names:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            print(f"  [WARN] Missing class dir: {class_dir}")
            continue

        true_idx = class_to_idx[class_name]
        img_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not img_files:
            continue

        if use_tta:
            # Manual TTA: original + hflip + vflip + hflip+vflip
            # Average softmax probabilities across all views
            for img_path in img_files:
                img = Image.open(img_path).convert("RGB")
                views = [
                    img,
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    img.transpose(Image.FLIP_TOP_BOTTOM),
                    img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
                ]
                probs_sum = np.zeros(n_classes, dtype=np.float64)
                for view in views:
                    results = model.predict(source=view, verbose=False, imgsz=IMG_SIZE)
                    probs_sum += results[0].probs.data.cpu().numpy()
                pred_idx = int(np.argmax(probs_sum))
                cm[true_idx, pred_idx] += 1
        else:
            results = model.predict(
                source=[str(p) for p in img_files],
                verbose=False,
                imgsz=IMG_SIZE,
            )
            for r in results:
                pred_idx = r.probs.top1
                cm[true_idx, pred_idx] += 1

        n_correct = int(cm[true_idx, true_idx])
        n_total_cls = int(cm[true_idx, :].sum())
        print(f"  {class_name:<30s}  {n_correct}/{n_total_cls} correct")

    total   = int(cm.sum())
    correct = int(np.diag(cm).sum())
    overall_acc = correct / total
    print(f"\nTest accuracy: {overall_acc*100:.2f}%  ({correct}/{total})")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(cm, class_names)

    worst_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                worst_pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    worst_pairs.sort(key=lambda x: -x[2])

    # ── Outputs ───────────────────────────────────────────────────────────────
    write_report(overall_acc, metrics, worst_pairs, out / "eval_report.txt")
    plot_confusion_matrix(cm, class_names, out / "confusion_matrix.png")

    with open(out / "eval_metrics.json", "w") as f:
        json.dump({
            "overall_accuracy": float(overall_acc),
            "per_class":        metrics,
            "worst_pairs":      worst_pairs[:20],
        }, f, indent=2)

    print(f"\n  Next step: python 9_infer_clf.py --folder ./output/<image_stem>/")


if __name__ == "__main__":
    main()
