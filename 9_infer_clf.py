"""
9_infer_clf.py
==============
Classifies individual leaf crops produced by Part 1 (4_infer.py)
using the trained YOLO11 model.

PIPELINE CONNECTION:
  Part 1 output:
    output/<image_stem>/
      leaf_001_conf0.91.jpg   ← 256×256 cropped leaf
      leaf_002_conf0.87.jpg

  Part 2 input (this script):
    Same folder → classify each image

WHAT IT DOES:
  For each leaf crop image:
    1. Feed through YOLO11 classify model (256×256 input)
    2. Softmax → probabilities over 8 classes
    3. Returns: top-1 prediction + confidence
                top-3 predictions + confidences
    4. Writes _diseases.json and _diseases.txt in same folder

RUN:
  python 9_infer_clf.py --folder ./output/my_photo/
  python 9_infer_clf.py --all --out_root ./output/
  python 9_infer_clf.py --folder ./output/my_photo/ \\
                         --weights ./clf_runs/yolo11_tomatovillage/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS    = "./clf_runs/yolo11_tomatovillage_v2/weights/best.pt"
TOP_K      = 3
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--folder",   help="Path to one leaf-crop folder (Part 1 output)")
    g.add_argument("--all",      action="store_true",
                   help="Process ALL subfolders in --out_root")
    p.add_argument("--out_root", default="./output",
                   help="Root of Part 1 output (used with --all)")
    p.add_argument("--weights",  default=WEIGHTS)
    return p.parse_args()


def classify_folder(folder: Path, model) -> dict:
    """
    Classifies all image files in folder.
    Returns dict: {filename: {top1: ..., top3: [...]}}
    """
    if not folder.exists():
        raise SystemExit(f"[ERROR] Folder not found: {folder}")

    exts = {".jpg", ".jpeg", ".png"}
    # Try leaf_*.jpg first, then fall back to any image
    leaf_imgs = sorted(folder.glob("leaf_*.jpg"))
    if not leaf_imgs:
        leaf_imgs = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
            and not p.name.startswith("_")
        )
    if not leaf_imgs:
        print(f"  [SKIP] No images in: {folder}")
        return {}

    results_dict = {}
    predictions = model.predict(
        source=[str(p) for p in leaf_imgs],
        verbose=False,
    )

    for img_path, r in zip(leaf_imgs, predictions):
        probs = r.probs
        top_k = min(TOP_K, len(model.names))
        top_indices = probs.top5[:top_k]
        top_confs = probs.top5conf[:top_k]

        top3 = [
            {"class": model.names[int(idx)], "confidence": round(float(conf), 4)}
            for idx, conf in zip(top_indices, top_confs)
        ]

        results_dict[img_path.name] = {
            "top1": top3[0],
            "top3": top3,
        }

    return results_dict


def write_results(folder: Path, results: dict):
    if not results:
        return

    # JSON
    json_path = folder / "_diseases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Human-readable text
    txt_lines = [
        f"Tomato Disease Classification — {folder.name}",
        "=" * 60,
        "",
    ]
    for fname, r in results.items():
        top1 = r["top1"]
        disease = top1["class"]
        conf = top1["confidence"] * 100
        flag = "⚠ " if "healthy" not in disease.lower() else "✓ "
        txt_lines.append(f"{flag}{fname}")
        txt_lines.append(f"   {disease}  ({conf:.1f}% confidence)")

        if top1["confidence"] < 0.6:
            txt_lines.append("   ↳ Low confidence. Next candidates:")
            for alt in r["top3"][1:]:
                txt_lines.append(f"     • {alt['class']}  ({alt['confidence']*100:.1f}%)")
        txt_lines.append("")

    txt_path = folder / "_diseases.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"  → {len(results)} leaves classified")
    print(f"     {json_path}")
    print(f"     {txt_path}")

    # Print summary
    diseased = [r["top1"]["class"] for r in results.values()
                if "healthy" not in r["top1"]["class"].lower()]
    healthy  = len(results) - len(diseased)
    print(f"     Healthy: {healthy}   Diseased: {len(diseased)}")
    if diseased:
        from collections import Counter
        for disease, count in Counter(diseased).most_common():
            print(f"       • {disease}: {count}")


def main():
    args = parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    wp = Path(args.weights)
    if not wp.exists():
        raise SystemExit(
            f"[ERROR] Weights not found: {wp}\n"
            f"        Run python 7_train_clf.py first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] pip install ultralytics>=8.3.0")

    model = YOLO(str(wp))
    n_classes = len(model.names)

    print(f"\nModel  : YOLO11 (classify)")
    print(f"Classes: {n_classes}")
    print(f"Weights: {wp}\n")

    # ── Collect folders to process ────────────────────────────────────────────
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"[ERROR] Folder not found: {folder}")
        folders = [folder]
    else:
        out_root = Path(args.out_root)
        if not out_root.exists():
            raise SystemExit(f"[ERROR] Output root not found: {out_root}")
        folders = sorted([d for d in out_root.iterdir() if d.is_dir()])
        if not folders:
            raise SystemExit(f"[ERROR] No subfolders in: {out_root}")
        print(f"Found {len(folders)} leaf-crop folders in {out_root}\n")

    # ── Process ───────────────────────────────────────────────────────────────
    for folder in folders:
        print(f"[{folder.name}]")
        results = classify_folder(folder, model)
        write_results(folder, results)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
