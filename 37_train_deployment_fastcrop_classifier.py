"""
Train a classifier for the exact fast Pi deployment crop.

Why this exists:
  The current classifier is already trained on clf_dataset, but it was not
  trained on the cheap deployment crop used to avoid the slow detector on Pi.
  Detector crops get high accuracy but are too slow. Fast mask/center crops are
  fast enough but shift the input distribution. This script fine-tunes the
  classifier on that deployment distribution.

What it does in one run:
  1. Builds a cropped class-folder dataset from clf_dataset.
  2. Balances the train split by oversampling minority classes with crop variants.
  3. Keeps val/test as the exact target deployment crop.
  4. Fine-tunes from models/classifier.pt by default.
  5. Evaluates the resulting best.pt on the cropped test split.
  6. Optionally exports the best classifier to OpenVINO.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from inference_rpi import center_crop, downscale_if_needed, mask_crop
from runtime_export_common import export_variant
from runtime_suite_common import EXPORTS_DIR, write_json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_float_list(value: str) -> list[float]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune classifier for fast Pi crop deployment")
    p.add_argument("--src", default="clf_dataset", help="Source class-folder dataset")
    p.add_argument("--out", default="clf_dataset_deploy_fastcrop", help="Prepared cropped dataset")
    p.add_argument("--base-model", default="models/classifier.pt", help="Classifier weights to fine-tune")
    p.add_argument("--project", default="clf_runs")
    p.add_argument("--name", default="deploy_fastcrop_mask055_224")
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="", help="Optional Ultralytics device, e.g. 0 or cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-edge", type=int, default=256)
    p.add_argument("--target-crop-mode", choices=["mask", "center"], default="mask")
    p.add_argument("--target-crop-scale", type=float, default=0.55)
    p.add_argument("--target-crop-pad", type=float, default=0.05)
    p.add_argument("--train-crop-scales", default="0.50,0.55,0.60")
    p.add_argument("--train-crop-pads", default="0.02,0.05,0.08")
    p.add_argument("--include-center-variants", action="store_true")

    p.add_argument("--balance-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-balanced-per-class", type=int, default=1200)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--overwrite-run",
        action="store_true",
        help="Allow reusing an existing training run directory",
    )
    p.add_argument(
        "--replace-model",
        action="store_true",
        help="Explicitly replace an existing candidate destination after validation",
    )
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")

    p.add_argument("--copy-best-to", default="models/classifier_deploy_fastcrop.pt")
    p.add_argument("--export-openvino", action="store_true")
    p.add_argument("--openvino-out", default=str(EXPORTS_DIR / "classifier_openvino_model"))
    return p.parse_args()


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def iter_images(split_dir: Path):
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
                yield class_dir.name, image_path


def collect_split(split_dir: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    if not split_dir.is_dir():
        return {}
    for class_name, image_path in iter_images(split_dir):
        by_class[class_name].append(image_path)
    return dict(by_class)


def crop_image(img, *, mode: str, scale: float, pad: float, max_edge: int):
    img = downscale_if_needed(img, max_edge)
    if mode == "mask":
        return mask_crop(img, crop_pad=pad, fallback_scale=scale)
    if mode == "center":
        return center_crop(img, crop_scale=scale)
    raise ValueError(mode)


def crop_variants(args) -> list[tuple[str, float, float]]:
    variants = []
    for scale in parse_float_list(args.train_crop_scales):
        for pad in parse_float_list(args.train_crop_pads):
            variants.append(("mask", scale, pad))
    if args.include_center_variants:
        for scale in parse_float_list(args.train_crop_scales):
            variants.append(("center", scale, 0.0))

    target = (args.target_crop_mode, args.target_crop_scale, args.target_crop_pad)
    if target not in variants:
        variants.insert(0, target)
    return variants


def safe_name(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def candidate_run_dirs(project: str, name: str) -> list[Path]:
    project_path = Path(project)
    candidates = [project_path / name]

    # For classify tasks, Ultralytics may place a relative project under
    # runs/classify even when project="clf_runs" is supplied.
    if not project_path.is_absolute():
        candidates.extend(
            [
                Path("runs") / "classify" / project_path / name,
                Path("runs") / project_path / name,
            ]
        )
        candidates.extend(sorted((Path("runs") / "classify").glob(f"**/{name}")))
        candidates.extend(sorted(Path("runs").glob(f"**/{name}")))

    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_best_weights(project: str, name: str) -> Path | None:
    for run_dir in candidate_run_dirs(project, name):
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            return best
    return None


def write_crop(src_image: Path, dest_dir: Path, dest_stem: str, *, mode: str, scale: float, pad: float, max_edge: int):
    img = cv2.imread(str(src_image))
    if img is None:
        return False
    crop = crop_image(img, mode=mode, scale=scale, pad=pad, max_edge=max_edge)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{dest_stem}.jpg"
    return bool(cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def prepare_dataset(args) -> Path:
    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"[ERROR] Source dataset not found: {src}")
    for split in ("train", "val", "test"):
        if not (src / split).is_dir():
            raise SystemExit(f"[ERROR] Missing split: {src / split}")
    if out == src or src.is_relative_to(out):
        raise SystemExit("[ERROR] Prepared dataset output must not contain or overwrite the source dataset")

    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"[ERROR] Output exists; use --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train = collect_split(src / "train")
    val = collect_split(src / "val")
    test = collect_split(src / "test")
    variants = crop_variants(args)

    manifest = {
        "source": str(src),
        "target_crop": {
            "mode": args.target_crop_mode,
            "scale": args.target_crop_scale,
            "pad": args.target_crop_pad,
            "max_edge": args.max_edge,
        },
        "train_variants": [
            {"mode": mode, "scale": scale, "pad": pad}
            for mode, scale, pad in variants
        ],
        "balance_train": args.balance_train,
        "splits": {},
    }

    train_counts = {class_name: len(paths) for class_name, paths in train.items()}
    max_count = max(train_counts.values()) if train_counts else 0
    target_count = min(max_count, args.max_balanced_per_class)

    for class_name, paths in sorted(train.items()):
        dest_dir = out / "train" / class_name
        if args.balance_train:
            wanted = max(len(paths), target_count)
            order = paths[:]
            rng.shuffle(order)
            written = 0
            for index in range(wanted):
                image_path = order[index % len(order)]
                mode, scale, pad = variants[index % len(variants)]
                stem = (
                    f"{image_path.stem}__{index:05d}"
                    f"__{mode}_s{safe_name(scale)}_p{safe_name(pad)}"
                )
                if write_crop(
                    image_path,
                    dest_dir,
                    stem,
                    mode=mode,
                    scale=scale,
                    pad=pad,
                    max_edge=args.max_edge,
                ):
                    written += 1
        else:
            written = 0
            for image_path in paths:
                if write_crop(
                    image_path,
                    dest_dir,
                    image_path.stem,
                    mode=args.target_crop_mode,
                    scale=args.target_crop_scale,
                    pad=args.target_crop_pad,
                    max_edge=args.max_edge,
                ):
                    written += 1
        manifest["splits"].setdefault("train", {})[class_name] = written

    for split_name, split_data in (("val", val), ("test", test)):
        for class_name, paths in sorted(split_data.items()):
            dest_dir = out / split_name / class_name
            written = 0
            for image_path in paths:
                if write_crop(
                    image_path,
                    dest_dir,
                    image_path.stem,
                    mode=args.target_crop_mode,
                    scale=args.target_crop_scale,
                    pad=args.target_crop_pad,
                    max_edge=args.max_edge,
                ):
                    written += 1
            manifest["splits"].setdefault(split_name, {})[class_name] = written

    for metadata_name in ("class_map.json", "split_stats.json"):
        source_metadata = src / metadata_name
        if source_metadata.exists():
            shutil.copy2(source_metadata, out / metadata_name)

    write_json(out / "deployment_fastcrop_manifest.json", manifest)
    print(f"Prepared deployment-crop dataset -> {out}")
    for split_name in ("train", "val", "test"):
        counts = manifest["splits"].get(split_name, {})
        print(f"  {split_name}: {sum(counts.values())} images")
        for class_name, count in sorted(counts.items(), key=lambda item: item[1]):
            print(f"    {class_name:<28} {count:5d}")
    return out


def train_classifier(args, data_root: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] ultralytics is not installed")

    base_model = Path(args.base_model)
    if not base_model.exists():
        raise SystemExit(f"[ERROR] Base classifier not found: {base_model}")

    model = YOLO(str(base_model), task="classify")
    train_kwargs = {
        "data": str(data_root),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "optimizer": "AdamW",
        "lr0": 1e-4,
        "lrf": 0.01,
        "weight_decay": 8e-4,
        "warmup_epochs": 3,
        "cos_lr": True,
        "dropout": 0.25,
        "label_smoothing": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.35,
        "hsv_v": 0.35,
        "fliplr": 0.5,
        "flipud": 0.5,
        "erasing": 0.20,
        "scale": 0.15,
        "translate": 0.05,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.overwrite_run,
        "pretrained": True,
        "seed": args.seed,
        "workers": args.workers,
        "amp": True,
        "plots": True,
        "verbose": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    best = find_best_weights(args.project, args.name)
    if best is None:
        raise SystemExit("[ERROR] Training finished but best.pt was not found")
    print(f"Best classifier -> {best}")
    return best


def evaluate_classifier(weights: Path, data_root: Path, imgsz: int, batch: int, out_dir: Path) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] ultralytics is not installed")

    test_dir = data_root / "test"
    if not test_dir.is_dir():
        raise SystemExit(f"[ERROR] Test split not found: {test_dir}")

    model = YOLO(str(weights), task="classify")
    names = (
        {int(k): str(v) for k, v in model.names.items()}
        if isinstance(model.names, dict)
        else {index: str(value) for index, value in enumerate(model.names)}
    )

    rows = []
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    true_labels = sorted(d.name for d in test_dir.iterdir() if d.is_dir())

    for true_label in true_labels:
        class_dir = test_dir / true_label
        true_norm = normalize_label(true_label)
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        for start in range(0, len(image_paths), batch):
            batch_paths = image_paths[start:start + batch]
            results = model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=imgsz,
                verbose=False,
            )
            if len(results) != len(batch_paths):
                raise RuntimeError(
                    f"Classifier returned {len(results)} results for {len(batch_paths)} test images"
                )
            for path, result in zip(batch_paths, results):
                pred_idx = int(result.probs.top1)
                pred_label = names[pred_idx]
                pred_norm = normalize_label(pred_label)
                correct = pred_norm == true_norm
                per_class[true_label]["total"] += 1
                if correct:
                    per_class[true_label]["correct"] += 1
                rows.append(
                    {
                        "image": str(path),
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": float(result.probs.top1conf),
                        "correct": correct,
                    }
                )

    correct = sum(1 for row in rows if row["correct"])
    total = len(rows)
    if total == 0:
        raise SystemExit("[ERROR] Evaluation produced no readable test images")
    accuracy = correct / total if total else 0.0
    per_class_rows = []
    for class_name in sorted(per_class):
        stats = per_class[class_name]
        class_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
        per_class_rows.append(
            {
                "class": class_name,
                "accuracy": class_acc,
                "correct": stats["correct"],
                "total": stats["total"],
            }
        )

    payload = {
        "weights": str(weights.resolve()),
        "data": str(data_root.resolve()),
        "imgsz": imgsz,
        "overall": {"accuracy": accuracy, "correct": correct, "total": total},
        "per_class": per_class_rows,
        "errors": [row for row in rows if not row["correct"]][:200],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "deployment_eval.json", payload)

    lines = [
        "=" * 72,
        "Deployment Fast-Crop Classifier Evaluation",
        "=" * 72,
        f"Weights : {weights}",
        f"Data    : {data_root}",
        f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})",
        "",
        f"{'Class':<32} {'Acc':>8} {'Correct':>8} {'Total':>8}",
        "-" * 72,
    ]
    for row in per_class_rows:
        lines.append(
            f"{row['class']:<32} {row['accuracy'] * 100:>7.2f}% "
            f"{row['correct']:>8} {row['total']:>8}"
        )
    (out_dir / "deployment_eval.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return payload


def copy_best_model(best: Path, dest: str, *, overwrite: bool = False) -> Path | None:
    if not dest:
        return None
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if best.resolve() == dest_path.resolve():
        print(f"Best classifier already at -> {dest_path}")
        return dest_path
    if dest_path.exists() and not overwrite:
        raise SystemExit(
            f"[ERROR] Candidate destination exists; refusing to replace it: {dest_path}. "
            "Choose a versioned --copy-best-to path or pass --replace-model after review."
        )
    temporary = dest_path.with_name(f".{dest_path.name}.tmp")
    try:
        shutil.copy2(best, temporary)
        temporary.replace(dest_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Copied best classifier -> {dest_path}")
    return dest_path


def export_openvino(best: Path, imgsz: int, final_path: str) -> Path:
    final = Path(final_path)
    exported = export_variant(
        best,
        task="classify",
        runtime="openvino",
        imgsz=imgsz,
        final_path=final,
        half=False,
        int8=False,
    )
    print(f"OpenVINO classifier -> {exported}")
    return exported


def print_next_steps(model_path: Path | None, args):
    model_ref = model_path or Path(args.copy_best_to)
    print("\nNext Pi commands:")
    print(
        "  scp "
        f"{model_ref} pi@192.168.1.25:~/agribot/models/{Path(model_ref).name}"
    )
    print(
        "  cd ~/agribot && source ~/agribot_venv/bin/activate && "
        "python3 10_export_runtime_models.py "
        f"--classifier models/{Path(model_ref).name} "
        "--detector models/detector.pt --runtimes openvino "
        f"--clf-imgsz {args.imgsz} --det-imgsz 256"
    )
    print(
        "  cd ~/agribot && source ~/agribot_venv/bin/activate && "
        "python3 24_benchmark_runtime_openvino.py --limit 0 --max-edge 256 "
        f"--conf 0.00 --crop-mode {args.target_crop_mode} "
        f"--crop-scale {args.target_crop_scale} --crop-pad {args.target_crop_pad}"
    )


def main():
    args = parse_args()
    data_root = Path(args.out)
    needs_data = not (args.skip_prepare and args.skip_train and args.skip_eval)

    if not args.skip_prepare:
        data_root = prepare_dataset(args)
    elif needs_data and not data_root.is_dir():
        raise SystemExit(f"[ERROR] Prepared dataset not found: {data_root}")

    best = Path(args.base_model)
    if not args.skip_train:
        best = train_classifier(args, data_root)

    run_dir = best.parent.parent if best.name == "best.pt" and best.parent.name == "weights" else Path(args.project) / args.name
    if not args.skip_eval:
        evaluate_classifier(best, data_root, args.imgsz, args.batch, run_dir)

    copied = copy_best_model(best, args.copy_best_to, overwrite=args.replace_model)
    if args.export_openvino:
        export_openvino(best, args.imgsz, args.openvino_out)

    print_next_steps(copied or best, args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
