"""
1_download.py
=============
Downloads FieldPlant + PlantDoc from Roboflow Universe using a free API key.

HOW TO GET A FREE API KEY:
  1. Go to https://roboflow.com  →  Sign Up (free)
  2. Top-right corner → Settings → API Keys → copy your Private API Key
  3. Paste it below or set env variable:  export ROBOFLOW_API_KEY=your_key

WHY THESE TWO DATASETS:
  FieldPlant  — 5,170 real field photos, multiple leaves per image, annotated
                by plant pathologists. Complex backgrounds, overlapping leaves.
                This is the PRIMARY dataset for our task.

  PlantDoc    — 2,482 internet-scraped multi-leaf images, 29 class labels.
                Adds visual diversity (different cameras, lighting, species).

  Both are FREE, CC-licensed, and available on Roboflow Universe.
  Both will have ALL class labels collapsed to a single class "leaf"
  in the next step, because we only need to know WHERE leaves are.

RUN:
  python 1_download.py --key YOUR_API_KEY
  # or set env var and run:
  python 1_download.py
"""

import argparse
import os
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_default_api_key() -> str:
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key:
        return env_key

    dotenv_path = Path(".env")
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "ROBOFLOW_API_KEY":
                return value.strip().strip('"').strip("'")

    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--key",
        default=load_default_api_key(),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    p.add_argument(
        "--out",
        default="./raw_datasets",
        help="Where to save downloaded datasets (default: ./raw_datasets)",
    )
    return p.parse_args()


def download(api_key: str, out_dir: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("[ERROR] Run: pip install roboflow")

    if not api_key:
        raise SystemExit(
            "[ERROR] No API key found.\n"
            "  Option A: python 1_download.py --key YOUR_KEY\n"
            "  Option B: export ROBOFLOW_API_KEY=YOUR_KEY\n"
            "  Get a free key at: https://roboflow.com"
        )
    rf = Roboflow(api_key=api_key)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── FieldPlant (PRIMARY) ──────────────────────────────────────────────────
    # 5,170 field images · multiple leaves per image · expert bounding boxes
    # universe.roboflow.com/plant-disease-detection/fieldplant
    print("\n[1/2] Downloading FieldPlant (primary dataset)...")
    fp_project = rf.workspace("plant-disease-detection").project("fieldplant")
    fp_project.version(11).download(
        model_format="yolov8",
        location=str(Path(out_dir) / "fieldplant"),
        overwrite=True,
    )
    print("      ✓ FieldPlant downloaded")

    # ── PlantDoc (SECONDARY) ─────────────────────────────────────────────────
    # 2,482 multi-leaf images · 29 species labels · good scene diversity
    # universe.roboflow.com/joseph-nelson/plantdoc
    print("\n[2/2] Downloading PlantDoc (secondary dataset)...")
    pd_project = rf.workspace("joseph-nelson").project("plantdoc")
    pd_project.version(3).download(
        model_format="yolov8",
        location=str(Path(out_dir) / "plantdoc"),
        overwrite=True,
    )
    print("      ✓ PlantDoc downloaded")

    print(f"\n✓ Both datasets saved to: {Path(out_dir).resolve()}")
    print("  Next step: python 2_prepare.py")


if __name__ == "__main__":
    args = parse_args()
    download(api_key=args.key, out_dir=args.out)
