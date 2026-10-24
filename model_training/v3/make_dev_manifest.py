"""Photo-level development manifest for calibration/policy selection.

Uses only the grouped calib partition: Tomato-Village calib photos and PlantDoc
calib source photos (all author boxes), never the locked test or PlantVillage.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import scan_plantdoc

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=str(Path.home() / "agribot-data"))
p.add_argument("--work", default=str(Path.home() / "agribot-data/work"))
args = p.parse_args()
training = json.loads((Path(args.work) / "data_manifest.json").read_text())
pd_split = {}
for row in training:
    if row["source"] == "plantdoc":
        pd_split[row["group"]] = row["split"]
pd_train, _, _ = scan_plantdoc(Path(args.data_root) / "PlantDoc")
dev = []
for row in training:
    if row["source"] == "tomatovillage" and row["split"] == "calib":
        item = dict(row); item.update(id="tv-dev:" + Path(item["source_path"]).name, split="dev_calib")
        dev.append(item)
for record in pd_train:
    if pd_split.get(record["group"]) != "calib":
        continue
    item = dict(record); item.update(id="pd-dev:" + Path(item["source_path"]).stem, split="dev_calib",
                                     analysis_eligible=True)
    dev.append(item)
Path(args.work, "dev_manifest.json").write_text(json.dumps(dev, indent=1))
from collections import Counter
print(json.dumps(dict(n=len(dev), by_source=dict(Counter(i["source"] for i in dev)),
                      by_label=dict(Counter(i["label"] for i in dev))), indent=1))
