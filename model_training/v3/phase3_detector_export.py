"""Export an evaluated generic leaf detector with the existing Android interface."""

from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
from PIL import Image
import torch
from torch import nn
from libreyolo import LibreYOLO9
from app_eval import canon


class DetectorGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, nhwc):
        return self.model(nhwc.permute(0, 3, 1, 2))["predictions"]


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter

    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    assert not json.loads((root / "data_complete.json").read_text()).get(
        "quarantined", False
    ), "Do not export the rejected region-as-leaf experiment"
    candidate = json.loads((root / "candidate_validation/candidate.json").read_text())
    assert candidate[
        "passed"
    ], "Detector validation failed; do not export as a promotion candidate"
    checkpoint = Path(candidate["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == candidate["sha256"]
    size = candidate["input_size"]
    out = root / "export"
    (out / "models").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = DetectorGraph(
        LibreYOLO9(str(checkpoint), size="t", nb_classes=1, device="cpu")
        .model.cpu()
        .eval()
    ).eval()
    x = torch.zeros(1, size, size, 3)
    with torch.no_grad():
        model(x)
    path = out / "models/leaf_detector.tflite"
    litert_torch.convert(model, (x,)).export(str(path))
    runtime = Interpreter(model_path=str(path), num_threads=4)
    runtime.allocate_tensors()
    ops = sorted({r["op_name"] for r in runtime._get_ops_details()})
    assert not any(
        o == "CUSTOM" or o.startswith("Flex") or "STABLEHLO" in o for o in ops
    ), ops
    parity = []
    for file in sorted(
        (Path(__file__).resolve().parents[1] / "v2/bundle/golden").glob("*.png")
    ):
        tensor, _ = canon.detector_input(
            np.asarray(Image.open(file).convert("RGB")), size
        )
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()
        runtime.set_tensor(runtime.get_input_details()[0]["index"], tensor)
        runtime.invoke()
        actual = runtime.get_tensor(runtime.get_output_details()[0]["index"])
        parity.append(
            dict(
                file=file.name,
                max_box_error=float(abs(expected[:, :4] - actual[:, :4]).max()),
                max_score_error=float(abs(expected[:, 4] - actual[:, 4]).max()),
            )
        )
    report = dict(
        candidate=candidate,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        parity=parity,
        ops=ops,
        passed=all(
            r["max_box_error"] < 0.05 and r["max_score_error"] < 0.0001 for r in parity
        ),
    )
    (out / "export_verification.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    assert report["passed"], report
    assets = root.parent / "export_cnn_safe"
    manifest = json.loads((assets / "model_manifest.json").read_text())
    detector = manifest["detector"]
    detector.update(
        file=path.name,
        sha256=report["sha256"],
        input=[1, size, size, 3],
        candidates=sum((size // s) ** 2 for s in [8, 16, 32]),
    )
    manifest["bundle_id"] = "agribot-v3-field-detector-" + report["sha256"][:8]
    manifest["status"] = (
        "EXPERIMENTAL: requires full app regressions, independent field acceptance and device verification"
    )
    manifest["detector_training"] = dict(
        checkpoint_sha256=candidate["sha256"],
        protocol=json.loads((root / "training_protocol.json").read_text()),
        data=json.loads((root / "data_complete.json").read_text()),
    )
    manifest.pop("heldout_results", None)
    for member in manifest["classifier"]["members"]:
        target = out / "models" / member["file"]
        if not target.exists():
            target.symlink_to(assets / "models" / member["file"])
    (out / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(dict(passed=True, sha256=report["sha256"], size=size)), flush=True)


if __name__ == "__main__":
    main()
