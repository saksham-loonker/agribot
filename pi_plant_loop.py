#!/usr/bin/env python3
"""
CPU-only Raspberry Pi plant decision loop.

The loop samples selected frames at 5 FPS. Each plant decision gets two primary
frames and one backup frame. A high-confidence agreement on the two primary
frames emits immediately; disagreement or low confidence consumes the backup
frame and otherwise emits Uncertain.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cv2

from inference_rpi import (
    CLF_CONF_THRESHOLD,
    DEFAULT_CLF_MODEL,
    DEFAULT_DET_MODEL,
    DET_IMGSZ,
    HIGH_CONF_THRESHOLD,
    MAX_EDGE,
    MAX_FRAME_LATENCY_MS,
    MAX_LEAVES,
    MIN_FRAME_FPS,
    PlantDecisionGate,
    _find_healthy_idx,
    infer_frame,
    load_models,
)


_STOP = False


def _signal_handler(sig, frame):
    global _STOP
    _STOP = True


signal.signal(signal.SIGINT, _signal_handler)


def parse_args():
    p = argparse.ArgumentParser(description="5 FPS Raspberry Pi plant inference loop")
    p.add_argument("--camera", default="0", help="Camera index or video path")
    p.add_argument("--det-model", default=str(DEFAULT_DET_MODEL))
    p.add_argument("--clf-model", default=str(DEFAULT_CLF_MODEL))
    p.add_argument("--det-imgsz", type=int, default=DET_IMGSZ)
    p.add_argument("--det-conf", type=float, default=0.30)
    p.add_argument("--det-iou", type=float, default=0.45)
    p.add_argument("--max-edge", type=int, default=MAX_EDGE)
    p.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    p.add_argument("--crop-mode", choices=["none", "center", "mask"], default="mask")
    p.add_argument("--crop-scale", type=float, default=0.55)
    p.add_argument("--crop-pad", type=float, default=0.05)
    p.add_argument("--conf", type=float, default=CLF_CONF_THRESHOLD)
    p.add_argument("--high-conf", type=float, default=HIGH_CONF_THRESHOLD)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fps", type=float, default=MIN_FRAME_FPS)
    p.add_argument("--with-detector", action="store_true")
    p.add_argument("--jsonl", default="", help="Optional path for decision logs")
    return p.parse_args()


def open_capture(source: str):
    try:
        camera_index = int(source)
        cap = cv2.VideoCapture(camera_index)
    except ValueError:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open camera/video source: {source}")
    return cap


def emit(decision, plant_id: int, late_frames: int, output_handle):
    payload = {
        "plant_id": plant_id,
        "label": decision.label,
        "confidence": round(decision.confidence, 6),
        "status": decision.status,
        "frames_used": decision.frames_used,
        "reason": decision.reason,
        "late_frames": late_frames,
        "time": time.time(),
    }
    line = json.dumps(payload, separators=(",", ":"))
    print(line, flush=True)
    if output_handle is not None:
        output_handle.write(line + "\n")
        output_handle.flush()


def main():
    args = parse_args()
    if args.fps < MIN_FRAME_FPS:
        raise SystemExit(f"[ERROR] --fps must be at least {MIN_FRAME_FPS:.1f}")

    det_model, clf_model = load_models(
        det_model_path=args.det_model,
        clf_model_path=args.clf_model,
        threads=args.threads,
        with_detector=args.with_detector,
        det_imgsz=args.det_imgsz,
    )
    healthy_idx = _find_healthy_idx(clf_model.names)
    cap = open_capture(args.camera)

    output_handle = None
    if args.jsonl:
        output_path = Path(args.jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("a", encoding="utf-8")

    gate = PlantDecisionGate(high_conf=args.high_conf)
    frame_period = 1.0 / args.fps
    next_slot = time.perf_counter()
    plant_id = 1
    late_frames = 0

    try:
        while not _STOP:
            now = time.perf_counter()
            if now < next_slot:
                time.sleep(next_slot - now)
            next_slot += frame_period

            ok, frame = cap.read()
            if not ok:
                break

            started = time.perf_counter()
            prediction = infer_frame(
                det_model,
                clf_model,
                frame,
                with_detector=args.with_detector,
                det_conf=args.det_conf,
                det_iou=args.det_iou,
                det_imgsz=args.det_imgsz,
                max_edge=args.max_edge,
                max_leaves=args.max_leaves,
                crop_mode=args.crop_mode,
                crop_scale=args.crop_scale,
                crop_pad=args.crop_pad,
                clf_conf=args.conf,
                healthy_idx=healthy_idx,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            prediction["latency_ms"] = latency_ms
            if latency_ms > MAX_FRAME_LATENCY_MS:
                late_frames += 1

            decision = gate.add(prediction)
            if decision is not None:
                emit(decision, plant_id, late_frames, output_handle)
                plant_id += 1
                late_frames = 0
                gate.reset()
    finally:
        cap.release()
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
