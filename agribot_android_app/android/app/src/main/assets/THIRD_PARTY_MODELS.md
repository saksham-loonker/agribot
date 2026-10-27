# Model attribution

The normal app bundle uses DINOv2 ViT-S/14 (Meta), MobileNetV3-Small through timm,
and the retained YOLOv9-T leaf detector from LibreYOLO. Research schema-5 bundles
add SigLIP 2 ViT-B/16 (Google, converted for timm), auxiliary heads and copied
final blocks. Those rejected research model assets are not bundled in the
normal app. Model hashes, tensor contracts and calibration are in
`model_manifest.json`.

- DINOv2: https://github.com/facebookresearch/dinov2 — Apache License 2.0.
- timm / MobileNetV3: https://github.com/huggingface/pytorch-image-models —
  Apache License 2.0.
- SigLIP 2 source: https://huggingface.co/timm/vit_base_patch16_siglip_256.v2_webli
  and https://huggingface.co/google/siglip2-base-patch16-256 — Apache License 2.0.
  The pinned timm revision is `ac4cc431026674dcac29e34a00d3bd2048390eb3`;
  original weight-file SHA-256 is
  `13ab9d238865f89b14005c5e92a099add04cbfab331a64245acab813509cb2b0`.
  Attribution: Tschannen and colleagues, *SigLIP 2: Multilingual Vision-Language
  Encoders with Improved Semantic Understanding, Localization, and Dense
  Features* (2025), https://arxiv.org/abs/2502.14786.
- LibreYOLO: https://github.com/LibreYOLO — MIT; detector identity and existing
  conversion provenance are retained from the original Agribot bundle.

The Apache License text is included in `licenses/Apache-2.0.txt`. Upstream
notices and the repository's training-data provenance remain applicable. No
Apple MobileCLIP parameters, teachers or predictions enter this candidate.
