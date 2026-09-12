# Online Field-Media Stimulation Record

**Run date:** 2026-08-02  
**Purpose:** independent local evidence for image decoding, capture-quality gating, and detector-box coordinate safety.  
**Status:** **PASS for the bounded stimulation; not a field-readiness sign-off.**

## What was tested

The Android pipeline has two different kinds of correctness that must not be
confused:

1. **Geometry and safety:** a detector output must be mapped from the 256 × 256
   letterboxed model input back into the camera frame without producing boxes
   outside the frame, inverted boxes, or boxes that exist only in letterbox
   padding.
2. **Model generalisation:** the embedded detector and classifier must recognise
   unseen real field imagery with acceptable recall and calibrated confidence.

This run proves the first item against a decoded public image. It does not claim
the second item because this host has no Android camera session or TFLite
interpreter suitable for executing the shipped model. The Android-native test
suite remains the executable local gate for parser, crop, tracker, decision-gate,
and frame-quality behaviour.

## Public media used

The source catalogue is intentionally recorded so that a future reviewer can
repeat the exercise and check the original licensing terms.

| Source | Licence | Local result |
|---|---|---|
| [Healthy tomato leaves](https://commons.wikimedia.org/wiki/File:Healthy_tomato_leaves_(7871755330).jpg) | CC BY 2.0 | Downloaded and decoded successfully |
| [Tomatoes in the field](https://commons.wikimedia.org/wiki/File:Tomatoes_in_the_field_(4648307711).jpg) | CC BY 2.0 | Wikimedia request rate-limited during this run; retained as a repeatable source |
| [Lufa Farms tomato rows](https://commons.wikimedia.org/wiki/File:Lufa_Farms_Tomato_Rows.jpg) | CC BY-SA 2.0 | Wikimedia request rate-limited during this run; retained as a repeatable source |
| [Powdery mildew on tomato leaves](https://commons.wikimedia.org/wiki/File:O%C3%AFdium_sur_Feuilles_de_Plants_de_Tomate.JPG) | CC BY-SA/GFDL | Wikimedia request rate-limited during this run; retained as a repeatable source |

The downloaded file is kept outside the repository at
$env:TEMP\agribot-field-media\healthy_leaves.jpg, so a large binary is not
silently committed to the application. Its run-specific measurements are in
[ONLINE_FIELD_MEDIA_STIMULATION.json](ONLINE_FIELD_MEDIA_STIMULATION.json).

## Results

The downloaded stimulus was a 960 × 643 RGB JPEG. The bounded image-quality
stimulation observed finite sample pixels, non-extreme brightness, and usable
contrast, so it was accepted rather than incorrectly treated as a dark, flat,
or corrupted frame.

The geometry stimulation covered:

- the complete camera frame;
- boxes touching each of the four camera-frame corners;
- inverse mapping through the aspect-ratio-preserving letterbox transform;
- clamping to `[0, frameWidth] × [0, frameHeight]`;
- rejection of inverted/non-finite boxes; and
- rejection of boxes located entirely in model-input padding.

All checks passed. The local command was:

```powershell
python 50_stimulate_field_media.py --image "$env:TEMP\agribot-field-media\healthy_leaves.jpg" --output ONLINE_FIELD_MEDIA_STIMULATION.json
```

The pure-Python regression tests also cover the boundary mapping and invalid
box cases:

```powershell
python -m unittest discover -s tests -p "test_field_media_stimulation.py"
```

## What this evidence does not prove

It does **not** prove that the detector finds every plant in the image, that a
disease label is correct, or that confidence values are calibrated. It also
does not prove that the phone's camera exposure, focus, motion blur, GPS, step
sensor, or accelerometer behave like this desktop stimulus. Those claims need a
physical ARM64 phone run with reviewed frame-level evidence and a completed
field export.

The release gate therefore remains intentionally strict:

1. run the embedded models on the target phone over representative Indian field
   conditions (sun, shade, glare, dust, overlap, different crop stages, and
   phone orientations);
2. review false positives, false negatives, duplicate tracks, and uncertain
   cases against human-labelled boxes;
3. verify that motion values are non-zero when the phone moves and unavailable
   rather than zero when no usable source exists;
4. verify GPS is used only as a field-envelope/reference source, never as a
   fabricated plant-size measurement; and
5. validate the exported evidence chain and provenance with
   `49_validate_android_field_export.py`.

Until those gates are complete, the app should present its output as an
**observed, reviewable field estimate**, not as an autonomous treatment
recommendation.
