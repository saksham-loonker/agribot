# Optional field-media fixtures

No third-party images are bundled. To run the optional visual-preprocessing
stimulation, place one or more appropriately licensed JPEGs in the local
fixtures directory and keep their source and license information with the
local evaluation record.

The runner checks image decoding, the RGB mask path, letterbox geometry, and
annotated-box round trips. It does not measure model accuracy or establish
field readiness.

Run from the repository root:

    python tests/field_stimulation/stimulate_field_media.py

The regular Python unit suite does not require external imagery. It also checks
that a missing local fixture is reported as a failure rather than an empty pass.
