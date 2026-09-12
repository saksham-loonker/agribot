# Android Golden Test Assets

This folder holds mobile golden fixtures for the native Agribot Android app.
The current repository includes the schema and example manifest first, so field
assets can be added without changing the test contract.

Expected structure:

```text
android_test_assets/
  side_scan/
    healthy/
    disease/
    uncertain/
  front_overview/
    clear_two_rows/
    occluded/
    low_light/
    ambiguous_geometry/
  expected/
    side_scan_expected.json
    front_overview_expected.json
    front_overview_manifest.jsonl
    front_overview_manifest.schema.json
```

Each front overview fixture must have one JSON object in
`expected/front_overview_manifest.jsonl`. The object records source media,
camera pose, reviewer identity, expected boxes, row side, plant mapping, and
acceptance rules. Unknown plant mappings must be written explicitly as `null`
or with `geometry_status` explaining why review is required.
