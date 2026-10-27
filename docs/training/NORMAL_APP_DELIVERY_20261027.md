# Normal app 2.0.4: completed offline build and verification

27 October 2026, 23:00 IST. The normal app now contains the original published
classifier and detector, with the completed persistence, lifecycle, scan-window
and evidence-scope repairs. **No rejected replacement model is bundled.**
Recognition repair remains unqualified; PR #6 stays draft under the user's
requirement to finish app and recognition together.

The classifier is SHA-256
`d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39`.
Manifest SHA-256 is
`b9ecdf557015e847bb4a4d76b622ab66100c036effa86e5a63c151e32de15f4f`.
Rejected schema-5 assets, source snapshots and previous research APKs remain
preserved separately, so the default app can be reviewed and built safely.

## Completed changes

Quick check shares its stopping rules with the actual Kotlin replay. Primary
recovery, when explicitly enabled by a matching model manifest, waits for seven
distinct frames; an early Other verdict cannot cut that window short. Duplicate
callbacks do not fabricate observations. Named disease evidence remains
protected. The original baseline manifest does not enable research recovery or
fabricate its auxiliary scores.

Primary-image evidence retains actual leaf counts and its visible-view scope
through Room persistence and UI restoration. English and Hindi cards use that
scope rather than claiming the entire plant is healthy. Row walking clears
off-plant primary views. Existing save-failure retry, atomic transactions and
background-inference rejection remain covered.

Strict manifest parsing supports schemas 2–5. The primary-model golden test
requires an exact research manifest when that path is enabled, and verifies
the baseline returns no invented primary evidence otherwise. In both modes,
the packaged model list must match the manifest exactly.

## Validation and packaging

- 133 native unit checks pass without skips.
- All 22 API-35 emulator checks pass without failures or skips, including native
  logits/detections, packaged model list, persistence, lifecycle, navigation,
  export, English/Hindi rendering and the primary-window regression.
- Release lint: zero errors, seven warnings.
- Debug, Android-test and unsigned-release APK builds pass.
- Python suite: 189 checks, zero failures, one optional downloaded-media skip.
- The 6,742-case policy and temporal replay fixtures are gzip-compressed in
  source control. Their decompressed bytes and SHA-256 values are unchanged;
  no case or assertion was removed.

A packaging audit caught an unused 453 MB rejected context model retained by
incremental debug assets after its source removal. The app module was rebuilt
cleanly; other useful module caches were retained. Both native instrumentation
and `apk_asset_delta.py` now reject undeclared, duplicate or missing model
assets. The final debug APK contains only the declared original two models and
is about 138 MB, rather than the stale 607 MB artifact. Its release counterpart
is about 111 MB. The stale artifact is not a delivery build.

The clean full build took 59.65 seconds and final emulator verification took
24.16 seconds on the PC. A subsequent CI check exposed a parser test that had
read the normal manifest while expecting research-only fields. Its research
contract now comes from a fixed unit-test resource, while bundled-contract
tests continue reading the real manifest. Gradle explicitly tracks that external
manifest as a test input, preventing stale results after model swaps. The final
validation build took 17.10 seconds; all 133 unit and 22 emulator checks pass.
Both APK hashes remain unchanged because this correction affects only tests and
their build inputs. Byte-exact APK reconstruction reuses verified local
model payloads and transfers approximately 26.6 MB for debug and 5.3 MB for
release. Every reconstructed APK matches the PC source hash before signing.

The local debug APK was re-signed with the existing laptop debug key. Its
certificate matches 2.0.3, and all ZIP payload hashes remain unchanged by
re-signing. This preserves upgrade compatibility with that debug installation.
The release APK remains unsigned pending the owner's release key.

## Local deliverables

Files are under `C:\Users\Manisha\agribot\.toolchains\app-candidates\2.0.4-normal`:

| File | SHA-256 |
| --- | --- |
| `agribot-2.0.4-normal-debug.apk` | `59e15456ffc2ec65a0a8f371ec66d28cd5f3d75a3e99d0b67a6f610a660733a0` |
| `agribot-2.0.4-normal-release-unsigned.apk` | `d3ef6ffbd9959768232b79f0d30a4b1f976e3aa759c8aded277ecdff4a398a75` |

[Evidence](evidence/normal_app_20261027/summary.json) includes native unit XML,
all 22 emulator case names, lint, exact source/model identities, APK delivery
and signing identities, and Python verification. Neither emulator verification
nor signing establishes physical-phone latency, memory or camera behavior.
The phone remains disconnected; no installation was attempted.

Read [the recognition continuation](OFFLINE_RECOGNITION_20261027.md) for failed
supervised, copied-encoder, kernel, logit-reward and routing controls, corrected
cache identities, permanently consumed AD/OLID status and fresh-source limits.
App verification does not authorize a recognition success claim or removal of
the remaining model and physical-device gates. Do not auto-merge PR #6.
