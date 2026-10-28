# App delivery and recognition continuation — 28 October 2026

This checkpoint supersedes the recurring PR-monitor instructions and older
sealed-OLID statements. The user explicitly requested app and recognition work
take priority and cancelled the recurring prompts. The Windows task
`Agribot-PR6-Review-4Hourly` is disabled; its private script also checks a user
cancellation marker. Do not reenable it or treat queued monitor messages as a
new objective. PR #6 remains draft until recognition qualification is complete.

## App: actual minified release exercised

The normal 2.0.4 app retains the original classifier and detector. The debug
delivery APK and unsigned ARM64 release are already built; see
[delivery identities and prior checks](NORMAL_APP_DELIVERY_20261027.md).

An additional runtime smoke check used the **actual minified ARM64 release**,
signed with the existing PC debug key exclusively for emulator testing. The
API-35 emulator advertises `x86_64,arm64-v8a` and uses
`libndk_translation.so`. No alternate ABI build, test keep-rules or unminified
substitute was needed. Every non-signature ZIP payload matches the delivered
unsigned release, SHA-256
`d3ef6ffbd9959768232b79f0d30a4b1f976e3aa759c8aded277ecdff4a398a75`.

Onboarding, camera permission, camera entry, both original LiteRT model
initializations, back navigation, settings, English/Hindi rendering, row-walk
setup, and retained Hindi after a forced cold restart passed. Logcat contained
no fatal exception. The emulator was stopped after evidence collection to free
RAM for training. A subsequent imagefile-camera run also exercised CameraX
RGBA conversion, detection and classification using an existing golden image.
It displayed Early Blight, saved/reopened the result and retained it after a
forced restart. A two-position manual row walk completed with one attention
result and one unsure result, zero Healthy; its record survived another restart.
CSV, PDF and ZIP generation reached Android's chooser with the expected file
names; no external recipient was selected. Export contents retain their earlier
instrumentation coverage and were not independently re-parsed in this smoke.
These repeated simulated camera frames are not independent photos. This is
runtime/loading coverage, not new recognition
accuracy evidence or a physical-phone latency test. The phone remains absent.
The PC debug signature must not replace the laptop upgrade APK's signature or
be described as an owner production signing identity.

The full local Python suite now runs **198 checks, zero failures, one optional
downloaded-media skip**, in 21.50 seconds. Six provenance checks cover cache
identity, lossless storage/corruption, valid geometry, repeated-texture rejection,
encoded/decoded photo identities and member CRC integrity. Three more checks
ensure inactive routed cases retain their full-cohort denominator and fixed
verdict contribution; duplicate/missing photographs fail explicitly.
Existing app evidence remains 133 unit and 22 emulator checks passing; no app
source or model bytes were changed in this continuation.

## Completed recognition controls

The phase-105 image-bag gradient-boosted tree used only matching cached original
DINO/CNN and whole-image features. It required **zero new encoder inputs** and
took 195.84 seconds. Selection stopped at 125 iterations after the prescribed
stale rounds. Consumed OLID validation Healthy remained 13/40. Full Healthy was
98/223, below the unchanged 70% requirement, and false-Healthy increased from
16 to 17. Original disease/earlier retention checks passed; recognition and
full safety did not. Do not export or continue this tree.

The phase-108 background pilot selected fixed consumed training families,
leaving original app inputs and weights unchanged. Its conservative
border-connected neutral-background mask was inadmissible in every case:
**zero safe masks and zero new encoder inputs**. It stopped in 4.86 seconds.
This is a coverage failure of this mask rule, not evidence that arbitrary
background removal is ineffective or that forcing masks would be safe. Do
not tune masks against the consumed validation or acceptance labels.

One further supervised control, phase 110, permits low-rate full-backbone CNN
adaptation after the frozen-prefix/spatial-suffix approaches plateaued. It
warm-starts the selected phase-101 auxiliary CNN suffix, verifies restoration,
freezes BatchNorm statistics, preserves the original deployed nine-class
classifier, and retains the same source/family split and input quarantine.
It uses consumed development only, at most 12 epochs and four stale routed
validation epochs. Full-cohort inference is forbidden unless the cheap
validation screen reaches 28/40 Healthy without a source false-Healthy
increase. The valid run completed all 12 bounded epochs in **186.36 seconds**.
Its selected epoch 9 had no source validation false-Healthy increase but
recovered only **18/40 Healthy**; it failed the cheap gate. Full-cohort inference,
restored verification and export were correctly skipped. Do not extend this
stopped full-backbone control. Initialization/checkpoint-write failures preceding the valid run
are preserved separately and are not experimental recognition results.

The final phase-117 control tested the strongest remaining representation
alternative: low-rate adaptation of the **entire copied DINO backbone**,
warm-starting phase 97's selected suffix rather than restarting it. Only the
existing consumed development collection was used; its source/family split,
exact input quarantine, full 40-photo denominator, auxiliary calibration and
routing cutoffs stayed fixed. Prefix weights now changed, so their frozen
feature cache could not replace training forwards; exact canonical input
pixels were reused instead. The deployed classifier remained untouched.

It stopped after nine bounded epochs in **86.23 seconds**. Selected epoch 4
recognized **19/40 Healthy** with a source validation false-Healthy increase
of one, already inadmissible. The higher-recall epoch reached 25/40 while
introducing six source safety violations. Neither met 28/40 or safety.
Full-cohort verification/export were skipped. Do not extend this stopped
full-DINO control or describe the unsafe higher-recall epoch as a repair.

## Source integrity and sealed data

The neural-free prior-overlap audit fingerprints 36,101 previous identities,
including image/ROI sources and consumed AD/OLID. No prior identities were
missing. A visual investigation caught a geometric-verification bug: many
query features could match the same target corner. A purported cross-label
match had 38 tentative matches but only **seven unique target points** and was
mostly paper background, not proof of contradictory leaf labels.

The verifier now requires unique target correspondences before homography.
Its regression test rejects repeated texture. The corrected phase-109 audit
reused **all** stored fingerprints without feature extraction or neural
inference and finished in 92.51 seconds. It flagged 64 TCP images in 60 inferred
families; quarantining complete families leaves 2,461 provisional images:
791 Healthy and 1,670 Other. Approximate retrieval can select different
candidates between runs; these are conservative screening results, not a
proof of independent captures. Legacy phase-106 results are superseded.
`acceptance_ready=false`, `independence_established=false`, no diagnostic
predictions opened, and no training used remain explicit. Keep this provisional
TCP reservation closed until suitability, candidate and protocol are frozen.

The new publisher source [Tomato dataset](https://data.mendeley.com/datasets/xc94xbg239/1),
DOI `10.17632/xc94xbg239.1`, was assigned **development-only** before sampling.
Its ZIP catalog contains 1,013 Healthy_Leaf and 1,050 Infected_Leaf images;
fruit categories are excluded. The latter label means broadly diseased, not
unsupported Other or one of our seven named diseases. A fixed 64-photo audit
found class-correlated orientation and camera metadata, plus visually similar
leaf poses. The corrected geometry check found zero confirmed pairs in that
sample; visual resemblance is not a verified specimen identity. The earlier
one-pair report is superseded and must not be called a label contradiction.

The complete 4,423,476,057-byte archive is now publisher-SHA-256 verified:
`5121ccfd42622e2728cc754d5f00ea63250d65954fe0dbd94df6ee911fd50d13`.
Only the 2,063 leaf photos were extracted, checking member CRC and encoded and
decoded identities. All encoded and decoded images are unique, with zero exact
cross-label pixel conflicts. Healthy photos are 987/1,013 portrait; diseased
photos are 1,020/1,050 landscape. Every Healthy photo lacks camera metadata;
every diseased photo records iPhone 14 Pro. Missing EXIF does not establish a
different actual camera, but capture/processing comparability remains unresolved.
The pHash screen produced 1,776 approximate image families, largest 52 photos;
these are not verified specimens and must not serve as independent plant IDs.

The prior-overlap screen compared this collection against the 36,101 cached old
identities and found zero matches in 131.31 seconds, without neural inference.
Before fitting, the 256 proposed training photos were separately screened
against all 2,461 provisional TCP reservation photos. That used existing
fingerprints, took 4.77 seconds and found zero matches. No clinical predictions
from the TCP reservation were opened, and no TCP photo was used for training.
Approximate geometry remains a screen, not proof of capture independence.

## Bounded new-supervision result

Given the source issues, only a fixed, predeclared **128 photos per binary
class** entered an exploratory development probe. The entire collection stays
training-only; its unused remainder cannot become independent acceptance.
Every selected input received all four right-angle rotations equally, retaining
clinical colour and texture. Supervision remained one image-level label over
its actual ROI bag and both crop views; no clinical label was assigned to an
individual detection, and broadly infected images were not renamed Other.

Phase 114 reused the old original-encoder feature bank, computing only **2,176
new inputs** plus 256 new detector inputs in 49.36 seconds. All 256 photos had
actual detected leaves. Its exact-identity SQLite cache and feature files are
retained; never repeat the old encoder or these new inputs unchanged.

Phase 115 warm-started one existing MLP, kept calibration and routing cutoffs
fixed, and selected actual validation verdicts under the same source safety
constraints. The new source was entirely excluded from validation. A guard
caught that only 31 Healthy validation cases are in the active route; the
required denominator remains the **complete 40-photo cohort**. The corrected
helper includes fixed inactive verdicts without weakening the 28/40 gate.
The earlier failed initialization did not fit a model.

The valid run stopped after six stale routed-validation epochs: selected epoch
15, **13/40 Healthy**, no source validation false-Healthy increase. It took
**3.51 seconds** and recomputed zero old encoder inputs. It failed the cheap
recognition gate, so complete cached head screening, restored-model inference
and export were skipped. Do not extend this stopped probe or infer full-cohort
accuracy that was not measured. Camera/style ambiguity remains unresolved;
this experiment does not validate this source for acceptance or prove that all
possible additional data would fail.

A separate metadata-only inspection of [EgyPLI's publisher paper](https://www.nature.com/articles/s41597-025-06539-8)
and its linked Kaggle version cataloged all 3,588 files. It contains eight
species directories, including 159 Tomato images, with no health-label
directories or non-image annotation files. The publisher notebook also trains
species labels. Although the paper describes health annotations, the inspected
release does not supply an explicit Healthy/diseased mapping. It is unsuitable
for clinical acceptance as distributed; do not label every Tomato image Healthy.
No pixels, clinical predictions or training were opened for this source.

## Additional recognition repair

The subsequent recovery-only campaign is documented in
[Recognition repair](RECOGNITION_REPAIR_20261028.md). It separates Healthy
admission from legacy vetoes, adds class-balanced replay and clinical anchors,
compares local/native-resolution features and LoRA, and reuses exact-identity
caches. New expected-reward and hard-negative controls retain the original
recognition and safety gates. Complete source/result identities and the
full-source supervised control are recorded there; no failed candidate is
exported or bundled into Android.

## Remaining qualification and evidence

AD and OLID are permanently consumed development. Keep all original recognition
and safety gates unchanged, prioritize real app verdicts, and preserve named
disease behavior. A candidate must pass complete cheap development checks,
then full restored inference/parity, then a genuinely suitable prospective
acceptance evaluation frozen in advance. Model export and replacement-model
Android integration remain deferred until recognition gates pass. The normal
app's baseline rebuild/runtime smoke is separate from promoting a replacement.

PC campaign root is
`/home/genesis/agribot-data/recognition_campaign_20261024`.
SSH remains `agribot-5090`; useful jobs run under persistent `tmux` session
`agribot`. Preserve dirty source overlays, exact-identity caches, optimizer/RNG
state and unrelated edits. Do not blindly pull or auto-merge PR #6.

Inspectable evidence is under [completion_20261028](evidence/completion_20261028/).
The minified-release result and UI XML, cached-tree metrics, mask pilot,
corrected/legacy provenance audits and source-diversity audit are retained.
Measured control times are individual observations, not a delivery promise.
Recognition has not qualified, so a final model-delivery ETA is not supportable.
