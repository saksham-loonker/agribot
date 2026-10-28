# Inactive proposal: distinguish corrections from recognition regressions

No active promotion gate or app verdict has changed. The proposal evaluator in
[clinical_retention_proposal.py](../../model_training/v3/clinical_retention_proposal.py)
is deliberately not imported by `OpenedGuardScreen`.

The current `original_disease_and_healthy_exact` check in
[opened_guard_screen.py](../../model_training/v3/opened_guard_screen.py)
requires every earlier Healthy or named-disease output to remain byte-for-byte
equal, including Healthy predictions on publisher-labeled diseased images.
This protects correct earlier decisions, but also rejects possible corrections.
This is distinct from a numeric safety or Healthy-recall requirement.

The fixed .05, both-crop, four-orientation veto audit supplies a concrete case:
the exhaustive supervised head would change 18 protected earlier verdicts,
all on publisher-labeled non-Healthy images, with zero changes on protected
publisher-labeled Healthy images. The ranking head changes 37 such earlier
non-Healthy cases, again with zero Healthy-label changes. Both variants also
remove two of the 16 consumed OLID false-Healthy outputs and retain its 87
correct Healthy outputs. These are consumed publisher-label comparisons, not
fresh clinical adjudication or independent acceptance. Rotations of one photo
are not independent observations.

Both candidates still fail the current exact-retention gate. Neither is a
qualified replacement: Healthy remains 87/223, below 157/223, and Healthy gain
is zero. This proposal does not solve or waive those failures.

If explicitly authorized, the limited alternative would retain:

- Exact preservation of every earlier named-disease output.
- Exact preservation of every earlier publisher-labeled correct Healthy output,
  and Healthy outputs without a usable clinical label.
- Changes to earlier labeled false-Healthy outputs only as unlabeled `UNSURE`;
  no new named diagnosis or treatment follows from the veto.
- Every existing source-retention, coarse, confidence-interval, Healthy-recall,
  paired-gain, false-Healthy, prospective-acceptance and phone requirement.

The existing strict check would remain computed and reported. Authorization
would be required before substituting this limited alternative in the active
promotion conjunction. No threshold tuning to these evaluation labels, dataset
membership rules at inference, silent relabeling, or blind gate removal is
proposed. Suitable unconsumed prospective acceptance remains necessary before
production use.
