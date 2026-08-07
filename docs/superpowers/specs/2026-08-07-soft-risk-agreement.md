# Soft risk relaxation vs. the real Detector, on real footage

`training.soft_risk.soft_flagged_area` relaxes `detector.flash`'s hard
thresholds through a sigmoid so the risk criterion has a gradient. Unit
tests already confirmed the relaxation agrees with the Detector to within
0.001-0.006 on synthetic pairs at four area fractions. This document
measures the same thing on real footage, which is the only place the loss
will actually be applied.

Script: `scripts/soft_risk_report.py`. Profile: `configs/profiles/itu.json`
(`max_area_ratio = 0.25`). Default `tau = 0.02`, unchanged from
`training/soft_risk.py`. Frames were resized so their longer edge is at
most 480px (`--max-dim 480`, the script default), matching the profile the
spec's confidence guard was tuned against. `estimate_global_shift` was used
with its default `min_response = 0.30`, exactly as `detector/pipeline.py`
calls it, so both sides of the comparison see the same shift (or the same
refusal to shift).

## Results (tau = 0.02)

| Clip | Pairs | Mean \|soft-hard\| | p95 \|soft-hard\| | Max \|soft-hard\| | Verdict agreement | Hard mean | Soft mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `test_mp4/wonbon_640.mp4` | 930 | 0.0134 | 0.0272 | 0.0508 | 97.6% | 0.1167 | 0.1301 |
| `test_mp4/anyma_640.mp4` | 1080 | 0.0094 | 0.0146 | 0.0214 | 100.0% | 0.0138 | 0.0232 |
| `test_mp4/cera_640.mp4` | 506 | 0.0112 | 0.0264 | 0.0518 | 99.2% | 0.1182 | 0.1286 |
| `data/raw/real_flicker_eval/videoplayback.mp4` | 184 | 0.0035 | 0.0104 | 0.1609 | 100.0% | 0.7298 | 0.7306 |
| `data/raw/real_flicker_eval/16046-269122203_medium.mp4` | 492 | 0.0070 | 0.0118 | 0.0259 | 99.8% | 0.0455 | 0.0523 |
| `data/raw/real_flicker_eval/1257-144566582_medium.mp4` | 1404 | 0.0116 | 0.0291 | 0.0770 | 99.4% | 0.1339 | 0.1447 |

All six clips decoded fully; no clip failed to decode or ran out of memory.
Every clip's verdict agreement is above the 95% bar set in the brief for
triggering a `tau` sweep, so **no sweep was run** — the brief's Step 3
condition ("if verdict agreement is below 95% on any clip") was never met.
`tau` stayed at its shipped default of 0.02 for every row above; it was not
edited in `training/soft_risk.py`.

## Verdict

The relaxation tracks the Detector well enough to train against. Verdict
agreement ranges from 97.6% to 100.0% across all six clips — all of it real
footage, none synthetic — comfortably above both the 95% bar that would
have triggered a `tau` sweep and the 90% bar the brief treats as a Plan-B
blocker. The worst clip by verdict agreement is `wonbon_640.mp4` at 97.6%
(930 pairs, 22 of them disagreeing on which side of `max_area_ratio =
0.25` they fall), with a mean absolute error of 0.0134 and a max of 0.0508
— consistent with the sigmoid softening a handful of frames that sit right
at the threshold rather than the relaxation drifting from the rule it
stands in for. The largest single-pair error anywhere is on
`videoplayback.mp4` (max 0.1609 on one pair, against a mean error of only
0.0035 over its 184 pairs and 100% verdict agreement) — that clip's hard
mean area (0.7298) sits far above the 0.25 limit for almost its whole
length, so a handful of frames near a local intensity extreme can produce a
large area-fraction gap without changing which side of the pass/fail line
either estimator lands on. Since no clip's agreement is below 90%, this
task finds no evidence that the risk term steers toward a criterion the
Detector does not share on real footage, and there is no blocker here for
starting Plan B.
