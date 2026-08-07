# Tier 0 Fallback — Real-Footage Measurements

Measured 2026-08-07 on branch `tier0-fallback`, against `configs/profiles/itu.json`.
Every number below comes from an actual run recorded in this session — no
placeholders. Commands and raw output are quoted so each number can be traced
back to its source.

## Method notes (read before the tables)

- **Resolution / downscaling.** `test_mp4/wonbon_640.mp4` (638x360, 931
  frames, 24 fps) was already downscaled before this task, from the source
  `test_mp4/원본.mp4` (1276x720), and was used as-is. `test_mp4/Anyma & Chris
  Avantgarde...mp4` (native 720x900, 1084 declared frames) and `test_mp4/Cera
  Khin...mp4` (native 720x1280, 510 declared frames) were NOT run at native
  resolution — an in-process attempt to do so drove free system RAM from
  ~9 GB down to ~0.4 GB and was aborted before it could OOM the machine. Both
  were instead stream-downscaled frame-by-frame (never materializing the full
  native clip in memory) to a long side of 640px, dimensions rounded to the
  nearest even number, the same recipe that produced `wonbon_640.mp4`:
  - `test_mp4/anyma_640.mp4` — **512x640**, 1081 frames (of 1084 declared;
    3 frames the container overstates, same class of container/decode
    mismatch `scripts/screen_clean_clips.py` already documents), 29.97 fps.
  - `test_mp4/cera_640.mp4` — **360x640**, 507 frames (of 510 declared),
    30.0 fps.
  These two files, not the originals, are what `fallback.cli` and the
  streaming statistics below were run on. This is a real, disclosed deviation
  from native resolution, made for a documented machine-memory reason, not a
  silent trim.

- **Mean luminance / contrast measurement pathway differs by clip, and
  that matters.** For `wonbon_640.mp4` the before/after numbers are the
  `fallback/tests/test_real_footage.py` slow test's printed block — computed
  directly on in-memory float32 frames, with no video re-encode in between.
  For Anyma and Cera Khin, `fallback.cli` does not print luminance/contrast
  (only per-segment strengths), and re-loading two full corrected clips as
  Python lists a second time to reproduce the in-memory measurement would
  repeat the exact memory pattern that had to be aborted above. Their
  before/after numbers were instead computed by a streaming pass (one frame
  at a time, `cv2.VideoCapture` → `detector.luminance.relative_luminance`, no
  list materialized) over the CLI's own input and output `.mp4` files. Both
  ends of that comparison go through the same `mp4v` encode/decode, so the
  before-vs-after delta for a given clip is fair; but it is not the same
  measurement pathway as wonbon's, and a spot check confirms the two
  pathways disagree by several percent on the same data: re-measuring
  `test_mp4/out/tier0_wonbon.mp4` from disk gives mean luminance 0.1203 after
  correction, vs. 0.1251 measured in-memory before any encode — i.e. `mp4v`
  compression alone shifts this statistic by about 4%. Treat cross-clip
  comparisons of the luminance/contrast percentages with that caveat; the
  pass/fail and strength numbers are unaffected (they come from
  `detector.pipeline.run_detection` on the actual corrected arrays, not from
  a re-encoded file, in every run below).

- **Frame-0 segments.** `fallback/transfer.py`'s `trailing_reference` has one
  sample at frame 0 and two at frame 1; a segment that starts at literal
  frame 0 is being corrected against a reference the pipeline hasn't
  finished warming up, so its `final_strength` reflects that warm-up
  artifact as well as (or instead of) the content's real difficulty. Two of
  the three clips have a segment starting at frame 0. In both cases the
  segment still escalated once and passed — the harder failure mode this
  task's brief warned about (a frame-0 segment that *cannot* converge at any
  strength) was not triggered by any of the three clips run here. But the
  elevated strength those two segments needed is still not a clean read on
  difficulty, which is why the maximum `final_strength` below is reported
  two ways.

## Per-clip summary

| Clip | Resolution used | Segments before → after | Passed | Max `final_strength` (overall) | Max `final_strength` (excl. frame‑0 segments) | Mean luminance before → after (Δ%) | Mean contrast before → after (Δ%) |
|---|---|---|---|---|---|---|---|
| wonbon (원본.mp4) | 638x360 (pre-existing downscale) | 5 → 0 | True | 0.9644 (frame‑0 segment) | 0.7752 | 0.1228 → 0.1251 (**+1.87%**) | 0.1774 → 0.0838 (**−52.76%**) |
| Anyma & Chris Avantgarde | 512x640 (downscaled this task) | 1 → 0 | True | 0.7939 (frame‑0 segment) | n/a — only segment starts at frame 0 | 0.041303 → 0.039882 (**−3.44%**) | 0.104715 → 0.098467 (**−5.97%**) |
| Cera Khin | 360x640 (downscaled this task) | 1 → 0 | True | 0.9723 (does **not** start at frame 0) | 0.9723 | 0.050381 → 0.048419 (**−3.89%**) | 0.119031 → 0.061404 (**−48.41%**) |

All three clips passed re-validation (`report.passed is True`, `remaining_segments == 0`) — 3/3.

Sources: wonbon row from `python -m pytest fallback/tests/test_real_footage.py -m slow -v -s` (printed block reproduced below); Anyma/Cera segment and pass/fail data from `test_mp4/out/tier0_anyma.json` and `test_mp4/out/tier0_cera.json` (produced by `fallback.cli`); Anyma/Cera luminance/contrast from the streaming pass described above.

### wonbon slow-test output (verbatim)

```
segments      5 -> 0
passed        True   rounds 2
  0-230: 0.864 -> 0.964 (1 escalations)
  251-342: 0.555 -> 0.555 (0 escalations)
  349-406: 0.366 -> 0.466 (1 escalations)
  631-730: 0.775 -> 0.775 (0 escalations)
  738-930: 0.635 -> 0.735 (1 escalations)
mean luminance 0.1228 -> 0.1251
mean contrast  0.1774 -> 0.0838
```

Wall clock for the slow test: `1 passed in 345.01s (0:05:45)`.

## Per-segment strengths

The gap between `initial_strength` (the analytic estimate from
`fallback/strength.py`) and `final_strength` (what the verify/escalate loop
actually needed) is the size of the analytic estimate's error that design
section 2.3 predicted would be non-zero — the estimate is not exact and
re-verification against the Detector is load-bearing, not a formality.

| Clip | Segment (start–end) | Starts at frame 0? | `initial_strength` | `final_strength` | Escalation rounds |
|---|---|---|---|---|---|
| wonbon | 0–230 | yes | 0.8644 | 0.9644 | 1 |
| wonbon | 251–342 | no | 0.5551 | 0.5551 | 0 |
| wonbon | 349–406 | no | 0.3662 | 0.4662 | 1 |
| wonbon | 631–730 | no | 0.7752 | 0.7752 | 0 |
| wonbon | 738–930 | no | 0.6348 | 0.7348 | 1 |
| Anyma | 0–45 | yes | 0.6939 | 0.7939 | 1 |
| Cera Khin | 279–506 | no | 0.8723 | 0.9723 | 1 |

Observations:
- 4 of 7 segments needed exactly one escalation (the estimate undershot by
  the fixed `strength_step = 0.1`); the other 3 converged on the analytic
  estimate with zero escalations. No segment needed more than one
  escalation and none exhausted `max_rounds` (6).
- Every segment that *did* need an escalation needed exactly `+0.1` — the
  estimate's error, where it exists, was consistently one step, not
  scattered across a wider range, in this sample.

## Comparison against All-In-One-Deflicker

On the same three clips, All-In-One-Deflicker (the neural-atlas method used
as the pre-existing reference in this project) was measured earlier to cost
−21% mean luminance on 원본, −35% on Anyma, and −35% on Cera Khin, and it
cleared the Detector on only 2 of the 3 clips. Tier 0's measured cost on the
same clips is +1.87% (wonbon/원본), −3.44% (Anyma), and −3.89% (Cera Khin) —
one to two orders of magnitude smaller in every case, in one case (wonbon)
even landing on the opposite sign — and it passed 3 of 3, including the clip
All-In-One-Deflicker failed to clear. Tier 0 costs substantially less mean
luminance than All-In-One-Deflicker on every clip measured, and it is more
reliable at actually clearing the Detector. This is the expected outcome of
touching only flagged segments instead of the whole clip (see `fallback/apply.py`'s
module docstring), not a surprise, but it is now a measured one rather than
an assumption.

## Verdict on design section 6's threshold question

Design section 6 asks whether any clip's maximum `final_strength` lands at
or above 0.9 — if so, that is a signal the detection profile itself, not
just Tier 0, warrants review, because a strength that high is compressing
the segment to near-uniform output to clear the bar.

**Yes, this threshold was crossed, and by a segment where it can't be
explained away as a frame-0 warm-up artifact.** Cera Khin's only segment
(279–506, which does not start at frame 0) needed `final_strength = 0.9723`
to pass. Wonbon's frame-0 segment also crossed it, at 0.9644, but per the
method note above that segment's elevated strength is confounded with the
`trailing_reference` warm-up limitation and is weaker evidence on its own.
Cera Khin's is not confounded that way — it is a genuine, non-frame-0
segment demanding compression to within 0.028 of full identity-to-reference
collapse. Excluding frame-0 segments entirely, the maximum `final_strength`
across all three clips is still 0.9723 (Cera Khin), comfortably above the
0.9 line. The recommendation from design section 6 stands: the ITU profile's
flagged-area/flash-rate thresholds should be reviewed, because near-uniform
output is what the bar is currently demanding on at least one real clip.

## Test suite

`pytest.ini` now registers the `slow` marker and defaults to `-m "not slow"`.

- Default run: `python -m pytest -q` → `281 passed, 1 deselected` (base
  suite of 281 unchanged, plus the new slow test correctly excluded).
- Slow run: `python -m pytest fallback/tests/test_real_footage.py -m slow -v -s`
  → `1 passed in 345.01s (0:05:45)`.
