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

  **The effect of downscaling on the strength numbers themselves is
  disclosed but not assessed, and it should be.** `required_strength`'s
  quantile is taken over the per-pixel motion-compensated luminance delta
  between consecutive frames. Downscaling to a 640px long side is a low-pass
  operation, and it low-passes exactly those per-pixel deltas before the
  quantile ever sees them — a hard single-pixel edge at native resolution
  becomes a softer, spatially-averaged gradient at 640px, which can shift
  the quantile value in either direction depending on the flash's spatial
  structure. Every strength in this document, including the ≥0.9 values
  discussed below, is therefore a 640px value. Whether it over- or
  under-estimates the strength `required_strength` would compute at native
  resolution is unknown — that comparison has not been run.

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

  **That caveat was scoped too narrowly above — it also bears on the
  within-clip deltas, not just cross-clip comparisons.** Anyma's and Cera
  Khin's before/after numbers both pass through the same `mp4v`
  encode/decode on both ends, and the claim that this makes the
  before-vs-after *delta* "fair" (the two encode passes' shifts cancel) is
  asserted above, not measured — no spot check like the wonbon one was run
  for a `mp4v`-encoded-on-both-ends pair to confirm the shifts actually
  cancel rather than partially compound. That matters here specifically
  because Anyma's −3.44% and Cera Khin's −3.89% are the same order of
  magnitude as the ~4% shift the wonbon spot check measured for a single
  encode pass. If the two encode passes on either clip do not cancel as
  cleanly as assumed, a meaningful fraction of those two reported deltas
  could be re-encode artifact rather than the correction's real effect.
  Until the cancellation assumption is checked directly, Anyma's and Cera
  Khin's luminance/contrast deltas should be read as bounded by, not clean
  of, that ~4% pathway noise.

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

| Clip | Resolution used | Segments before → after | Frames touched (coverage) | Passed | Max `final_strength` (overall) | Max `final_strength` (excl. frame‑0 segments) | Mean luminance before → after (Δ%) | Mean contrast before → after (Δ%) |
|---|---|---|---|---|---|---|---|---|
| wonbon (원본.mp4) | 638x360 (pre-existing downscale) | 5 → 0 | 674 / 931 (**72.4%**) | True | 0.9644 (frame‑0 segment) | 0.7752 | 0.1228 → 0.1251 (**+1.87%**) | 0.1774 → 0.0838 (**−52.76%**) |
| Anyma & Chris Avantgarde | 512x640 (downscaled this task) | 1 → 0 | 46 / 1081 (**4.3%**) | True | 0.7939 (frame‑0 segment) | n/a — only segment starts at frame 0 | 0.041303 → 0.039882 (**−3.44%**) | 0.104715 → 0.098467 (**−5.97%**) |
| Cera Khin | 360x640 (downscaled this task) | 1 → 0 | 228 / 507 (**45.0%**) | True | 0.9723 (does **not** start at frame 0) | 0.9723 | 0.050381 → 0.048419 (**−3.89%**) | 0.119031 → 0.061404 (**−48.41%**) |

All three clips passed re-validation (`report.passed is True`, `remaining_segments == 0`) — 3/3.

**Coverage matters for reading the luminance column above.** These are
whole-clip means, but Tier 0 only ever touches the risk segment(s), so the
mean dilutes each clip's real within-segment change by the fraction of
frames left untouched. Anyma's whole-clip −3.44% comes from a correction
applied to only 4.3% of its frames — the other 95.7% contribute exactly 0 to
that delta. The whole-clip number is not a fair read of how hard the
correction hit the frames it actually touched; see the comparison section
below for why contrast, not luminance, is the honest cost figure regardless.

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
- The estimate's error is ≤ 0.1 in 7 of 7 cases. Note that this is a weaker
  claim than it sounds: `strength_step = 0.1` is the *only* step size the
  escalation loop tried in this sample, so "every miss was one step" cannot
  by itself distinguish a tightly-bounded estimation error from a loosely
  bounded one — a segment whose true requirement was 0.35 above the estimate
  would also show up as "needed escalations" without revealing how many
  steps it actually took to discover, since the loop stops as soon as the
  Detector passes. A step size smaller than 0.1 (e.g. the 0.02 recommended
  below for the Cera Khin bisection) is needed before "the estimate's error
  is small" can be claimed with any precision finer than one decimal place.

## Comparison against All-In-One-Deflicker

On the same three clips, All-In-One-Deflicker (the neural-atlas method used
as the pre-existing reference in this project) was measured earlier to cost
−21% mean luminance on 원본, −35% on Anyma, and −35% on Cera Khin, and it
cleared the Detector on only 2 of the 3 clips. **Those All-In-One-Deflicker
figures trace only to an assertion in this project's earlier notes** — there
is no recorded run of All-In-One-Deflicker in this repository to point to,
and the resolution it was measured at is not stated. They are quoted here
because they are the only reference numbers available, not because they have
been independently verified; treat the comparison below as directional, not
exact.

Tier 0's measured whole-clip mean-luminance cost on the same clips is +1.87%
(wonbon/원본), −3.44% (Anyma), and −3.89% (Cera Khin). Taking the ratio of
each All-In-One-Deflicker figure to Tier 0's on the same clip gives 11.2×
(원본: 21/1.87), 10.2× (Anyma: 35/3.44), and 9.0× (Cera Khin: 35/3.89) — this
is **roughly one order of magnitude smaller, not "one to two orders of
magnitude"** as an earlier draft of this section claimed; 9–11× is a single
order of magnitude. Tier 0 also passed 3 of 3 on the Detector, including the
clip All-In-One-Deflicker failed to clear.

Two further corrections to how that ratio should be read:

1. **The luminance figures are diluted by coverage, not just gentler by
   design.** All-In-One-Deflicker corrects the whole clip; Tier 0 corrects
   only the risk segment(s) — 72.4% of wonbon's frames, 4.3% of Anyma's, and
   45.0% of Cera Khin's (see the coverage column above). Anyma's whole-clip
   −3.44% is therefore mostly zero contribution from the 95.7% of frames
   Tier 0 never touched, diluting whatever the correction did to the 4.3% it
   did touch — that is 96% dilution, not evidence the correction itself was
   gentle. The 11.2×/10.2×/9.0× ratios above compare a whole-clip cost
   against a *partial-clip* cost, which flatters Tier 0 on any clip where
   its risk segments are a small share of the frames (Anyma most severely).
   A within-segment mean-luminance delta, or an All-In-One-Deflicker number
   measured only over the same frame ranges, would be the apples-to-apples
   comparison; neither has been computed here.

2. **Mean luminance is close to the wrong statistic to compare on at all,
   by construction.** `compress_contrast` compresses every pixel toward the
   segment's trailing luminance mean (design section 2.1) — that is
   specifically a transform that is near mean-*preserving*, not one that
   happens to score well on a mean-luminance metric. wonbon's own number
   demonstrates this directly: its mean luminance *increased* by 1.87%
   rather than decreasing, because compression pulled dark flash troughs up
   toward the mean as often as it pulled bright peaks down. A method that is
   near-invariant on a statistic by construction will always look
   inexpensive on that statistic, independent of how much it actually
   altered the footage. **Contrast (standard deviation) is the honest cost
   figure for this transform**, and it is not small: −52.76% for wonbon and
   −48.41% for Cera Khin (Anyma is milder at −5.97%, consistent with its
   correction touching only 4.3% of frames). Those numbers are already in
   the per-clip table above but were absent from this comparison paragraph
   in the original draft — a reader who stopped at the luminance figures
   would have concluded Tier 0's cost is uniformly tiny, when for two of the
   three clips it is compressing contrast by roughly half within the
   segments it touches.

Net: Tier 0 is still the only method of the two that clears the Detector on
all three clips, and its whole-clip mean-luminance cost is genuinely far
smaller than All-In-One-Deflicker's reported figures. But "far smaller" is
partly a coverage artifact and partly a statistic this transform cannot
score badly on by construction; the contrast numbers, not the luminance
ratio, are the number to carry into any comparison against the neural
Mitigator's own cost.

## Verdict on design section 6's threshold question

Design section 6 asks whether any clip's maximum `final_strength` lands at
or above 0.9 — if so, that is a signal the detection profile itself, not
just Tier 0, warrants review, because a strength that high is compressing
the segment to near-uniform output to clear the bar.

**This conclusion is not yet established, and an earlier draft of this
section overstated it.** Cera Khin's only segment (279–506, which does not
start at frame 0) needed `final_strength = 0.9723` to pass — but
`final_strength` is not the segment's true minimum passing strength; it is
that minimum rounded *up* to the nearest multiple of `strength_step = 0.1`
by the escalation loop (design section 4). The loop stops the instant the
Detector passes, so it never checks whether
anything below the value that passed would also have worked. All this
number establishes is that Cera Khin's true minimum passing strength lies
somewhere in the interval **(0.8723, 0.9723]** — and 0.88, comfortably below
the 0.9 threshold this conclusion turns on, is inside that interval and has
not been ruled out.

Two further, independent sources of inflation compound this, both already
described in design section 2.3 and `fallback/strength.py`'s docstring but
not previously connected to the 0.9 verdict:

1. **`required_strength` takes the max over frame pairs across the whole
   segment.** Cera Khin's segment is 228 frames (279–506) at 30 fps — 7.6
   seconds. The reported strength is driven entirely by whichever single
   frame pair in those 7.6 seconds had the worst motion-compensated delta
   quantile; every other pair in the segment needed less.
2. **That single worst-pair value is then applied uniformly to the entire
   7.6-second segment** (`fallback/apply.py`'s `_apply_segments`), not just
   to the frames near the worst pair. Milder pairs elsewhere in the segment
   are over-corrected as a direct consequence.

Each of these three effects (quantization, max-over-pairs, uniform
application of that max) pushes the reported 0.9723 upward relative to what
a more targeted or finer-grained method would need. None of them has been
measured, so their combined size is unknown — they could be small, or they
could be enough to put Cera Khin's real difficulty comfortably under 0.9.

**This does not mean the observation should be discarded — it means it is
not yet a finding.** 0.9723 is a real number this branch measured on real
footage, and a value that high, even as an upper bound, is still notable.
But "the ITU profile's thresholds warrant review" is a conclusion that
should follow evidence, not precede it, and right now the evidence only
supports "somewhere in (0.8723, 0.9723], possibly below 0.9." The
recommended next step is a bisection search on Cera Khin's segment 279–506
at `strength_step = 0.02` (five times finer than the 0.1 used here) to
narrow that interval before anyone treats the ≥0.9 threshold as crossed and
acts on it.

### Resolved: the bisection was run, and 0.9 is not crossed

Run on `test_mp4/cera_640.mp4` (507 frames, 360x640, 30 fps, segment
279–506) by calling `_apply_segments` at a fixed strength and re-running
`run_detection` over the whole clip at each step — the same pass/fail test
the escalation loop uses:

```
s=0.980 -> PASS   (upper anchor)
s=0.800 -> FAIL   (lower anchor)
s=0.890 -> PASS
s=0.845 -> FAIL
s=0.867 -> FAIL
s=0.879 -> PASS

minimum passing strength lies in (0.867, 0.879]
```

**The true minimum is 0.879, not 0.9723.** The ≥0.9 threshold is not
crossed, and the "the ITU profile's thresholds warrant review" observation
is withdrawn: this clip clears the ITU bar below 0.9, so nothing here
argues the profile is unachievable. The suggestion may be revived only on
new evidence, not on this measurement.

Two consequences follow, and they matter more than the withdrawal.

**The analytic estimate was accurate to 0.007, not 0.1.** `required_strength`
predicted 0.8723; the true minimum is 0.879. The entire 0.093 gap between
the estimate and the reported 0.9723 came from the escalation step size —
0.8723 failed by a hair, and the loop's only available response was to add
a full 0.1. The "error ≤ 0.1 in 7 of 7 cases" statement above is therefore
not just circular but badly pessimistic on at least this segment; the
closed-form estimate is far better than this branch's own numbers suggest.

**Tier 0 therefore over-corrected this clip by roughly 9 percentage points
of strength.** It applied 0.9723 where 0.879 would have passed, destroying
contrast that the threshold never demanded — on 45% of the clip's frames.
`strength_step` is a plain CLI argument, so lowering it to 0.02 costs only
extra detection passes (each round is one `run_detection` over the clip)
and buys a strictly gentler correction. That trade should be measured
before Tier 0 is used for anything beyond measurement; the current 0.1
default was chosen for convergence speed, with no evidence about its
damage cost, and this measurement is the first evidence that the cost is
real.

## Test suite

`pytest.ini` now registers the `slow` marker and defaults to `-m "not slow"`.

- Default run at the time these numbers were measured: `python -m pytest -q`
  → `281 passed, 1 deselected` (base suite of 281 unchanged, plus the new
  slow test correctly excluded).
- Slow run: `python -m pytest fallback/tests/test_real_footage.py -m slow -v -s`
  → `1 passed in 345.01s (0:05:45)`.
- **Post-fix-wave update (2026-08-07, this document's own correction pass):**
  the code fixes above (`SegmentOutcome.rounds` → `escalations` and its
  snapshot bug, the `max_rounds < 1` guard, `fallback/cli.py`'s
  `VideoReadError` handling) added 3 regression tests. Targeted run:
  `python -m pytest fallback/tests/test_apply.py fallback/tests/test_cli.py
  detector/tests -v` → `89 passed`. Default run: `python -m pytest -q` →
  `284 passed, 1 deselected`. The slow real-footage test was not re-run for
  this pass (machine was decoding video for another process at the time;
  see this fix wave's report) — the wonbon numbers above are unchanged from
  the original measurement run and have not been re-verified against the
  renamed/fixed fields.
