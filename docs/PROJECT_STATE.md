# Project state — 2026-08-09

Written so a fresh Claude Code session on another machine can pick up where
this one left off. Claude Code's memory lives outside the repo and is keyed
to a machine and a project path, so it does not travel; this file does.

**Read this first, then `git log --oneline -15` for what actually landed.**

---

## Where things stand

The self-supervised rewrite is **merged to `master`** (`6be89c2`), 331 tests
passing. The mitigator no longer trains to reconstruct a clean reference —
an objective real footage cannot supply. It learns from the degraded frames
alone: a temporal-consistency term against the motion-compensated
neighbour, plus a differentiable relaxation of the detector's own risk
criterion.

Also on `master`: the **Tier 0 fallback**, the project's first output to
pass Detector re-validation (3/3 real clips).

Phase 1 and **phase 2 are both finished**. Training on real footage works —
2 of the 6 evaluation clips now pass, against 0 before, and one of them
beats the Tier 0 fallback on cost as well as on risk. The other four do not.
Read the next two sections before planning anything.

---

## Phase 2 result — 2 of 6, and where the other 4 fail

Trained on 14 real clips (8,190 frames, ingested at 512). Two runs, same
settings, differing only in initialisation. Both reach 30 epochs in under
an hour: **110 s per epoch**, 5,865 train examples.

**`--init-from` won**, by the criterion the design spec set: total remaining
risk segments first, total triggering frames as the tiebreak.

| | from scratch | `--init-from` phase 1 |
|---|---|---|
| best epoch | 28 | **16** |
| `val_loss` | 0.2674 | **0.2538** |
| `soft_area` | 0.0489 | **0.0452** |
| triggering frames, 3 Tier-0 clips | 213 | **157** |

### The clips with a Tier 0 baseline — 2 of 3

| clip | Tier 0 | phase 1 best | **phase 2** |
|---|---|---|---|
| wonbon | 5 → 0 | 5 → 2 | **5 → 0** |
| Anyma | 1 → 0 | 1 → 0 | **1 → 0** |
| Cera Khin | 1 → 0 | 1 → 1 | 1 → 1 |

**wonbon is the first output in this project to clear both halves of spec
§7 at once** — zero remaining segments *and* a smaller in-segment contrast
cost than Tier 0 (−48.6% against −52.8%). All 574 of its triggering frames
are gone and peak windowed area fell 64.5%.

Anyma passes on risk and loses on cost: −29.9% against Anyma's own Tier 0
figure of **−6.0%**. (Not −52.8% — see the misattribution note under
Known-deferred items. Read against the wrong baseline this looks like a win.)

### The untouched holdout — 0 of 3

These three were never read until the final evaluation. They are harder,
and the spec chose them precisely to be:

| clip | segments | triggering frames | peak area |
|---|---|---|---|
| `1257-144566582_medium` | 4 → **6** | 867 → 580 (−33%) | +0.3% |
| `16046-269122203_medium` | 1 → 1 | 71 → 27 (−62%) | **+16.0%** |
| `videoplayback` | 1 → 1 | 158 → **158** | 1.000 → 1.000 |

Three distinct failures, not one:

- **`1257` got worse by the bar that counts.** Triggering frames fell a
  third, but the segment count rose from 4 to 6 — the correction punched
  holes in long risky stretches and split them. Fewer risky frames, more
  risky *segments*, and the bar counts segments.
- **`16046` over-corrects.** It is the deliberately weak case (area 0.331),
  chosen to test exactly this. Triggering frames fell 62% while peak
  windowed area rose 16%.
- **`videoplayback` is untouched.** 158 → 158 triggering frames, peak area
  1.000 → 1.000, nothing moved at all. Yet mean luminance fell 19.9% and
  contrast 16.8%, so the model *did* alter the video — it changed the
  picture without changing the verdict. This is the whole-frame-flagged
  case, and it is the clearest evidence that the remaining problem is the
  correction's quality inside the mask, not its reach.

**Do not report phase 2 as "2 of 3".** It is 2 of 6. The three clips with
Tier 0 baselines are the easier half.

### The output is visibly purple, and the objective is why

Watched for the first time on 2026-08-09. Measured on wonbon, means over
the 674 in-segment frames:

| channel | original | mitigated | ratio |
|---|---|---|---|
| R | 0.1084 | 0.1084 | **1.000** |
| G | 0.1084 | 0.0825 | **0.761** |
| B | 0.1084 | 0.1084 | **1.000** |

Outside the segments all three channels sit at 0.98, so this is the
correction, not the codec.

**Those means mislead, and the per-pixel figures are the real story.** Over
120 corrected frames, mean per-pixel `|delta|` is R 0.0390, G 0.1412,
B 0.0144, and roughly 72% of pixels move in *every* channel — red and blue
change plenty, their means just cancel. What matters is the ratio,
**G : R : B = 10.1 : 2.8 : 1**, against the luminance coefficients
`0.7152 : 0.2126 : 0.0722` = **9.9 : 2.9 : 1**.

The correction is distributed across channels in proportion to each
channel's luminance weight. That is not a quirk — it is the exact minimum-
norm solution to "change luminance by this much, with the least RGB
change", which is what the objective asks for. Every loss term is computed
on `relative_luminance` (`detector/luminance.py:14`) and `fidelity`
penalises raw RGB change, so the model is solving the stated problem
correctly.

The hue shifts because a *fixed-direction* subtraction is not a *scaling*.
Reducing a pixel proportionally to its own (R, G, B) preserves its colour;
subtracting a constant-direction vector does not.

**Nothing in the loss asks it to preserve colour.** Same failure mode as
the removed histogram path recorded in `mitigator/arch.py`: the objective
was satisfiable in a way nobody intended, and the model complied.

**FIXED, without retraining — `mitigate_frame(..., preserve_hue=True)`,
`--preserve-hue` on `mitigator.cli`.** Instead of adding the residual, take
only the luminance it implies and apply that as a scale on *linear* RGB.
Scaling linear RGB by a scalar leaves chromaticity unchanged by
construction, so the hue shift becomes impossible rather than penalised.
(Linear light matters: multiplying encoded sRGB would shift colour again
through the gamma curve. Pixels below a luminance floor have no
chromaticity and fall back to the additive result.)

Measured on wonbon, 27.5M lit pixels over 120 corrected frames:

| | additive | `--preserve-hue` |
|---|---|---|
| mean chromaticity shift | 0.23748 | **0.02035** (−91.4%) |
| risk segments | 5 → 0 | 5 → **0** |
| triggering frames | 574 → 0 | 574 → **0** |
| peak windowed area | −64.5% | −64.2% |
| in-segment contrast | −48.6% | **−48.3%** |

**The colour comes free.** The verdict is unchanged and the contrast cost
is fractionally *lower*, still under Tier 0's −52.8%. This works because
every term that decides the verdict reads luminance only, so preserving
chromaticity costs the objective nothing.

Off by default so far, and the additive path is byte-for-byte unchanged —
only wonbon has been checked. Verify on the other five evaluation clips
before making it the default.

It also revises the deferral of the boundary-taper work below. That
deferral was argued on brightness steps alone, measured against each
clip's own p95 frame-to-frame luminance change. A 24% green step at a
segment edge is a colour discontinuity, which is not in that comparison at
all and is far more visible.

### Two things real data settled that synthetic data could not

**The risk term is finally live.** `soft_area` starts at **0.2146** on this
corpus against 0.0774 on synthetic init, and `risk` at 0.0705 against
0.0063 — eleven times higher. The note further down about `--lambda-risk`
sitting in the softplus's flat region was true *of synthetic data*: real
footage starts just under the ITU `max_area_ratio` of 0.25, where the
softplus has slope. `--lambda-risk` is worth tuning now; it was not before.

**Global shift estimation is more reliable here, not less.**
`trusted_fraction` is **0.9442** against 0.8842 on synthetic. The design
spec called camera motion its largest unvalidated risk. It is not the
problem.

### What phase 2 says to do next, in order

1. **~~Fix the hue shift.~~ DONE** on wonbon — `--preserve-hue`, no
   retraining, verdict unchanged. Re-check it on the other five clips and
   then make it the default.
2. **Fix segment fragmentation.** `1257` is the only clip that got *worse*
   by the bar, going 4 → 6 segments while losing a third of its triggering
   frames. A correction that thins a long risky stretch without clearing it
   splits one segment into several, and the bar counts segments. This is
   the cheapest failure to understand and may be shared with Cera.
3. **Tune `--lambda-risk`** — see above; it now has gradient to give.
4. **Look at `videoplayback`.** Verdict completely unmoved while luminance
   fell 19.9%: the model changed the picture and not the risk. Whatever is
   wrong there is not reach, not motion, and not data volume.
5. **`16046` over-corrects** (peak area +16%). Watch it as the
   over-correction canary the spec chose it to be.

---

## Phase 1 is done, and it did not clear the bar

30 epochs at `--lambda-temporal 10 --batch-size 16 --patch-size 256`, 5.7
hours, no divergence and no collapse. `best.pt` is epoch 27.

| | init | best (e27) | target below |
|---|---|---|---|
| `soft_area` | 0.0774 | **0.0370** (−52%) | ~0.050 — beaten |
| `temporal` | 0.0258 | **0.0105** (−59%) | ~0.010 — reached |
| `fidelity` | 0.0217 | 0.0623 | ~0.02 — not held |
| `val_loss` | 0.2324 (e0) | 0.1683 | — |

**The loss design works. It does not transfer.** Both flicker terms beat
their targets on synthetic validation. The checkpoint was then run over the
three real clips the Tier 0 baseline was measured on:

| clip | Tier 0 | after 3 epochs | after 30 epochs |
|---|---|---|---|
| wonbon | 5 → **0** | 5 → 2 | 5 → **4** |
| Anyma | 1 → **0** | 1 → **0** | 1 → **1** |
| Cera Khin | 1 → **0** | 1 → 1 | 1 → **1** |

**More training made the real-clip verdict worse: 1/3 → 0/3.** Anyma passed
at 3 epochs (30 triggering frames → 0) and fails at 30 (30 → 30). wonbon's
peak windowed area went from −51.2% at 3 epochs to **+7.0%** at 30.

In-segment contrast cost fell everywhere over the same stretch — wonbon
−48.9% → −41.0%, Anyma −28.8% → −23.9%, Cera −53.6% → −50.4%. Both
movements are one fact: **the model got gentler.** `fidelity` agrees
(0.0695 at e2, 0.0623 at e27).

So **`val_loss` does not track the goal.** It improved 28% across the run
while the criterion the project is graded on went backwards. Do not select
a checkpoint, or decide to stop, on it alone.

The likely cause is the synthetic-to-real gap rather than the objective.
Synthetic `soft_area` starts at 0.0774 — a world where 8% of the frame is
flagged. These three clips peak at 0.630 / 0.401 / 0.658, and all 17 clips
in `data/raw/real_flicker/` peak at **1.000**. The model spent 30 epochs
specialising to a regime real footage is not in. λ tuning does not fix
that. Phase 2 does, and this is the measurement showing it is not optional.

The 3-epoch checkpoint — the better of the two on the real bar — is kept at
`mitigator/weights/phase1/epoch2_evaluated.pt`. Note also that the trainer
only ever writes `best.pt` and `latest.pt`, so the epoch with the lowest
`soft_area` (e22, 0.0328) was not saved and cannot be evaluated.

---

## The one thing that will waste your time if you miss it

**Never train with the default `--lambda-temporal`.** It is `1.0`, and at
1.0 the model collapses to the trivial solution: predict a zero residual,
satisfy `fidelity` perfectly, leave the flicker untouched.

Measured here — a 3-epoch run produced a model that corrected *less than a
randomly initialised one*:

| | \|delta\| / \|input\| |
|---|---|
| fresh init | 3.94% |
| after 3 epochs at λ_t = 1 | **0.13%** |

Only `fidelity` improved (0.02156 → 0.00052). `temporal`, `risk` and
`soft_area` all got slightly worse. Validation loss went 0.0614 → 0.0614 →
0.0599 across three epochs.

A 150-step sweep on one real validation batch, fresh init each time:

| λ_temporal | \|delta\|/\|in\| | soft_area | fidelity | temporal | risk |
|---|---|---|---|---|---|
| 1.0 | 0.458% | 0.07834 | 0.00118 | 0.02532 | 0.00637 |
| **10.0** | **7.330%** | **0.05043** | **0.02043** | **0.01020** | 0.00409 |
| 50.0 | 13.877% | 0.04571 | 0.05818 | 0.00933 | 0.00200 |
| 200.0 | 111.815% | 0.02530 | 0.30283 | 0.00853 | 0.00060 |
| *init* | 10.059% | 0.07744 | 0.02166 | 0.02578 | 0.00628 |

λ_t = 10 is close to free: temporal −60%, soft_area −35%, fidelity
essentially unchanged. λ_t = 200 destroys the image — the correction is
larger than the frame itself.

**`--lambda-risk` cannot rescue this.** `soft_area` starts at 0.077, three
times *below* the ITU `max_area_ratio` of 0.25, so the risk term's softplus
sits in its flat region: it adds to the loss value and supplies almost no
gradient. Only λ_temporal moves the output.

So:

```bash
python -m training.train_mitigator \
  --data-dir data/synthetic \
  --profile configs/profiles/itu.json \
  --checkpoint-dir mitigator/weights/phase1 \
  --epochs 30 --batch-size 8 --patch-size 256 \
  --lambda-temporal 10 --num-workers 2
```

The sweep is one batch at 150 steps. The direction is solid, the exact
optimum is not — run 2–3 epochs and watch `soft_area` and `fidelity` before
committing to a long run.

---

## How to judge the run you just started

Watch three numbers per epoch. The reference points are from a fresh init
on one real validation batch:

| | init | healthy after training | collapse |
|---|---|---|---|
| `fidelity` | 0.0217 | ~0.02, roughly flat | **→ 0.001** |
| `temporal` | 0.0258 | falling toward ~0.010 | flat or rising |
| `soft_area` | 0.0774 | falling toward ~0.050 | flat or rising |

**`fidelity` is the signal that matters.** If it collapses toward 0.001 the
model has chosen the trivial solution and nothing else in the log means
anything — stop and raise λ_temporal. If it climbs past ~0.06 you are into
over-correction; λ_t = 200 measured 0.303 and produced a correction larger
than the frame itself.

**The ~0.06 line is softer than it reads.** The completed 30-epoch run sat
at 0.060–0.073 for almost its whole length and was *under*-correcting on
real footage, not over-correcting — the clips it failed, it failed by
leaving flicker in. Treat 0.06 as "watch this", and 0.30 as the number that
actually meant a ruined picture. Judge over-correction on the contrast
figure from `scripts/evaluate_mitigation.py`, not on `fidelity` alone.

**None of these three numbers predicts the Detector verdict.** All three
moved the right way across 30 epochs while the real-clip result went from
1/3 to 0/3. They tell you the run is healthy; they do not tell you it is
working. Only re-validation on real clips does that, so run it — at 3
epochs as well as at the end, because the two disagreed here.

Larger batches mean fewer optimiser steps per epoch: 9259 train examples is
1157 steps/epoch at `--batch-size 8` but only 579 at 16. A small measured
improvement over 3 epochs is not failure — check the *direction* first, and
add epochs before changing anything else.

**Expect the epoch numbers to oscillate.** Over 30 epochs `val_loss` swung
±0.02 on a roughly 3-epoch period on top of a real downward trend, and two
consecutive worse epochs twice looked like the onset of collapse and twice
were not. `fidelity` and the flicker terms move *against* each other epoch
to epoch (e17: 0.0550 / temporal 0.0129; e18: 0.0715 / 0.0108), so a flat
`val_loss` can hide both still moving. Do not act on fewer than three
consecutive epochs in the same direction.

Use `--log-every N` to print a running mean of every term every N steps;
without it an epoch is 11+ minutes of silence. It reports the mean over the
last N steps, not a cumulative average, so a term that starts to diverge is
visible rather than averaged away. Those step figures come from 256×256
crops and are **not** comparable to the epoch line's full-resolution
validation values — compare step lines to step lines only.

## What to do after phase 1

Roughly in order. Items in the same group are independent of each other.

**1. ~~Decide what replaces the mask blend.~~ RETIRED — the premise was
wrong.** Measured on 2026-08-08; do not re-open it without reading this.

`max_area` is a per-clip maximum of a **1-second window**, not a per-frame
mask saturation rate. Per frame, masks are effectively full on **5 of 674**
in-segment frames for wonbon (0.7%) and **0 of 234** for Cera. In-segment
mask area averages 0.351 and 0.554 — the blend is protecting most of the
frame, not nothing.

The follow-up hypothesis — that the blend *confines* the correction — is
also wrong, and structurally so. `compute_prior` builds `mask_i` as an OR
over recent raw masks, so it always contains `raw_i`: of 23.5M flagged
pixels on wonbon and 11.2M on Cera, **zero** fell outside the given mask.
88% (wonbon) and 95% (Cera) of the flicker that survives correction sits
*inside* the region the model was free to edit. The problem is the quality
of the correction there, and phase 2's `videoplayback` result says the same
thing (whole frame flagged, verdict completely unmoved).

One real defect remains: the blend's hard spatial edge *creates* flicker.
5–12% of post-correction flagged pixels were not flagged before and lie
outside the mask — a pixel corrected at frame i−1 and left original at
frame i steps. Same class as the un-ramped segment boundary. Both are
deferred; see Known-deferred items.

**2. ~~Decide the frame-0 and warm-up handling.~~ DONE.** The first
`round(fps)` frames are excluded as training centres (they stay in
`degraded/` so `compute_prior` still sees real neighbours; trimming the
clip would just move the warm-up). Original note kept for context: The Detector fabricates an
all-ones mask for frame 0 and mask persistence ORs it forward ~0.2 s;
`uncertain_frames` equals fps exactly on every clip. Real segments start at
frame 0, so phase 2 trains on whole-frame corrections there unless
something changes.

**3. Normalise the corpus.** Resolutions span 360×640 to 3840×2160 and the
ITU criterion is an area ratio. Drop one of the two
`56426-479655491*` files (same source, two resolutions, and `clip_split`
hashes the id so they can straddle the train/val split). Check whether the
two Travis Scott clips are the same performance. Decide whether the
61-frame clip is usable at all.

**4. Build the three missing pieces** (see the section below): the
injection-free ingestion path, `--init-from`, and the inverted screener.

**5. Define what phase-2 success means.** Synthetic data had a reference;
real footage does not. The tools are Detector re-validation plus the
within-frame contrast now reported by `scripts/evaluate_mitigation.py` —
but no target numbers have been agreed. Tier 0's cost is the bar to beat:
see `docs/` and the Tier 0 notes for its measured contrast loss.

**6. Consider implementing spec §7's periodic detector evaluation.** In
phase 1 you can live without it. In phase 2 it is the only window into
whether training is actually reducing risk.

## Data

**Synthesis is ITU-only.** `configs/profiles/` ships six profiles;
`data/profiles_itu/` holds just `itu.json`. Pointing `training/cli.py` at
`configs/profiles` synthesises against all six and produces roughly six
times the samples — a different dataset from the one every measurement here
was taken on. All 188 existing samples are ITU (90 `general`, 98 `red`,
from 98 of the 101 clips in `data/clips_final/`).

```bash
python -m training.cli --clips-dir data/clips_final \
    --profiles-dir data/profiles_itu --output data/synthetic --seed 0
```

Synthesis is a pure function of `(seed, sample_id)`, so the same clips at
the same seed reproduce the same samples.

`data/synthetic/` is 24.11 GB, of which **`clean/` is 12.07 GB and is no
longer read** — the self-supervised objective needs no reference. Only
`degraded/`, `meta.json` and `prior_cache.npz` matter now.

`prior_cache.npz` is at **schema version 3**; a stale cache fails loudly.
Regenerating all 188 takes ~28 minutes on a free CPU (~11 s/sample).

`mitigator/weights/best.pt` (the old 35.66 dB checkpoint) **can no longer
be loaded** — `in_channels` went from `9+1+16` to `9+1+1`.

---

## The phase-2 tooling that now exists

All of it landed on 2026-08-08. Do not rebuild any of it.

- **`scripts/ingest_real_clips.py`** — mp4 directory to sample directories,
  no injection. Long side capped at 512 (matching `mitigator/infer.py`'s
  `_MAX_PROCESSING_DIM`, so training and inference see one resolution).
  `clip_id` is an ASCII slug plus a hash of the original filename, because
  the corpus has emoji, `#`, `!`, Korean and doubled spaces.
- **`--init-from PATH`** on the trainer — weights only, fresh optimizer,
  epoch 0. Passing it together with `--resume` is now an error rather than
  a silent precedence rule.
- **`--split-file PATH`** — explicit `clip_id` → `train`/`val` map, checked
  against the directory in *both* directions. Omitted, the `clip_id` hash
  still decides, so the synthetic path is unchanged.
- **`--snapshot-every N`** — keeps `epoch_NNN.pt`. Phase 1 lost its lowest
  `soft_area` epoch because only `best.pt` and `latest.pt` survived.
- **`--log-every N`** — running mean of every loss term every N steps.

**The "inverted screener" was deliberately not built.** All 17 real clips
are already screened, with the results in
`data/raw/real_flicker_screening.json` and zero passes. It is a tool for
footage not yet collected — build it when new clips actually arrive. The
three missing pieces were two.

Corpus and split are settled: 14 clips / 8,190 frames in `data/real`
(dropping `7301-199291564_medium` as unusable, `56426-479655491_medium` as
a duplicate source, and `1258-144566586_medium` for stock-id adjacency to
an eval clip), split by `configs/splits/phase2.json` into 11 train (5,865
examples) and 3 dev (484).

**512 downscaling does not change the Detector's verdict on this corpus.**
Measured on six clips including three 1080p ones — identical segment counts
and in-segment percentages at 512 and at native resolution. Worth knowing
because `read_frames_lenient`'s docstring warns it can, the ITU criterion
being an area ratio.

---

## What the real footage actually looks like

All 17 clips in `data/raw/real_flicker/` screened against ITU. Zero
failures. Results in `data/raw/real_flicker_screening.json`.

- **`max_area` is 1.000 on 17 of 17.** At peak the *entire frame* is
  flagged, four times the ITU limit. This is the normal case, not a tail.
- **16 of 17 exercise the red path.** Nothing in the synthetic corpus ever
  reached `soft_red`; that gap is closed.
- **7 clips are risky end to end** (`in-segment` 100%), and none is below
  27.9%. Total flagged material: 264.2 seconds.

The full-frame result voids `mitigate_frame`'s safety guarantee:

```python
return mask * restored + (1 - mask) * center
```

`arch.py` calls this "a hard guarantee rather than a hope" — pixels outside
the mask are the original, unconditionally. When the mask is all ones,
`(1 - mask)` is zero and it protects nothing. **It holds on synthetic data
and dissolves on real data. Decide what replaces it before phase 2.**

Related: `uncertain_frames` equals fps exactly on every clip — a
one-second Detector warm-up. Harmless on long clips, fatal on short ones
(`7301-199291564_medium.mp4` is 61 frames with 60 uncertain). The detector
also fabricates an all-ones mask for frame 0, and mask persistence ORs it
forward ~0.2 s; real segments start at frame 0 (원본 0–230, Anyma 0–45), so
phase 2 will train on whole-frame corrections there.

**Corpus hygiene:** `56426-479655491.mp4` (3840×2160) and
`56426-479655491_medium.mp4` (2560×1440) are the same source at two
resolutions, and `clip_split` hashes the clip id — they can land on
opposite sides of the train/val split. Use one. Resolutions span 360×640 to
3840×2160, and since the ITU criterion is an **area ratio**, the same flash
reads differently across that span.

`data/raw/real_flicker_eval/` (3 clips) is held out. Do not train on it.

---

## Measured costs on the development laptop (i5-11300H, RTX 3050 4 GB)

- **89–102 min per epoch** at `--batch-size 8 --patch-size 256`. 30 epochs
  is ~45 hours, not an overnight job.
- Full-resolution training does **not** fit in 4 GB: `bs=1` at 480×854 peaks
  at 5.75 GB and spills to system memory (4.2 s/step); `bs=2` OOMs. Even
  16 GB would only reach `bs=2`, so patch training is effectively mandatory.
- `bs=8 patch 256` peaks at 2.15 GB / 393 ms per step. `bs=16` hits a cliff
  at 4.27 GB and 13.9 s/step.
- Splits: 9259 train examples from 147 samples, 2790 val from 41.
  `trusted_fraction` 0.8842.

**Patch training changes what the area criterion means.** The ITU threshold
is a fraction of the *frame*; `soft_area` computed on a 256×256 crop is a
fraction of the crop. Unmeasured, but worth suspecting if phase 1 results
look strange.

That suspicion did **not** pay off for the completed run. Validation is
deliberately never cropped (`collate_fn=None` on the val loader), so the
`soft_area` in the epoch log is already a frame-level figure — and it
improved while the real-clip verdict got worse. Whatever caused that, it is
not the crop-versus-frame mismatch. The caveat still stands for the
training term.

### Second machine: RTX 4090 Laptop, 16 GB, 20 cores

The numbers above are all from the 4 GB card. On a 4090 laptop the shape
changes enough to re-plan around:

- **11–12 min per epoch** at `--batch-size 16 --patch-size 256`. 30 epochs
  is **5.7 hours**, not 45.
- `bs=16` peaks at ~5.7 GB and does not hit the 4 GB card's 13.9 s/step
  cliff — that cliff was spillover to system memory, not a real limit.
- `--num-workers 8` (of 20 cores) keeps GPU utilisation at 80–100%.
- The GPU sat at 70–75 °C throughout, power-capped rather than
  thermally capped (`clocks_throttle_reasons.active` = `0x4`, SW power cap;
  neither thermal bit ever set).

Check which machine you are on before trusting either table.

---

## Known-deferred items

- `self_supervised_loss`'s `weights` argument silently broadcasts a wrong
  shape — `(2,1)` against a `(2,)` per-item tensor yields `(2,2)` and a
  plausible wrong number. The only caller passes `.view(-1)`; future ones
  are unguarded. A one-line shape assert is cheap.
- `mitigator_losses.py:245-246` recomputes `relative_luminance_torch` that
  `soft_risk.py:76-77` computes again — two extra full-res gamma passes with
  grad per step.
- `weights.sum().item()` forces a host sync 3× per step.
- `README.md:132` and `mitigator/infer.py:7` still name PSNR/SSIM-vs-GT,
  which spec §7 calls unusable for this task.
- Spec §7's "full detector evaluation after training and every 5 epochs"
  was never implemented. Do not assume it is present. The 30-epoch run is
  the argument for implementing it: the epoch log looked healthy the whole
  way while the real verdict decayed, and nothing in the loop would have
  told you.
- Dated 2026-08-03 design docs still describe the removed histogram path.
- **The Tier 0 contrast baseline is misattributed in two docs.** −52.8% is
  **wonbon**, not Anyma; Anyma's is **−6.0%** and Cera Khin's −48.4%. The
  per-clip table in
  `docs/superpowers/specs/2026-08-07-tier0-fallback-measurements.md` is the
  correct source — the +1.87% mean-luminance figure quoted alongside −52.8%
  comes from wonbon's row, which is what pins it. Fixed in
  `scripts/evaluate_mitigation.py`; still wrong in spec §7 and in
  `docs/superpowers/plans/2026-08-07-self-supervised-pipeline.md`. This
  matters: believing Anyma's bar is −52.8% turns a 4.8×-worse result into
  an apparent win.
- **Segment boundaries step, and nothing ramps them.**
  `_assign_segment_strengths` is a step function — full strength inside a
  segment, exactly 0 outside. On the 3-epoch output the worst boundary jump
  was 0.0780 (wonbon), 5–29× the original's jump at the same frame and in
  the 93rd–99.7th percentile of the corrected clip. Deliberately not fixed:
  every one of those steps is still *below* its clip's original p95
  frame-to-frame delta, none trips the Detector, and the magnitude scales
  with however much the final model darkens — so it is worth re-measuring
  only once something clears the bar. The segments already carry ±0.5 s of
  margin, but that margin is the 1-second criterion window's reach, not
  spare room, so tapering into it trades protection for smoothness. If it
  is taken up: taper the output blend, not `required_strength` — the blend
  is monotone by construction, survives the all-ones mask, and does not
  contradict `_assign_segment_strengths`' stated design.

---

## Why the objective is shaped this way

The design spec is `docs/superpowers/specs/2026-08-07-self-supervised-mitigation-design.md`
and the executed plan is `docs/superpowers/plans/2026-08-07-self-supervised-pipeline.md`.

Short version: we train for *restoration* but grade for *constraint
satisfaction*. The Detector's verdict is the success criterion, and it is
computable on real footage without any reference — which is the whole reason
the objective had to stop depending on clean frames.

`arch.py`'s own docstring records why the 16-bin illumination histogram was
removed: it described normal illumination using the frame's non-masked
pixels, which is sound on synthetic data but wrong on concert footage, where
inside the mask is stage lighting and outside is a dark crowd. It asked the
model to make the lights as dark as the audience, and the model complied to
a median 14.6%.
