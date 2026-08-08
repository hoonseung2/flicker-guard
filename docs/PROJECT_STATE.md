# Project state — 2026-08-08

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

Phase 1 training has been **started and diagnosed, not completed**. Read the
next section before launching anything.

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

Larger batches mean fewer optimiser steps per epoch: 9259 train examples is
1157 steps/epoch at `--batch-size 8` but only 579 at 16. A small measured
improvement over 3 epochs is not failure — check the *direction* first, and
add epochs before changing anything else.

## What to do after phase 1

Roughly in order. Items in the same group are independent of each other.

**1. Decide what replaces the mask blend.** `max_area` is 1.000 on 17 of 17
real clips, so `mitigate_frame`'s "pixels outside the mask stay original"
guarantee protects nothing on real footage. This is the heaviest open
question and it changes how ingestion should work, so decide it before
building anything for phase 2.

**2. Decide the frame-0 and warm-up handling.** The Detector fabricates an
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

## Phase 2 is blocked on three missing pieces

1. **No ingestion path.** `MitigatorDataset` wants
   `<sample>/meta.json` (needs `fps`) plus `<sample>/degraded/000000.png…`.
   `training/cli.py` only produces that by *injecting* flicker into clean
   sources, and has no flag to skip injection. Phase 2 must not inject.

2. **No `--init-from`.** `--resume` loads `latest.pt` from the *same*
   `--checkpoint-dir` and restores model **and optimizer and epoch
   counter**. Copying a phase-1 checkpoint into a phase-2 directory and
   passing `--resume` restores optimizer momentum from a different dataset
   and sets `start_epoch = 30`, so `range(30, 30)` runs **zero epochs and
   exits cleanly**. A weights-only initialisation flag does not exist.

3. **Screening runs the wrong direction.** `scripts/screen_clean_clips.py`
   keeps clips the Detector does *not* flag — valid clean sources for
   synthesis. Phase 2 needs the inverse.

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
  was never implemented. Do not assume it is present.
- Dated 2026-08-03 design docs still describe the removed histogram path.

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
