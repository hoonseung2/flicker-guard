# Overnight run log — 2026-08-08

Judgment calls made without the user present, so they can be audited in the
morning. The standing rule agreed before starting:

1. Where the spec states a criterion, follow it.
2. Where it does not, pick the option that stays reversible and leaves a
   measurement behind.
3. If a decision the user made turns out to be wrong, adapt, record it
   loudly here, and keep going rather than stopping.
4. Never delete data. Never push.

Plan: `2026-08-08-phase2-data-path.md`. Spec:
`../specs/2026-08-08-phase2-data-path-design.md`.

---

## Checks that passed, and so changed nothing

**512 downscale does not change the Detector's verdict.** This was the
largest risk to the plan — `read_frames_lenient`'s own docstring warns that
downscaling changes what gets flagged, because the ITU criterion is an area
ratio, and the corpus figures the dev-set choice rests on were measured at
the screening resolution.

Measured on six clips, including three 1080p ones (a 3.75× downscale):

| clip | native seg / % | @512 seg / % |
|---|---|---|
| `11832-233049403_medium` | 2 / 27.9 | 2 / 27.9 |
| `녹음 2026-08-08 000241` | 2 / 50.9 | 2 / 50.9 |
| `disco lights` | 1 / 100.0 | 1 / 100.0 |
| `56426-479655491` | 2 / 95.3 | 2 / 95.3 |
| `57523-484978976_medium` | 1 / 100.0 | 1 / 100.0 |
| `1258-144566586_medium` | 6 / 60.6 | 6 / 60.6 |

Identical in every case. The 512 decision stands and the dev set is unchanged.

**Ingestion produced exactly what the plan predicted:** 14 clips, 8,190
frames, long side capped at 512. Split file: train 11 / dev 3.

---

## Decisions taken

**Rewrote a test rather than deleting it.**
`test_dataset_includes_every_frame_of_a_segment_regardless_of_warm_up`
asserted that a segment starting at frame 0 contributes frame 0 — the
opposite of the warm-up exclusion the design calls for. Its reasoning was
about the retired histogram-validity gate and is correct on synthetic data,
where the injected segment is ground truth. The replacement asserts the new
behaviour and records why real footage breaks the old one, so the history
is not lost.

**Fixed one pre-existing test double.** `_fake_dataset` in
`test_main_val_loader_handles_differently_shaped_examples` took
`(data_dir, profile, split)` positionally and broke when `main` began
passing `split_map`. It was the only double not already accepting
`**kwargs`. Signature change only.

---

## Deviations from the plan

**Tasks 4–6 were implemented as one commit, not three.** The plan noted
that Task 4's test needs `split_map` from Task 6, and suggested running
Task 6 first. Doing all three dataset changes together was simpler and the
tests cover them separately.

**The plan's Task 10 Step 5 (Tier 0 baselines for the dev clips) is
deferred to the follow-up work.** Those numbers are only consumed by the
selection constraint, which is in the out-of-scope follow-up plan. Running
them tonight would spend wall-clock that training needs. Recorded here so
it is not mistaken for done.

---

## Status

_Updated as the night proceeds._

- Implementation: complete, 221 tests passing (baseline was 198).
- Ingestion: complete — 14 clips, 8,190 frames.
- Split file: `configs/splits/phase2.json`, train 11 / dev 3.
- Smoke run (1 epoch): passed, exit 0.
- Run 1 (from scratch, 30 epochs): complete, best epoch 28. **2 of 3 pass.**
- Run 2 (--init-from phase-1 best.pt): running.

## What the smoke run showed

Splits: **5,865 train examples from 11/11 clips, 484 dev from 3/3.** Every
clip contributes; the floor set before starting was 1,000 examples.

An epoch is **142 s**, against ~690 s in phase 1 — fewer examples (5,865
vs 9,259) and every clip capped at 512. 30 epochs is ~71 minutes, so both
initialisation experiments fit in about 2.4 hours rather than the 8-10 the
plan budgeted.

`trusted_fraction` is **0.9442**, against 0.8842 on synthetic data. Global
shift estimation is *more* reliable on this footage than on the synthetic
corpus, not less — the opposite of the risk the design spec flagged as its
largest unvalidated assumption.

**The risk term is live on real data for the first time.** `soft_area`
starts at **0.2146**, against 0.0774 on synthetic init, and `risk` at
0.0705 against 0.0063 — eleven times higher. PROJECT_STATE recorded that on
synthetic data `soft_area` sat three times below the ITU `max_area_ratio`
of 0.25, leaving the risk term's softplus in its flat region contributing
almost no gradient, and concluded `--lambda-risk` could not rescue that
run. Real footage starts just under the limit, where the softplus has
slope. This is direct evidence for the regime gap the spec blamed for
phase 1's 0/3, measured rather than inferred.

After one epoch: `soft_area` 0.2146 → 0.0927 (val), `temporal` 0.0667 →
0.0276, `risk` 0.0705 → 0.0182.


---

## Run 1 result: from scratch, 30 epochs, best epoch 28

`val_loss` 0.3186 -> 0.2674. Epochs ran 110 s each; the whole run took
about 55 minutes.

Re-validated on the three clips the Tier 0 baseline was measured on:

| clip | Tier 0 | phase 1 best | phase 2 scratch |
|---|---|---|---|
| wonbon | 5 -> 0 | 5 -> 2 | **5 -> 0** |
| Anyma | 1 -> 0 | 1 -> 0 | **1 -> 0** |
| Cera Khin | 1 -> 0 | 1 -> 1 | 1 -> 1 |

**2 of 3, against phase 1's best of 1 of 3.** In-segment contrast against
each clip's own Tier 0 figure:

| clip | phase 2 | Tier 0 | |
|---|---|---|---|
| wonbon | **-48.3%** | -52.8% | cheaper |
| Anyma | -30.5% | **-6.0%** | 5x more expensive |
| Cera | -54.1% | -48.4% | more expensive, and still fails |

**wonbon is the first result in this project to clear both halves of the
spec's bar at once** — zero remaining risk segments *and* a smaller
contrast cost than the classical fallback. Its peak windowed area fell
62.9% and every one of its 574 triggering frames is gone.

Anyma passes the segment bar and loses the cost one. This is where the
mislabelled baseline would have mattered: -30.5% read against -52.8% looks
like a comfortable win, and against Anyma's real -6.0% it is a five-fold
loss.

Cera is stuck exactly where phase 1 left it — peak windowed area +0.5%,
essentially unmoved, with its segment running from frame 279 to the end of
the clip. Whatever Cera needs, training on real footage did not supply it.
