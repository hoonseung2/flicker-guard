import numpy as np

from scripts.evaluate_mitigation import contrast_stats


def test_contrast_falls_when_a_frame_is_compressed_toward_its_mean():
    rng = np.random.default_rng(0)
    frames = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
    flattened = [f * 0.2 + f.mean() * 0.8 for f in frames]

    assert contrast_stats(flattened)["mean_contrast"] < contrast_stats(frames)["mean_contrast"]


def test_mean_luminance_can_stay_flat_while_contrast_collapses():
    # This is why contrast is the reported cost and mean luminance is not.
    # Compressing toward the mean leaves the mean almost untouched -- on a
    # real clip it rose 1.87% while contrast fell 52.8%.
    #
    # NOTE on this test's construction: the task plan's original snippet
    # flattened frames with `f * factor + f.mean() * (1 - factor)`, done
    # directly on sRGB-encoded pixel values. That is not what the Tier 0
    # correction actually does (see fallback/transfer.py:compress_contrast),
    # and it is not innocuous -- an affine blend applied to gamma-encoded
    # values, then measured through relative_luminance's sRGB->linear curve,
    # shifts the mean by ~29% here (verified: consistent across 5 seeds),
    # not the near-zero shift the docstring above describes. That's Jensen's
    # inequality: srgb_to_linear is convex, so reducing a distribution's
    # variance while holding its arithmetic mean fixed changes the mean of
    # the convex-transformed values. Using the naive snippet, this test
    # would never pass regardless of how contrast_stats is implemented.
    #
    # The real correction avoids this by compressing in *linear* light
    # toward a *linear*-domain reference (an affine map is mean-invariant
    # in the space it's applied in, not in a differently-curved space it's
    # measured in afterwards) -- so this test reproduces that with the
    # actual production function instead of a hand-rolled approximation.
    from detector.luminance import srgb_to_linear
    from fallback.transfer import compress_contrast, trailing_reference
    from scripts.evaluate_mitigation import luminance_stats

    rng = np.random.default_rng(1)
    frames = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
    linear_frames = [srgb_to_linear(f) for f in frames]
    reference = trailing_reference(linear_frames, index=len(frames) - 1, window_frames=len(frames))
    flattened = [compress_contrast(f, reference, strength=0.85) for f in frames]

    before_mean = luminance_stats(frames)["mean_luminance"]
    after_mean = luminance_stats(flattened)["mean_luminance"]
    assert abs(after_mean - before_mean) / before_mean < 0.10

    before_contrast = contrast_stats(frames)["mean_contrast"]
    after_contrast = contrast_stats(flattened)["mean_contrast"]
    assert after_contrast < before_contrast * 0.5


def test_contrast_stats_ignores_a_nan_contaminated_frame_instead_of_nanning_the_report():
    # The mitigator can pass a frame through with NaN pixels in it (see
    # mitigator/infer.py's isfinite guard -- that guard catches NaN *model
    # output* by falling back to the original frame, but nothing stops a
    # NaN from reaching evaluate_mitigation through some other path, e.g. a
    # corrupt decode). A single NaN frame must not silently turn the whole
    # report's mean/p95 into nan and hide every other frame's real contrast.
    rng = np.random.default_rng(2)
    frames = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(5)]
    contaminated = list(frames)
    contaminated[2] = contaminated[2].copy()
    contaminated[2][0, 0, 0] = np.nan

    clean_result = contrast_stats(frames)
    result = contrast_stats(contaminated)

    assert np.isfinite(result["mean_contrast"])
    assert np.isfinite(result["p95_contrast"])
    # The four uncontaminated frames should still dominate the result --
    # it should be close to (not wildly different from) the all-clean case.
    assert abs(result["mean_contrast"] - clean_result["mean_contrast"]) < 0.05


def test_contrast_stats_empty_list_returns_zero():
    result = contrast_stats([])
    assert result["mean_contrast"] == 0.0
    assert result["p95_contrast"] == 0.0
