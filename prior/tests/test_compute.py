import numpy as np

from detector.pipeline import run_detection_with_masks
from detector.profiles import ThresholdProfile
from prior.compute import compute_prior
from training.injection import inject_general_flash
from training.params import InjectionWindow

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _clean_clip(n, h=20, w=20, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_compute_prior_marks_injected_window_pixels_in_mask():
    clean = _clean_clip(n=30)
    window = InjectionWindow(
        start_frame=5, end_frame=20, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=1, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    results = compute_prior(degraded, fps=10.0, profile=PROFILE)

    assert len(results) == 30
    assert results[6].mask.all()


def test_compute_prior_returns_frame_prior_per_frame_with_matching_indices():
    clean = _clean_clip(n=10)
    results = compute_prior(clean, fps=10.0, profile=PROFILE)
    assert [r.frame_index for r in results] == list(range(10))


def test_compute_prior_accepts_a_one_shot_generator():
    # detector.cli.read_video_frames returns a generator; compute_prior consumes
    # frames twice, so it must materialise them rather than silently returning [].
    frames = (frame for frame in _clean_clip(n=12))
    results = compute_prior(frames, fps=10.0, profile=PROFILE)
    assert len(results) == 12
    assert [r.frame_index for r in results] == list(range(12))


def test_compute_prior_mask_persists_after_the_transition_ends():
    # The Detector's per-frame mask flags pixels that *changed* since the
    # previous frame, which is right for deciding whether content is
    # hazardous but wrong for deciding what to repair: while a flash holds
    # its bright state there is no transition, so the mask empties even
    # though those pixels are still wrong. Measured on the real training
    # set, the transition mask covered only 10.4% (median) of the actually
    # damaged pixels on hold frames, versus 92.1% on transition frames.
    clean = _clean_clip(n=80, value=0.5)
    window = InjectionWindow(
        start_frame=20, end_frame=60, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=4, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    plain = compute_prior(degraded, fps=10.0, profile=PROFILE, mask_persistence_seconds=0.0)
    held = compute_prior(degraded, fps=10.0, profile=PROFILE, mask_persistence_seconds=0.5)

    # Inside the injected window, the transition mask empties on every frame
    # where the pattern is merely holding its state.
    hold_frames = [
        p.frame_index for p in plain
        if window.start_frame < p.frame_index <= window.end_frame and not p.mask.any()
    ]
    assert hold_frames, "expected at least one hold frame with an empty transition mask"

    # Persistence carries the preceding transition forward, so almost all of
    # them come back flagged. Not *all*: a hold frame with no transition
    # anywhere in the window behind it has nothing to carry forward -- which
    # is why this asserts a large majority rather than every frame.
    refilled = sum(1 for i in hold_frames if held[i].mask.any())
    assert refilled > 0.8 * len(hold_frames)


def test_compute_prior_zero_persistence_matches_the_plain_transition_mask():
    # persistence is an opt-out-able widening, not a redefinition -- with it
    # switched off the mask must be exactly the Detector's own output.
    clean = _clean_clip(n=60, value=0.5)
    window = InjectionWindow(
        start_frame=20, end_frame=45, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=3, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    _, _, detector_masks = run_detection_with_masks(degraded, fps=10.0, profile=PROFILE)
    priors = compute_prior(degraded, fps=10.0, profile=PROFILE, mask_persistence_seconds=0.0)

    for prior, mask in zip(priors, detector_masks):
        assert np.array_equal(prior.mask, mask)


def test_compute_prior_persistence_window_scales_with_fps():
    # The window is specified in seconds, so the same clip at a higher frame
    # rate must hold the mask for the same wall-clock duration rather than
    # the same number of frames.
    clean = _clean_clip(n=90, value=0.5)
    window = InjectionWindow(
        start_frame=30, end_frame=70, mask_top=0, mask_left=0,
        mask_height=20, mask_width=20, period_frames=5, ramp_frames=10,
    )
    degraded = inject_general_flash(clean, window, dark_target=0.2, bright_target=0.8)

    slow = compute_prior(degraded, fps=10.0, profile=PROFILE, mask_persistence_seconds=0.5)
    fast = compute_prior(degraded, fps=30.0, profile=PROFILE, mask_persistence_seconds=0.5)

    # 0.5s is 5 frames at 10fps but 15 at 30fps -- the wider window can only
    # flag more, never less.
    assert sum(p.mask.sum() for p in fast) > sum(p.mask.sum() for p in slow)


def test_frame_prior_carries_the_segments_required_strength():
    # The model sees a three-frame window; how hard this segment needs to be
    # pushed is global context it cannot derive locally. Measured: the
    # previous model reached a median 14.6% of its own target, and a
    # histogram describing "what normal looks like" never told it how far to
    # move.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(0)
    base = rng.random((24, 24, 3)).astype(np.float32) * 0.2
    frames = []
    for i in range(40):
        frame = base.copy()
        if 15 <= i < 30 and i % 2 == 0:
            frame[:, :16] = 0.95
        frames.append(frame)

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)

    flagged = [p for p in priors if p.required_strength > 0.0]
    assert flagged, "the flickering stretch should carry a nonzero strength"
    assert all(0.0 <= p.required_strength <= 1.0 for p in priors)


def test_frames_outside_a_risk_segment_have_zero_strength():
    # A frame the Detector did not flag needs no correction, and a nonzero
    # strength there would tell the model to move a frame that is already
    # compliant.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(1)
    frames = [rng.random((24, 24, 3)).astype(np.float32) * 0.1 for _ in range(30)]

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)
    assert all(p.required_strength == 0.0 for p in priors)


def test_every_frame_in_one_segment_shares_that_segments_strength():
    # required_strength is a per-segment quantity: it is the max over the
    # segment's frame pairs, so handing different frames of one segment
    # different values would be incoherent.
    import numpy as np
    from prior.compute import compute_prior

    rng = np.random.default_rng(2)
    base = rng.random((24, 24, 3)).astype(np.float32) * 0.2
    frames = []
    for i in range(40):
        frame = base.copy()
        if 15 <= i < 30 and i % 2 == 0:
            frame[:, :16] = 0.95
        frames.append(frame)

    priors = compute_prior(frames, fps=20.0, profile=PROFILE)
    values = {round(p.required_strength, 6) for p in priors if p.required_strength > 0.0}
    assert len(values) == 1, f"expected one strength across the segment, got {values}"
