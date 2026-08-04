import numpy as np
import torch

from detector.profiles import ThresholdProfile
from mitigator.arch import MitigatorNet
from mitigator.infer import mitigate_segment

PROFILE = ThresholdProfile(
    name="test", max_flashes_per_second=3, max_area_ratio=0.10,
    general_flash_dark_threshold=0.80, general_flash_delta_threshold=0.10,
    red_saturation_ratio_threshold=0.80,
)


def _flat_frames(n, h, w, value=0.5):
    return [np.full((h, w, 3), value, dtype=np.float32) for _ in range(n)]


def test_mitigate_segment_returns_same_length_and_shape():
    torch.manual_seed(0)
    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    assert len(result) == len(frames)
    for out_frame, in_frame in zip(result, frames):
        assert out_frame.shape == in_frame.shape


def test_mitigate_segment_passes_through_frames_with_no_target_histogram():
    # fps=10 means the first ~10 frames of any clip have target_histogram
    # None (Detector's trailing window hasn't filled yet) -- those frames
    # must come back byte-identical to the input, untouched by the network.
    torch.manual_seed(0)
    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    assert np.array_equal(result[0], frames[0])


def test_mitigate_segment_downscales_large_frames_for_bottleneck_attention(monkeypatch):
    # Regression test: the bottleneck's self-attention is global over all
    # H*W positions, so its cost is quadratic in pixel count. Training only
    # ever exercised it on <=512x512 crops; feeding a real video's native
    # resolution (e.g. 720x1280) unmodified overwhelmed a T4 GPU's memory in
    # practice, with no Python traceback. mitigate_segment must downscale
    # before the model and upscale the result back, rather than passing
    # large frames through unbounded -- this proves it survives a
    # resolution above the cap and that pixels strictly outside the mask
    # come back byte-identical to the input (no interpolation bleed).
    import mitigator.infer as infer_module
    from prior.compute import FramePrior

    torch.manual_seed(0)
    model = MitigatorNet()
    h, w = 600, 800  # above _MAX_PROCESSING_DIM (512) on both axes
    frames = _flat_frames(3, h, w)
    mask = np.zeros((h, w), dtype=bool)
    mask[100:200, 100:200] = True
    no_mask = np.zeros((h, w), dtype=bool)
    fake_priors = [
        FramePrior(frame_index=0, target_histogram=None, mask=no_mask),
        FramePrior(frame_index=1, target_histogram=np.full(64, 1.0 / 64, dtype=np.float32), mask=mask),
        FramePrior(frame_index=2, target_histogram=None, mask=no_mask),
    ]
    monkeypatch.setattr(infer_module, "compute_prior", lambda frames, fps, profile: fake_priors)

    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)

    assert result[1].shape == (h, w, 3)
    outside_mask = ~mask
    assert np.array_equal(result[1][outside_mask], frames[1][outside_mask])


def test_mitigate_segment_passes_through_frame_on_nan_output(monkeypatch):
    # An untrained or misbehaving model can output NaN/Inf. mitigate_segment
    # must never let that reach the returned frames -- fall back to the
    # original input for that frame instead.
    import mitigator.infer as infer_module

    def _nan_mitigate_frame(window, mask, histogram, model):
        return torch.full((1, 3, window.shape[-2], window.shape[-1]), float("nan"))

    monkeypatch.setattr(infer_module, "mitigate_frame", _nan_mitigate_frame)

    model = MitigatorNet()
    frames = _flat_frames(20, 16, 16)
    result = mitigate_segment(frames, fps=10.0, profile=PROFILE, model=model)
    for out_frame, in_frame in zip(result, frames):
        assert np.array_equal(out_frame, in_frame)
