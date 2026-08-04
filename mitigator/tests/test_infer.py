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
