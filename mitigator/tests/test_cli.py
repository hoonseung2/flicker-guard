from pathlib import Path

import cv2
import numpy as np
import torch

from mitigator.arch import MitigatorNet
from mitigator.cli import load_model, main, write_video
from training.train_mitigator import save_checkpoint

SHIPPED_PROFILE = Path(__file__).parent.parent.parent / "configs" / "profiles" / "itu.json"


def _write_test_video(path: Path, n: int = 20, h: int = 16, w: int = 16, fps: float = 10.0) -> None:
    rng = np.random.default_rng(0)
    frame = (rng.uniform(0.1, 0.9, size=(h, w, 3)) * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        for _ in range(n):
            writer.write(frame)
    finally:
        writer.release()


def test_load_model_restores_saved_weights(tmp_path):
    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(checkpoint_path, model, optimizer, epoch=3)

    loaded = load_model(checkpoint_path)

    for original_param, loaded_param in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(original_param, loaded_param.cpu())


def test_write_video_round_trips_frame_count_and_resolution(tmp_path):
    frames = [np.full((12, 20, 3), 0.5, dtype=np.float32) for _ in range(7)]
    output_path = tmp_path / "out.mp4"

    write_video(frames, output_path, fps=15.0)

    cap = cv2.VideoCapture(str(output_path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 20
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 12
    finally:
        cap.release()


def test_write_video_pixel_values_survive_uint8_round_trip(tmp_path):
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    frame[:, :, 0] = 0.2  # R
    frame[:, :, 1] = 0.6  # G
    frame[:, :, 2] = 0.9  # B
    output_path = tmp_path / "out.mp4"

    write_video([frame], output_path, fps=10.0)

    cap = cv2.VideoCapture(str(output_path))
    try:
        ok, bgr_uint8 = cap.read()
    finally:
        cap.release()
    assert ok
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    assert np.allclose(rgb, frame, atol=0.02)  # mp4v/uint8 quantization tolerance


def test_main_runs_video_through_mitigate_segment_and_writes_output(tmp_path):
    input_path = tmp_path / "in.mp4"
    output_path = tmp_path / "out.mp4"
    checkpoint_path = tmp_path / "model.pt"
    _write_test_video(input_path, n=20, h=16, w=16, fps=10.0)

    torch.manual_seed(0)
    model = MitigatorNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0)

    exit_code = main([
        "--input", str(input_path),
        "--output", str(output_path),
        "--checkpoint", str(checkpoint_path),
        "--profile", str(SHIPPED_PROFILE),
    ])

    assert exit_code == 0
    assert output_path.exists()
    cap = cv2.VideoCapture(str(output_path))
    try:
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 16
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 16
    finally:
        cap.release()
