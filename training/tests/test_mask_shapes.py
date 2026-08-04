import numpy as np

from training.mask_shapes import (
    beam_area,
    circle_area,
    rect_area,
    render_beam_mask,
    render_circle_mask,
    render_rect_mask,
)


def test_rect_area_formula():
    assert rect_area(half_height=2, half_width=3) == 5 * 7


def test_render_rect_mask_pixel_count_matches_rect_area():
    mask = render_rect_mask(frame_height=50, frame_width=50, center_row=25, center_col=25, half_height=4, half_width=6)
    assert mask.shape == (50, 50)
    assert mask.dtype == bool
    assert mask.sum() == rect_area(4, 6)


def test_render_rect_mask_is_centered():
    mask = render_rect_mask(frame_height=20, frame_width=20, center_row=10, center_col=10, half_height=2, half_width=2)
    assert mask[10, 10]
    assert mask[8, 10] and mask[12, 10]
    assert not mask[7, 10] and not mask[13, 10]


def test_circle_area_formula():
    assert circle_area(radius=10) == round(np.pi * 100)


def test_render_circle_mask_pixel_count_approximately_matches_circle_area():
    # Pixel rasterization of a circle is never pixel-exact -- allow 15% slack.
    mask = render_circle_mask(frame_height=100, frame_width=100, center_row=50, center_col=50, radius=20)
    expected = circle_area(20)
    assert abs(mask.sum() - expected) / expected < 0.15


def test_render_circle_mask_is_centered_and_round():
    mask = render_circle_mask(frame_height=40, frame_width=40, center_row=20, center_col=20, radius=5)
    assert mask[20, 20]
    assert mask[20, 25] and mask[25, 20]  # exactly on the radius, axis-aligned -- inside
    assert not mask[20, 30]  # well outside
    # Diagonal offset (3, 3): distance = sqrt(18) ~= 4.24 < radius=5 -- inside.
    assert mask[23, 23]
    # Diagonal offset (4, 4): distance = sqrt(32) ~= 5.66 > radius=5 -- outside.
    assert not mask[24, 24]


def test_beam_area_formula():
    assert beam_area(half_length=10, half_thickness=2) == 21 * 5


def test_render_beam_mask_pixel_count_matches_beam_area_at_zero_angle():
    mask = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=15, half_thickness=3, angle_degrees=0.0,
    )
    assert mask.sum() == beam_area(15, 3)


def test_render_beam_mask_long_axis_follows_angle():
    # At angle=0, the beam is wide along columns (the "length" axis) and
    # narrow along rows (the "thickness" axis).
    mask_0 = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=20, half_thickness=2, angle_degrees=0.0,
    )
    assert mask_0[30, 45]       # far along columns from center -- inside
    assert not mask_0[45, 30]   # far along rows from center -- outside

    # At angle=90, the beam's long axis rotates to follow rows instead.
    mask_90 = render_beam_mask(
        frame_height=60, frame_width=60, center_row=30, center_col=30,
        half_length=20, half_thickness=2, angle_degrees=90.0,
    )
    assert mask_90[45, 30]       # far along rows -- now inside
    assert not mask_90[30, 45]   # far along columns -- now outside


def test_render_masks_stay_within_frame_bounds_near_an_edge():
    # A shape centered near/at a corner must not raise and must return a
    # full-sized array -- pixels beyond the frame are simply not covered.
    for render, kwargs in [
        (render_rect_mask, dict(half_height=5, half_width=5)),
        (render_circle_mask, dict(radius=5)),
        (render_beam_mask, dict(half_length=10, half_thickness=3, angle_degrees=45.0)),
    ]:
        mask = render(frame_height=20, frame_width=20, center_row=0, center_col=0, **kwargs)
        assert mask.shape == (20, 20)
        assert mask.any()
