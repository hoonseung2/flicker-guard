"""Samples synthesis parameters guaranteed to exceed a ThresholdProfile's
thresholds with a comfortable margin, so Detector re-validation in
training.synth is a confirmation step rather than blind trial and error.

Per-profile, not global (spec Global Constraint): every value here is
computed from the specific ThresholdProfile passed in, never from a fixed
min/max spanning all profiles.

Frequency note: `ThresholdProfile.max_flashes_per_second` is compared
against Detector's `flagged_frame_count_last_second` (flagged *frames* in a
trailing one-second window), not the visual flash rate (see
detector/scoring.py). `period_frames` here is chosen so the number of
flagged frames inside any trailing one-second window exceeds
`max_flashes_per_second` with margin.

Every target here is clamped (`_MAX_AREA_RATIO`, `_MAX_SATURATION_TARGET`,
`bright_target <= 1.0`), so a profile sitting close enough to one of those
caps could silently receive a target that does not actually exceed its own
threshold — whose only symptom would be every synthesis attempt exhausting
its retries with no diagnostic. `_require_exceeds` turns that into an
immediate, legible configuration error instead.
"""
from dataclasses import dataclass, field

import numpy as np

from detector.profiles import ThresholdProfile
from training import mask_shapes

_AREA_MARGIN = 0.10
_MAX_AREA_RATIO = 0.90
_RATIO_MARGIN = 0.05
_MAX_SATURATION_TARGET = 0.99


class ClipTooShortError(ValueError):
    """The clip has fewer frames than one injection window needs.

    A distinct type (rather than a bare ValueError) so callers can tell "this
    particular clip is unusable, skip it and carry on" apart from "this
    profile can never be satisfied", which is a fail-fast config error.
    """


@dataclass
class LightShape:
    kind: str  # "rect" | "circle" | "beam"
    start_row: float
    start_col: float
    end_row: float
    end_col: float
    half_height: int | None = None    # rect only
    half_width: int | None = None     # rect only
    radius: int | None = None         # circle only
    half_length: int | None = None    # beam only
    half_thickness: int | None = None  # beam only
    angle_degrees: float | None = None  # beam only


@dataclass
class InjectionWindow:
    start_frame: int
    end_frame: int  # inclusive
    # Fallback static rectangle. Always populated, but when `lights` below is
    # non-empty these do NOT describe the actual injected region -- the real
    # injected pixels are lights's per-frame rendered union instead (see
    # injection.py's _static_region_mask vs _lights_mask_at_offset).
    mask_top: int
    mask_left: int
    mask_height: int
    mask_width: int
    period_frames: int  # frames per alternation half-cycle (state held this long)
    ramp_frames: int  # Detector's trailing-window length (round(fps)) at sample time
    # Populated only by sample_synthesis_params_realistic (Task 5) -- empty
    # for the flat-mode path and any window built by plain
    # sample_synthesis_params, which keeps using the mask_top/left/height/
    # width rectangle above via injection.py's fallback path.
    lights: list[LightShape] = field(default_factory=list)


@dataclass
class SynthParams:
    pattern: str  # "general" | "red"
    window: InjectionWindow
    dark_target: float | None = None
    bright_target: float | None = None
    red_rgb: tuple[float, float, float] | None = None
    baseline_rgb: tuple[float, float, float] | None = None


def _require_exceeds(
    achieved: float,
    threshold: float,
    profile_name: str,
    threshold_field: str,
    quantity: str,
    cap_reason: str,
) -> None:
    """Fail loudly when a clamp/rounding cap has eaten the safety margin.

    Every target here is capped (`_MAX_AREA_RATIO`, `_MAX_SATURATION_TARGET`,
    `bright_target <= 1.0`). A profile close enough to a cap silently stops
    getting a target that actually exceeds its own threshold, and the only
    symptom would be every synthesis attempt exhausting its retries with no
    diagnostic. Turn that into an immediate configuration error instead.
    """
    if achieved > threshold:
        return
    raise ValueError(
        f"profile {profile_name!r}: {quantity} {achieved:.4f} does not exceed "
        f"{threshold_field}={threshold:.4f} — {cap_reason}. This profile "
        f"cannot be synthesized against; widen the cap or relax the profile."
    )


def _mask_dims(frame_height: int, frame_width: int, target_area_ratio: float) -> tuple[int, int, int, int]:
    target_area_ratio = min(target_area_ratio, _MAX_AREA_RATIO)
    side_ratio = target_area_ratio ** 0.5
    mask_height = max(1, min(frame_height, round(frame_height * side_ratio)))
    mask_width = max(1, min(frame_width, round(frame_width * side_ratio)))
    mask_top = (frame_height - mask_height) // 2
    mask_left = (frame_width - mask_width) // 2
    return mask_top, mask_left, mask_height, mask_width


def _period_frames(profile: ThresholdProfile, fps: float) -> int:
    window_frames = max(1, round(fps))
    # Halve the just-crossing period for margin, so the steady-state flagged
    # count comfortably exceeds max_flashes_per_second rather than sitting
    # right at the boundary.
    period = window_frames / (2 * (profile.max_flashes_per_second + 1))
    return max(1, int(period))


def sample_synthesis_params(
    profile: ThresholdProfile,
    pattern: str,
    clip_frame_count: int,
    frame_height: int,
    frame_width: int,
    fps: float,
    rng: np.random.Generator,
) -> SynthParams:
    if pattern not in ("general", "red"):
        raise ValueError(f"unknown pattern: {pattern!r}")

    window_frames = max(1, round(fps))
    period_frames = _period_frames(profile, fps)
    # window_frames of ramp-up (counter filling) + 4 periods of guaranteed
    # steady-state tail once the trailing window is fully inside the pattern.
    duration_frames = window_frames + 4 * period_frames

    if clip_frame_count < duration_frames + 1:
        raise ClipTooShortError(
            f"clip has {clip_frame_count} frames, need at least "
            f"{duration_frames + 1} for fps={fps} and profile {profile.name!r}"
        )

    latest_start = clip_frame_count - duration_frames
    # PriorCalc's TargetHistogramSmoother needs a stretch of genuinely clean
    # (non-risky, non-uncertain) frames before the injected segment to ever
    # compute a non-None target histogram. Measured on the real 170-sample
    # dataset: samples whose injected segment started early (median start
    # ~15% into the clip) contributed zero training examples, while samples
    # that started around the clip's middle (median ~48%) contributed dozens
    # each -- see docs/superpowers/specs/2026-08-04-mask-diversification-design.md
    # section 3. When a clip is too short to offer the full desired runway,
    # give it as much as the clip can provide rather than rejecting it
    # outright (ClipTooShortError above already handles clips too short for
    # the injection window itself).
    desired_runway = 2 * window_frames
    min_start = min(desired_runway, latest_start)
    start_frame = int(rng.integers(min_start, latest_start + 1))
    end_frame = start_frame + duration_frames - 1

    mask_top, mask_left, mask_height, mask_width = _mask_dims(
        frame_height, frame_width, profile.max_area_ratio + _AREA_MARGIN
    )
    achieved_area_ratio = (mask_height * mask_width) / (frame_height * frame_width)
    _require_exceeds(
        achieved_area_ratio, profile.max_area_ratio, profile.name,
        "max_area_ratio", "injected mask area ratio",
        f"the mask is capped at {_MAX_AREA_RATIO:.2f} of the frame "
        f"(_MAX_AREA_RATIO) and rounded to whole pixels on "
        f"{frame_height}x{frame_width}",
    )

    window = InjectionWindow(
        start_frame=start_frame,
        end_frame=end_frame,
        mask_top=mask_top,
        mask_left=mask_left,
        mask_height=mask_height,
        mask_width=mask_width,
        period_frames=period_frames,
        ramp_frames=window_frames,
    )

    if pattern == "general":
        dark_target = profile.general_flash_dark_threshold * 0.5
        bright_target = min(1.0, dark_target + profile.general_flash_delta_threshold + 0.1)
        _require_exceeds(
            bright_target - dark_target, profile.general_flash_delta_threshold, profile.name,
            "general_flash_delta_threshold", "injected luminance delta",
            "bright_target is clamped to 1.0",
        )
        return SynthParams(pattern="general", window=window, dark_target=dark_target, bright_target=bright_target)

    target_ratio = min(_MAX_SATURATION_TARGET, profile.red_saturation_ratio_threshold + _RATIO_MARGIN)
    _require_exceeds(
        target_ratio, profile.red_saturation_ratio_threshold, profile.name,
        "red_saturation_ratio_threshold", "injected red saturation ratio",
        f"the target is capped at {_MAX_SATURATION_TARGET} (_MAX_SATURATION_TARGET)",
    )
    r = 0.9
    gb_sum = r / target_ratio - r
    red_rgb = (r, gb_sum / 2, gb_sum / 2)
    baseline_rgb = (0.3, 0.3, 0.3)
    return SynthParams(pattern="red", window=window, red_rgb=red_rgb, baseline_rgb=baseline_rgb)


_LIGHT_KINDS = ("rect", "circle", "beam")
_MAX_LIGHTS = 3
_BEAM_ASPECT = 4  # half_length = _BEAM_ASPECT * half_thickness
# Deriving a shape's half-extent/radius from a target area via round() can
# overshoot the target by several percent per light (squaring amplifies the
# rounding error), and summing that overshoot across up to _MAX_LIGHTS lights
# compounds it further -- empirically, two rect lights each targeting 4500px^2
# out of a 10000px^2 frame achieved a combined 9522px^2 (95.2%), which would
# have silently defeated _require_exceeds for a profile whose max_area_ratio
# sits at 0.95 (right where _MAX_AREA_RATIO would otherwise cap it). Capping
# the combined *lights* target well below _MAX_AREA_RATIO leaves enough
# headroom to absorb that per-light overshoot; the legacy single-rectangle
# path (_mask_dims) is unaffected and keeps using _MAX_AREA_RATIO directly.
#
# NOTE on _require_exceeds below (sum(_light_area(...)) vs what actually
# renders): the analytic area sum is an upper bound on rendered area, not a
# safety margin / "conservative" floor. Measured on a 480x854 frame across
# the shipped 0.25-threshold profiles: the analytic sum averaged ratio 0.352
# while the real rendered union (what synthesize_sample_realistic's Detector
# re-validation actually sees) averaged only 0.216 -- a ~39% shortfall.
# Two causes: (a) light centers are fractional/interpolated, so
# rect_area/beam_area's integer-pixel-count formulas overstate what actually
# renders (e.g. rect(half=5): analytic 121px vs rendered ~100px); (b)
# _sample_one_light draws centers uniformly across the *whole* frame, so a
# shape centered near a corner can lose most of its area to off-frame
# clipping when rendered. No shipped profile is affected (Detector uses a
# rolling-max area test across a window, not a strict per-frame test; 144/144
# real acceptance measured across all shipped profiles), but a profile closer
# to the edge could hit exactly the silent-retry-exhaustion failure mode
# _require_exceeds exists to prevent.
_MAX_LIGHTS_AREA_RATIO = 0.60
# Profiles with max_area_ratio above roughly 0.40-0.45 see a rapidly
# increasing chance that sample_lights raises a plain ValueError (not
# ClipTooShortError) because _require_exceeds rejects a per-light area
# shortfall -- measured (200 draws/point, 480x854 frame): <=0.40 -> 0%,
# 0.45-0.50 -> ~11.5%, 0.55 -> ~19.5%, 0.60 -> ~33%, >=0.70 -> 100% (not just
# unlikely -- structurally impossible, since _MAX_LIGHTS_AREA_RATIO itself
# caps the achievable combined area at/below what a threshold >= ~0.60 would
# need to exceed). All 6 shipped profiles (0.10-0.25) are comfortably in the
# 0% range and unaffected. Because this is a plain ValueError, cli.py's batch
# runner (which only catches ClipTooShortError specifically) does not catch
# it -- it propagates and aborts the whole batch run, potentially partway
# through since which draw triggers it is stochastic. Revisit before
# onboarding any profile near or above max_area_ratio ~0.60; the fix would
# need a real design decision (retry with a different shape kind, exclude
# beam from large single-light draws, or redistribute target area across
# lights) and is out of scope here.


def _light_area(light: LightShape) -> int:
    if light.kind == "rect":
        return mask_shapes.rect_area(light.half_height, light.half_width)
    if light.kind == "circle":
        return mask_shapes.circle_area(light.radius)
    if light.kind == "beam":
        return mask_shapes.beam_area(light.half_length, light.half_thickness)
    raise ValueError(f"unknown light kind: {light.kind!r}")


def _achieved_area(kind: str, target_area: float, frame_height: int, frame_width: int) -> int:
    """Area `_sample_one_light` actually produces for this kind and target.

    Asks `_sample_one_light` directly rather than re-deriving its size
    formulas, so the two can never drift apart. The generator only
    positions the shape -- `_light_area` reads the size fields alone -- so
    the answer is deterministic despite the throwaway rng.

    The result is usually below `target_area`: each branch clamps its size
    to at most half the frame, and rounding to whole pixels loses more.
    Measured on 480x854 at a 0.60 target: rect reaches 0.56, beam 0.52,
    circle only 0.44.
    """
    probe = _sample_one_light(kind, target_area, frame_height, frame_width, np.random.default_rng(0))
    return _light_area(probe)


def _feasible_kinds(
    target_area: float,
    min_required_area: float,
    frame_height: int,
    frame_width: int,
) -> tuple[str, ...]:
    """Kinds whose area at `target_area` clears `min_required_area`.

    Note the two different areas. `target_area` is what the shape aims for;
    `min_required_area` is this light's share of what `sample_lights`'
    `_require_exceeds` check actually demands (the profile's
    `max_area_ratio`). Filtering on the target instead of the requirement
    would be both too strict and too lax: too strict because a shape landing
    a hair under a generous target is still far above the threshold, and too
    lax because at a target no shape can reach, every kind gets filtered out
    and the fallback below hands back the same unfiltered choice that caused
    the failure.

    Picking uniformly from all kinds regardless was the direct cause of
    `sample_lights` raising a plain (batch-aborting) ValueError on a
    measured ~33% of draws at a 0.60 target: shapes clamp to half the frame
    and lose more to whole-pixel rounding, so on 480x854 at a 0.60 target
    rect reaches 0.56, beam 0.52, but circle only 0.44 -- a circle drawn for
    a 0.50-threshold profile simply cannot satisfy the check.

    Falls back to every kind when none qualifies, so a genuinely
    unsatisfiable profile still surfaces as `_require_exceeds`'s explanatory
    ValueError rather than an IndexError on an empty tuple.
    """
    feasible = tuple(
        kind for kind in _LIGHT_KINDS
        if _achieved_area(kind, target_area, frame_height, frame_width) > min_required_area
    )
    return feasible or _LIGHT_KINDS


def _sample_one_light(
    kind: str,
    target_area: float,
    frame_height: int,
    frame_width: int,
    rng: np.random.Generator,
) -> LightShape:
    start_row, start_col = float(rng.uniform(0, frame_height)), float(rng.uniform(0, frame_width))
    end_row, end_col = float(rng.uniform(0, frame_height)), float(rng.uniform(0, frame_width))
    max_half = max(1, min(frame_height, frame_width) // 2)

    if kind == "rect":
        half = max(1, min(round((target_area ** 0.5) / 2), max_half))
        return LightShape(
            kind="rect", start_row=start_row, start_col=start_col,
            end_row=end_row, end_col=end_col, half_height=half, half_width=half,
        )
    if kind == "circle":
        radius = max(1, min(round((target_area / np.pi) ** 0.5), max_half))
        return LightShape(
            kind="circle", start_row=start_row, start_col=start_col,
            end_row=end_row, end_col=end_col, radius=radius,
        )
    half_thickness = max(1, min(round((target_area / (4 * _BEAM_ASPECT)) ** 0.5), max_half))
    # This half_length clamp (and the max_half clamp on half_thickness above)
    # is the proximate cause of the sample_lights ValueError-rate blowup
    # documented above _MAX_LIGHTS_AREA_RATIO: for a large single-light
    # target, clamping half_length to the frame's half-side caps the beam's
    # achievable area well below its per-light target, so the combined-area
    # check in sample_lights fails increasingly often as max_area_ratio rises.
    half_length = max(1, min(_BEAM_ASPECT * half_thickness, max(frame_height, frame_width) // 2))
    angle_degrees = float(rng.uniform(0, 180))
    return LightShape(
        kind="beam", start_row=start_row, start_col=start_col,
        end_row=end_row, end_col=end_col, half_length=half_length,
        half_thickness=half_thickness, angle_degrees=angle_degrees,
    )


def sample_lights(
    profile: ThresholdProfile,
    frame_height: int,
    frame_width: int,
    rng: np.random.Generator,
) -> list[LightShape]:
    n_lights = int(rng.integers(1, _MAX_LIGHTS + 1))
    target_total_area = min(
        (profile.max_area_ratio + _AREA_MARGIN) * frame_height * frame_width,
        _MAX_LIGHTS_AREA_RATIO * frame_height * frame_width,
    )
    per_light_area = target_total_area / n_lights

    # Each light must clear its equal share of the profile's threshold, so
    # that the sum the _require_exceeds check below looks at clears the
    # whole threshold.
    min_required_per_light = profile.max_area_ratio * frame_height * frame_width / n_lights
    kinds = _feasible_kinds(per_light_area, min_required_per_light, frame_height, frame_width)
    lights = [
        _sample_one_light(
            kinds[int(rng.integers(0, len(kinds)))],
            per_light_area, frame_height, frame_width, rng,
        )
        for _ in range(n_lights)
    ]

    achieved_area_ratio = sum(_light_area(light) for light in lights) / (frame_height * frame_width)
    _require_exceeds(
        achieved_area_ratio, profile.max_area_ratio, profile.name,
        "max_area_ratio", "combined injected light area ratio",
        f"light sizes are derived from an equal per-light share of "
        f"{_MAX_LIGHTS_AREA_RATIO:.2f} of the frame (_MAX_LIGHTS_AREA_RATIO) "
        f"and rounded to whole pixels on {frame_height}x{frame_width}",
    )
    return lights


def sample_synthesis_params_realistic(
    profile: ThresholdProfile,
    pattern: str,
    clip_frame_count: int,
    frame_height: int,
    frame_width: int,
    fps: float,
    rng: np.random.Generator,
) -> SynthParams:
    """Same placement/timing as sample_synthesis_params, but replaces the
    single static rectangle with a set of moving lights (see
    docs/superpowers/specs/2026-08-04-mask-diversification-design.md) --
    used only by the realistic injection path (synthesize_sample_realistic).
    The flat path keeps calling sample_synthesis_params directly, so its
    single static rectangle is unaffected."""
    params = sample_synthesis_params(profile, pattern, clip_frame_count, frame_height, frame_width, fps, rng)
    params.window.lights = sample_lights(profile, frame_height, frame_width, rng)
    return params
