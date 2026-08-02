"""Loads regional PSE threshold profiles. Values sourced from
flicker-guard/README.md section 2 (regulatory summary table).

This module owns *every* numeric detection knob, per the plan's Global
Constraint "all threshold values must be configurable per profile, never
hardcoded in logic". The per-pixel thresholds
(`general_flash_dark_threshold`, `general_flash_delta_threshold`,
`red_saturation_ratio_threshold`) lived as module constants in
`detector/flash.py` until final-review finding I3 moved them here; `flash.py`
now takes them as parameters and the pipeline threads them through from the
active profile.

Profile JSON files may omit the per-pixel fields, in which case the
`DEFAULT_*` values below apply — they reproduce the previously hardcoded
behaviour exactly.

MVP limitation (see also `detector/pipeline.py`): this schema encodes only a
flash-frequency limit and a flagged-area limit. Regulatory rules that are not
reducible to those two numbers — Ofcom's dark-scene sub-rule (luminance < 160
with contrast >= 20) and Japan's high-contrast pattern-density rule — are NOT
encoded here and are therefore NOT detected. External validation (Harding FPA
or equivalent) is required before production use.
"""
import json
from dataclasses import dataclass
from pathlib import Path

# Defaults reproduce the constants that used to live in detector/flash.py.
DEFAULT_GENERAL_FLASH_DARK_THRESHOLD = 0.80
DEFAULT_GENERAL_FLASH_DELTA_THRESHOLD = 0.10
DEFAULT_RED_SATURATION_RATIO_THRESHOLD = 0.80


@dataclass(eq=True)
class ThresholdProfile:
    """Thresholds for one regulatory profile.

    `max_flashes_per_second` is the regulatory flashes-per-second limit from
    README section 2. Note that the detector compares it against
    `FlickerScore.flagged_frame_count_last_second`, which counts flagged
    *frames* (~2x the visual flash rate) — see `detector/scoring.py`. That
    comparison is deliberately conservative (roughly 2x stricter than the
    regulation), never permissive.
    """

    name: str
    max_flashes_per_second: float
    max_area_ratio: float
    # Per-pixel general-flash test: a transition counts only when the darker
    # of the two frames is below `general_flash_dark_threshold` and the
    # relative-luminance change exceeds `general_flash_delta_threshold`.
    general_flash_dark_threshold: float = DEFAULT_GENERAL_FLASH_DARK_THRESHOLD
    general_flash_delta_threshold: float = DEFAULT_GENERAL_FLASH_DELTA_THRESHOLD
    # WCAG "saturated red" test: R / (R + G + B) >= this ratio.
    red_saturation_ratio_threshold: float = DEFAULT_RED_SATURATION_RATIO_THRESHOLD


def load_profile(path: Path) -> ThresholdProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThresholdProfile(
        name=data["name"],
        max_flashes_per_second=data["max_flashes_per_second"],
        max_area_ratio=data["max_area_ratio"],
        general_flash_dark_threshold=data.get(
            "general_flash_dark_threshold", DEFAULT_GENERAL_FLASH_DARK_THRESHOLD
        ),
        general_flash_delta_threshold=data.get(
            "general_flash_delta_threshold", DEFAULT_GENERAL_FLASH_DELTA_THRESHOLD
        ),
        red_saturation_ratio_threshold=data.get(
            "red_saturation_ratio_threshold", DEFAULT_RED_SATURATION_RATIO_THRESHOLD
        ),
    )
