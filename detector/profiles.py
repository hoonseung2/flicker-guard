"""Loads regional PSE threshold profiles. Values sourced from
flicker-guard/README.md section 2 (regulatory summary table)."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(eq=True)
class ThresholdProfile:
    name: str
    max_flashes_per_second: float
    max_area_ratio: float


def load_profile(path: Path) -> ThresholdProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ThresholdProfile(
        name=data["name"],
        max_flashes_per_second=data["max_flashes_per_second"],
        max_area_ratio=data["max_area_ratio"],
    )
