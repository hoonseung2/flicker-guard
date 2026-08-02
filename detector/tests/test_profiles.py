import json
from pathlib import Path
import pytest
from detector.profiles import ThresholdProfile, load_profile

PROFILE_DIR = Path(__file__).parent.parent.parent / "configs" / "profiles"

def test_load_profile_reads_json(tmp_path):
    profile_path = tmp_path / "custom.json"
    profile_path.write_text(json.dumps({
        "name": "custom",
        "max_flashes_per_second": 3,
        "max_area_ratio": 0.10,
    }))
    profile = load_profile(profile_path)
    assert profile == ThresholdProfile(name="custom", max_flashes_per_second=3, max_area_ratio=0.10)

@pytest.mark.parametrize("filename,expected_area_ratio", [
    ("kr.json", 0.10),
    ("jp.json", 0.25),
    ("itu.json", 0.25),
    ("ofcom.json", 0.25),
    ("w3c.json", 0.25),
    ("netflix.json", 0.25),
])
def test_shipped_profiles_match_readme_thresholds(filename, expected_area_ratio):
    profile = load_profile(PROFILE_DIR / filename)
    assert profile.max_area_ratio == expected_area_ratio
    assert profile.max_flashes_per_second == 3

def test_threshold_profile_defaults_match_legacy_hardcoded_values():
    # I3: the per-pixel knobs moved out of flash.py; a profile built from only
    # the original three fields must reproduce the old hardcoded behaviour.
    profile = ThresholdProfile(name="legacy", max_flashes_per_second=3, max_area_ratio=0.10)
    assert profile.general_flash_dark_threshold == 0.80
    assert profile.general_flash_delta_threshold == 0.10
    assert profile.red_saturation_ratio_threshold == 0.80

@pytest.mark.parametrize(
    "filename",
    ["kr.json", "jp.json", "itu.json", "ofcom.json", "w3c.json", "netflix.json"],
)
def test_shipped_profiles_declare_per_pixel_thresholds_explicitly(filename):
    # I3: every shipped profile spells the detection knobs out, so tuning one
    # region never silently depends on a code default.
    data = json.loads((PROFILE_DIR / filename).read_text(encoding="utf-8"))
    assert data["general_flash_dark_threshold"] == 0.80
    assert data["general_flash_delta_threshold"] == 0.10
    assert data["red_saturation_ratio_threshold"] == 0.80

def test_load_profile_reads_per_pixel_thresholds_from_json(tmp_path):
    profile_path = tmp_path / "strict.json"
    profile_path.write_text(json.dumps({
        "name": "strict",
        "max_flashes_per_second": 2,
        "max_area_ratio": 0.05,
        "general_flash_dark_threshold": 0.90,
        "general_flash_delta_threshold": 0.02,
        "red_saturation_ratio_threshold": 0.70,
    }))
    profile = load_profile(profile_path)
    assert profile.general_flash_dark_threshold == 0.90
    assert profile.general_flash_delta_threshold == 0.02
    assert profile.red_saturation_ratio_threshold == 0.70
