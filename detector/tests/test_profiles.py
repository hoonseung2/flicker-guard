import json
from pathlib import Path
import pytest
from detector.profiles import ThresholdProfile, load_profile

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
])
def test_shipped_profiles_match_readme_thresholds(filename, expected_area_ratio):
    path = Path(__file__).parent.parent.parent / "configs" / "profiles" / filename
    profile = load_profile(path)
    assert profile.max_area_ratio == expected_area_ratio
    assert profile.max_flashes_per_second == 3
