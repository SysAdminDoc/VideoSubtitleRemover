"""Cross-source dependency-floor consistency guard.

Version floors live in four places that must agree:

* ``requirements.txt``            -- ``>=`` floors (and ``<`` ceilings)
* ``dependency_profiles.json``    -- exact ``==`` reviewed pins
* ``backend/dependency_caps.py``  -- ``TRACKED_PACKAGES`` (min, max-exclusive)
* ``backend/security_checks.py``  -- the libpng floor and the shared Pillow floor

A drift between any of them is the root cause of stale security advisories,
so this test fails the moment the sources disagree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from packaging.version import Version

import backend.dependency_caps as caps
import backend.release_verification as rv

REPO_ROOT = Path(__file__).resolve().parent.parent


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _tracked() -> dict[str, tuple[str, str]]:
    return {_norm(n): (mn, mx) for n, mn, mx in caps.TRACKED_PACKAGES}


def _requirements_floors() -> dict[str, tuple[str, str]]:
    """Parse uncommented ``name>=min[,<max]`` lines from requirements.txt."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    floors: dict[str, tuple[str, str]] = {}
    line_re = re.compile(
        r"^([A-Za-z0-9_.-]+)\s*>=\s*([0-9][0-9.]*)"
        r"(?:\s*,\s*<\s*([0-9][0-9.]*))?\s*$"
    )
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if m:
            floors[_norm(m.group(1))] = (m.group(2), m.group(3) or "")
    return floors


def _profile_pins() -> dict[str, str]:
    """All exact ``name==version`` pins from dependency_profiles.json
    (common constraints plus every profile)."""
    data = json.loads(
        (REPO_ROOT / "dependency_profiles.json").read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    constraints = list(data.get("commonConstraints", []))
    for profile in data.get("profiles", {}).values():
        constraints.extend(profile.get("constraints", []))
    for entry in constraints:
        if "==" in entry:
            name, _, version = entry.partition("==")
            pins[_norm(name)] = version.strip()
    return pins


def test_requirements_floors_match_tracked_caps():
    tracked = _tracked()
    reqs = _requirements_floors()
    checked = 0
    for name, (floor, ceiling) in reqs.items():
        if name not in tracked:
            continue
        checked += 1
        t_min, t_max = tracked[name]
        assert Version(floor) == Version(t_min), (
            f"{name}: requirements floor {floor} != caps min {t_min}")
        if ceiling:
            assert t_max and Version(ceiling) == Version(t_max), (
                f"{name}: requirements ceiling {ceiling} != caps max {t_max}")
    # Guard against a silently-empty parse masking real drift.
    assert checked >= 3, f"expected to cross-check >=3 packages, got {checked}"


def test_profile_pins_satisfy_tracked_caps():
    tracked = _tracked()
    pins = _profile_pins()
    checked = 0
    for name, version in pins.items():
        if name not in tracked:
            continue
        checked += 1
        t_min, t_max = tracked[name]
        assert Version(version) >= Version(t_min), (
            f"{name}: profile pin {version} below caps floor {t_min}")
        if t_max:
            assert Version(version) < Version(t_max), (
                f"{name}: profile pin {version} not below caps max {t_max}")
    assert checked >= 4, f"expected to cross-check >=4 pins, got {checked}"


def test_pillow_floor_single_source():
    tracked = _tracked()
    reqs = _requirements_floors()
    pins = _profile_pins()
    floor = caps.PILLOW_MINIMUM_VERSION
    assert rv.PILLOW_MINIMUM_VERSION == floor
    assert tracked["pillow"][0] == floor
    assert reqs["pillow"][0] == floor
    assert pins["pillow"] == floor


def test_onnxruntime_directml_pin_matches_caps_window():
    pins = _profile_pins()
    assert pins["onnxruntime-directml"] == caps.ONNXRUNTIME_DIRECTML_VERSION
    assert Version(caps.ONNXRUNTIME_DIRECTML_VERSION) < Version(
        caps.ONNXRUNTIME_DIRECTML_MAX_EXCLUSIVE)
