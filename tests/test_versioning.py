from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / ".github" / "scripts" / "versioning.py"


def _load_versioning():
    spec = importlib.util.spec_from_file_location("ci_versioning", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_release_uses_base_version_when_no_tags_exist():
    versioning = _load_versioning()

    assert versioning.compute_next_version([], base_version="0.1.0") == "v0.1.0"


def test_increments_patch_from_latest_tag():
    versioning = _load_versioning()

    assert versioning.compute_next_version(["v0.1.0", "v0.1.1"]) == "v0.1.2"


def test_compares_versions_numerically_not_lexicographically():
    versioning = _load_versioning()

    assert versioning.compute_next_version(["v0.1.2", "v0.1.9", "v0.1.10"]) == "v0.1.11"


def test_ignores_non_semver_tags():
    versioning = _load_versioning()

    tags = ["v0.1.0", "beta", "v1.0", "release-1"]
    assert versioning.compute_next_version(tags) == "v0.1.1"


def test_respects_manual_base_version_bump_above_latest_tag():
    versioning = _load_versioning()

    assert versioning.compute_next_version(["v0.1.5"], base_version="0.2.0") == "v0.2.0"


def test_format_release_title_prepends_release_prefix():
    versioning = _load_versioning()

    assert versioning.format_release_title("v0.1.3") == "Release v0.1.3"


def test_read_pyproject_version_returns_valid_semver():
    versioning = _load_versioning()

    version = versioning.read_pyproject_version(ROOT_DIR / "pyproject.toml")
    parsed = versioning.parse_plain_version(version)
    assert len(parsed) == 3
