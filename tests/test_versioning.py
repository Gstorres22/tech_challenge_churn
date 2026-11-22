from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / ".github" / "scripts" / "versioning.py"


def _load_versioning():
    spec = importlib.util.spec_from_file_location("ci_versioning", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test Bot")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0.1.0"\n', "utf-8")
    (repo / "app.txt").write_text("v0\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "develop")


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


def test_format_release_branch_normalizes_and_validates_version():
    versioning = _load_versioning()

    assert versioning.format_release_branch("0.1.0") == "release/v0.1.0"
    assert versioning.format_release_branch("v0.1.12") == "release/v0.1.12"
    with pytest.raises(ValueError):
        versioning.format_release_branch("invalid")


def test_find_active_release_branches_filters_and_sorts_numerically():
    versioning = _load_versioning()

    branches = [
        "origin/develop",
        "origin/release/beta",
        "origin/release/v0.1.2",
        "refs/remotes/origin/release/v0.1.10",
        "refs/heads/release/v0.1.9",
        "release/v0.1.10",
    ]
    assert versioning.find_active_release_branches(branches) == [
        "release/v0.1.10",
        "release/v0.1.9",
        "release/v0.1.2",
    ]


def test_decide_release_action_respects_single_active_release_rule():
    versioning = _load_versioning()

    no_changes = versioning.decide_release_action(0, [], "v0.1.2")
    assert no_changes["action"] == "skip_no_changes"

    create = versioning.decide_release_action(3, [], "v0.1.2")
    assert create["action"] == "create_release"
    assert create["branch"] == "release/v0.1.2"
    assert create["title"] == "Release v0.1.2"

    frozen = versioning.decide_release_action(4, ["origin/release/v0.1.2"], "v0.1.2")
    assert frozen["action"] == "frozen_existing"
    assert frozen["branch"] == "release/v0.1.2"
    assert frozen["version"] == "v0.1.2"


def test_needs_backmerge_false_when_release_has_no_homol_edits(tmp_path):
    versioning = _load_versioning()
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    (repo / "app.txt").write_text("feature-1\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "feat: feature 1")

    _git(repo, "checkout", "-b", "release/v0.1.0")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "release/v0.1.0", "-m", "Merge release/v0.1.0")

    assert versioning.needs_backmerge_to_develop("main", "develop", repo_root=repo) is False


def test_needs_backmerge_false_when_develop_moves_ahead_without_homol_edits(tmp_path):
    versioning = _load_versioning()
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    (repo / "app.txt").write_text("feature-1\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "feat: feature 1")

    _git(repo, "checkout", "-b", "release/v0.1.0")

    _git(repo, "checkout", "develop")
    (repo / "next_feature.txt").write_text("feature-2\n", encoding="utf-8")
    _git(repo, "add", "next_feature.txt")
    _git(repo, "commit", "-m", "feat: feature 2 on develop")

    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "release/v0.1.0", "-m", "Merge release/v0.1.0")

    assert versioning.needs_backmerge_to_develop("main", "develop", repo_root=repo) is False


def test_needs_backmerge_true_when_homol_hotfix_merged_and_false_after_sync(tmp_path):
    versioning = _load_versioning()
    repo = tmp_path / "repo"
    _init_git_repo(repo)

    (repo / "app.txt").write_text("feature-1\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "feat: feature 1")

    _git(repo, "checkout", "-b", "release/v0.1.0")
    (repo / "homol_fix.txt").write_text("fix-in-homol\n", encoding="utf-8")
    _git(repo, "add", "homol_fix.txt")
    _git(repo, "commit", "-m", "fix: ajuste em homologacao")

    _git(repo, "checkout", "develop")
    (repo / "next_feature.txt").write_text("feature-2\n", encoding="utf-8")
    _git(repo, "add", "next_feature.txt")
    _git(repo, "commit", "-m", "feat: feature 2 on develop")

    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "release/v0.1.0", "-m", "Merge release/v0.1.0")

    assert versioning.needs_backmerge_to_develop("main", "develop", repo_root=repo) is True

    _git(repo, "checkout", "develop")
    _git(repo, "merge", "--no-ff", "main", "-m", "chore(sync): back-merge Release v0.1.0")

    assert versioning.needs_backmerge_to_develop("main", "develop", repo_root=repo) is False


def test_cli_plan_release_branch_and_check_backmerge_with_remote(tmp_path, capsys):
    versioning = _load_versioning()
    remote = tmp_path / "remote.git"
    remote.mkdir()
    _git(remote, "init", "--bare")

    repo = tmp_path / "work"
    _init_git_repo(repo)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "push", "-u", "origin", "develop")

    # Sem commits à frente e sem release/* no remoto: fetch wildcard deve passar limpo.
    _git(
        repo,
        "fetch",
        "origin",
        "+refs/heads/main:refs/remotes/origin/main",
        "+refs/heads/release/*:refs/remotes/origin/release/*",
        "--tags",
    )
    assert versioning.main(["plan-release", "--pyproject", str(repo / "pyproject.toml")]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "skip_no_changes"

    # Novo commit em develop -> deve planejar create_release para release/v0.1.0.
    (repo / "app.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "app.txt")
    _git(repo, "commit", "-m", "feat: v1")
    assert versioning.main(["plan-release", "--pyproject", str(repo / "pyproject.toml")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["action"] == "create_release"
    assert plan["branch"] == "release/v0.1.0"

    # Publica release/v0.1.0 no remoto -> próximo plan-release deve congelar Homol.
    _git(repo, "push", "origin", "HEAD:refs/heads/release/v0.1.0")
    _git(
        repo,
        "fetch",
        "origin",
        "+refs/heads/main:refs/remotes/origin/main",
        "+refs/heads/release/*:refs/remotes/origin/release/*",
        "--tags",
    )
    assert versioning.main(["plan-release", "--pyproject", str(repo / "pyproject.toml")]) == 0
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["action"] == "frozen_existing"
    assert frozen["branch"] == "release/v0.1.0"

    assert versioning.main(["branch", "--version", "v0.1.2"]) == 0
    assert capsys.readouterr().out.strip() == "release/v0.1.2"

    assert (
        versioning.main(
            [
                "check-backmerge",
                "--pyproject",
                str(repo / "pyproject.toml"),
                "--source-ref",
                "main",
                "--develop-ref",
                "develop",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "false"
