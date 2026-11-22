"""Utilitário de versionamento sequencial e controle de releases (Dev -> Homol -> Prod)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
PLAIN_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
RELEASE_BRANCH_PREFIX = "release/"
RELEASE_BRANCH_RE = re.compile(r"^release/v(\d+)\.(\d+)\.(\d+)$")


def parse_semver_tag(tag: str) -> tuple[int, int, int] | None:
    """Converte uma tag Git no formato ``vX.Y.Z`` para tupla numérica."""
    match = SEMVER_TAG_RE.match(tag.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_plain_version(version: str) -> tuple[int, int, int]:
    """Converte uma versão ``X.Y.Z`` (com ou sem prefixo ``v``) para tupla numérica."""
    match = PLAIN_VERSION_RE.match(version.strip())
    if match is None:
        raise ValueError(f"Versão inválida: {version!r}. Esperado formato X.Y.Z.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def read_pyproject_version(pyproject_path: Path | str) -> str:
    """Lê ``[project].version`` do ``pyproject.toml``."""
    path = Path(pyproject_path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"Campo [project].version ausente ou inválido em {path}.")

    parse_plain_version(version)
    return version.strip()


def compute_next_version(tags: Iterable[str], base_version: str = "0.1.0") -> str:
    """Calcula a próxima versão sequencial ``vX.Y.Z`` a partir das tags existentes.

    Regras:
    - Ignora tags que não seguem estritamente ``vX.Y.Z``.
    - Sem tags válidas: inicia em ``v{base_version}``.
    - Se ``base_version`` for maior que a maior tag: adota ``v{base_version}``.
    - Caso contrário: incrementa +1 no componente ``patch`` da maior tag.
    """
    base_tuple = parse_plain_version(base_version)
    valid_tags = [parsed for tag in tags if (parsed := parse_semver_tag(tag)) is not None]
    if not valid_tags:
        return f"v{base_tuple[0]}.{base_tuple[1]}.{base_tuple[2]}"

    max_tag = max(valid_tags)
    if base_tuple > max_tag:
        return f"v{base_tuple[0]}.{base_tuple[1]}.{base_tuple[2]}"

    return f"v{max_tag[0]}.{max_tag[1]}.{max_tag[2] + 1}"


def format_release_title(version: str) -> str:
    """Formata o título padrão de Pull Request e Release no GitHub."""
    return f"Release {version}"


def format_release_branch(version: str) -> str:
    """Formata o nome padrão da branch de homologação ``release/vX.Y.Z``."""
    major, minor, patch = parse_plain_version(version)
    return f"{RELEASE_BRANCH_PREFIX}v{major}.{minor}.{patch}"


def parse_release_branch(branch_name: str) -> tuple[int, int, int] | None:
    """Extrai a tupla semver de um nome de branch ``release/vX.Y.Z`` (local ou remoto)."""
    normalized = branch_name.strip()
    for prefix in ("refs/remotes/origin/", "refs/heads/", "origin/"):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break

    match = RELEASE_BRANCH_RE.match(normalized)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def find_active_release_branches(branches: Iterable[str]) -> list[str]:
    """Filtra e ordena (mais recente primeiro) branches válidas ``release/vX.Y.Z``."""
    found: dict[tuple[int, int, int], str] = {}
    for branch in branches:
        parsed = parse_release_branch(branch)
        if parsed is not None:
            found[parsed] = f"{RELEASE_BRANCH_PREFIX}v{parsed[0]}.{parsed[1]}.{parsed[2]}"

    return [found[key] for key in sorted(found, reverse=True)]


def decide_release_action(
    ahead_of_main: int,
    active_release_branches: Iterable[str],
    next_version: str,
) -> dict[str, object]:
    """Define a ação de corte de release a partir de ``develop`` (máx. 1 ativa por vez)."""
    if ahead_of_main <= 0:
        return {
            "action": "skip_no_changes",
            "version": "",
            "branch": "",
            "title": "",
            "reason": "Nenhum commit à frente da main. Nova release não necessária.",
        }

    active = find_active_release_branches(active_release_branches)
    if active:
        existing_branch = active[0]
        existing_version = existing_branch.removeprefix(RELEASE_BRANCH_PREFIX)
        return {
            "action": "frozen_existing",
            "version": existing_version,
            "branch": existing_branch,
            "title": format_release_title(existing_version),
            "reason": (
                f"Release ativa já existente ({existing_branch}). "
                "Escopo de Homol permanece congelado até o merge na main."
            ),
        }

    branch = format_release_branch(next_version)
    return {
        "action": "create_release",
        "version": next_version,
        "branch": branch,
        "title": format_release_title(next_version),
        "reason": f"Pronto para criar {branch} e abrir PR para main.",
    }


def list_git_tags(repo_root: Path | None = None) -> list[str]:
    """Lista todas as tags Git do repositório local."""
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_remote_release_branches(
    repo_root: Path | None = None,
    remote: str = "origin",
) -> list[str]:
    """Lista branches remotas ``release/vX.Y.Z`` conhecidas no repositório local."""
    result = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote}/release/"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return find_active_release_branches(result.stdout.splitlines())


def count_commits_ahead(
    base_ref: str,
    head_ref: str = "HEAD",
    repo_root: Path | None = None,
) -> int:
    """Conta quantos commits existem em ``head_ref`` que ainda não estão em ``base_ref``."""
    result = subprocess.run(
        ["git", "rev-list", "--count", f"{base_ref}..{head_ref}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def needs_backmerge_to_develop(
    source_ref: str = "HEAD",
    develop_ref: str = "origin/develop",
    repo_root: Path | None = None,
) -> bool:
    """Verifica se ``source_ref`` (Homol/Main) traz alterações de conteúdo ausentes na ``develop``.

    Compara o ``merge-base(develop_ref, source_ref)`` contra ``source_ref``:
    - Sem ajustes exclusivos em Homol/Main (mesmo que ``develop`` tenha avançado): ``False``.
    - Com hotfix/ajuste em Homol/Main não presente na ``develop``: ``True``.
    """
    merge_base = subprocess.run(
        ["git", "merge-base", develop_ref, source_ref],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    diff_proc = subprocess.run(
        ["git", "diff", "--quiet", merge_base, source_ref],
        cwd=repo_root,
        check=False,
    )
    if diff_proc.returncode == 0:
        return False
    if diff_proc.returncode == 1:
        return True
    raise RuntimeError(
        f"Falha ao comparar diff entre {merge_base} e {source_ref} (exit {diff_proc.returncode})."
    )


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada CLI para versionamento, planejamento de release e back-merge."""
    default_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    parser = argparse.ArgumentParser(description="Versionamento e controle de releases CI/CD.")
    parser.add_argument(
        "command",
        choices=["next", "title", "branch", "plan-release", "check-backmerge"],
        help="Operação a executar.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=default_pyproject,
        help="Caminho para o pyproject.toml.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Versão explícita (opcional, usada por 'branch' e 'title').",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Referência base para 'plan-release' (padrão: origin/main).",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Referência alvo para 'plan-release' (padrão: HEAD).",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Nome do remote Git para listar branches release/*.",
    )
    parser.add_argument(
        "--source-ref",
        default="HEAD",
        help="Referência liberada para 'check-backmerge' (padrão: HEAD).",
    )
    parser.add_argument(
        "--develop-ref",
        default="origin/develop",
        help="Referência da branch develop para 'check-backmerge'.",
    )
    args = parser.parse_args(argv)
    repo_root = args.pyproject.resolve().parent

    if args.command == "check-backmerge":
        needed = needs_backmerge_to_develop(
            source_ref=args.source_ref,
            develop_ref=args.develop_ref,
            repo_root=repo_root,
        )
        print("true" if needed else "false")
        return 0

    if args.version is not None:
        major, minor, patch = parse_plain_version(args.version)
        version = f"v{major}.{minor}.{patch}"
    else:
        base_version = read_pyproject_version(args.pyproject)
        tags = list_git_tags(repo_root)
        version = compute_next_version(tags, base_version=base_version)

    if args.command == "next":
        print(version)
    elif args.command == "title":
        print(format_release_title(version))
    elif args.command == "branch":
        print(format_release_branch(version))
    elif args.command == "plan-release":
        ahead = count_commits_ahead(args.base_ref, args.head_ref, repo_root=repo_root)
        active = list_remote_release_branches(repo_root=repo_root, remote=args.remote)
        plan = decide_release_action(ahead, active, next_version=version)
        print(json.dumps(plan, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
