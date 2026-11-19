"""Utilitário de versionamento sequencial para a esteira CI/CD (Dev -> Homol -> Prod)."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path

SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
PLAIN_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


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


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada CLI para consulta da próxima versão e título do PR."""
    default_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"

    parser = argparse.ArgumentParser(description="Cálculo sequencial de versão de release.")
    parser.add_argument("command", choices=["next", "title"], help="Informação a imprimir.")
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=default_pyproject,
        help="Caminho para o pyproject.toml.",
    )
    args = parser.parse_args(argv)

    base_version = read_pyproject_version(args.pyproject)
    tags = list_git_tags(args.pyproject.resolve().parent)
    version = compute_next_version(tags, base_version=base_version)

    if args.command == "next":
        print(version)
    else:
        print(format_release_title(version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
