"""Download automatizado do dataset Telco Customer Churn (Kaggle)."""

from __future__ import annotations

import io
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import dotenv
import pandas as pd

from churn.config import RAW_DATA_FILE, ROOT_DIR

KAGGLE_DATASET: str = "blastchar/telco-customer-churn"
KAGGLE_DOWNLOAD_URL: str = (
    "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
)
DATASET_FILENAME: str = RAW_DATA_FILE.name


def get_kaggle_token(env_path: Path | str | None = ROOT_DIR / ".env") -> str:
    """Lê ``KAGGLE_API_TOKEN`` do ambiente ou de um arquivo ``.env``."""
    if env_path is not None:
        env_file = Path(env_path)
        if env_file.exists():
            dotenv.load_dotenv(env_file)

    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "KAGGLE_API_TOKEN não definido. Configure-o no arquivo .env ou nas variáveis "
            "de ambiente antes de executar o download."
        )
    return token


def _extract_csv_from_zip(zip_bytes: bytes, target_filename: str = DATASET_FILENAME) -> bytes:
    """Extrai de um ZIP em memória o conteúdo do CSV esperado."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).name != target_filename:
                continue
            content = archive.read(member)
            if not content.strip():
                raise RuntimeError(
                    f"O arquivo '{target_filename}' está vazio dentro do ZIP baixado."
                )
            return content

    raise RuntimeError(
        f"O arquivo '{target_filename}' não foi encontrado no ZIP retornado pelo Kaggle."
    )


def _download_with_kagglehub(destination: Path) -> Path:
    """Fallback com kagglehub caso o endpoint Bearer retorne 401/403."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Download via Bearer retornou 401/403 e o pacote 'kagglehub' não está instalado."
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    candidates = [path for path in dataset_dir.rglob(DATASET_FILENAME) if path.is_file()]
    if not candidates:
        raise RuntimeError(
            f"O arquivo '{DATASET_FILENAME}' não foi encontrado no download via kagglehub."
        )

    source_file = candidates[0]
    if source_file.stat().st_size == 0:
        raise RuntimeError(f"O arquivo '{DATASET_FILENAME}' baixado via kagglehub está vazio.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination)
    return destination


def download_telco_dataset(
    output_path: Path | str = RAW_DATA_FILE,
    *,
    force: bool = False,
    env_path: Path | str | None = ROOT_DIR / ".env",
) -> Path:
    """Baixa o CSV Telco Customer Churn do Kaggle e salva em ``output_path``."""
    destination = Path(output_path)
    if destination.exists() and not force:
        return destination

    token = get_kaggle_token(env_path=env_path)
    request = urllib.request.Request(
        KAGGLE_DOWNLOAD_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "churn-ml-pipeline/0.1",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            zip_bytes = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return _download_with_kagglehub(destination)
        raise RuntimeError(f"Falha ao baixar dataset do Kaggle (HTTP {exc.code}).") from exc

    csv_bytes = _extract_csv_from_zip(zip_bytes, target_filename=DATASET_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_bytes)
    return destination


def main() -> None:
    """Executa o download do dataset bruto e exibe um resumo tabular."""
    csv_path = download_telco_dataset()
    df = pd.read_csv(csv_path)
    print(f"Dataset salvo em: {csv_path}")
    print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]} ({', '.join(df.columns)})")


if __name__ == "__main__":
    main()
