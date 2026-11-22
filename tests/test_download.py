from __future__ import annotations

import io
import urllib.request
import zipfile

import pytest


class _DummyHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _DummyHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_download_constants():
    from churn.download import KAGGLE_DATASET, KAGGLE_DOWNLOAD_URL

    assert KAGGLE_DATASET == "blastchar/telco-customer-churn"
    assert (
        KAGGLE_DOWNLOAD_URL
        == "https://www.kaggle.com/api/v1/datasets/download/blastchar/telco-customer-churn"
    )


def test_get_kaggle_token_raises_when_missing(tmp_path, monkeypatch):
    from churn.download import get_kaggle_token

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError):
        get_kaggle_token(env_path=None)

    with pytest.raises(RuntimeError):
        get_kaggle_token(env_path=tmp_path / "missing.env")


def test_get_kaggle_token_raises_when_empty(tmp_path, monkeypatch):
    from churn.download import get_kaggle_token

    monkeypatch.setenv("KAGGLE_API_TOKEN", "   ")
    with pytest.raises(RuntimeError):
        get_kaggle_token(env_path=None)

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("KAGGLE_API_TOKEN=\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        get_kaggle_token(env_path=empty_env)


def test_get_kaggle_token_reads_from_environment(monkeypatch):
    from churn.download import get_kaggle_token

    monkeypatch.setenv("KAGGLE_API_TOKEN", "  dummy-env-token  ")

    assert get_kaggle_token(env_path=None) == "dummy-env-token"


def test_get_kaggle_token_reads_from_temp_dotenv_file(tmp_path, monkeypatch):
    from churn.download import get_kaggle_token

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    temp_env = tmp_path / ".env"
    temp_env.write_text("KAGGLE_API_TOKEN=dummy-file-token\n", encoding="utf-8")

    assert get_kaggle_token(env_path=temp_env) == "dummy-file-token"


def test_download_telco_dataset_extracts_csv_from_zip(tmp_path, monkeypatch):
    from churn.download import KAGGLE_DOWNLOAD_URL, download_telco_dataset

    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-download-token")
    csv_bytes = b"customerID,Churn\n0001-A,No\n0002-B,Yes\n"
    zip_bytes = _make_zip_bytes({"WA_Fn-UseC_-Telco-Customer-Churn.csv": csv_bytes})
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout=60):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        return _DummyHTTPResponse(zip_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    output_path = tmp_path / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    result = download_telco_dataset(output_path=output_path, env_path=None)

    assert result == output_path
    assert output_path.exists()
    assert output_path.read_bytes() == csv_bytes
    assert captured["url"] == KAGGLE_DOWNLOAD_URL
    assert captured["auth"] == "Bearer dummy-download-token"


def test_download_telco_dataset_skips_when_exists_and_not_force(tmp_path, monkeypatch):
    from churn.download import download_telco_dataset

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    output_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path.write_text("cached-content", encoding="utf-8")

    def _unexpected_urlopen(*args, **kwargs):
        raise AssertionError("urlopen não deve ser chamado quando force=False e arquivo existe")

    monkeypatch.setattr(urllib.request, "urlopen", _unexpected_urlopen)

    result = download_telco_dataset(output_path=output_path, force=False, env_path=None)

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "cached-content"


def test_download_telco_dataset_overwrites_when_force_true(tmp_path, monkeypatch):
    from churn.download import download_telco_dataset

    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-download-token")
    output_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path.write_text("stale-content", encoding="utf-8")

    fresh_bytes = b"customerID,Churn\n0009-Z,Yes\n"
    zip_bytes = _make_zip_bytes({"WA_Fn-UseC_-Telco-Customer-Churn.csv": fresh_bytes})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: _DummyHTTPResponse(zip_bytes),
    )

    result = download_telco_dataset(output_path=output_path, force=True, env_path=None)

    assert result == output_path
    assert output_path.read_bytes() == fresh_bytes


def test_download_telco_dataset_raises_when_csv_missing_in_zip(tmp_path, monkeypatch):
    from churn.download import download_telco_dataset

    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-download-token")
    zip_bytes = _make_zip_bytes({"unexpected.csv": b"col1,col2\n1,2\n"})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: _DummyHTTPResponse(zip_bytes),
    )

    output_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    with pytest.raises(RuntimeError):
        download_telco_dataset(output_path=output_path, env_path=None)

    assert not output_path.exists()


def test_download_telco_dataset_raises_when_csv_empty_in_zip(tmp_path, monkeypatch):
    from churn.download import download_telco_dataset

    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-download-token")
    zip_bytes = _make_zip_bytes({"WA_Fn-UseC_-Telco-Customer-Churn.csv": b""})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=60: _DummyHTTPResponse(zip_bytes),
    )

    output_path = tmp_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    with pytest.raises(RuntimeError):
        download_telco_dataset(output_path=output_path, env_path=None)
