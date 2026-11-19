from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from churn.config import FEATURES, SEED, TARGET
from churn.data import clean
from churn.train import build_pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / ".github" / "scripts" / "model_quality_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("ci_model_quality_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save_dummy_pipeline(tmp_path: Path, sample_raw_df) -> Path:
    df = clean(sample_raw_df)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipeline.fit(df[FEATURES], df[TARGET])
    model_path = tmp_path / "churn_model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_gate_skips_when_model_missing_and_optional(tmp_path):
    gate = _load_gate_module()

    result = gate.evaluate_model_gate(tmp_path / "missing.joblib", require_model=False)

    assert result.passed is True
    assert result.skipped is True


def test_gate_fails_when_model_missing_and_required(tmp_path):
    gate = _load_gate_module()

    result = gate.evaluate_model_gate(tmp_path / "missing.joblib", require_model=True)

    assert result.passed is False
    assert result.skipped is False


def test_gate_passes_with_valid_serialized_pipeline(tmp_path, sample_raw_df):
    gate = _load_gate_module()
    model_path = _save_dummy_pipeline(tmp_path, sample_raw_df)

    result = gate.evaluate_model_gate(model_path, require_model=True)

    assert result.passed is True
    assert result.skipped is False
    assert result.sample_probability is not None
    assert 0.0 <= result.sample_probability <= 1.0


def test_gate_fails_when_metrics_below_threshold(tmp_path, sample_raw_df):
    gate = _load_gate_module()
    model_path = _save_dummy_pipeline(tmp_path, sample_raw_df)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"roc_auc": 0.60, "f1": 0.40}), encoding="utf-8")

    result = gate.evaluate_model_gate(
        model_path,
        require_model=True,
        metrics_path=metrics_path,
        min_roc_auc=0.75,
        min_f1=0.50,
    )

    assert result.passed is False
    assert "roc_auc" in result.message


def test_gate_passes_when_metrics_meet_threshold(tmp_path, sample_raw_df):
    gate = _load_gate_module()
    model_path = _save_dummy_pipeline(tmp_path, sample_raw_df)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"roc_auc": 0.82, "f1": 0.62}), encoding="utf-8")

    result = gate.evaluate_model_gate(
        model_path,
        require_model=True,
        metrics_path=metrics_path,
        min_roc_auc=0.75,
        min_f1=0.50,
    )

    assert result.passed is True
    assert result.skipped is False
