import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import FEATURES, SEED, TARGET
from churn.data import clean
from churn.evaluate import classification_metrics
from churn.train import build_pipeline, get_candidate_models


def test_classification_metrics_computes_perfect_scores():
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_classification_metrics_respects_custom_threshold():
    y_true = [0, 0, 1, 1]
    y_proba = [0.2, 0.4, 0.45, 0.9]

    default_threshold = classification_metrics(y_true, y_proba, threshold=0.5)
    lower_threshold = classification_metrics(y_true, y_proba, threshold=0.42)

    assert default_threshold["recall"] == pytest.approx(0.5)
    assert lower_threshold["recall"] == pytest.approx(1.0)


def test_candidate_models_include_required_families_with_fixed_seed():
    models = get_candidate_models()

    assert set(models) == {"logistic_regression", "random_forest", "mlp"}
    for model in models.values():
        assert model.random_state == SEED


def test_build_pipeline_fits_and_returns_valid_probabilities(sample_raw_df):
    df = clean(sample_raw_df)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipeline.fit(df[FEATURES], df[TARGET])

    probs = pipeline.predict_proba(df[FEATURES])[:, 1]
    assert len(probs) == len(df)
    assert all(0.0 <= p <= 1.0 for p in probs)
