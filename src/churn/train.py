"""Treino, validação cruzada e avaliação de negócio dos modelos de churn."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn.config import (
    BUSINESS_ACTION_COST,
    BUSINESS_LTV_MONTHS,
    BUSINESS_RETENTION_SUCCESS_RATE,
    COMPARISON_REPORT_PATH,
    CV_FOLDS,
    FEATURES,
    METRICS_PATH,
    MIN_F1,
    MIN_RECALL,
    MODEL_PATH,
    RAW_DATA_FILE,
    SEED,
    TARGET,
    TEST_SIZE,
)
from churn.data import clean, load_raw
from churn.evaluate import business_metrics, classification_metrics, find_optimal_threshold
from churn.features import build_preprocessor


def get_candidate_models() -> dict[str, object]:
    """Modelos exigidos pelo enunciado, todos com seed fixa."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=500,
            random_state=SEED,
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    """Encapsula pré-processamento + estimador em um único artefato serializável."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def split_data(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide o dataset limpo em treino e holdout de forma estratificada e determinística."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def _evaluate_predictions(
    y_true,
    y_proba,
    monthly_charges,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Combina métricas técnicas e de negócio para um conjunto de probabilidades."""
    cls = classification_metrics(y_true, y_proba, threshold=threshold)
    biz = business_metrics(y_true, y_proba, monthly_charges, threshold=threshold)
    return {"threshold": float(threshold), **cls, **biz}


def _evaluate_single_candidate_cv(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    skf: StratifiedKFold,
) -> tuple[dict[str, float | str], dict[str, object]]:
    """Executa CV estratificada e otimização OOF de limiar para um único candidato."""
    oof_proba = np.zeros(len(X_train), dtype=float)
    train_scores: list[dict[str, float]] = []
    val_scores: list[dict[str, float]] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = build_pipeline(clone(model))
        pipeline.fit(X_fold_train, y_fold_train)

        train_proba = pipeline.predict_proba(X_fold_train)[:, 1]
        val_proba = pipeline.predict_proba(X_fold_val)[:, 1]
        oof_proba[val_idx] = val_proba

        train_scores.append(classification_metrics(y_fold_train, train_proba, threshold=0.5))
        val_scores.append(classification_metrics(y_fold_val, val_proba, threshold=0.5))

    cv_train_roc_auc_mean = float(np.mean([m["roc_auc"] for m in train_scores]))
    cv_roc_auc_mean = float(np.mean([m["roc_auc"] for m in val_scores]))
    cv_roc_auc_std = float(np.std([m["roc_auc"] for m in val_scores]))
    overfitting_gap = float(cv_train_roc_auc_mean - cv_roc_auc_mean)

    charges_train = X_train["MonthlyCharges"]
    oof_default = _evaluate_predictions(y_train, oof_proba, charges_train, threshold=0.5)
    opt_threshold, oof_opt = find_optimal_threshold(
        y_train,
        oof_proba,
        charges_train,
        min_f1=MIN_F1,
        min_recall=MIN_RECALL,
    )
    oof_optimal = {"threshold": float(opt_threshold), **oof_opt}

    summary_row: dict[str, float | str] = {
        "model": model_name,
        "cv_train_roc_auc_mean": cv_train_roc_auc_mean,
        "cv_roc_auc_mean": cv_roc_auc_mean,
        "cv_roc_auc_std": cv_roc_auc_std,
        "overfitting_gap": overfitting_gap,
        "cv_pr_auc_mean": float(np.mean([m["pr_auc"] for m in val_scores])),
        "cv_f1_mean": float(np.mean([m["f1"] for m in val_scores])),
        "cv_f1_std": float(np.std([m["f1"] for m in val_scores])),
        "cv_precision_mean": float(np.mean([m["precision"] for m in val_scores])),
        "cv_recall_mean": float(np.mean([m["recall"] for m in val_scores])),
        "cv_accuracy_mean": float(np.mean([m["accuracy"] for m in val_scores])),
        "opt_threshold": float(opt_threshold),
        "oof_roc_auc": float(oof_opt["roc_auc"]),
        "oof_pr_auc": float(oof_opt["pr_auc"]),
        "oof_f1_opt": float(oof_opt["f1"]),
        "oof_precision_opt": float(oof_opt["precision"]),
        "oof_recall_opt": float(oof_opt["recall"]),
        "oof_accuracy_opt": float(oof_opt["accuracy"]),
        "oof_net_retained_revenue": float(oof_opt["net_retained_revenue"]),
        "oof_roi_opt": float(oof_opt["roi"]),
        "oof_gain_vs_target_all": float(oof_opt["gain_vs_target_all"]),
    }
    detail_entry: dict[str, object] = {
        **summary_row,
        "oof_default": oof_default,
        "oof_optimal": oof_optimal,
    }
    return summary_row, detail_entry


def evaluate_candidates_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict[str, object] | None = None,
    *,
    cv_folds: int = CV_FOLDS,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Avalia candidatos por CV estratificada e seleciona limiar OOF sem data leakage."""
    candidate_models = get_candidate_models() if models is None else models
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    rows: list[dict[str, float | str]] = []
    details: dict[str, dict[str, object]] = {}
    for model_name, model in candidate_models.items():
        summary_row, detail_entry = _evaluate_single_candidate_cv(
            model_name,
            model,
            X_train,
            y_train,
            skf=skf,
        )
        rows.append(summary_row)
        details[model_name] = detail_entry

    df_summary = (
        pd.DataFrame(rows)
        .sort_values(
            by=["cv_roc_auc_mean", "oof_net_retained_revenue", "oof_f1_opt"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return df_summary, details


def _build_comparison_row(
    rank: int,
    is_champion: bool,
    cv_row: dict[str, object],
    holdout_default: dict[str, float | int],
    holdout_optimal: dict[str, float | int],
) -> dict[str, object]:
    """Consolida métricas de CV, OOF e holdout para uma linha da tabela comparativa."""
    return {
        "rank": rank,
        "model": cv_row["model"],
        "is_champion": is_champion,
        "cv_train_roc_auc_mean": cv_row["cv_train_roc_auc_mean"],
        "cv_roc_auc_mean": cv_row["cv_roc_auc_mean"],
        "cv_roc_auc_std": cv_row["cv_roc_auc_std"],
        "overfitting_gap": cv_row["overfitting_gap"],
        "cv_pr_auc_mean": cv_row["cv_pr_auc_mean"],
        "cv_f1_mean": cv_row["cv_f1_mean"],
        "cv_f1_std": cv_row["cv_f1_std"],
        "cv_precision_mean": cv_row["cv_precision_mean"],
        "cv_recall_mean": cv_row["cv_recall_mean"],
        "cv_accuracy_mean": cv_row["cv_accuracy_mean"],
        "opt_threshold": cv_row["opt_threshold"],
        "oof_f1_opt": cv_row["oof_f1_opt"],
        "oof_recall_opt": cv_row["oof_recall_opt"],
        "oof_net_retained_revenue": cv_row["oof_net_retained_revenue"],
        "oof_roi_opt": cv_row["oof_roi_opt"],
        "holdout_roc_auc": holdout_optimal["roc_auc"],
        "holdout_pr_auc": holdout_optimal["pr_auc"],
        "holdout_f1_05": holdout_default["f1"],
        "holdout_precision_05": holdout_default["precision"],
        "holdout_recall_05": holdout_default["recall"],
        "holdout_accuracy_05": holdout_default["accuracy"],
        "holdout_net_revenue_05": holdout_default["net_retained_revenue"],
        "holdout_roi_05": holdout_default["roi"],
        "holdout_f1_opt": holdout_optimal["f1"],
        "holdout_precision_opt": holdout_optimal["precision"],
        "holdout_recall_opt": holdout_optimal["recall"],
        "holdout_accuracy_opt": holdout_optimal["accuracy"],
        "holdout_net_revenue_opt": holdout_optimal["net_retained_revenue"],
        "holdout_roi_opt": holdout_optimal["roi"],
        "holdout_gain_vs_target_all_opt": holdout_optimal["gain_vs_target_all"],
    }


def _build_metrics_payload(
    *,
    champion_name: str,
    champion_threshold: float,
    champion_detail: dict[str, object],
    champion_holdout_default: dict[str, float | int],
    champion_holdout_optimal: dict[str, float | int],
    dataset_info: dict[str, int | float],
    all_models: list[dict[str, object]],
) -> dict[str, object]:
    """Monta o dicionário completo de métricas serializado em models/metrics.json."""
    business_keys = (
        "gross_saved_revenue",
        "campaign_cost",
        "net_retained_revenue",
        "roi",
        "targeted_customers",
        "true_positives_caught",
        "false_positives",
        "baseline_target_all_net_revenue",
        "gain_vs_target_all",
    )
    return {
        "champion_model": champion_name,
        "threshold": float(champion_threshold),
        "roc_auc": float(champion_holdout_optimal["roc_auc"]),
        "pr_auc": float(champion_holdout_optimal["pr_auc"]),
        "f1": float(champion_holdout_optimal["f1"]),
        "precision": float(champion_holdout_optimal["precision"]),
        "recall": float(champion_holdout_optimal["recall"]),
        "accuracy": float(champion_holdout_optimal["accuracy"]),
        "net_retained_revenue": float(champion_holdout_optimal["net_retained_revenue"]),
        "roi": float(champion_holdout_optimal["roi"]),
        "cv_roc_auc_mean": float(champion_detail["cv_roc_auc_mean"]),
        "cv_roc_auc_std": float(champion_detail["cv_roc_auc_std"]),
        "cv_train_roc_auc_mean": float(champion_detail["cv_train_roc_auc_mean"]),
        "overfitting_gap": float(champion_detail["overfitting_gap"]),
        "baseline_target_all_net_revenue": float(
            champion_holdout_optimal["baseline_target_all_net_revenue"]
        ),
        "gain_vs_target_all": float(champion_holdout_optimal["gain_vs_target_all"]),
        "holdout_default_threshold": champion_holdout_default,
        "holdout_optimal_threshold": champion_holdout_optimal,
        "business_metrics": {
            "assumptions": {
                "ltv_months": BUSINESS_LTV_MONTHS,
                "retention_success_rate": BUSINESS_RETENTION_SUCCESS_RATE,
                "action_cost": BUSINESS_ACTION_COST,
            },
            "optimal_threshold": float(champion_threshold),
            **{key: champion_holdout_optimal[key] for key in business_keys},
            "default_threshold_net_retained_revenue": float(
                champion_holdout_default["net_retained_revenue"]
            ),
            "gain_vs_default_threshold": float(
                champion_holdout_optimal["net_retained_revenue"]
                - champion_holdout_default["net_retained_revenue"]
            ),
        },
        "dataset_info": dataset_info,
        "all_models": all_models,
    }


def train_and_evaluate(
    raw_data_path: Path | str = RAW_DATA_FILE,
    *,
    model_output_path: Path | str = MODEL_PATH,
    metrics_output_path: Path | str = METRICS_PATH,
    comparison_output_path: Path | str = COMPARISON_REPORT_PATH,
    cv_folds: int = CV_FOLDS,
) -> dict[str, object]:
    """Executa split, CV sem data leakage, re-treino no treino e avaliação no holdout."""
    df = clean(load_raw(raw_data_path))
    X_train, X_test, y_train, y_test = split_data(df, test_size=TEST_SIZE, random_state=SEED)

    models = get_candidate_models()
    cv_df, cv_details = evaluate_candidates_cv(
        X_train,
        y_train,
        models=models,
        cv_folds=cv_folds,
        random_state=SEED,
    )

    champion_name = str(cv_df.iloc[0]["model"])
    champion_pipeline: Pipeline | None = None
    champion_holdout_default: dict[str, float | int] = {}
    champion_holdout_optimal: dict[str, float | int] = {}
    champion_threshold = float(cv_df.iloc[0]["opt_threshold"])

    comparison_rows: list[dict[str, object]] = []
    all_models: list[dict[str, object]] = []
    test_charges = X_test["MonthlyCharges"]

    for rank, cv_row in enumerate(cv_df.to_dict(orient="records"), start=1):
        model_name = str(cv_row["model"])
        opt_threshold = float(cv_row["opt_threshold"])
        is_champion = model_name == champion_name

        pipeline = build_pipeline(clone(models[model_name]))
        pipeline.fit(X_train, y_train)
        test_proba = pipeline.predict_proba(X_test)[:, 1]

        holdout_default = _evaluate_predictions(y_test, test_proba, test_charges, threshold=0.5)
        holdout_optimal = _evaluate_predictions(
            y_test,
            test_proba,
            test_charges,
            threshold=opt_threshold,
        )

        if is_champion:
            champion_pipeline = pipeline
            champion_holdout_default = holdout_default
            champion_holdout_optimal = holdout_optimal
            champion_threshold = opt_threshold

        comparison_rows.append(
            _build_comparison_row(
                rank=rank,
                is_champion=is_champion,
                cv_row=cv_row,
                holdout_default=holdout_default,
                holdout_optimal=holdout_optimal,
            )
        )
        all_models.append(
            {
                "rank": rank,
                "model": model_name,
                "is_champion": is_champion,
                "cv": cv_row,
                "oof_default_threshold": cv_details[model_name]["oof_default"],
                "oof_optimal_threshold": cv_details[model_name]["oof_optimal"],
                "holdout_default_threshold": holdout_default,
                "holdout_optimal_threshold": holdout_optimal,
            }
        )

    comparison_path = Path(comparison_output_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False, lineterminator="\n")

    model_path = Path(model_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_pipeline, model_path, compress=3)

    dataset_info: dict[str, int | float] = {
        "raw_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_count": int(len(FEATURES)),
        "churn_rate_total": float(df[TARGET].mean()),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
        "test_size": float(TEST_SIZE),
        "cv_folds": int(cv_folds),
        "random_state": int(SEED),
    }
    metrics_payload = _build_metrics_payload(
        champion_name=champion_name,
        champion_threshold=champion_threshold,
        champion_detail=cv_details[champion_name],
        champion_holdout_default=champion_holdout_default,
        champion_holdout_optimal=champion_holdout_optimal,
        dataset_info=dataset_info,
        all_models=all_models,
    )

    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(metrics_payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    return metrics_payload


def main() -> None:
    """Ponto de entrada CLI para treino, avaliação e persistência dos artefatos."""
    metrics = train_and_evaluate()
    print(
        f"Campeao: {metrics['champion_model']} | "
        f"Threshold otimo: {metrics['threshold']:.2f} | "
        f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} +/- {metrics['cv_roc_auc_std']:.4f} | "
        f"Overfitting gap: {metrics['overfitting_gap']:.4f}"
    )
    print(
        f"Holdout -> ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )
    print(
        f"Negocio (Holdout) -> Receita retida liquida: ${metrics['net_retained_revenue']:,.2f} | "
        f"ROI: {metrics['roi']:.2f}x | "
        f"Ganho vs target-all: ${metrics['gain_vs_target_all']:,.2f}"
    )


if __name__ == "__main__":
    main()
