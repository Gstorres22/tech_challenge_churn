"""Quality gate do modelo serializado e métricas para Homol/Prod."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib

from churn.api.schemas import CustomerFeatures
from churn.predict import predict_proba

SAMPLE_CUSTOMER: dict[str, object] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.2,
}


@dataclass(frozen=True)
class GateResult:
    """Resultado da verificação de qualidade do artefato de ML."""

    passed: bool
    skipped: bool
    message: str
    sample_probability: float | None = None


def evaluate_model_gate(
    model_path: Path | str,
    *,
    require_model: bool = False,
    metrics_path: Path | str | None = None,
    min_roc_auc: float = 0.80,
    min_f1: float = 0.55,
    min_recall: float = 0.60,
) -> GateResult:
    """Valida carga, contrato de entrada, saída probabilística e métricas do modelo."""
    path = Path(model_path)
    if not path.exists():
        if require_model:
            return GateResult(
                passed=False,
                skipped=False,
                message=f"FAIL: modelo obrigatório não encontrado em {path}.",
            )
        return GateResult(
            passed=True,
            skipped=True,
            message=f"SKIP: modelo não encontrado em {path} (require_model=False).",
        )

    CustomerFeatures(**SAMPLE_CUSTOMER)
    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        return GateResult(
            passed=False,
            skipped=False,
            message=f"FAIL: artefato em {path} não possui método predict_proba.",
        )

    probs = predict_proba([SAMPLE_CUSTOMER], model=model)
    if len(probs) != 1 or not math.isfinite(probs[0]) or not (0.0 <= probs[0] <= 1.0):
        return GateResult(
            passed=False,
            skipped=False,
            message=f"FAIL: probabilidade inválida gerada pelo modelo: {probs!r}.",
        )

    prob = float(probs[0])

    if metrics_path is not None:
        m_path = Path(metrics_path)
        if m_path.exists():
            metrics = json.loads(m_path.read_text(encoding="utf-8"))
            failures: list[str] = []
            if "roc_auc" in metrics and float(metrics["roc_auc"]) < min_roc_auc:
                failures.append(
                    f"roc_auc={float(metrics['roc_auc']):.4f} < mínimo {min_roc_auc:.4f}"
                )
            if "f1" in metrics and float(metrics["f1"]) < min_f1:
                failures.append(f"f1={float(metrics['f1']):.4f} < mínimo {min_f1:.4f}")
            if "recall" in metrics and float(metrics["recall"]) < min_recall:
                failures.append(f"recall={float(metrics['recall']):.4f} < mínimo {min_recall:.4f}")
            if "net_retained_revenue" in metrics and float(metrics["net_retained_revenue"]) <= 0.0:
                failures.append(
                    f"net_retained_revenue={float(metrics['net_retained_revenue']):.2f} <= 0.00"
                )
            if failures:
                return GateResult(
                    passed=False,
                    skipped=False,
                    message=f"FAIL: métricas abaixo do limiar ({'; '.join(failures)}).",
                    sample_probability=prob,
                )

    return GateResult(
        passed=True,
        skipped=False,
        message=f"OK: modelo em {path} aprovado no quality gate (prob={prob:.4f}).",
        sample_probability=prob,
    )


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada CLI do quality gate de ML."""
    parser = argparse.ArgumentParser(description="Valida modelo treinado e métricas de qualidade.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/churn_model.joblib"),
        help="Caminho do modelo serializado (.joblib).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("models/metrics.json"),
        help="Caminho opcional do JSON de métricas.",
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.80,
        help="Valor mínimo aceito para roc_auc.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.55,
        help="Valor mínimo aceito para f1.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.60,
        help="Valor mínimo aceito para recall.",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Falha caso o arquivo do modelo não exista.",
    )
    args = parser.parse_args(argv)

    result = evaluate_model_gate(
        args.model_path,
        require_model=args.require_model,
        metrics_path=args.metrics_path,
        min_roc_auc=args.min_roc_auc,
        min_f1=args.min_f1,
        min_recall=args.min_recall,
    )
    print(result.message)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
