"""Treino e comparação de modelos (Etapa 2).

Esqueleto: define os candidatos (linear, árvore/ensemble e MLP) e o fluxo
de validação cruzada. A lógica completa será implementada após a EDA.

Uso:
    python -m churn.train
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn.config import SEED
from churn.features import build_preprocessor


def get_candidate_models() -> dict[str, object]:
    """Modelos exigidos pelo enunciado, todos com seed fixa."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=SEED
        ),
    }


def build_pipeline(model: object) -> Pipeline:
    """Encapsula pré-processamento + estimador em um único artefato serializável."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def main() -> None:
    # TODO (Etapa 2):
    #   1. load_raw() + clean()
    #   2. train_test_split estratificado (TEST_SIZE, SEED)
    #   3. StratifiedKFold(CV_FOLDS) + cross_validate para cada candidato
    #   4. salvar tabela comparativa em reports/
    #   5. re-treinar o campeão e salvar com joblib.dump em MODEL_PATH
    raise NotImplementedError("Treino ainda não implementado.")


if __name__ == "__main__":
    main()
