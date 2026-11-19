"""Pipeline de pré-processamento (scikit-learn)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Cria o ColumnTransformer usado por todos os modelos.

    Numéricas: imputação pela mediana + padronização.
    Categóricas: imputação pela moda + one-hot (ignora categorias novas na inferência).
    """
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )
