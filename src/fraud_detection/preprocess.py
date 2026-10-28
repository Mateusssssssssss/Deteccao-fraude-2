"""Amostragem só durante fit, dentro de cada fold."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data import FEATURES


def build_pipeline(
    estimator,
    config: dict,
    sampling: str = "none",
    *,
    standardize: bool = False,
    features: list[str] | None = None,
) -> Pipeline:
    # A mediana e, quando usado, o scaler aprendem só com os dados recebidos no fit.
    numeric = SkPipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "scale",
                StandardScaler()
                if standardize or sampling in {"under_smote", "smote"}
                else "passthrough",
            ),
        ]
    )
    # ColumnTransformer mantém apenas as features solicitadas, na ordem explícita.
    steps = [
        (
            "preprocess",
            ColumnTransformer([("numeric", numeric, FEATURES if features is None else features)]),
        )
    ]
    # Receita histórica: reduz legítimas e depois cria exemplos sintéticos de fraude.
    if sampling == "under_smote":
        steps += [
            (
                "under",
                RandomUnderSampler(
                    sampling_strategy=config["under_ratio"], random_state=config["seed"]
                ),
            ),
            (
                "smote",
                SMOTE(
                    sampling_strategy=config["smote_ratio"],
                    k_neighbors=config["smote_neighbors"],
                    random_state=config["seed"],
                ),
            ),
        ]
    # Nova ablação: preserva todas as legítimas e aumenta só a minoria.
    elif sampling == "smote":
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=config["smote_ratio"],
                    k_neighbors=config["smote_neighbors"],
                    random_state=config["seed"],
                ),
            )
        )
    elif sampling != "none":
        raise ValueError(f"Amostragem desconhecida: {sampling}")
    # imblearn.Pipeline executa samplers só no fit; predict não reamostra entradas.
    return Pipeline(steps + [("model", estimator)])
