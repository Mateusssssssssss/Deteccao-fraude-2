"""Duas famílias solicitadas e baseline ingênuo."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from fraud_detection.preprocess import build_pipeline


def candidates(config: dict) -> dict:
    # Dummy fornece o desempenho de referência sem aprender padrões das features.
    result = {"dummy": build_pipeline(DummyClassifier(strategy="prior"), config)}
    for sampling in config["sampling_strategies"]:
        # Copiar parâmetros evita alterar a configuração compartilhada entre receitas.
        params = dict(config["models"]["random_forest"])
        # Evita acumular pesos balanceados e reamostragem nesta comparação.
        if sampling != "none":
            params["class_weight"] = None
        rf = RandomForestClassifier(**params, random_state=config["seed"], n_jobs=config["n_jobs"])
        xgb = XGBClassifier(
            **config["models"]["xgboost"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config["seed"],
            n_jobs=config["n_jobs"],
        )
        result[f"random_forest__{sampling}"] = build_pipeline(rf, config, sampling)
        result[f"xgboost__{sampling}"] = build_pipeline(xgb, config, sampling)
    return result
