"""Seleção supervisionada recalculada somente no treino recebido."""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fraud_detection.data import FEATURES, TARGET, temporal_split
from fraud_detection.model import candidates


def removal_mask(correlation, importance, correlation_cutoff=0.09, importance_cutoff=0.1):
    """Retira apenas se AMBOS os critérios estritos forem satisfeitos."""
    return (np.abs(correlation) < correlation_cutoff) & (importance < importance_cutoff)


def audit_features(train: pd.DataFrame, config: dict, settings: dict) -> pd.DataFrame:
    """Pearson no treino; permutação numa janela interna, usando referência XGBoost."""
    # A importância usa uma janela interna; a validação externa nunca participa.
    fit, validation, _metadata = temporal_split(
        train, settings["selection_validation_time_fraction"]
    )
    # Referência fixa XGBoost permite aplicar a mesma regra às duas famílias comparadas.
    reference = candidates(config)["xgboost__none"].fit(fit[FEATURES], fit[TARGET])
    importance = permutation_importance(
        reference,
        validation[FEATURES],
        validation[TARGET],
        scoring="average_precision",
        n_repeats=settings["importance_repeats"],
        random_state=config["seed"],
        n_jobs=1,
    )
    # O módulo será usado na regra: associação negativa forte também é preservada.
    correlation = train[FEATURES].corrwith(train[TARGET]).fillna(0)
    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "pearson": correlation.to_numpy(),
            "abs_pearson": correlation.abs().to_numpy(),
            "permutation_ap": importance.importances_mean,
            "permutation_std": importance.importances_std,
        }
    )
    # AND: baixa correlação sozinha ou baixa importância sozinha não basta para excluir.
    table["remove"] = removal_mask(
        table.pearson,
        table.permutation_ap,
        settings["correlation_cutoff"],
        settings["importance_cutoff"],
    )
    table["keep"] = ~table["remove"]
    table["reference_fit_time_max"] = float(fit.Time.max())
    table["reference_validation_time_min"] = float(validation.Time.min())
    table["reference_validation_time_max"] = float(validation.Time.max())
    table["selection_input_time_max"] = float(train.Time.max())
    table["reference"] = "XGBoost sem reamostragem; permutação em validação interna"
    if not table.keep.any():
        raise ValueError("A regra removeu todas as features; reveja os limites.")
    return table
