"""CV expansiva nos tempos únicos, sem atravessar empates."""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from fraud_detection.data import TARGET

SCORING = {
    key: key
    for key in ["average_precision", "f1", "recall", "precision", "roc_auc", "balanced_accuracy"]
}


def temporal_folds(train, config: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    # Divide instantes únicos para que empates de Time não cruzem fronteiras de folds.
    times = np.sort(train.Time.unique())
    folds = []
    for fit_times, val_times in TimeSeriesSplit(n_splits=config["cv_splits"]).split(times):
        # Converte os instantes do fold em posições de linhas do TRAIN.
        fit_idx = np.flatnonzero(train.Time.isin(times[fit_times]))
        val_idx = np.flatnonzero(train.Time.isin(times[val_times]))
        # Falhar explicitamente é melhor que reportar AP/ROC-AUC de um fold inválido.
        for idx in [fit_idx, val_idx]:
            if train.iloc[idx][TARGET].nunique() != 2:
                raise ValueError("Fold temporal com uma só classe; reveja número de folds/janela.")
        if "under_smote" in config["sampling_strategies"]:
            counts = train.iloc[fit_idx][TARGET].value_counts()
            if (
                counts[1] <= config["smote_neighbors"]
                or counts[1] / counts[0] >= config["under_ratio"]
            ):
                raise ValueError("Fold incompatível com ratios/k_neighbors de under+SMOTE.")
        folds.append((fit_idx, val_idx))
    return folds


def evaluate_cv(pipeline, X, y, folds) -> dict:
    # Mesmos folds e scorers para todos os candidatos; erros de fit não são escondidos.
    return cross_validate(
        pipeline,
        X,
        y,
        cv=folds,
        scoring=SCORING,
        return_estimator=True,
        return_train_score=True,
        error_score="raise",
        n_jobs=1,
    )
