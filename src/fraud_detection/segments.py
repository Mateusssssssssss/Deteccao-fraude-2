"""Taxas de fraude observada e erros do modelo por faixas numéricas."""

import numpy as np
import pandas as pd

from fraud_detection.data import FEATURES, TARGET


def fraud_rates_by_bins(train: pd.DataFrame, features=FEATURES, q: int = 10) -> pd.DataFrame:
    """Quantis no TRAIN original; inclui ausentes e constantes explicitamente."""
    rows = []
    # Denominador global da parcela de fraudes; a taxa da faixa usa outro denominador.
    total_frauds = int(train[TARGET].sum())
    for feature in features:
        values = train[feature]
        # Ausências/constantes não podem simplesmente desaparecer da contagem por faixa.
        if values.dropna().nunique() < 2:
            groups = pd.Series(np.where(values.isna(), "Ausente", "Constante"), index=train.index)
        else:
            groups = (
                pd.qcut(values, q=q, duplicates="drop")
                .cat.add_categories("Ausente")
                .fillna("Ausente")
            )
        for category, part in train.groupby(groups, sort=True, observed=True):
            # Fraude observada descreve rótulos; não é uma taxa de erro do classificador.
            n, frauds = len(part), int(part[TARGET].sum())
            rows.append(
                {
                    "feature": feature,
                    "bin": str(category),
                    "size": n,
                    "fraud_count": frauds,
                    "legitimate_count": n - frauds,
                    "mean": frauds / n,
                    "fraud_rate_pct": 100 * frauds / n,
                    "share_of_train_frauds_pct": 100 * frauds / total_frauds
                    if total_frauds
                    else np.nan,
                    "source": "TRAIN original; faixas por quantis; sem reamostragem",
                }
            )
    return pd.DataFrame(rows)


def error_rates_by_bins(fit, validation, probability, threshold, fold, features=FEATURES):
    """Aprende fronteiras no treino do fold e mede erros só na validação externa."""
    rows = []
    y = validation[TARGET].to_numpy()
    predicted = np.asarray(probability) >= threshold
    for feature in features:
        # Fronteiras aprendidas no treino do fold: a validação não define os próprios bins.
        nonmissing = fit[feature].dropna()
        edges = np.unique(nonmissing.quantile(np.linspace(0, 1, 11)).to_numpy())
        if len(edges) < 2:
            groups = pd.Series(
                np.where(validation[feature].isna(), "Ausente", "Faixa única"),
                index=validation.index,
            )
        else:
            # Extremos abertos cobrem valores futuros fora do alcance observado no treino.
            edges[0], edges[-1] = -np.inf, np.inf
            groups = pd.cut(validation[feature], bins=edges, include_lowest=True)
            groups = groups.astype("string").fillna("Ausente")
        for category in groups.unique():
            mask = (groups == category).to_numpy(dtype=bool)
            actual, pred = y[mask], predicted[mask]
            # Cada faixa recebe sua própria matriz de confusão e seus denominadores.
            tp = int(np.sum((actual == 1) & pred))
            fp = int(np.sum((actual == 0) & pred))
            fn = int(np.sum((actual == 1) & ~pred))
            tn = int(np.sum((actual == 0) & ~pred))

            def percent(a, b):
                return 100 * a / b if b else np.nan

            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "bin": str(category),
                    "count": int(mask.sum()),
                    "fraud_count": tp + fn,
                    "legitimate_count": tn + fp,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "fraud_rate_pct": percent(tp + fn, mask.sum()),
                    "recall_pct": percent(tp, tp + fn),
                    "precision_pct": percent(tp, tp + fp),
                    "false_positive_rate_pct": percent(fp, fp + tn),
                    "missed_fraud_rate_pct": percent(fn, tp + fn),
                    "false_alert_share_pct": percent(fp, tp + fp),
                    "alert_rate_pct": percent(tp + fp, mask.sum()),
                    "threshold": threshold,
                    "source": "Validação temporal externa TRAIN; limites aprendidos no treino do fold",
                }
            )
    return pd.DataFrame(rows)
