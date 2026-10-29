"""Apresentação legível de tabelas; preserva os valores numéricos dos artefatos."""

import pandas as pd

LABELS = {
    "feature": "Feature",
    "bin": "Faixa de valores",
    "size": "Transações",
    "count": "Quantidade",
    "fraud_count": "Fraudes confirmadas",
    "legitimate_count": "Transações legítimas",
    "mean": "Taxa de fraude (proporção)",
    "fraud_rate_pct": "Fraude na faixa (%)",
    "share_of_train_frauds_pct": "Parcela das fraudes do TRAIN (%)",
    "model": "Receita",
    "source": "Origem",
    "threshold": "Limiar de alerta",
    "fold": "Fold temporal",
    "policy": "Política de threshold",
    "f1": "F1 (%)",
    "f2": "F2 (%)",
    "f3": "F3 (%)",
    "recall": "Recall (%)",
    "precision": "Precision (%)",
    "specificity": "Especificidade (%)",
    "balanced_accuracy": "Balanced Accuracy (%)",
    "average_precision": "AP (score)",
    "roc_auc": "ROC-AUC (score)",
    "log_loss": "Log Loss",
    "brier": "Brier score",
    "tn": "TN: legítimas liberadas",
    "fp": "FP: falsos alertas",
    "fn": "FN: fraudes não detectadas",
    "tp": "TP: fraudes detectadas",
    "alert_rate": "Taxa de alertas (%)",
    "recall_pct": "Recall da faixa (%)",
    "precision_pct": "Precision da faixa (%)",
    "false_positive_rate_pct": "Falsos positivos entre legítimas (%)",
    "missed_fraud_rate_pct": "Fraudes não detectadas (%)",
    "false_alert_share_pct": "Alertas que eram falsos (%)",
    "alert_rate_pct": "Taxa de alertas da faixa (%)",
    "pearson": "Pearson com Class",
    "spearman": "Spearman com Class",
    "abs_pearson": "|Pearson|",
    "permutation_ap": "Queda de AP por permutação",
    "permutation_std": "Desvio da permutação",
    "tree_importance": "Importância interna da árvore",
    "keep": "Manter?",
    "remove": "Remover?",
    "n_features": "Features utilizadas",
    "proportion": "Proporção da classe (%)",
    "missing": "Ausências",
    "cardinality": "Valores distintos",
    "skew": "Assimetria",
    "iqr_outliers": "Valores fora de 1,5 IQR",
    "dtype": "Tipo de dado",
    "strategy": "Estratégia de balanceamento",
    "family": "Família do modelo",
    "legitimate_before": "Legítimas antes do sampler",
    "fraud_before": "Fraudes antes do sampler",
    "legitimate_after": "Legítimas após sampler",
    "fraud_after": "Fraudes após sampler",
    "legitimate_removed": "Legítimas removidas",
    "synthetic_frauds": "Fraudes sintéticas criadas",
    "ratio_after": "Razão fraudes/legítimas após sampler",
    "selected_last_fold": "Usada no último fold?",
    "fit_seconds": "Tempo de fit (segundos)",
    "fit_rows": "Transações de treino",
    "validation_rows": "Transações de validação",
    "fit_frauds": "Fraudes no treino",
    "validation_frauds": "Fraudes na validação",
    "fit_time_max": "Último Time do treino",
    "validation_time_min": "Primeiro Time da validação",
    "validation_time_max": "Último Time da validação",
}
FRACTIONS = {
    "f1",
    "f2",
    "f3",
    "recall",
    "precision",
    "specificity",
    "balanced_accuracy",
    "alert_rate",
    "proportion",
    "disagreement",
    "double_fault",
    "both_correct",
    "only_a_wrong",
    "only_b_wrong",
}


def readable_table(frame: pd.DataFrame):
    """Exibe percentuais com denominadores descritos no guia, sem alterar o DataFrame."""
    # Elimina só duplicatas de exibição: a versão proporcional já será mostrada como %.
    # Por exemplo, recall e recall_pct representam o mesmo valor em unidades diferentes.
    frame = frame.drop(
        columns=[c for c in frame.columns if c.endswith("_pct") and c[:-4] in frame.columns]
    )
    labels, formats = {}, {}
    for column in frame.columns:
        base = column.removesuffix("_std").removeprefix("train_")
        if column == "mean":
            if "size" in frame.columns:
                labels[column] = "Taxa de fraude (%)"
                formats[column] = lambda value: f"{100 * value:.4f}%".replace(".", ",")
            else:
                labels[column] = "Média"
        elif column.endswith("_pct"):
            labels[column] = LABELS.get(column, column.replace("_pct", " (%)"))
            digits = 4 if "fraud_rate" in column else 2
            formats[column] = lambda value, d=digits: f"{value:.{d}f}%".replace(".", ",")
        elif base in FRACTIONS:
            labels[column] = LABELS.get(column, column)
            if column.endswith("_std"):
                labels[column] = (
                    LABELS.get(column[:-4], column[:-4]).replace("(%)", "") + "desvio (p.p.)"
                )
            formats[column] = lambda value: f"{100 * value:.2f}%".replace(".", ",")
            if column.endswith("_std"):
                formats[column] = lambda value: f"{100 * value:.2f} p.p.".replace(".", ",")
        else:
            labels[column] = LABELS.get(column, column)
    # Os CSVs continuam numéricos; apenas o objeto de exibição recebe rótulos e formatação.
    shown = frame.rename(columns=labels)
    return shown.style.format(
        {labels[k]: v for k, v in formats.items()},
        precision=4,
        na_rep="Não definido",
        thousands=".",
        decimal=",",
    )
