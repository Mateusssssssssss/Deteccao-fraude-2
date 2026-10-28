"""Métricas probabilísticas e gráficos com origem explícita."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics(y, probability, threshold: float = 0.5) -> dict:
    # Probabilidades medem ordenação/calibração; o threshold define as decisões binárias.
    predicted = (np.asarray(probability) >= threshold).astype(int)
    # Ordem fixa [0,1] garante a leitura TN, FP, FN, TP na matriz.
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "specificity": float(tn / (tn + fp)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def comparison_plot(table, path) -> None:
    table = table.sort_values("average_precision", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(table))
    for i, (metric, label) in enumerate(
        [("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]
    ):
        bars = ax.bar(
            x + (i - 1) * 0.25,
            table[metric],
            0.25,
            yerr=table[f"{metric}_std"],
            label=label,
            capsize=3,
        )
        labels = ax.bar_label(bars, fmt="%.2f", fontsize=8, label_type="center")
        for label, bar in zip(labels, bars):
            label.set_color("black" if bar.get_height() < 0.1 else "white")
    ax.set(
        xticks=x,
        xticklabels=table.model,
        ylim=(0, 1),
        ylabel="Score",
        title="CV temporal no TRAIN — média ± desvio-padrão; threshold 0,5",
    )
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnostics(y, probability, threshold: float, folder, prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(
        y,
        probability >= threshold,
        ax=axes[0],
        display_labels=["Não fraude", "Fraude"],
        colorbar=False,
    )
    PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[1], name=prefix)
    axes[1].axhline(np.mean(y), linestyle="--", color="gray", label="Prevalência")
    axes[1].legend()
    fig.suptitle(f"{prefix} — threshold {threshold}")
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}_diagnostics.png", dpi=160)
    plt.close(fig)


def target_correlation_importance_plot(table, path) -> None:
    """Compara Pearson com Class no TRAIN e permutação na validação do TRAIN."""
    # Alinha a mesma feature nos dois painéis, sem equiparar as escalas dos eixos.
    table = table.sort_values("permutation_ap", ascending=True, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(14, 13), sharey=True)
    colors = np.where(table.pearson >= 0, "#2878a5", "#c05252")
    bars = axes[0].barh(table.feature, table.pearson, color=colors)
    axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axes[0].axvline(0, color="gray", linewidth=0.8)
    axes[0].set(
        title="Correlação com Class — somente TRAIN",
        xlabel="Pearson (associação ponto-bisserial; com sinal)",
    )
    axes[0].margins(x=0.2)
    bars = axes[1].barh(
        table.feature,
        table.permutation_ap,
        xerr=table.permutation_std,
        color="#36896d",
        capsize=2,
    )
    # Rótulos na ponta da barra de erro para evitar sobreposição.
    for bar, value, error in zip(bars, table.permutation_ap, table.permutation_std):
        axes[1].annotate(
            f"{value:.4f}",
            (value + error, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axes[1].axvline(0, color="gray", linewidth=0.8)
    axes[1].set(
        title="Importância — XGBoost, validação dentro do TRAIN",
        xlabel="Queda de AP por permutação (média ± desvio; 3 repetições)",
    )
    axes[1].margins(x=0.25)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Correlação com a target × importância das features — escalas distintas")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.text(
        0.5,
        0.01,
        "Importância condicionada à receita já selecionada. Associação e importância não implicam causalidade.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
