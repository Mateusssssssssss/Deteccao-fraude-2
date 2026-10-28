"""EDA reproduzível exclusivamente no TRAIN cronológico."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.evaluation import target_correlation_importance_plot
from fraud_detection.segments import fraud_rates_by_bins


def generate(config: dict, output: str | Path = ROOT) -> pd.DataFrame:
    train, _, split_meta, _ = read_split(config)
    output = Path(output)
    figures, metric_dir = output / "reports/figures", output / "reports/metrics"
    figures.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    # Perfil dos dados originais: qualidade, escala, assimetria e valores extremos.
    summary = train[FEATURES].describe().T
    summary["missing"] = train[FEATURES].isna().sum()
    summary["cardinality"] = train[FEATURES].nunique()
    summary["dtype"] = train[FEATURES].dtypes.astype(str)
    summary["skew"] = train[FEATURES].skew()
    q1, q3 = train[FEATURES].quantile(0.25), train[FEATURES].quantile(0.75)
    summary["iqr_outliers"] = (
        (train[FEATURES] < q1 - 1.5 * (q3 - q1)) | (train[FEATURES] > q3 + 1.5 * (q3 - q1))
    ).sum()
    summary.to_csv(metric_dir / "eda_train_summary.csv")
    # A prevalência real é calculada antes de qualquer balanceamento sintético.
    classes = train[TARGET].value_counts().sort_index().rename("count").to_frame()
    classes["proportion"] = classes["count"] / len(train)
    classes.to_csv(metric_dir / "eda_train_classes.csv")
    (metric_dir / "eda_metadata.json").write_text(
        json.dumps(
            {
                "source": "TRAIN temporal sem reamostragem",
                "rows": len(train),
                "columns": len(train.columns),
                "duplicates_train": int(train.duplicated().sum()),
                "missing_total": int(train.isna().sum().sum()),
                **split_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Não fraude (0)", "Fraude (1)"], classes["count"])
    ax.bar_label(bars, padding=3)
    ax.set(yscale="log", ylabel="Transações (log)", title="Classes no TRAIN original")
    fig.tight_layout()
    fig.savefig(figures / "eda_classes.png", dpi=160)
    plt.close(fig)
    axes = train[FEATURES].hist(bins=40, figsize=(17, 18))
    fig = axes.ravel()[0].figure
    fig.suptitle("Distribuições — TRAIN; PCs sem significado semântico disponível")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures / "eda_distributions.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].boxplot(
        [np.log1p(train.loc[train.Class == c, "Amount"].dropna()) for c in [0, 1]],
        tick_labels=["Não fraude", "Fraude"],
        showfliers=False,
    )
    axes[0].set(title="Valor por classe — TRAIN", ylabel="log(1 + Amount); extremos ocultos")
    bucket = pd.cut(train.Time, bins=16)
    temporal = train.groupby(bucket, observed=True).Class.agg(["size", "mean"])
    temporal.to_csv(metric_dir / "eda_time_prevalence.csv")
    axes[1].plot(np.arange(len(temporal)), temporal["mean"], "o-")
    axes[1].set(
        title="Prevalência temporal — TRAIN", xlabel="Faixa temporal", ylabel="Taxa de fraude"
    )
    fig.tight_layout()
    fig.savefig(figures / "eda_amount_time.png", dpi=160)
    plt.close(fig)
    # A associação com Class é restrita ao TRAIN para não orientar decisões com TEST.
    corr = train[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(
        xticks=range(len(corr)),
        yticks=range(len(corr)),
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        title="Pearson — somente TRAIN",
    )
    ax.tick_params(axis="x", rotation=90)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "eda_correlations.png", dpi=140)
    plt.close(fig)
    association = pd.DataFrame(
        {
            "pearson": train[FEATURES].corrwith(train.Class),
            "spearman": train[FEATURES].corrwith(train.Class, method="spearman"),
        }
    )
    association.to_csv(metric_dir / "eda_associations.csv")
    # Taxa por quantis: captura formas não lineares sem selecionar/remover features.
    selected = ["Amount", "Time", *association.pearson.abs().nlargest(4).index]
    selected = list(dict.fromkeys(selected))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    nonlinear = []
    for ax, feature in zip(axes.ravel(), selected):
        groups = pd.qcut(train[feature], q=10, duplicates="drop")
        stats = train.groupby(groups, observed=True).Class.agg(["size", "mean"])
        stats["feature"] = feature
        stats["bin"] = stats.index.astype(str)
        nonlinear.append(stats.reset_index(drop=True))
        ax.plot(range(len(stats)), stats["mean"], "o-")
        ax.set(title=feature, xlabel="Quantil (ordem crescente)", ylabel="Taxa de fraude")
    fig.suptitle("Relações por segmentos — TRAIN; associações exploratórias, sem causalidade")
    fig.tight_layout()
    fig.savefig(figures / "eda_nonlinear.png", dpi=160)
    plt.close(fig)
    # Contagens e percentuais claros para todas as features, não só as seis desenhadas.
    fraud_rates_by_bins(train).to_csv(metric_dir / "eda_feature_bins.csv", index=False)
    # Interação exploratória explícita: valor x componente, usando mediana apenas TRAIN.
    top = association.pearson.abs().drop(["Amount", "Time"]).idxmax()
    interaction = (
        train.assign(
            amount_band=np.where(train.Amount <= train.Amount.median(), "baixo", "alto"),
            component_band=np.where(train[top] <= train[top].median(), "baixo", "alto"),
        )
        .groupby(["amount_band", "component_band"])
        .Class.agg(["size", "mean"])
    )
    interaction.to_csv(metric_dir / "eda_interaction.csv")
    # Diagnóstico pós-modelagem opcional; nunca treina modelo nem acessa TEST.
    importance_path = metric_dir / "association_importance.csv"
    if importance_path.exists():
        target_correlation_importance_plot(
            pd.read_csv(importance_path), figures / "eda_target_correlation_importance.png"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    generate(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
