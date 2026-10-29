"""SMOTE isolado x undersampling isolado, comparados em CV temporal somente TRAIN."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.evaluation import metrics
from fraud_detection.features import audit_features
from fraud_detection.model import candidates
from fraud_detection.preprocess import build_pipeline
from fraud_detection.segments import error_rates_by_bins
from fraud_detection.validation import temporal_folds

LOGGER = logging.getLogger(__name__)


def make_pipeline(family: str, strategy: str, selected: list[str], config: dict, settings: dict):
    """Somente o sampler muda entre receitas; RF usa pesos uniformes em todas."""
    model = clone(candidates(config)[f"{family}__none"].named_steps["model"])
    if family == "random_forest":
        model.set_params(class_weight=None)
    experiment = dict(
        config, smote_ratio=settings["sampling_ratio"], under_ratio=settings["sampling_ratio"]
    )
    return build_pipeline(model, experiment, strategy, standardize=True, features=selected)


def sampling_counts(pipeline, y) -> dict:
    """Lê o sampler ajustado sem executar fit_resample novamente."""
    counts = pd.Series(y).value_counts()
    before = {0: int(counts[0]), 1: int(counts[1])}
    after = dict(before)
    if "smote" in pipeline.named_steps:
        sampler = pipeline.named_steps["smote"]
        for label, additional in sampler.sampling_strategy_.items():
            after[int(label)] += int(additional)
    elif "under" in pipeline.named_steps:
        selected_y = np.asarray(y)[pipeline.named_steps["under"].sample_indices_]
        after = {label: int(np.sum(selected_y == label)) for label in [0, 1]}
    return {
        "legitimate_before": before[0],
        "fraud_before": before[1],
        "legitimate_after": after[0],
        "fraud_after": after[1],
        "legitimate_removed": before[0] - after[0],
        "synthetic_frauds": after[1] - before[1],
        "ratio_after": after[1] / after[0],
    }


def sampling_metrics(y, probability, threshold: float) -> dict:
    """Métricas desta rodada: F1 é principal; não inclui F2."""
    score = metrics(y, probability, threshold)
    score["alerts"] = score["tp"] + score["fp"]
    score["alert_rate"] = score["alerts"] / len(y)
    return score


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio dos mesmos folds; F1 é a métrica principal."""
    rows = []
    for (strategy, model), part in folds.groupby(["strategy", "model"], sort=False):
        row = {
            "strategy": strategy,
            "model": model,
            "source": "CV temporal TRAIN; média dos mesmos folds",
            "threshold": 0.5,
        }
        for metric in [
            "f1",
            "recall",
            "precision",
            "average_precision",
            "roc_auc",
            "balanced_accuracy",
            "specificity",
            "log_loss",
            "brier",
            "fp",
            "fn",
            "tp",
            "tn",
            "alert_rate",
            "fit_seconds",
        ]:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1", "average_precision"], ascending=False)


def boosting_diagnostic(pipeline, fit, validation, settings, strategy):
    """Curvas por árvores no último fold; quantidades prefixadas, sem early stopping."""
    model = pipeline.named_steps["model"]
    n_estimators = model.n_estimators
    points = sorted({n_estimators, *settings["iteration_points"]})
    rows = []
    # O sampler não é aplicado no diagnóstico: métricas usam distribuições originais.
    transformed = {
        "train": pipeline.named_steps["preprocess"].transform(fit[FEATURES]),
        "validation": pipeline.named_steps["preprocess"].transform(validation[FEATURES]),
    }
    for count in points:
        if count > n_estimators:
            continue
        row = {"strategy": strategy, "trees": count, "source": "Último fold temporal TRAIN"}
        for name, frame in [("train", fit), ("validation", validation)]:
            probability = model.predict_proba(transformed[name], iteration_range=(0, count))[:, 1]
            score = sampling_metrics(frame[TARGET], probability, settings["threshold"])
            row[f"{name}_average_precision"] = score["average_precision"]
            row[f"{name}_log_loss"] = score["log_loss"]
        rows.append(row)
    return pd.DataFrame(rows)


def run(config: dict, feature_settings: dict, settings: dict, output: str | Path = ROOT):
    """Ajusta receitas apenas nos treinos dos folds; não pontua o TEST histórico."""
    import time

    output = Path(output)
    md, ed, fd = [output / p for p in ["reports/metrics", "reports/experiments", "reports/figures"]]
    for folder in [md, ed, fd]:
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ed / "sampling_comparison_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Experimento já calculado; use --output para reprodução separada.")
    if settings["threshold"] != 0.5 or settings["primary_metric"] != "f1":
        raise ValueError("O protocolo compara F1 no threshold comum 0,5.")
    if set(settings["strategies"]) != {"smote", "under", "none"}:
        raise ValueError("Compare smote, under e controle none, sem combinação entre samplers.")
    if not 0 < settings["sampling_ratio"] < 1:
        raise ValueError("A razão de amostragem deve estar entre 0 e 1.")
    train, _, split_metadata, source = read_split(config)
    folds = temporal_folds(train, dict(config, sampling_strategies=["none"]))
    feature_audits, selections, fold_manifest = [], {}, []
    # Mesma seleção de features para todas as receitas, reaprendida no treino de cada fold.
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
        LOGGER.info("Preparando features do fold %s/%s, sem usar sua validação", fold, len(folds))
        audit = audit_features(fit, config, feature_settings)
        selected = audit.loc[audit.keep, "feature"].tolist()
        selections[fold] = selected
        feature_audits.append(audit.assign(fold=fold))
        fold_manifest.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(validation),
                "fit_frauds": int(fit[TARGET].sum()),
                "validation_frauds": int(validation[TARGET].sum()),
                "fit_time_max": float(fit.Time.max()),
                "validation_time_min": float(validation.Time.min()),
                "validation_time_max": float(validation.Time.max()),
                "n_features": len(selected),
            }
        )
    pd.concat(feature_audits).to_csv(md / "sampling_feature_selection.csv", index=False)
    pd.DataFrame(fold_manifest).to_csv(ed / "sampling_fold_manifest.csv", index=False)
    rows, sampler_rows, predictions, last_models = [], [], {}, {}
    # A ordem atende ao pedido: primeiro só SMOTE, depois só under; controle ao final.
    for strategy in settings["strategies"]:
        for family in settings["families"]:
            model_name = f"{family}__{strategy}"
            oof = pd.Series(np.nan, index=train.index, dtype=float)
            for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
                fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
                LOGGER.info("%s — fold %s/%s", model_name, fold, len(folds))
                pipeline = make_pipeline(family, strategy, selections[fold], config, settings)
                start = time.perf_counter()
                pipeline.fit(fit[FEATURES], fit[TARGET])
                elapsed = time.perf_counter() - start
                probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
                train_probability = pipeline.predict_proba(fit[FEATURES])[:, 1]
                score = sampling_metrics(validation[TARGET], probability, 0.5)
                train_score = sampling_metrics(fit[TARGET], train_probability, 0.5)
                rows.append(
                    {
                        "strategy": strategy,
                        "family": family,
                        "model": model_name,
                        "fold": fold,
                        "source": "Validação temporal TRAIN; distribuição original",
                        "n_features": len(selections[fold]),
                        "fit_seconds": elapsed,
                        **score,
                        **{
                            f"train_{key}": train_score[key]
                            for key in [
                                "average_precision",
                                "f1",
                                "recall",
                                "precision",
                                "log_loss",
                            ]
                        },
                    }
                )
                sampler_rows.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "fold": fold,
                        **sampling_counts(pipeline, fit[TARGET]),
                    }
                )
                oof.iloc[val_idx] = probability
                # Somente o último estimador de cada receita fica em memória para diagnósticos.
                last_models[(strategy, family)] = pipeline
                pd.DataFrame(rows).to_csv(md / "sampling_folds.csv", index=False)
            predictions[model_name] = oof

    # Baseline ingênuo com prevalência aprendida só no treino de cada fold.
    dummy_oof = pd.Series(np.nan, index=train.index, dtype=float)
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        dummy = DummyClassifier(strategy="prior").fit(
            train.iloc[fit_idx][FEATURES], train.iloc[fit_idx][TARGET]
        )
        probability = dummy.predict_proba(train.iloc[val_idx][FEATURES])[:, 1]
        dummy_oof.iloc[val_idx] = probability
        rows.append(
            {
                "strategy": "dummy",
                "family": "dummy",
                "model": "dummy",
                "fold": fold,
                "source": "Validação temporal TRAIN; baseline ingênuo",
                "n_features": 0,
                "fit_seconds": 0.0,
                **sampling_metrics(train.iloc[val_idx][TARGET], probability, 0.5),
            }
        )
    predictions["dummy"] = dummy_oof
    scores = pd.DataFrame(rows)
    summary = summarize(scores)
    scores.to_csv(md / "sampling_folds.csv", index=False)
    summary.to_csv(md / "sampling_comparison.csv", index=False)
    pd.DataFrame(sampler_rows).to_csv(md / "sampling_counts.csv", index=False)
    covered = dummy_oof.dropna().index
    oof_scores = [
        {
            "model": name,
            "source": "OOF temporal TRAIN agregado; prefixo inicial sem previsão",
            "rows": len(covered),
            **sampling_metrics(train.loc[covered, TARGET], p.loc[covered].to_numpy(), 0.5),
        }
        for name, p in predictions.items()
    ]
    pd.DataFrame(oof_scores).to_csv(md / "sampling_oof_metrics.csv", index=False)

    # Figuras e métricas são pós-seleção e descritivas; nenhum resultado gera retuning.
    last_fit_idx, last_val_idx = folds[-1]
    last_fit, last_validation = train.iloc[last_fit_idx], train.iloc[last_val_idx]
    winners = {}
    for strategy in ["smote", "under"]:
        LOGGER.info("Gerando diagnósticos: %s", strategy)
        mode_summary = summary.loc[summary.strategy.isin([strategy, "none", "dummy"])]
        mode_summary.to_csv(md / f"{strategy}_model_comparison.csv", index=False)
        winners[strategy] = summary.loc[summary.strategy.eq(strategy)].iloc[0].model
        plot_bars(mode_summary, fd / f"{strategy}_comparison_f1_recall_precision.png", strategy)
        plot_generalization(
            scores.loc[scores.strategy.eq(strategy)], fd / f"{strategy}_train_validation.png"
        )
        iterations = boosting_diagnostic(
            last_models[(strategy, "xgboost")], last_fit, last_validation, settings, strategy
        )
        iterations.to_csv(md / f"{strategy}_boosting_iterations.csv", index=False)
        plot_boosting(iterations, fd / f"{strategy}_boosting_iterations.png")
        associations, sensitivities, segments = [], [], []
        for family in settings["families"]:
            name = f"{family}__{strategy}"
            pipeline = last_models[(strategy, family)]
            probability = predictions[name].loc[covered].to_numpy()
            plot_oof(
                train.loc[covered, TARGET], probability, fd / f"{strategy}_{family}_oof.png", name
            )
            LOGGER.info("%s: permutação na última validação TRAIN", name)
            importance = permutation_importance(
                pipeline,
                last_validation[FEATURES],
                last_validation[TARGET],
                scoring="average_precision",
                n_repeats=settings["permutation_repeats"],
                random_state=config["seed"],
                n_jobs=1,
            )
            numeric_names = pipeline.named_steps["preprocess"].transformers_[0][2]
            tree = dict(zip(numeric_names, pipeline.named_steps["model"].feature_importances_))
            association = pd.DataFrame(
                {
                    "model": name,
                    "feature": FEATURES,
                    "pearson": [train[c].corr(train[TARGET]) for c in FEATURES],
                    "spearman": [train[c].corr(train[TARGET], method="spearman") for c in FEATURES],
                    "permutation_ap": importance.importances_mean,
                    "permutation_std": importance.importances_std,
                    "tree_importance": [tree.get(c, 0.0) for c in FEATURES],
                    "selected_last_fold": [c in numeric_names for c in FEATURES],
                    "source": "Pearson/Spearman TRAIN; importância última validação TRAIN",
                }
            )
            associations.append(association)
            plot_association(
                association, fd / f"{strategy}_{family}_association_importance.png", name
            )
            last_p = predictions[name].loc[last_validation.index].to_numpy()
            for threshold in settings["threshold_diagnostic_grid"]:
                sensitivities.append(
                    {
                        "model": name,
                        "source": "Última validação TRAIN; diagnóstico, sem seleção",
                        **sampling_metrics(last_validation[TARGET], last_p, threshold),
                    }
                )
            for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
                part = error_rates_by_bins(
                    train.iloc[fit_idx],
                    train.iloc[val_idx],
                    predictions[name].iloc[val_idx].to_numpy(),
                    0.5,
                    fold,
                )
                segments.append(part.assign(model=name))
        pd.concat(associations).to_csv(md / f"{strategy}_association_importance.csv", index=False)
        sensitivity = pd.DataFrame(sensitivities)
        sensitivity.to_csv(md / f"{strategy}_threshold_sensitivity.csv", index=False)
        plot_sensitivity(sensitivity, fd / f"{strategy}_threshold_sensitivity.png")
        pd.concat(segments).to_csv(md / f"{strategy}_category_errors.csv", index=False)
        errors = {
            family: (predictions[f"{family}__{strategy}"].loc[covered].to_numpy() >= 0.5)
            != train.loc[covered, TARGET].to_numpy()
            for family in settings["families"]
        }
        a, b = [errors[family] for family in settings["families"]]
        pd.DataFrame(
            [
                {
                    "model_a": settings["families"][0],
                    "model_b": settings["families"][1],
                    "source": "OOF temporal TRAIN alinhado",
                    "disagreement": float(np.mean(a != b)),
                    "double_fault": float(np.mean(a & b)),
                    "both_correct": float(np.mean(~a & ~b)),
                    "only_a_wrong": float(np.mean(a & ~b)),
                    "only_b_wrong": float(np.mean(~a & b)),
                }
            ]
        ).to_csv(md / f"{strategy}_error_diversity.csv", index=False)

    manifest = {
        "config": config,
        "feature_settings": feature_settings,
        "settings": settings,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "train_rows": len(train),
        "train_frauds": int(train[TARGET].sum()),
        **split_metadata,
        "oof_rows": len(covered),
        "oof_coverage": len(covered) / len(train),
        "positive_class": 1,
        "threshold_policy": "Fixo 0.5 em todas as receitas",
        "selection_metric": "F1 média na CV",
        "winners": winners,
        "sampling": {
            "smote": "Somente SMOTE",
            "under": "Somente RandomUnderSampler",
            "none": "Sem sampler, pesos uniformes",
            "smotec_notebook_alias": "SMOTE numérico",
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]
        },
        "test_evaluated": False,
        "primary_model_replaced": False,
        "limitations": [
            "Mesmos folds TRAIN reutilizados de rodadas anteriores; comparação exploratória.",
            "Importância é diagnóstico condicionado à seleção; não confirma causalidade.",
            "Threshold 0.5 fixado para isolar receitas; sensibilidade não seleciona outro limiar.",
            "Mudanças de prevalência de treino podem afetar probabilidades não calibradas.",
            "SMOTENC não se aplica às entradas exclusivamente numéricas deste dataset.",
            "Não há novo refit/promoção de modelo nem avaliação do TEST histórico nesta rodada.",
        ],
    }
    write_report(summary, manifest, output)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Experimento concluído: %s", winners)
    return manifest


def plot_bars(table, path, strategy) -> None:
    """Dois modelos, três barras por grupo: F1, Recall e Precision."""
    # Controles permanecem nas tabelas; o gráfico principal mostra só o sampler escolhido.
    table = table.loc[table.strategy.eq(strategy)].sort_values("f1", ascending=False)
    names = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
    labels = [names[name.split("__", 1)[0]] for name in table.model]
    title = "SMOTE isolado" if strategy == "smote" else "Undersampling isolado"
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(table))
    for i, (metric, label, color) in enumerate(
        [
            ("f1", "F1-score", "#3578b8"),
            ("recall", "Recall", "#e49032"),
            ("precision", "Precision", "#37936b"),
        ]
    ):
        values = 100 * table[metric].to_numpy()
        bars = ax.bar(
            x + (i - 1) * 0.24,
            values,
            0.24,
            yerr=100 * table[f"{metric}_std"].to_numpy(),
            capsize=4,
            label=label,
            color=color,
        )
        ax.bar_label(
            bars,
            labels=[f"{v:.2f}%".replace(".", ",") for v in values],
            label_type="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylim=(0, 100),
        ylabel="Score (%)",
        title=f"{title} — F1, Recall e Precision por modelo\nCV temporal TRAIN; média ± desvio; threshold 0,50",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncols=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_generalization(table, path) -> None:
    """F1 de treino e validação por janela; tamanho e período variam juntos."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, part in table.groupby("model"):
        part = part.sort_values("fold")
        ax.plot(part.fold, 100 * part.train_f1, "--o", label=f"{model} — treino")
        ax.plot(part.fold, 100 * part.f1, "-o", label=f"{model} — validação")
    ax.set(
        xlabel="Fold temporal expansivo",
        ylabel="F1 (%)",
        ylim=(0, 100),
        title="F1: treino versus validação — não é learning curve controlada",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boosting(table, path) -> None:
    """AP e Log Loss por quantidade prefixada de árvores do XGBoost."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["average_precision", "log_loss"]):
        for label in ["train", "validation"]:
            ax.plot(table.trees, table[f"{label}_{metric}"], "o-", label=label)
        ax.set(xlabel="Árvores", ylabel=metric, title=f"Último fold TRAIN — {metric}")
        ax.legend()
    fig.suptitle("XGBoost: curvas diagnósticas; sem retuning de n_estimators")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_oof(y, probability, path, name) -> None:
    """Matriz e curvas usam as mesmas previsões OOF, com cobertura parcial do TRAIN."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ConfusionMatrixDisplay.from_predictions(
        y,
        probability >= 0.5,
        ax=axes[0],
        colorbar=False,
        display_labels=["Legítima", "Fraude"],
    )
    axes[0].set(title="Matriz — limiar 0,5", xlabel="Previsto", ylabel="Real")
    PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[1], name="OOF TRAIN")
    axes[1].axhline(np.mean(y), linestyle="--", color="gray", label="Prevalência")
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall / AP")
    RocCurveDisplay.from_predictions(y, probability, ax=axes[2], name="OOF TRAIN")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].set(xlabel="Taxa de falsos positivos", ylabel="Recall", title="ROC")
    fig.suptitle(f"{name} — OOF temporal TRAIN (prefixo inicial sem previsão)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_association(table, path, name) -> None:
    """Correlação e permutação alinhadas, sem equiparar escalas."""
    table = table.sort_values("permutation_ap", kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(14, 13), sharey=True)
    bars = axes[0].barh(
        table.feature, table.pearson, color=np.where(table.pearson < 0, "#c05252", "#2878a5")
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
    axes[0].set(title="Pearson com Class — TRAIN", xlabel="Associação linear com sinal")
    axes[0].margins(x=0.2)
    axes[1].barh(table.feature, table.permutation_ap, xerr=table.permutation_std, color="#36896d")
    for i, row in enumerate(table.itertuples()):
        axes[1].annotate(
            f"{row.permutation_ap:.4f}",
            (row.permutation_ap + row.permutation_std, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
        )
    axes[1].set(
        title="Permutação — última validação TRAIN", xlabel="Queda de AP ± desvio (escala distinta)"
    )
    axes[1].margins(x=0.3)
    for ax in axes:
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{name} — diagnóstico condicionado à seleção; sem causalidade")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(table, path) -> None:
    """Barras e rótulos de F1/Recall/Precision por limiar; diagnóstico sem tuning."""
    # Um painel por linha dá espaço aos valores de cada uma das três barras.
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    names = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
    for ax, (name, part) in zip(axes, table.groupby("model", sort=False)):
        part = part.sort_values("threshold")
        x = np.arange(len(part))
        for i, (metric, label, color) in enumerate(
            [
                ("f1", "F1-score", "#3578b8"),
                ("recall", "Recall", "#e49032"),
                ("precision", "Precision", "#37936b"),
            ]
        ):
            values = 100 * part[metric].to_numpy()
            bars = ax.bar(x + (i - 1) * 0.25, values, 0.25, label=label, color=color)
            # Rótulos verticais cabem nas barras estreitas; nas baixas ficam acima.
            for bar, value in zip(bars, values):
                inside = value >= 15
                ax.annotate(
                    f"{value:.2f}%".replace(".", ","),
                    xy=(bar.get_x() + bar.get_width() / 2, value / 2 if inside else value),
                    xytext=(0, 0 if inside else 3),
                    textcoords="offset points",
                    ha="center",
                    va="center" if inside else "bottom",
                    rotation=90,
                    color="white" if inside else "#222222",
                    fontsize=7,
                    fontweight="bold",
                )
        labels = [f"{value:g}".replace(".", ",") for value in part.threshold]
        ax.set(
            xticks=x,
            xticklabels=labels,
            title=names.get(name.split("__", 1)[0], name),
            xlabel="Threshold",
            ylabel="Score (%)",
            ylim=(0, 100),
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="upper left", ncols=3, bbox_to_anchor=(0, 1.02))
    fig.suptitle("Sensibilidade por threshold — última validação TRAIN; sem escolha de novo limiar")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(summary, manifest, output) -> None:
    """Relatório atual orientado por F1; não altera as avaliações históricas."""
    rows = summary.loc[~summary.strategy.eq("dummy")]
    lines = [
        "# SMOTE isolado e undersampling isolado — F1 principal",
        "",
        (
            "Médias dos mesmos folds temporais do TRAIN, pesos uniformes e threshold comum 0,50. "
            "F1 equilibra Precision e Recall com pesos simétricos. O relatório atual não usa F2."
        ),
        "",
        (
            "As previsões e contagens já calculadas foram preservadas. Trocar a métrica de "
            "seleção não muda Precision ou Recall no mesmo threshold. Não houve novo treino "
            "nem reteste para atualizar estes relatórios."
        ),
        "",
        "| Receita | F1 médio | Recall médio | Precision média | AP média |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples():
        lines.append(
            f"| {row.model} | {row.f1:.2%} | {row.recall:.2%} | "
            f"{row.precision:.2%} | {row.average_precision:.4f} |"
        )
    lines += [
        "",
        "## Mudança frente ao controle sem reamostragem",
        "",
        "| Modelo | Sampler | Delta F1 (p.p.) | Delta Recall (p.p.) | Delta Precision (p.p.) |",
        "|---|---|---:|---:|---:|",
    ]
    for family in manifest["settings"]["families"]:
        baseline = rows.loc[rows.model.eq(f"{family}__none")].iloc[0]
        for strategy in ["smote", "under"]:
            row = rows.loc[rows.model.eq(f"{family}__{strategy}")].iloc[0]
            lines.append(
                f"| {family} | {strategy} | {100 * (row.f1 - baseline.f1):+.2f} | "
                f"{100 * (row.recall - baseline.recall):+.2f} | "
                f"{100 * (row.precision - baseline.precision):+.2f} |"
            )
    lines += [
        "",
        "## Gráficos e notebooks",
        "",
        (
            "Cada gráfico principal contém dois grupos (Random Forest e XGBoost), com três "
            "barras por grupo: F1, Recall e Precision. Os controles aparecem nas tabelas. "
            "A sensibilidade por threshold também usa barras agrupadas dessas três métricas."
        ),
        "",
        (
            "Notebooks: notebooks/metricas_smotec.ipynb e notebooks/metricas_under.ipynb. "
            "SMOTEC é o nome de arquivo solicitado; o sampler é SMOTE numérico."
        ),
        "",
        (
            "Os gráficos de treino/validação usam F1. PR/AP, ROC, matriz OOF, curvas AP/Log Loss "
            "do XGBoost, correlação/importance e diversidade de erros continuam complementares."
        ),
        "",
        (
            "**Limites:** mesmos folds reutilizados; diferenças de médias não provam "
            "superioridade estatística. Threshold 0,50 é comum e não foi otimizado nesta "
            "atualização. OOF agregado não é a média dos folds. "
            "O artefato histórico com limiar 0,70 continua preservado como histórico; "
            "nenhum novo modelo orientado por F1 foi promovido nem avaliado no TEST."
        ),
        "",
    ]
    (output / "reports/sampling_comparison_results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def refresh_reports(root: str | Path = ROOT, settings: dict | None = None) -> dict:
    """Recalcula F1 das contagens já salvas e regenera relatórios, sem fit ou TEST."""
    root = Path(root)
    md, ed, fd = [
        root / part for part in ["reports/metrics", "reports/experiments", "reports/figures"]
    ]
    settings = (
        load_config(ROOT / "configs/sampling_comparison.yaml") if settings is None else settings
    )
    if settings.get("primary_metric") != "f1" or settings.get("threshold") != 0.5:
        raise ValueError("A atualização usa F1 e mantém o threshold 0,5.")
    manifest_path = ed / "sampling_comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_settings = dict(manifest["settings"])
    current_settings = dict(settings)
    previous_metric = previous_settings.pop("primary_metric")
    current_settings.pop("primary_metric")
    # Reaproveitar métricas só é válido se sampler, dados/folds e limiar não mudaram.
    if previous_settings != current_settings:
        raise ValueError("Configuração mudou além da métrica; execute um novo experimento.")

    def clean(frame):
        # Preserva as métricas independentes da seleção e remove F2 das saídas atuais.
        frame = frame.drop(
            columns=[
                c
                for c in frame.columns
                if c == "f2" or c == "f2_std" or c.startswith(("train_f2", "validation_f2"))
            ]
        )
        if {"tp", "fp", "fn"}.issubset(frame.columns):
            denominator = 2 * frame.tp + frame.fp + frame.fn
            recalculated = np.divide(
                2 * frame.tp,
                denominator,
                out=np.zeros(len(frame), dtype=float),
                where=denominator.to_numpy() != 0,
            )
            if "f1" in frame and not np.allclose(frame.f1, recalculated, atol=1e-12):
                raise ValueError("Contagens e F1 persistido não correspondem.")
            frame["f1"] = recalculated
        return frame

    names = ["sampling_folds.csv", "sampling_oof_metrics.csv"]
    for mode in ["smote", "under"]:
        names.extend([f"{mode}_threshold_sensitivity.csv", f"{mode}_boosting_iterations.csv"])
    # Carrega e valida antes de sobrescrever qualquer arquivo.
    updated = {name: clean(pd.read_csv(md / name)) for name in names}
    summary = summarize(updated["sampling_folds.csv"])
    old_hash = hashlib.sha256((md / "sampling_folds.csv").read_bytes()).hexdigest()
    for name, frame in updated.items():
        frame.to_csv(md / name, index=False)
    summary.to_csv(md / "sampling_comparison.csv", index=False)
    for mode in ["smote", "under"]:
        selected = summary.loc[summary.strategy.isin([mode, "none", "dummy"])]
        selected.to_csv(md / f"{mode}_model_comparison.csv", index=False)
        plot_bars(selected, fd / f"{mode}_comparison_f1_recall_precision.png", mode)
        plot_generalization(
            updated["sampling_folds.csv"].loc[
                lambda f, selected_mode=mode: f.strategy.eq(selected_mode)
            ],
            fd / f"{mode}_train_validation.png",
        )
        plot_sensitivity(
            updated[f"{mode}_threshold_sensitivity.csv"], fd / f"{mode}_threshold_sensitivity.png"
        )

    manifest.setdefault("previous_reporting_primary_metric", previous_metric)
    manifest.setdefault("fold_metrics_before_refresh_sha256", old_hash)
    manifest["settings"] = settings
    manifest["selection_metric"] = "F1 média na CV"
    manifest["metrics_recomputed_without_refit"] = True
    manifest.pop("primary_f2_model_replaced", None)
    manifest["primary_model_replaced"] = False
    manifest["winners"] = {
        mode: summary.loc[summary.strategy.eq(mode)].iloc[0].model for mode in ["smote", "under"]
    }
    write_report(summary, manifest, root)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("F1 recalculado das contagens; gráficos e tabelas atualizados sem novo fit.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--features-config", type=Path, default=ROOT / "configs/ablation.yaml")
    parser.add_argument(
        "--sampling-config", type=Path, default=ROOT / "configs/sampling_comparison.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument(
        "--refresh-reports",
        action="store_true",
        help="Reagrega F1 e figuras usando contagens já calculadas, sem retreinar.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.refresh_reports:
        refresh_reports(args.output, load_config(args.sampling_config))
        return
    run(
        load_config(args.config),
        load_config(args.features_config),
        load_config(args.sampling_config),
        args.output,
    )


if __name__ == "__main__":
    main()
