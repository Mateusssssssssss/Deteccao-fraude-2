"""Busca temporal de XGBoost/Random Forest com SMOTE e threshold em janela reservada."""

import argparse
import importlib.metadata
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import FixedThresholdClassifier, GridSearchCV, ParameterGrid

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.final_test import (
    calibration_table,
    complete_metrics,
    file_hash,
    metric_dictionary,
    save_figures,
)
from fraud_detection.sampling_comparison import plot_association
from fraud_detection.segments import error_rates_by_bins
from fraud_detection.validation import temporal_folds
from fraud_detection.xgboost_gridsearch import (
    build_search_pipeline,
    iteration_metrics,
    load_reusable_results,
    parameter_key,
    plot_comparison,
    plot_iterations,
)

LOGGER = logging.getLogger(__name__)


def select_threshold_index(results: dict) -> int:
    """F1 decide; empates usam Precision, Recall e maior limiar, nessa ordem."""
    return int(
        np.lexsort(
            (
                np.asarray(results["param_threshold"], dtype=float),
                results["mean_test_recall"],
                results["mean_test_precision"],
                results["mean_test_f1"],
            )
        )[-1]
    )


def search_threshold(fitted, fit, tune, settings):
    """GridSearch de limiar sem refit: o modelo nunca viu a janela que escolhe threshold."""
    values = np.asarray(settings["thresholds"], dtype=float)
    if (
        settings["threshold_objective"] != "f1"
        or len(values) == 0
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("A grade de threshold deve estar em [0,1], com objetivo F1.")
    combined = pd.concat([fit, tune], ignore_index=True)
    grid = GridSearchCV(
        FixedThresholdClassifier(
            FrozenEstimator(fitted), response_method="predict_proba", pos_label=1
        ),
        {"threshold": values.tolist()},
        scoring={
            "f1": make_scorer(f1_score, zero_division=0),
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "average_precision": "average_precision",
        },
        cv=[(np.arange(len(fit)), np.arange(len(fit), len(combined)))],
        refit=select_threshold_index,
        n_jobs=1,
        error_score="raise",
    )
    grid.fit(combined[FEATURES], combined[TARGET])
    probability = fitted.predict_proba(tune[FEATURES])[:, 1]
    table = pd.DataFrame(
        [
            {
                "threshold": value,
                **{
                    key: val
                    for key, val in complete_metrics(tune[TARGET], probability, value).items()
                    if key != "f2"
                },
            }
            for value in values
        ]
    )
    # As decisões calculadas para relatório precisam coincidir com o scorer da busca.
    np.testing.assert_allclose(table.f1, grid.cv_results_["mean_test_f1"])
    np.testing.assert_allclose(
        table.average_precision, grid.cv_results_["mean_test_average_precision"]
    )
    return float(grid.best_params_["threshold"]), table


def select_parameters(
    pipeline, config, features, settings, boundaries, original_folds, search_frame, dataset_hash
):
    """Escolhe por F1 ou AP apenas nos folds anteriores à janela de threshold."""
    metric = settings.get("primary_metric", "average_precision")
    lookup_settings = dict(settings)
    if metric == "f1":
        lookup_settings["threshold"] = settings.get("cv_reference_threshold", 0.5)
    cached, provenance = load_reusable_results(
        config,
        features,
        lookup_settings,
        boundaries,
        dataset_hash,
        probability_scores_only=metric == "average_precision",
    )
    metrics = ["average_precision", "roc_auc"]
    if metric == "f1":
        metrics = ["f1", "recall", "precision", *metrics]
    all_params = list(ParameterGrid({f"model__{k}": v for k, v in settings["param_grid"].items()}))
    records = {parameter_key(row["params"]): row for row in cached.to_dict("records")}
    pending = [params for params in all_params if parameter_key(params) not in records]
    folds = original_folds[:-1]
    LOGGER.info(
        "GridSearch %s: %s combinações; %s reutilizadas; %s novas x %s folds",
        settings.get("family", "xgboost"),
        len(all_params),
        len(cached),
        len(pending),
        len(folds),
    )
    if pending:
        candidates = [{key: [value] for key, value in params.items()} for params in pending]
        grid = GridSearchCV(
            pipeline,
            candidates,
            scoring={
                **(
                    {
                        "f1": make_scorer(f1_score, zero_division=0),
                        "recall": make_scorer(recall_score, zero_division=0),
                        "precision": make_scorer(precision_score, zero_division=0),
                    }
                    if metric == "f1"
                    else {}
                ),
                "average_precision": "average_precision",
                "roc_auc": "roc_auc",
            },
            refit=False,
            cv=folds,
            n_jobs=settings["search_n_jobs"],
            return_train_score=True,
            error_score="raise",
            verbose=2,
        )
        with joblib.parallel_backend("threading"):
            grid.fit(search_frame[FEATURES], search_frame[TARGET])
        records.update(
            {
                parameter_key(row["params"]): row
                for row in pd.DataFrame(grid.cv_results_).to_dict("records")
            }
        )
    rows = []
    for params in all_params:
        source = records[parameter_key(params)]
        row = {"params": params, **{f"param_{key}": value for key, value in params.items()}}
        for score_name in metrics:
            for phase in ["train", "test"]:
                scores = [
                    float(source[f"split{i}_{phase}_{score_name}"]) for i in range(len(folds))
                ]
                row.update(
                    {f"split{i}_{phase}_{score_name}": score for i, score in enumerate(scores)}
                )
                row[f"mean_{phase}_{score_name}"] = float(np.mean(scores))
                row[f"std_{phase}_{score_name}"] = float(np.std(scores))
        rows.append(row)
    table = pd.DataFrame(rows)
    table[f"rank_test_{metric}"] = (
        table[f"mean_test_{metric}"].rank(method="min", ascending=False).astype(int)
    )
    if metric == "f1":
        best_index = int(
            np.lexsort((table.mean_test_recall, table.mean_test_precision, table.mean_test_f1))[-1]
        )
    else:
        best_index = int(np.argmax(table.mean_test_average_precision))
    return table, table.loc[best_index, "params"], provenance, len(pending) * len(folds)


def plot_ap(table, path, metric="average_precision"):
    """Perfis do critério escolhido; a melhor combinação pode mudar nos demais parâmetros."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    names = [
        column.removeprefix("param_model__")
        for column in table
        if column.startswith("param_model__")
    ]
    for ax, name in zip(axes.flat, names):
        column = f"param_model__{name}"
        part = table.loc[table.groupby(column)[f"mean_test_{metric}"].idxmax()].sort_values(column)
        ax.errorbar(
            part[column],
            part[f"mean_test_{metric}"],
            yerr=part[f"std_test_{metric}"],
            fmt="-o",
            capsize=4,
        )
        ax.set(
            xlabel=name,
            ylabel=f"{metric} médio da CV anterior ao threshold",
            ylim=(0, 1),
            xticks=part[column],
        )
        ax.grid(alpha=0.2)
    axes.flat[-1].set_visible(False)
    fig.suptitle(f"Perfis por {metric} — melhor combinação em cada valor; sem efeito isolado")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_thresholds(table, threshold, path):
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(table))
    chosen = int(np.flatnonzero(np.isclose(table.threshold, threshold))[0])
    ax.axvspan(chosen - 0.45, chosen + 0.45, color="#e6e6e6")
    # Muitos limiares: rótulos verticais mantêm os três scores legíveis.
    for i, (key, label, color) in enumerate(
        [
            ("f1", "F1", "#3578b8"),
            ("recall", "Recall", "#e49032"),
            ("precision", "Precision", "#37936b"),
        ]
    ):
        bars = ax.bar(x + (i - 1) * 0.25, 100 * table[key], 0.25, label=label, color=color)
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.1f}%",
                (bar.get_x() + bar.get_width() / 2, value / 2 if value >= 15 else value + 1),
                ha="center",
                va="center" if value >= 15 else "bottom",
                rotation=90,
                fontsize=6,
                color="white" if value >= 15 else "black",
            )
    ax.set(
        xticks=x,
        xticklabels=[f"{v:g}" for v in table.threshold],
        ylim=(0, 100),
        xlabel="Threshold — cinza: escolhido por F1",
        ylabel="Score (%)",
        title="GridSearchCV do threshold — janela TRAIN separada da busca de parâmetros",
    )
    ax.tick_params(axis="x", rotation=70)
    ax.legend(ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config, features, settings, output=ROOT):
    """Três períodos ordenados: busca de parâmetros → threshold → validação, só TRAIN."""
    metric = settings["primary_metric"]
    if metric not in {"average_precision", "f1"}:
        raise ValueError("Use Average Precision ou F1 como critério explícito.")
    family = settings.get("family", "xgboost")
    label = "Random Forest" if family == "random_forest" else "XGBoost"
    name = f"{label} {metric} + SMOTE"
    started = time.perf_counter()
    output = Path(output)
    prefix = settings["experiment_name"]
    md, ed, fd, model_dir = [
        output / path
        for path in ["reports/metrics", "reports/experiments", "reports/figures", "models"]
    ]
    for folder in [md, ed, fd, model_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ed / f"{prefix}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Rodada já calculada; use outra saída.")
    train, _, split, source = read_split(config)
    a, b = temporal_folds(train, dict(config, sampling_strategies=["none"]))[-1]
    prefix_frame, validation = train.iloc[a], train.iloc[b]
    original_folds = temporal_folds(
        prefix_frame, dict(config, cv_splits=settings["cv_splits"], sampling_strategies=["none"])
    )
    fit_idx, tune_idx = original_folds[-1]
    search_frame, tune = prefix_frame.iloc[fit_idx], prefix_frame.iloc[tune_idx]
    boundaries = pd.DataFrame(
        [
            {
                "fold": i,
                "fit_rows": len(a),
                "validation_rows": len(b),
                "fit_frauds": int(prefix_frame.iloc[a][TARGET].sum()),
                "validation_frauds": int(prefix_frame.iloc[b][TARGET].sum()),
                "fit_time_max": float(prefix_frame.iloc[a].Time.max()),
                "validation_time_min": float(prefix_frame.iloc[b].Time.min()),
                "validation_time_max": float(prefix_frame.iloc[b].Time.max()),
            }
            for i, (a, b) in enumerate(original_folds, start=1)
        ]
    )
    pipeline = build_search_pipeline(
        config, features, settings, joblib.Memory(output / ".cache/xgboost_gridsearch", verbose=0)
    )
    table, params, reuse, new_fits = select_parameters(
        pipeline,
        config,
        features,
        settings,
        boundaries,
        original_folds,
        search_frame,
        file_hash(source),
    )
    table.sort_values(f"rank_test_{metric}").to_csv(md / f"{prefix}_results.csv", index=False)
    winner = table.loc[
        table.params.map(parameter_key).map(lambda key: key == parameter_key(params))
    ].iloc[0]
    LOGGER.info(
        "Parâmetros por %s: %s; score CV=%.6f. Ajustando modelo no prefixo pré-threshold.",
        metric,
        params,
        winner[f"mean_test_{metric}"],
    )
    fitted = clone(pipeline).set_params(**params).fit(search_frame[FEATURES], search_frame[TARGET])
    threshold, threshold_table = search_threshold(fitted, search_frame, tune, settings)
    threshold_table.to_csv(md / f"{prefix}_thresholds.csv", index=False)
    selection = {
        "best_params": {key.removeprefix("model__"): value for key, value in params.items()},
        "threshold": threshold,
        "primary_metric": metric,
        "family": family,
        "best_cv_score": float(winner[f"mean_test_{metric}"]),
        "threshold_objective": "f1",
        "best_cv_ap": float(winner.mean_test_average_precision),
        "validation_used_for_selection": False,
    }
    (ed / f"{prefix}_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info(
        "Threshold escolhido no TRAIN: %.4f. Parâmetros/limiar congelados antes da validação.",
        threshold,
    )
    probability = fitted.predict_proba(validation[FEATURES])[:, 1]
    raw = complete_metrics(validation[TARGET], probability, threshold)
    score = {key: value for key, value in raw.items() if key != "f2"}
    result = {
        "model": name,
        "source": "Validação temporal TRAIN posterior ao ajuste do threshold; histórico explorado",
        **score,
    }
    pd.DataFrame([result]).to_csv(md / f"{prefix}_validation.csv", index=False)
    (md / f"{prefix}_model_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metric_dictionary(raw).loc[lambda f: f.metric.ne("f2")].to_csv(
        md / f"{prefix}_metric_definitions.csv", index=False
    )
    calibration = calibration_table(validation[TARGET], probability)
    calibration.to_csv(md / f"{prefix}_calibration.csv", index=False)
    report = classification_report(
        validation[TARGET],
        probability >= threshold,
        labels=[0, 1],
        target_names=["Legítima (0)", "Fraude (1)"],
        output_dict=True,
        zero_division=0,
    )
    (md / f"{prefix}_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    error_rates_by_bins(
        search_frame,
        validation,
        probability,
        threshold,
        "Validação TRAIN após escolha do threshold",
    ).to_csv(md / f"{prefix}_category_errors.csv", index=False)
    fitted.named_steps["select"].audit_.to_csv(md / f"{prefix}_feature_selection.csv", index=False)
    importance = permutation_importance(
        fitted,
        validation[FEATURES],
        validation[TARGET],
        scoring="average_precision",
        n_repeats=settings["permutation_repeats"],
        random_state=config["seed"],
        n_jobs=1,
    )
    selected = fitted.named_steps["select"].selected_features_
    tree = dict(zip(selected, fitted.named_steps["model"].feature_importances_))
    association = pd.DataFrame(
        {
            "feature": FEATURES,
            "pearson": [search_frame[c].corr(search_frame[TARGET]) for c in FEATURES],
            "spearman": [
                search_frame[c].corr(search_frame[TARGET], method="spearman") for c in FEATURES
            ],
            "permutation_ap": importance.importances_mean,
            "permutation_std": importance.importances_std,
            "tree_importance": [tree.get(c, 0.0) for c in FEATURES],
            "selected_last_fold": [c in selected for c in FEATURES],
        }
    )
    association.to_csv(md / f"{prefix}_association_importance.csv", index=False)
    if family == "xgboost":
        iterations = iteration_metrics(
            fitted, search_frame, validation, dict(settings, threshold=threshold)
        )
        iterations.to_csv(md / f"{prefix}_iterations.csv", index=False)
        plot_iterations(iterations, fd / f"{prefix}_iterations.png")
    plot_comparison(
        pd.DataFrame([result]),
        fd / f"{prefix}_model_f1_recall_precision.png",
        f"{label} por {metric} + SMOTE — validação TRAIN após ambas as escolhas\nThreshold escolhido por GridSearchCV: {threshold:g}",
    )
    plot_thresholds(threshold_table, threshold, fd / f"{prefix}_thresholds.png")
    plot_ap(table, fd / f"{prefix}_parameters.png", metric)
    plot_association(
        association, fd / f"{prefix}_association_importance.png", f"{label} por {metric} + SMOTE"
    )
    save_figures(
        validation[TARGET].to_numpy(),
        probability,
        threshold,
        fd,
        calibration,
        prefix=prefix,
        label=f"{label} {metric} — validação TRAIN",
    )
    cv_scores = pd.DataFrame(
        [
            {
                "fold": i + 1,
                "source": "CV TRAIN anterior à janela de threshold",
                "average_precision": winner[f"split{i}_test_average_precision"],
                "train_average_precision": winner[f"split{i}_train_average_precision"],
                "roc_auc": winner[f"split{i}_test_roc_auc"],
                **(
                    {key: winner[f"split{i}_test_{key}"] for key in ["f1", "recall", "precision"]}
                    if metric == "f1"
                    else {}
                ),
                **(
                    {
                        f"train_{key}": winner[f"split{i}_train_{key}"]
                        for key in ["f1", "recall", "precision"]
                    }
                    if metric == "f1"
                    else {}
                ),
            }
            for i in range(len(original_folds) - 1)
        ]
    )
    cv_scores.to_csv(md / f"{prefix}_fold_metrics.csv", index=False)
    boundaries.iloc[:-1].to_csv(ed / f"{prefix}_folds.csv", index=False)
    fitted.memory = None
    artifact = model_dir / f"{prefix}_pipeline.joblib"
    joblib.dump(
        {
            "pipeline": fitted,
            "features": FEATURES,
            "selected_features": selected,
            "model": f"{family}__{metric}_threshold",
            "threshold": threshold,
            "threshold_objective": "F1 por GridSearchCV em janela TRAIN reservada",
            "test_evaluated": False,
        },
        artifact,
    )
    if family == "xgboost":
        table.loc[
            table.groupby("param_model__learning_rate")[f"mean_test_{metric}"].idxmax()
        ].to_csv(md / f"{prefix}_learning_rate_profiles.csv", index=False)
    pd.DataFrame([result]).to_csv(md / f"{prefix}_model_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["AP média da CV"],
        [winner.mean_test_average_precision],
        yerr=[winner.std_test_average_precision],
        capsize=5,
        color="#3578b8",
    )
    ax.text(
        0,
        winner.mean_test_average_precision / 2,
        f"{winner.mean_test_average_precision:.4f}",
        ha="center",
        color="white",
        fontweight="bold",
    )
    ax.set(
        ylim=(0, 1),
        ylabel="Average Precision",
        title=f"{label} — AP da CV anterior ao threshold (diagnóstico)",
    )
    fig.tight_layout()
    fig.savefig(fd / f"{prefix}_model_cv_ap.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cv_scores.fold, cv_scores.train_average_precision, "--o", label="Treino interno")
    ax.plot(cv_scores.fold, cv_scores.average_precision, "-o", label="Validação interna")
    ax.set(
        xticks=cv_scores.fold,
        ylim=(0, 1),
        xlabel="Fold temporal",
        ylabel="AP",
        title="AP por fold — períodos e tamanho do treino variam",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fd / f"{prefix}_model_train_validation.png", dpi=160)
    plt.close(fig)
    if metric == "f1":
        cv_row = {
            "model": name,
            **{key: winner[f"mean_test_{key}"] for key in ["f1", "recall", "precision"]},
            **{f"{key}_std": winner[f"std_test_{key}"] for key in ["f1", "recall", "precision"]},
        }
        plot_comparison(
            pd.DataFrame([cv_row]),
            fd / f"{prefix}_model_cv_f1_recall_precision.png",
            f"{label} selecionado por F1 — CV dos parâmetros\nReferência de decisão 0,50 na CV; limiar final é escolhido depois",
            True,
        )
    if metric == "f1":
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, key in zip(axes, ["f1", "recall", "precision"]):
            ax.plot(cv_scores.fold, 100 * cv_scores[f"train_{key}"], "--o", label="Treino interno")
            ax.plot(cv_scores.fold, 100 * cv_scores[key], "-o", label="Validação interna")
            ax.set(
                xticks=cv_scores.fold, ylim=(0, 100), xlabel="Fold temporal", ylabel=f"{key} (%)"
            )
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"{label} — CV de parâmetros em referência comum 0,50; threshold final escolhido depois"
        )
        fig.tight_layout()
        fig.savefig(fd / f"{prefix}_model_train_validation.png", dpi=160)
        plt.close(fig)
    manifest = {
        **selection,
        "config": config,
        "settings": settings,
        "feature_settings": features,
        "reuse": reuse,
        "combinations": len(table),
        "grid_fits": len(table) * (len(original_folds) - 1),
        "cv_seconds": time.perf_counter() - started,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]
        },
        "inner_folds": len(original_folds) - 1,
        "executed_grid_fits": new_fits,
        "reused_grid_fits": (len(table) * (len(original_folds) - 1) - new_fits),
        "train_rows": len(train),
        "search_fit_rows": len(search_frame),
        "threshold_rows": len(tune),
        "threshold_frauds": int(tune[TARGET].sum()),
        "threshold_time_min": float(tune.Time.min()),
        "threshold_time_max": float(tune.Time.max()),
        "search_fit_time_max": float(search_frame.Time.max()),
        "validation_time_min": float(validation.Time.min()),
        "validation_time_max": float(validation.Time.max()),
        "validation_rows": len(validation),
        "validation_frauds": int(validation[TARGET].sum()),
        "selected_features": selected,
        "dataset_sha256": file_hash(source),
        "artifact_sha256": file_hash(artifact),
        "test_evaluated": False,
        "selected_random_forest_replaced": False,
        "refit_after_threshold": False,
        "cv_reference_threshold": settings.get("cv_reference_threshold", 0.5)
        if metric == "f1"
        else None,
        "split": split,
        "limitations": [
            "F1 equilibra Precision e Recall; AP permanece diagnóstico quando F1 é a métrica de seleção.",
            "Seleção usa somente os folds anteriores à janela de threshold; limiar final é ajustado por F1 nessa janela.",
            (
                "Na CV de parâmetros, F1/Recall/Precision usam a referência comum 0,50; o limiar operacional não fica fixado nesse valor."
                if metric == "f1"
                else "AP usa scores contínuos, sem escolher parâmetros por um threshold fixo."
            ),
            "Validação TRAIN ficou fora de ambas as escolhas nesta rodada, mas o histórico já explorou esses dados.",
            "O mesmo pipeline congelado usado no threshold é aplicado à validação, sem refit posterior ou calibração.",
            "Não houve novo TEST; confirmação independente exige dados novos.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "Concluído: AP validação %.6f, F1 %.4f, Recall %.4f, Precision %.4f.",
        score["average_precision"],
        score["f1"],
        score["recall"],
        score["precision"],
    )
    return manifest


def refresh_cv_statistics(output: Path, prefix: str, family: str) -> None:
    """Apenas tabelas/figuras: não carrega estimadores nem executa fit/predict."""
    output = Path(output)
    md, fd = output / "reports/metrics", output / "reports/figures"
    scores = pd.read_csv(md / f"{prefix}_fold_metrics.csv")
    label = "Random Forest" if family == "random_forest" else "XGBoost"
    row = {
        "model": label,
        **{key: scores[key].mean() for key in ["f1", "recall", "precision"]},
        **{f"{key}_std": scores[key].std(ddof=0) for key in ["f1", "recall", "precision"]},
    }
    plot_comparison(
        pd.DataFrame([row]),
        fd / f"{prefix}_model_cv_f1_recall_precision.png",
        f"{label} — CV de parâmetros em referência comum 0,50\nLimiar operacional escolhido depois na janela reservada",
        True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, key in zip(axes, ["f1", "recall", "precision"]):
        ax.plot(scores.fold, 100 * scores[f"train_{key}"], "--o", label="Treino interno")
        ax.plot(scores.fold, 100 * scores[key], "-o", label="Validação interna")
        ax.set(xticks=scores.fold, ylim=(0, 100), xlabel="Fold temporal", ylabel=f"{key} (%)")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{label} — CV usada para escolher parâmetros; períodos e tamanho do treino variam"
    )
    fig.tight_layout()
    fig.savefig(fd / f"{prefix}_model_train_validation.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument(
        "--search-config", type=Path, help="Configuração explícita para reproduzir outra grade."
    )
    parser.add_argument(
        "--statistics-only", action="store_true", help="Regera somente gráficos da CV já salva."
    )
    args = parser.parse_args()
    filename = (
        "xgboost_f1_threshold.yaml" if args.family == "xgboost" else "random_forest_gridsearch.yaml"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = load_config(args.search_config or ROOT / "configs" / filename)
    if settings.get("family", "xgboost") != args.family:
        parser.error("A família da configuração não corresponde a --family.")
    if args.statistics_only:
        refresh_cv_statistics(args.output, settings["experiment_name"], args.family)
        return
    run(
        load_config(),
        load_config(ROOT / "configs/ablation.yaml"),
        settings,
        args.output,
    )


if __name__ == "__main__":
    main()
