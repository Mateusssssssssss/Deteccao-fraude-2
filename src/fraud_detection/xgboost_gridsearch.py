"""GridSearchCV de XGBoost + SMOTE, com validação temporal separada dentro do TRAIN."""

import argparse
import ast
import importlib.metadata
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.features import TrainFeatureSelector
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

LOGGER = logging.getLogger(__name__)
METRICS = ["f1", "recall", "precision", "average_precision", "roc_auc", "balanced_accuracy"]
PARAMETERS = ["max_depth", "n_estimators", "reg_alpha", "reg_lambda"]


def build_search_pipeline(config: dict, features: dict, settings: dict, cache=None) -> Pipeline:
    """Seleção → mediana/scaler → SMOTE → estimador; estatísticas ficam no fold."""
    numeric = SkPipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ]
    )
    if settings.get("family", "xgboost") == "random_forest":
        params = dict(config["models"]["random_forest"], class_weight=None)
        estimator = RandomForestClassifier(
            **params, random_state=config["seed"], n_jobs=settings["model_n_jobs"]
        )
    else:
        estimator = XGBClassifier(
            **config["models"]["xgboost"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config["seed"],
            n_jobs=settings["model_n_jobs"],
        )
    # O cache reaproveita apenas transformações do mesmo X/y, fold e configuração.
    return Pipeline(
        [
            ("select", TrainFeatureSelector(config, features)),
            ("preprocess", preprocess),
            (
                "smote",
                SMOTE(
                    sampling_strategy=settings["sampling_ratio"],
                    k_neighbors=config["smote_neighbors"],
                    random_state=config["seed"],
                ),
            ),
            ("model", estimator),
        ],
        memory=cache,
    )


def make_search(pipeline, folds, settings: dict, candidates=None) -> GridSearchCV:
    """F1 decide; métricas complementares e scores de treino são registrados na grade."""
    if settings["primary_metric"] != "f1" or settings["threshold"] != 0.5:
        raise ValueError("A busca atual usa F1 e threshold fixo 0,50; não otimiza limiar.")
    grid = settings["param_grid"]
    if set(grid) not in (set(PARAMETERS), set(PARAMETERS) | {"learning_rate"}):
        raise ValueError(
            "A grade deve conter max_depth, n_estimators, reg_alpha e reg_lambda; learning_rate é opcional."
        )
    for params in ParameterGrid(grid):
        if (
            params["max_depth"] < 1
            or params["n_estimators"] < 1
            or params["reg_alpha"] < 0
            or params["reg_lambda"] < 0
            or not 0 < params.get("learning_rate", 0.05) <= 1
        ):
            raise ValueError(
                "Profundidade/árvores positivas, regularizações não negativas e learning_rate em (0, 1]."
            )
    # A baseline precisa estar na mesma grade; falha antes de iniciar fits caros.
    initial = {f"model__{name}": pipeline.named_steps["model"].get_params()[name] for name in grid}
    expanded = ParameterGrid({f"model__{name}": values for name, values in grid.items()})
    if not any(params == initial for params in expanded):
        raise ValueError("Inclua os parâmetros da configuração inicial na grade.")
    return GridSearchCV(
        pipeline,
        (
            candidates
            if candidates is not None
            else {f"model__{key}": values for key, values in grid.items()}
        ),
        scoring={
            "f1": make_scorer(f1_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "precision": make_scorer(precision_score, zero_division=0),
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit=False,
        cv=folds,
        n_jobs=settings["search_n_jobs"],
        pre_dispatch=settings["search_n_jobs"],
        error_score="raise",
        return_train_score=True,
        verbose=2,
    )


def clean_metrics(y, probability, threshold: float) -> dict:
    """Tabela atual usa F1; F2 histórico não participa das saídas deste experimento."""
    return {
        key: value
        for key, value in complete_metrics(y, probability, threshold).items()
        if key != "f2"
    }


def grid_summary(results: dict, index: int, name: str) -> dict:
    row = {
        "model": name,
        "source": "CV interna TRAIN usada na seleção; média dos folds",
        "threshold": 0.5,
    }
    for key in METRICS:
        row[key] = float(results[f"mean_test_{key}"][index])
        row[f"{key}_std"] = float(results[f"std_test_{key}"][index])
        row[f"train_{key}"] = float(results[f"mean_train_{key}"][index])
    return row


def plot_comparison(table: pd.DataFrame, path: Path, title: str, dispersion: bool = False) -> None:
    """Barras agrupadas no padrão do projeto, com valores de F1/Recall/Precision."""
    table = table.sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(table))
    for i, (metric, label, color) in enumerate(
        [
            ("f1", "F1-score", "#3578b8"),
            ("recall", "Recall", "#e49032"),
            ("precision", "Precision", "#37936b"),
        ]
    ):
        values = 100 * table[metric].to_numpy()
        errors = 100 * table[f"{metric}_std"].to_numpy() if dispersion else None
        bars = ax.bar(
            x + (i - 1) * 0.24, values, 0.24, yerr=errors, capsize=4, label=label, color=color
        )
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.2f}%".replace(".", ","),
                (bar.get_x() + bar.get_width() / 2, value / 2 if value >= 15 else value),
                ha="center",
                va="center" if value >= 15 else "bottom",
                color="white" if value >= 15 else "black",
                fontsize=10,
                fontweight="bold",
            )
    ax.set(xticks=x, xticklabels=table.model, ylabel="Score (%)", ylim=(0, 100), title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncols=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_grid(table: pd.DataFrame, path: Path) -> None:
    """Perfis do melhor F1 por valor: demais parâmetros são maximizados, não constantes."""
    parameters = [
        name for name in [*PARAMETERS, "learning_rate"] if f"param_model__{name}" in table
    ]
    fig, axes = plt.subplots(
        (len(parameters) + 1) // 2, 2, figsize=(12, 4.5 * ((len(parameters) + 1) // 2))
    )
    for ax, parameter in zip(axes.flat, parameters):
        indices = table.groupby(f"param_model__{parameter}").mean_test_f1.idxmax()
        part = table.loc[indices].sort_values(f"param_model__{parameter}")
        ax.plot(
            part[f"param_model__{parameter}"],
            100 * part.mean_train_f1,
            "--o",
            label="Treino interno",
        )
        ax.errorbar(
            part[f"param_model__{parameter}"],
            100 * part.mean_test_f1,
            yerr=100 * part.std_test_f1,
            fmt="-o",
            capsize=3,
            label="CV interna",
        )
        ax.set(
            xlabel=parameter,
            ylabel="F1 (%)",
            ylim=(0, 100),
            xticks=part[f"param_model__{parameter}"],
        )
        ax.grid(alpha=0.2)
        ax.legend()
    for unused in list(axes.flat)[len(parameters) :]:
        unused.set_visible(False)
    fig.suptitle("Perfis da grade — melhor combinação em cada valor; não efeito isolado")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def iteration_metrics(pipeline, fit, validation, settings: dict) -> pd.DataFrame:
    """Avalia prefixos de árvores do modelo pronto; não decide n_estimators na validação."""
    rows = []
    model = pipeline.named_steps["model"]
    for trees in sorted({model.n_estimators, *settings["iteration_points"]}):
        if trees > model.n_estimators:
            continue
        row = {"trees": trees}
        for name, frame in [("train", fit), ("validation", validation)]:
            selected = pipeline.named_steps["select"].transform(frame[FEATURES])
            transformed = pipeline.named_steps["preprocess"].transform(selected)
            probability = model.predict_proba(transformed, iteration_range=(0, trees))[:, 1]
            values = clean_metrics(frame[TARGET], probability, settings["threshold"])
            for key in ["f1", "average_precision", "log_loss"]:
                row[f"{name}_{key}"] = values[key]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_iterations(table, path) -> None:
    """Curvas de diagnóstico do modelo congelado; não fazem early stopping."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["f1", "average_precision", "log_loss"]):
        for name, label in [("train", "Treino"), ("validation", "Validação separada")]:
            ax.plot(table.trees, table[f"{name}_{metric}"], "o-", label=label)
        ax.set(xlabel="Árvores utilizadas", ylabel=metric, title=metric)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("XGBoost otimizado — diagnóstico posterior; n_estimators escolhido somente na CV")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parameter_key(params: dict) -> tuple:
    """Identidade da combinação, independente da ordem do dicionário."""
    return tuple(sorted(params.items()))


def load_reusable_results(
    config, features, settings, boundaries, dataset_hash, *, probability_scores_only=False
):
    """Reusa scores somente sob o mesmo dataset, versões e protocolo de treinamento."""
    if not settings.get("reuse_from"):
        return pd.DataFrame(), {}
    origin = Path(settings["reuse_from"])
    if not origin.is_absolute():
        origin = ROOT / origin
    prefix = settings.get("reuse_prefix", "xgboost_gridsearch")
    manifest_path = origin / f"reports/experiments/{prefix}_manifest.json"
    if not manifest_path.is_file():
        LOGGER.info("Rodada anterior ausente: todos os candidatos serão executados.")
        return pd.DataFrame(), {"skipped_reason": "manifesto anterior ausente"}
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        previous["dataset_sha256"] != dataset_hash
        or previous["config"] != config
        or previous["feature_settings"] != features
    ):
        raise ValueError("Reutilização recusada: dataset ou configurações de treino diferentes.")
    family = settings.get("family", "xgboost")
    if previous["settings"].get("family", "xgboost") != family:
        raise ValueError("Reutilização recusada: família de modelos diferente.")
    # AP/ROC-AUC usam scores contínuos: trocar a métrica de escolha ou o limiar
    # não altera esses scores por fold. O chamador descarta métricas de decisões.
    protocol_keys = ["cv_splits", "sampling_ratio", "model_n_jobs"]
    if not probability_scores_only:
        protocol_keys += ["primary_metric", "threshold"]
    for key in protocol_keys:
        prior_value = (
            previous["settings"].get(
                "cv_reference_threshold", previous["settings"].get("threshold")
            )
            if key == "threshold"
            else previous["settings"][key]
        )
        if prior_value != settings[key]:
            raise ValueError(f"Reutilização recusada: protocolo diferente em {key}.")
    for name, version in previous["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Reutilização recusada: versão diferente de {name}.")
    old_boundaries = pd.read_csv(origin / f"reports/experiments/{prefix}_folds.csv")
    if (
        len(old_boundaries) == len(boundaries) - 1
        and previous.get("refit_after_threshold") is False
    ):
        # Buscas em três períodos salvam apenas os folds de parâmetros; o último
        # período da divisão original foi reservado ao threshold e não foi CV.
        pd.testing.assert_frame_equal(
            boundaries.iloc[:-1].reset_index(drop=True), old_boundaries, check_dtype=False
        )
        last = boundaries.iloc[-1]
        if (
            previous["search_fit_time_max"] != old_boundaries.validation_time_max.max()
            or previous["threshold_time_min"] != last.validation_time_min
            or previous["threshold_time_max"] != last.validation_time_max
        ):
            raise ValueError("Reutilização recusada: períodos de busca/threshold diferentes.")
    else:
        pd.testing.assert_frame_equal(boundaries, old_boundaries, check_dtype=False)
    scores_path = origin / f"reports/metrics/{prefix}_results.csv"
    table = pd.read_csv(scores_path, float_precision="round_trip")
    # A rodada original só buscou quatro parâmetros; explicita o learning_rate fixo.
    table["params"] = table.params.map(ast.literal_eval)
    for name in settings["param_grid"]:
        key = f"model__{name}"
        table["params"] = [
            {key: config["models"][family][name], **params} for params in table.params
        ]
        table[f"param_{key}"] = [params[key] for params in table.params]
    expected = {
        parameter_key(params)
        for params in ParameterGrid(
            {f"model__{key}": values for key, values in settings["param_grid"].items()}
        )
    }
    table = table.loc[table.params.map(parameter_key).isin(expected)].copy()
    table["reused_from_previous"] = True
    provenance = {
        "manifest_sha256": file_hash(manifest_path),
        "cv_scores_sha256": file_hash(scores_path),
        "origin_prefix": prefix,
        "candidates": len(table),
    }
    return table, provenance


def combine_results(
    previous: pd.DataFrame, current: pd.DataFrame, all_params: list
) -> pd.DataFrame:
    """Reordena como a grade completa e recalcula ranks, sem misturar folds ou scores."""
    rows = {}
    for frame in [previous, current]:
        for row in frame.to_dict("records"):
            key = parameter_key(row["params"])
            if key in rows:
                raise ValueError("Candidato duplicado ao combinar resultados da grade.")
            rows[key] = row
    if set(rows) != {parameter_key(params) for params in all_params}:
        raise ValueError("Resultados incompletos para a grade configurada.")
    table = pd.DataFrame([rows[parameter_key(params)] for params in all_params])
    for metric in METRICS:
        table[f"rank_test_{metric}"] = (
            table[f"mean_test_{metric}"].rank(method="min", ascending=False).astype(int)
        )
    return table


def run(config: dict, features: dict, settings: dict, output: str | Path = ROOT) -> dict:
    """Escolhe na CV interna, congela parâmetros e descreve a validação TRAIN separada."""
    output = Path(output)
    prefix = settings.get("experiment_name", "xgboost_gridsearch")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise ValueError("Nome de experimento inválido.")
    md, ed, fd, models = [
        output / path
        for path in ["reports/metrics", "reports/experiments", "reports/figures", "models"]
    ]
    for folder in (md, ed, fd, models):
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = ed / f"{prefix}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Busca já concluída. Para uma nova rodada, use --output em outra pasta."
        )
    train, _, split, source = read_split(config)
    # A última janela do TRAIN fica fora de GridSearchCV; TEST sequer recebe predict.
    fit_idx, val_idx = temporal_folds(train, dict(config, sampling_strategies=["none"]))[-1]
    fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
    inner_folds = temporal_folds(
        fit, dict(config, cv_splits=settings["cv_splits"], sampling_strategies=["none"])
    )
    boundaries = pd.DataFrame(
        [
            {
                "fold": i,
                "fit_rows": len(a),
                "validation_rows": len(b),
                "fit_frauds": int(fit.iloc[a][TARGET].sum()),
                "validation_frauds": int(fit.iloc[b][TARGET].sum()),
                "fit_time_max": float(fit.iloc[a].Time.max()),
                "validation_time_min": float(fit.iloc[b].Time.min()),
                "validation_time_max": float(fit.iloc[b].Time.max()),
            }
            for i, (a, b) in enumerate(inner_folds, start=1)
        ]
    )
    boundaries.to_csv(ed / f"{prefix}_folds.csv", index=False)
    # Cache do imblearn evita recalcular a seleção/SMOTE para cada candidato no mesmo fold.
    memory = joblib.Memory(output / ".cache/xgboost_gridsearch", verbose=0)
    pipeline = build_search_pipeline(config, features, settings, memory)
    # Valida a grade completa antes de tentar reaproveitar ou executar candidatos.
    make_search(pipeline, inner_folds, settings)
    all_params = list(
        ParameterGrid({f"model__{key}": values for key, values in settings["param_grid"].items()})
    )
    previous, reuse = load_reusable_results(
        config, features, settings, boundaries, file_hash(source)
    )
    cached_keys = {parameter_key(params) for params in previous.get("params", [])}
    pending = [params for params in all_params if parameter_key(params) not in cached_keys]
    started = time.perf_counter()
    LOGGER.info(
        "GridSearch: %s candidatos totais; %s reutilizados; %s novos x %s folds",
        len(all_params),
        len(previous),
        len(pending),
        len(inner_folds),
    )
    current = pd.DataFrame()
    if pending:
        candidates = [{key: [value] for key, value in params.items()} for params in pending]
        search = make_search(pipeline, inner_folds, settings, candidates)
        # Threads evitam processos adicionais e limitam concorrência conforme a config.
        with joblib.parallel_backend("threading"):
            search.fit(fit[FEATURES], fit[TARGET])
        current = pd.DataFrame(search.cv_results_)
        current["reused_from_previous"] = False
    results = combine_results(previous, current, all_params)
    best_index = int(np.argmax(results.mean_test_f1.to_numpy()))
    best_params = results.loc[best_index, "params"]
    best = clone(pipeline).set_params(**best_params).fit(fit[FEATURES], fit[TARGET])
    seconds = time.perf_counter() - started
    ordered = results.sort_values(
        ["rank_test_f1", "mean_test_average_precision"], ascending=[True, False]
    )
    ordered.to_csv(md / f"{prefix}_results.csv", index=False)
    ordered.to_json(ed / f"{prefix}_results.json", orient="records", indent=2)
    cv_results = results.to_dict(orient="list")
    # A baseline usa todos os parâmetros pesquisados, inclusive a taxa original 0,05.
    baseline_params = {
        f"model__{key}": config["models"]["xgboost"][key] for key in settings["param_grid"]
    }
    baseline_index = next(
        (i for i, params in enumerate(cv_results["params"]) if params == baseline_params), None
    )
    if baseline_index is None:
        raise ValueError("Inclua a configuração original na grade para comparar na mesma CV.")
    cv_comparison = pd.DataFrame(
        [
            grid_summary(cv_results, baseline_index, "XGBoost inicial + SMOTE"),
            grid_summary(cv_results, best_index, "XGBoost GridSearch + SMOTE"),
        ]
    )
    cv_comparison.to_csv(md / f"{prefix}_cv_comparison.csv", index=False)
    fold_scores = pd.DataFrame(
        [
            {
                "model": row["model"],
                "fold": fold + 1,
                "threshold": 0.5,
                **{key: float(cv_results[f"split{fold}_test_{key}"][idx]) for key in METRICS},
                **{
                    f"train_{key}": float(cv_results[f"split{fold}_train_{key}"][idx])
                    for key in METRICS
                },
            }
            for idx, row in [
                (baseline_index, cv_comparison.iloc[0]),
                (best_index, cv_comparison.iloc[1]),
            ]
            for fold in range(len(inner_folds))
        ]
    )
    fold_scores.to_csv(md / f"{prefix}_fold_metrics.csv", index=False)
    # Registra a decisão ANTES de prever a validação: ela não altera a escolha da grade.
    selection = {
        "best_params": {key.removeprefix("model__"): value for key, value in best_params.items()},
        "threshold": settings["threshold"],
        "selection": "Maior F1 médio na CV interna; empate segue ordem da grade",
        "dataset_sha256": file_hash(source),
        "validation_used_for_selection": False,
    }
    (ed / f"{prefix}_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    baseline = clone(pipeline).fit(fit[FEATURES], fit[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(fit[FEATURES], fit[TARGET])
    estimators = [
        ("XGBoost inicial + SMOTE", baseline),
        ("XGBoost GridSearch + SMOTE", best),
        ("Dummy (prevalência)", dummy),
    ]
    rows = []
    for name, estimator in estimators:
        probability = estimator.predict_proba(validation[FEATURES])[:, 1]
        rows.append(
            {
                "model": name,
                "source": "Última janela TRAIN, fora da busca atual; histórico já explorado",
                **clean_metrics(validation[TARGET], probability, settings["threshold"]),
            }
        )
    validation_table = pd.DataFrame(rows)
    validation_table.to_csv(md / f"{prefix}_validation.csv", index=False)
    probability = best.predict_proba(validation[FEATURES])[:, 1]
    score = complete_metrics(validation[TARGET], probability, settings["threshold"])
    definitions = metric_dictionary(score).loc[lambda f: f.metric.ne("f2")]
    definitions.to_csv(md / f"{prefix}_metric_definitions.csv", index=False)
    calibration = calibration_table(validation[TARGET], probability)
    calibration.to_csv(md / f"{prefix}_calibration.csv", index=False)
    report = classification_report(
        validation[TARGET],
        probability >= settings["threshold"],
        labels=[0, 1],
        target_names=["Legítima (0)", "Fraude (1)"],
        output_dict=True,
        zero_division=0,
    )
    (md / f"{prefix}_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    error_rates_by_bins(
        fit, validation, probability, settings["threshold"], "Validação TRAIN separada da busca"
    ).to_csv(md / f"{prefix}_category_errors.csv", index=False)
    best.named_steps["select"].audit_.to_csv(md / f"{prefix}_feature_selection.csv", index=False)
    LOGGER.info("Busca concluída: %s; gerando importância e gráficos", selection["best_params"])
    importance = permutation_importance(
        best,
        validation[FEATURES],
        validation[TARGET],
        scoring="average_precision",
        n_repeats=settings["permutation_repeats"],
        random_state=config["seed"],
        n_jobs=1,
    )
    selected = best.named_steps["select"].selected_features_
    tree = dict(zip(selected, best.named_steps["model"].feature_importances_))
    association = pd.DataFrame(
        {
            "feature": FEATURES,
            "pearson": [fit[c].corr(fit[TARGET]) for c in FEATURES],
            "spearman": [fit[c].corr(fit[TARGET], method="spearman") for c in FEATURES],
            "permutation_ap": importance.importances_mean,
            "permutation_std": importance.importances_std,
            "tree_importance": [tree.get(c, 0.0) for c in FEATURES],
            "selected_last_fold": [c in selected for c in FEATURES],
            "source": "Correlações no prefixo TRAIN; permutação na validação TRAIN fora da busca atual",
        }
    )
    association.to_csv(md / f"{prefix}_association_importance.csv", index=False)
    iterations = iteration_metrics(best, fit, validation, settings)
    iterations.to_csv(md / f"{prefix}_iterations.csv", index=False)
    plot_comparison(
        cv_comparison,
        fd / f"{prefix}_cv_f1_recall_precision.png",
        "XGBoost + SMOTE — CV interna TRAIN usada na seleção\nMédia ± desvio; threshold 0,50",
        True,
    )
    plot_comparison(
        validation_table,
        fd / f"{prefix}_validation_f1_recall_precision.png",
        "XGBoost + SMOTE — validação temporal TRAIN fora da busca\nMesmo período e threshold 0,50; não é TEST",
    )
    plot_grid(results, fd / f"{prefix}_parameters.png")
    plot_iterations(iterations, fd / f"{prefix}_iterations.png")
    plot_association(
        association,
        fd / f"{prefix}_association_importance.png",
        "XGBoost otimizado + SMOTE",
    )
    save_figures(
        validation[TARGET].to_numpy(),
        probability,
        settings["threshold"],
        fd,
        calibration,
        prefix=prefix,
        label="XGBoost — validação TRAIN",
    )
    # Artefato EXPERIMENTAL: fit só no prefixo de busca, sem substituir RF ou avaliar TEST.
    artifact = models / (
        "xgboost_smote_gridsearch.joblib"
        if prefix == "xgboost_gridsearch"
        else f"{prefix}_pipeline.joblib"
    )
    best.memory = None
    joblib.dump(
        {
            "pipeline": best,
            "features": FEATURES,
            "selected_features": selected,
            "threshold": settings["threshold"],
            "model": "xgboost__smote_gridsearch"
            if prefix == "xgboost_gridsearch"
            else f"xgboost__{prefix}",
            "fit_scope": "Prefixo TRAIN da busca; sem refit final no TRAIN completo",
            "threshold_objective": "F1 em threshold fixo 0,50; sem tuning de limiar",
            "test_evaluated": False,
        },
        artifact,
    )
    manifest = {
        "created_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
        "config": config,
        "feature_settings": features,
        "settings": settings,
        **selection,
        "best_cv_f1": float(results.loc[best_index, "mean_test_f1"]),
        "cv_seconds": seconds,
        "combinations": len(results),
        "inner_folds": len(inner_folds),
        "grid_fits": len(results) * len(inner_folds),
        "executed_grid_fits": len(pending) * len(inner_folds),
        "reused_grid_fits": len(previous) * len(inner_folds),
        "reuse": reuse,
        "experiment_name": prefix,
        "train_rows": len(train),
        "search_fit_rows": len(fit),
        "validation_rows": len(validation),
        "validation_frauds": int(validation[TARGET].sum()),
        "search_fit_time_max": float(fit.Time.max()),
        "validation_time_min": float(validation.Time.min()),
        "validation_time_max": float(validation.Time.max()),
        "selected_features": selected,
        "split": split,
        "artifact_sha256": file_hash(artifact),
        "test_evaluated": False,
        "selected_random_forest_replaced": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost"]
        },
        "limitations": [
            "CV interna participou da seleção e pode ser otimista; não é estimativa externa ao tuning.",
            "Última janela TRAIN ficou fora desta busca, mas já foi explorada no histórico do projeto.",
            "TEST histórico já consultado; não foi reavaliado nesta busca. Generalização independente exige dados novos.",
            "Threshold 0,50 e razão SMOTE 0,1 fixos; não houve tuning de threshold ou calibração.",
            "Melhor combinação somente entre os valores da grade, sem garantia de ótimo global.",
            "Curvas por árvores são diagnósticas após congelar parâmetros, sem early stopping nesta rodada.",
            "O artefato experimental não foi reajustado no TRAIN completo nem substitui o RF escolhido.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(manifest, validation_table, output)
    write_model_statistics(output, prefix)
    return manifest


def write_report(manifest: dict, validation: pd.DataFrame, output: Path) -> None:
    prefix = manifest.get("experiment_name", "xgboost_gridsearch")
    lines = [
        "# XGBoost + SMOTE — GridSearchCV",
        "",
        f"Grade: {manifest['combinations']} combinações × {manifest['inner_folds']} folds temporais. Seleção por F1; threshold 0,50.",
        "",
        "Melhores parâmetros na grade: `" + json.dumps(manifest["best_params"]) + "`.",
        "",
        f"F1 médio da melhor combinação na CV usada na seleção: **{manifest['best_cv_f1']:.2%}**.",
        "",
        "## Validação temporal dentro do TRAIN, fora desta busca",
        "",
        "| Receita | F1 | Recall | Precision | AP | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation.itertuples():
        lines.append(
            f"| {row.model} | {row.f1:.2%} | {row.recall:.2%} | {row.precision:.2%} | {row.average_precision:.4f} | {row.tp} | {row.fp} | {row.fn} |"
        )
    lines += [
        "",
        "Esta janela já foi explorada em rodadas anteriores. Não é confirmação independente de todas as decisões do projeto. O TEST não foi pontuado; o Random Forest escolhido permanece preservado.",
        "",
        f"Notebook de treino: notebooks/xgboost_gridsearch.ipynb. Estatísticas: notebooks/xgboost_model_metrics.ipynb. Resultados completos: reports/metrics/{prefix}_results.csv.",
        "",
    ]
    (output / f"reports/{prefix}_results.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_statistics(output: Path = ROOT, prefix: str = "xgboost_learning_rate") -> dict:
    """Gera a apresentação do XGB otimizado lendo scores prontos; não treina ou prevê."""
    output = Path(output)
    md, fd = output / "reports/metrics", output / "reports/figures"
    validation = pd.read_csv(md / f"{prefix}_validation.csv")
    name = "XGBoost GridSearch + SMOTE"
    selected = validation.loc[validation.model.eq(name)].copy()
    if len(selected) != 1:
        raise ValueError("Relatório precisa de exatamente um XGBoost otimizado.")
    cv = pd.read_csv(md / f"{prefix}_cv_comparison.csv")
    plot_comparison(
        cv,
        fd / f"{prefix}_cv_f1_recall_precision.png",
        "XGBoost + SMOTE — CV interna TRAIN usada na seleção\nMédia ± desvio; threshold 0,50",
        True,
    )
    plot_comparison(
        validation,
        fd / f"{prefix}_validation_f1_recall_precision.png",
        "XGBoost + SMOTE — validação temporal TRAIN fora da busca\nMesmo período e threshold 0,50; não é TEST",
    )
    cv_selected = cv.loc[cv.model.eq(name)]
    folds = pd.read_csv(md / f"{prefix}_fold_metrics.csv")
    folds = folds.loc[folds.model.eq(name)].sort_values("fold")
    selected.to_csv(md / f"{prefix}_model_metrics.csv", index=False)
    result = selected.iloc[0].to_dict()
    (md / f"{prefix}_model_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_comparison(
        selected,
        fd / f"{prefix}_model_f1_recall_precision.png",
        "XGBoost otimizado + SMOTE — somente o modelo escolhido na grade\nValidação TRAIN fora da busca; threshold 0,50",
    )
    plot_comparison(
        cv_selected,
        fd / f"{prefix}_model_cv_f1_recall_precision.png",
        "XGBoost otimizado + SMOTE — CV interna usada na seleção\nMédia ± desvio; threshold 0,50",
        True,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(
        axes, ["f1", "recall", "precision"], ["F1", "Recall", "Precision"]
    ):
        ax.plot(folds.fold, 100 * folds[f"train_{metric}"], "--o", label="Treino interno")
        ax.plot(folds.fold, 100 * folds[metric], "-o", label="Validação interna")
        ax.set(xticks=folds.fold, xlabel="Fold temporal", ylabel=f"{label} (%)", ylim=(0, 100))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("XGBoost otimizado — treino/CV usados na seleção; não é learning curve controlada")
    fig.tight_layout()
    fig.savefig(fd / f"{prefix}_model_train_validation.png", dpi=160)
    plt.close(fig)
    grid = pd.read_csv(md / f"{prefix}_results.csv")
    if "param_model__learning_rate" in grid:
        profiles = grid.loc[grid.groupby("param_model__learning_rate").mean_test_f1.idxmax()].copy()
        profiles.to_csv(md / f"{prefix}_learning_rate_profiles.csv", index=False)
        profiles["model"] = profiles.param_model__learning_rate.map(
            lambda value: f"learning_rate={value:g}"
        )
        for metric in ["f1", "recall", "precision"]:
            profiles[metric] = profiles[f"mean_test_{metric}"]
            profiles[f"{metric}_std"] = profiles[f"std_test_{metric}"]
        plot_comparison(
            profiles,
            fd / f"{prefix}_learning_rate_f1_recall_precision.png",
            "Melhor combinação em cada learning_rate — CV interna TRAIN\nDemais parâmetros podem mudar; média ± desvio",
            True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--features-config", type=Path, default=ROOT / "configs/ablation.yaml")
    parser.add_argument(
        "--search-config", type=Path, default=ROOT / "configs/xgboost_gridsearch.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument(
        "--statistics-only",
        action="store_true",
        help="Regera apenas estatísticas salvas, sem treino ou predição.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.statistics_only:
        settings = load_config(args.search_config)
        write_model_statistics(args.output, settings.get("experiment_name", "xgboost_gridsearch"))
        return
    run(
        load_config(args.config),
        load_config(args.features_config),
        load_config(args.search_config),
        args.output,
    )


if __name__ == "__main__":
    main()
