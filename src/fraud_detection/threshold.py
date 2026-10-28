"""GridSearchCV de threshold com F2 e validação temporal externa, só no TRAIN."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import FixedThresholdClassifier, GridSearchCV

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split, temporal_split
from fraud_detection.evaluation import metrics
from fraud_detection.model import candidates
from fraud_detection.validation import temporal_folds

LOGGER = logging.getLogger(__name__)


def alert_rate(y_true, y_pred) -> float:
    """Fração prevista positiva, para explicitar o custo em volume de revisão."""
    return float(np.mean(np.asarray(y_pred) == 1))


def select_f2_index(results: dict) -> int:
    """Maior F2; desempata por Recall, Precision e maior threshold, nesta ordem."""
    return int(
        np.lexsort(
            (
                np.asarray(results["param_threshold"], dtype=float),
                results["mean_test_precision"],
                results["mean_test_recall"],
                results["mean_test_f2"],
            )
        )[-1]
    )


def threshold_metrics(y, probability, threshold: float) -> dict:
    result = metrics(y, probability, threshold)
    result["f2"] = float(
        fbeta_score(y, np.asarray(probability) >= threshold, beta=2, zero_division=0)
    )
    result["alerts"] = result["tp"] + result["fp"]
    result["alert_rate"] = result["alerts"] / len(y)
    return result


def search_threshold(pipeline, train: pd.DataFrame, settings: dict):
    """Ajusta pipeline no prefixo; GridSearch vê somente a janela interna ao pontuar.

    FrozenEstimator impede refits na varredura. A janela interna não foi vista pelo
    modelo. Refit do GridSearch refaz apenas o wrapper, mantendo o pipeline congelado.
    O chamador faz o refit do pipeline no TRAIN apropriado depois de escolher o limiar.
    """
    # A grade inclui extremos para mostrar o custo de maximizar apenas Recall.
    thresholds = np.asarray(settings["thresholds"], dtype=float)
    if (
        thresholds.size == 0
        or not np.isfinite(thresholds).all()
        or ((thresholds < 0) | (thresholds > 1)).any()
    ):
        raise ValueError("Thresholds devem ser uma grade não vazia entre 0 e 1.")
    if settings["objective"] != "f2" or settings["beta"] != 2:
        raise ValueError("Este protocolo otimiza F2 (beta=2), conforme escolhido.")
    fit, tune, meta = temporal_split(train, settings["inner_validation_time_fraction"])
    # Modelo aprende no prefixo; a janela seguinte só escolhe o threshold.
    estimator = clone(pipeline).fit(fit[FEATURES], fit[TARGET])
    combined = pd.concat([fit, tune], ignore_index=True)
    # O wrapper expõe threshold como parâmetro; FrozenEstimator impede novo fit do modelo.
    search = GridSearchCV(
        FixedThresholdClassifier(
            FrozenEstimator(estimator), response_method="predict_proba", pos_label=1
        ),
        param_grid={"threshold": thresholds.tolist()},
        scoring={
            "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "precision": make_scorer(precision_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
            "alert_rate": make_scorer(alert_rate),
        },
        cv=[(np.arange(len(fit)), np.arange(len(fit), len(combined)))],
        refit=select_f2_index,
        error_score="raise",
        n_jobs=1,
    )
    search.fit(combined[FEATURES], combined[TARGET])
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            **{
                metric: search.cv_results_[f"mean_test_{metric}"]
                for metric in ["f2", "recall", "precision", "f1", "alert_rate"]
            },
        }
    )
    # Converte proporções de validação em contagens para tornar o volume de alertas visível.
    table["alerts"] = np.rint(table.alert_rate * len(tune)).astype(int)
    table["tp"] = np.rint(table.recall * int(tune[TARGET].sum())).astype(int)
    table["fp"] = table.alerts - table.tp
    table["fn"] = int(tune[TARGET].sum()) - table.tp
    meta.update(
        {
            "inner_fit_rows": len(fit),
            "inner_tune_rows": len(tune),
            "inner_fit_time_max": float(fit.Time.max()),
            "inner_tune_time_min": float(tune.Time.min()),
            "inner_tune_time_max": float(tune.Time.max()),
            "inner_tune_positives": int(tune[TARGET].sum()),
        }
    )
    return search, table, meta


def run(config: dict, settings: dict, output: str | Path = ROOT) -> dict:
    output = Path(output)
    metric_dir, experiment_dir, figure_dir, model_dir = [
        output / p for p in ["reports/metrics", "reports/experiments", "reports/figures", "models"]
    ]
    for folder in [metric_dir, experiment_dir, figure_dir, model_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    selection_path = experiment_dir / "threshold_selection.json"
    if selection_path.exists():
        raise FileExistsError("Busca já executada nesta saída; use --output para reprodução.")
    train, _, split_meta, source = read_split(config)
    if settings["model"] != "xgboost__none":
        raise ValueError("Receita fixa desta rodada: xgboost__none, sem nova seleção de família.")
    pipeline = candidates(config)[settings["model"]]
    # Fold externo mede a política inteira: subtreino -> threshold interno -> refit.
    fold_config = dict(config, sampling_strategies=["none"])
    folds = temporal_folds(train, fold_config)
    rows, grids, manifests = [], [], []
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        LOGGER.info("Threshold: fold externo %s/%s", fold, len(folds))
        outer_train, validation = train.iloc[fit_idx], train.iloc[val_idx]
        search, grid, meta = search_threshold(pipeline, outer_train, settings)
        threshold = float(search.best_params_["threshold"])
        estimator = clone(pipeline).fit(outer_train[FEATURES], outer_train[TARGET])
        probability = estimator.predict_proba(validation[FEATURES])[:, 1]
        for policy, value in [("fixed_0.5", 0.5), ("gridsearch_f2", threshold)]:
            rows.append(
                {
                    "fold": fold,
                    "model": settings["model"],
                    "policy": policy,
                    "source": "CV temporal externa TRAIN; threshold ajustado internamente",
                    **threshold_metrics(validation[TARGET], probability, value),
                }
            )
        grids.append(grid.assign(outer_fold=fold, source="janela interna TRAIN"))
        manifests.append(
            {
                "outer_fold": fold,
                "threshold": threshold,
                **meta,
                "outer_fit_rows": len(outer_train),
                "outer_validation_rows": len(validation),
                "outer_fit_time_max": float(outer_train.Time.max()),
                "outer_validation_time_min": float(validation.Time.min()),
                "outer_validation_time_max": float(validation.Time.max()),
            }
        )
    scores = pd.DataFrame(rows)
    scores.to_csv(metric_dir / "threshold_outer_folds.csv", index=False)
    summary = []
    for policy, part in scores.groupby("policy", sort=False):
        row = {"model": settings["model"], "policy": policy, "source": "CV externa temporal TRAIN"}
        for metric in [
            "f2",
            "recall",
            "precision",
            "f1",
            "average_precision",
            "alert_rate",
            "fp",
            "fn",
        ]:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(metric_dir / "threshold_comparison.csv", index=False)
    pd.concat(grids).to_csv(metric_dir / "threshold_inner_grids.csv", index=False)
    pd.DataFrame(manifests).to_csv(experiment_dir / "threshold_folds.csv", index=False)
    LOGGER.info("GridSearch final no TRAIN; sem avaliar o TEST.")
    search, grid, meta = search_threshold(pipeline, train, settings)
    grid.to_csv(metric_dir / "threshold_grid.csv", index=False)
    threshold = float(search.best_params_["threshold"])
    recall_best = grid.sort_values(["recall", "precision", "threshold"], ascending=False).iloc[0]
    result = {
        "model": settings["model"],
        "threshold": threshold,
        "objective": "F2",
        "positive_class": 1,
        "settings": settings,
        "experiment_config": config,
        "grid_size": len(grid),
        "selected_inner_scores": grid.iloc[search.best_index_].to_dict(),
        "recall_only_reference": recall_best.to_dict(),
        "inner_split": meta,
        "train_test_split": split_meta,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_evaluated": False,
        "limitations": [
            "Melhor threshold somente na grade e janela consideradas; não ótimo universal.",
            "CV externa valida a política de threshold; o limiar final é reajustado no TRAIN.",
            "Receita XGBoost já selecionada antes com os mesmos dados: diagnóstico condicionado à seleção.",
            "TEST histórico já visto e não reutilizado nesta rodada; falta avaliação externa nova.",
            "F2 é preferência de métrica, não equivale a custo financeiro conhecido.",
            "Refit no TRAIN completo pode mudar a distribuição dos scores; probabilidades não calibradas.",
        ],
    }
    selection_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    final = clone(pipeline).fit(train[FEATURES], train[TARGET])
    joblib.dump(
        {
            "pipeline": final,
            "threshold": threshold,
            "features": FEATURES,
            "model": settings["model"],
            "threshold_objective": "F2",
            "test_evaluated": False,
        },
        model_dir / "recall_pipeline.joblib",
    )
    plot_results(grid, summary, threshold, figure_dir)
    write_report(result, summary, output)
    LOGGER.info("Threshold F2 selecionado na grade: %.6f", threshold)
    return result


def plot_results(grid, summary, threshold: float, folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric in ["f2", "recall", "precision"]:
        axes[0].plot(grid.threshold, grid[metric], marker=".", label=metric)
    axes[0].axvline(threshold, color="gray", linestyle="--", label=f"Escolhido: {threshold:g}")
    axes[0].set(
        xlabel="Threshold",
        ylabel="Score",
        ylim=(0, 1),
        title="Janela interna final TRAIN — usada para seleção",
    )
    axes[0].legend()
    x = np.arange(len(summary))
    for i, metric in enumerate(["f2", "recall", "precision"]):
        axes[1].bar(
            x + (i - 1) * 0.25,
            summary[metric],
            0.25,
            yerr=summary[f"{metric}_std"],
            label=metric,
            capsize=3,
        )
    axes[1].set(
        xticks=x,
        xticklabels=summary.policy,
        ylim=(0, 1),
        title="CV temporal externa TRAIN — média ± desvio",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(folder / "threshold_recall_f2.png", dpi=160)
    plt.close(fig)


def write_report(selection: dict, summary: pd.DataFrame, output: Path) -> None:
    lines = [
        "# GridSearch de threshold com ênfase em Recall",
        "",
        (
            f"Receita fixa: {selection['model']}. Melhor threshold F2 na grade: "
            f"**{selection['threshold']:g}**. Foram pesquisados {selection['grid_size']} limiares."
        ),
        "",
        (
            "F2 (beta=2) prioriza Recall, mas penaliza falsos positivos. GridSearchCV varia "
            "threshold no FixedThresholdClassifier; FrozenEstimator preserva o modelo "
            "ajustado no prefixo, sem treinar na janela que escolhe o limiar."
        ),
        "",
        "## Avaliação da política — CV temporal externa somente TRAIN",
        "",
        "| Política | F2 média | Recall médio | Precision média | FP médios/fold | FN médios/fold |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.policy} | {row.f2:.4f} | {row.recall:.4f} | "
            f"{row.precision:.4f} | {row.fp:.1f} | {row.fn:.1f} |"
        )
    lines += [
        "",
        (
            "Cada fold escolhe seu threshold numa janela interna anterior à validação externa. "
            "O limiar final é escolhido novamente na janela interna do TRAIN completo; "
            "seu score de seleção não é avaliação independente."
        ),
        "",
        (
            "A receita já foi selecionada por AP na rodada anterior; a avaliação aqui é "
            "condicionada a essa escolha. Não é uma nova avaliação independente de toda a seleção."
        ),
        "",
        "![Curvas e comparação](figures/threshold_recall_f2.png)",
        "",
        "## Recall puro — referência, não política adotada",
        "",
        (
            f"Na janela interna final, maximizar só Recall escolheria threshold "
            f"{selection['recall_only_reference']['threshold']:g}, Recall "
            f"{selection['recall_only_reference']['recall']:.4f}, Precision "
            f"{selection['recall_only_reference']['precision']:.4f} e "
            f"{int(selection['recall_only_reference']['alerts'])} alertas."
        ),
        "",
        "## Artefato e limites",
        "",
        (
            "Novo artefato local: models/recall_pipeline.joblib, com refit no TRAIN completo. "
            "O pipeline anterior e os resultados históricos de TEST não foram alterados. "
            "**TEST não foi avaliado novamente.** É necessário outro período externo para "
            "avaliar esta nova política de forma independente. Melhor na grade não significa "
            "ótimo universal, custo ótimo ou probabilidade calibrada."
        ),
        "",
    ]
    (output / "reports/threshold_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--threshold-config", type=Path, default=ROOT / "configs/threshold.yaml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(load_config(args.config), load_config(args.threshold_config), args.output)


if __name__ == "__main__":
    main()
