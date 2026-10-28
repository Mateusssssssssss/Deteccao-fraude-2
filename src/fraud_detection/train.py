"""Comparação em TRAIN, congelamento e avaliação única do vencedor em TEST."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.evaluation import comparison_plot, diagnostics, metrics
from fraud_detection.model import candidates
from fraud_detection.validation import SCORING, evaluate_cv, temporal_folds

LOGGER = logging.getLogger(__name__)


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run(config: dict, output: str | Path = ROOT) -> tuple[pd.DataFrame, dict]:
    if config["primary_metric"] != "average_precision":
        raise ValueError("A métrica primária deste protocolo é average_precision.")
    if config["threshold"] != 0.5:
        raise ValueError("Esta comparação usa threshold fixo 0.5 nos scorers e inferência.")
    output = Path(output)
    figures, metric_dir, experiments, models = [
        output / p for p in ["reports/figures", "reports/metrics", "reports/experiments", "models"]
    ]
    for folder in [figures, metric_dir, experiments, models]:
        folder.mkdir(parents=True, exist_ok=True)
    # Protege a avaliação já realizada contra sobrescrita silenciosa.
    if (metric_dir / "test_metrics.json").exists():
        raise FileExistsError(
            "TEST já avaliado nesta saída. Use --output para uma reprodução separada; "
            "repetições não tornam o mesmo TEST independente."
        )
    train, test, split_meta, source = read_split(config)
    X, y = train[FEATURES], train[TARGET]
    folds = temporal_folds(train, config)
    fold_manifest = []
    for i, (fit_idx, val_idx) in enumerate(folds):
        fold_manifest.append(
            {
                "fold": i + 1,
                "fit_rows": len(fit_idx),
                "validation_rows": len(val_idx),
                "fit_positive": int(y.iloc[fit_idx].sum()),
                "validation_positive": int(y.iloc[val_idx].sum()),
                "fit_time_max": float(X.iloc[fit_idx].Time.max()),
                "validation_time_min": float(X.iloc[val_idx].Time.min()),
                "validation_time_max": float(X.iloc[val_idx].Time.max()),
            }
        )
    pd.DataFrame(fold_manifest).to_csv(experiments / "fold_manifest.csv", index=False)
    # Comparação inicial de modelos: mesmo conjunto de folds e threshold 0,5.
    rows, fold_rows, predictions, fitted = [], [], {}, {}
    for name, pipeline in candidates(config).items():
        LOGGER.info("CV temporal: %s", name)
        result = evaluate_cv(pipeline, X, y, folds)
        row = {
            "model": name,
            "source": "CV temporal TRAIN",
            "threshold": 0.5,
            "configuration": json.dumps(pipeline.get_params()["model"].get_params(), default=str),
        }
        for metric in SCORING:
            row[metric] = float(np.mean(result[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(result[f"test_{metric}"], ddof=1))
        rows.append(row)
        oof = pd.Series(np.nan, index=train.index)
        for i, ((fit_idx, val_idx), estimator) in enumerate(zip(folds, result["estimator"])):
            probability = estimator.predict_proba(X.iloc[val_idx])[:, 1]
            oof.iloc[val_idx] = probability
            fold_rows.append(
                {
                    "model": name,
                    "fold": i + 1,
                    "source": "CV temporal TRAIN",
                    **metrics(y.iloc[val_idx], probability),
                    **{f"train_{m}": float(result[f"train_{m}"][i]) for m in SCORING},
                }
            )
        predictions[name] = oof
        fitted[name] = result["estimator"][-1]
    comparison = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    comparison.to_csv(metric_dir / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(metric_dir / "cv_folds.csv", index=False)
    comparison_plot(comparison, figures / "model_comparison_f1_recall_precision.png")
    # Baseline também participa: não forçar seleção de candidato pior que o Dummy.
    best = comparison.iloc[0].model
    selected = {
        "model": best,
        "selection": "maior AP média em CV temporal TRAIN",
        "threshold": 0.5,
        "config": config,
        "features": FEATURES,
    }
    dump_json(experiments / "selection.json", selected)  # antes de qualquer score TEST
    # OOF contém só previsões de janelas futuras ao fit; o prefixo inicial fica sem score.
    oof = pd.DataFrame(predictions)
    covered = oof.dropna().index
    oof_metrics = [
        {
            "model": name,
            "source": "OOF temporal TRAIN (agregado, cobertura parcial)",
            **metrics(y.loc[covered], oof.loc[covered, name]),
        }
        for name in predictions
    ]
    pd.DataFrame(oof_metrics).to_csv(metric_dir / "oof_metrics.csv", index=False)
    # Não publicar previsões individuais de transações.
    errors = {
        name: (oof.loc[covered, name].to_numpy() >= 0.5) != y.loc[covered].to_numpy()
        for name in predictions
    }
    names = [n for n in predictions if n != "dummy"]
    diversity = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            diversity.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "disagreement": float(np.mean(errors[a] != errors[b])),
                    "double_fault": float(np.mean(errors[a] & errors[b])),
                    "both_correct": float(np.mean(~errors[a] & ~errors[b])),
                    "only_a_wrong": float(np.mean(errors[a] & ~errors[b])),
                    "only_b_wrong": float(np.mean(~errors[a] & errors[b])),
                }
            )
    pd.DataFrame(diversity).to_csv(metric_dir / "error_diversity.csv", index=False)
    diagnostics(y.loc[covered], oof.loc[covered, best].to_numpy(), 0.5, figures, "OOF_TRAIN")
    val_idx = folds[-1][1]
    estimator = fitted[best]
    LOGGER.info("Importância por permutação na última janela de validação do TRAIN")
    importance = permutation_importance(
        estimator,
        X.iloc[val_idx],
        y.iloc[val_idx],
        scoring="average_precision",
        n_repeats=3,
        random_state=config["seed"],
        n_jobs=1,
    )
    association = pd.DataFrame(
        {
            "feature": FEATURES,
            "pearson": [X[c].corr(y, method="pearson") for c in FEATURES],
            "spearman": [X[c].corr(y, method="spearman") for c in FEATURES],
            "permutation_ap": importance.importances_mean,
            "permutation_std": importance.importances_std,
            "tree_importance": getattr(
                estimator.named_steps["model"],
                "feature_importances_",
                np.full(len(FEATURES), np.nan),
            ),
        }
    )
    association.to_csv(metric_dir / "association_importance.csv", index=False)
    plot = association.sort_values("permutation_ap", ascending=False).head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    axes[0].barh(plot.feature, plot.spearman)
    axes[0].set_title("Spearman — TRAIN completo")
    axes[0].set_xlabel("Associação monotônica (com sinal)")
    axes[1].barh(plot.feature, plot.permutation_ap, xerr=plot.permutation_std)
    axes[1].set_title("Permutação — última validação TRAIN")
    axes[1].set_xlabel("Queda de AP (escala distinta)")
    fig.tight_layout()
    fig.savefig(figures / "association_importance.png", dpi=160)
    plt.close(fig)
    # Curvas diagnósticas de árvores: números fixados antes de olhar TEST.
    iteration_rows = []
    for name in [n for n in names if n.startswith("xgboost")]:
        pipe = fitted[name]
        model = pipe.named_steps["model"]
        for stage in sorted({10, 25, 50, 100, config["models"]["xgboost"]["n_estimators"]}):
            if stage > config["models"]["xgboost"]["n_estimators"]:
                continue
            row = {"model": name, "trees": stage}
            for label, idx in [("TRAIN_fold", folds[-1][0]), ("validation_TRAIN", val_idx)]:
                transformed = pipe.named_steps["preprocess"].transform(X.iloc[idx])
                p = model.predict_proba(transformed, iteration_range=(0, stage))[:, 1]
                row[label] = metrics(y.iloc[idx], p)["average_precision"]
            iteration_rows.append(row)
    if iteration_rows:
        iterations = pd.DataFrame(iteration_rows)
        iterations.to_csv(metric_dir / "boosting_iterations.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, part in iterations.groupby("model"):
            for label in ["TRAIN_fold", "validation_TRAIN"]:
                ax.plot(part.trees, part[label], marker="o", label=f"{name} / {label}")
        ax.set(xlabel="Árvores", ylabel="AP", title="Diagnóstico; sem reajuste após seleção")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "boosting_iterations.png", dpi=160)
        plt.close(fig)
    # Comparação treino/CV por tamanho das janelas (tempo e tamanho estão confundidos).
    scores = pd.DataFrame(fold_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        part = scores[scores.model == name]
        ax.plot(range(1, len(part) + 1), part.average_precision, "o-", label=f"{name} validação")
        ax.plot(
            range(1, len(part) + 1),
            part.train_average_precision,
            "--",
            alpha=0.6,
            label=f"{name} treino",
        )
    ax.set(
        xlabel="Fold expansivo (treino crescente, período diferente)",
        ylabel="AP",
        title="Generalização temporal — não é learning curve controlada",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "train_validation.png", dpi=160)
    plt.close(fig)
    LOGGER.info("Seleção congelada: %s. Refit no TRAIN completo.", best)
    final = candidates(config)[best].fit(X, y)
    joblib.dump(
        {"pipeline": final, "threshold": 0.5, "features": FEATURES, "model": best},
        models / "final_pipeline.joblib",
    )
    # TEST só é usado para avaliar o pipeline já congelado.
    probability = final.predict_proba(test[FEATURES])[:, 1]
    final_metrics = metrics(test[TARGET], probability)
    dump_json(
        metric_dir / "test_metrics.json",
        {
            "model": best,
            "source": "TEST temporal final",
            "rows": len(test),
            "positives": int(test[TARGET].sum()),
            **final_metrics,
        },
    )
    diagnostics(test[TARGET], probability, 0.5, figures, "TEST_final")
    bins = pd.cut(test.Time, bins=4, duplicates="drop")
    segment_rows = []
    for label in bins.cat.categories:
        mask = (bins == label).to_numpy()
        if test.loc[mask, TARGET].nunique() == 2:
            segment_rows.append(
                {
                    "time_interval": str(label),
                    "rows": int(mask.sum()),
                    **metrics(test.loc[mask, TARGET], probability[mask]),
                }
            )
    pd.DataFrame(segment_rows).to_csv(metric_dir / "test_time_segments.csv", index=False)
    versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "matplotlib"]
    }
    dump_json(
        experiments / "run_manifest.json",
        {
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset_file": source.name,
            "config": config,
            "versions": versions,
            **split_meta,
            "train_rows": len(train),
            "train_positives": int(y.sum()),
            "test_rows": len(test),
            "test_positives": int(test[TARGET].sum()),
            "oof_rows": len(covered),
            "oof_coverage": len(covered) / len(train),
            "positive_class": 1,
            "limitations": [
                "Apenas dois dias; dependência temporal e intervalos não quantificados.",
                "PCA fornecido pronto; ajuste original e rótulos não auditáveis.",
                "Sem identificador de cartão, não se garante independência por cliente.",
                "TEST histórico já explorado pelo projeto antigo; novo protocolo é retrospectivo.",
                "AP não é área trapezoidal sob PR. Threshold fixo, sem calibração.",
            ],
        },
    )
    LOGGER.info("TEST final: %s", final_metrics)
    return comparison, final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(load_config(args.config), args.output)


if __name__ == "__main__":
    main()
