"""Diagnóstico do Random Forest com SMOTE escolhido, sem pontuar o TEST."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split
from fraud_detection.final_test import (
    calibration_table,
    complete_metrics,
    file_hash,
    metric_dictionary,
    save_figures,
)
from fraud_detection.sampling_comparison import make_pipeline, plot_association
from fraud_detection.segments import error_rates_by_bins
from fraud_detection.validation import temporal_folds

LOGGER = logging.getLogger(__name__)
COLORS = ["#3578b8", "#e49032", "#37936b"]
SCORES = [("f1", "F1-score"), ("recall", "Recall"), ("precision", "Precision")]


def verify_reference(actual: dict, reference: dict) -> None:
    """Interrompe se a reconstrução divergir das métricas da execução original."""
    for key, value in actual.items():
        if key in reference and not np.isclose(value, reference[key], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Reconstrução divergiu da referência: {key}.")


def plot_metrics(result: dict, path: Path, title: str) -> None:
    """Três barras com valores percentuais; somente o modelo escolhido."""
    fig, ax = plt.subplots(figsize=(9, 6))
    values = [100 * result[key] for key, _ in SCORES]
    bars = ax.bar([label for _, label in SCORES], values, color=COLORS, width=0.6)
    ax.bar_label(
        bars,
        labels=[f"{value:.2f}%".replace(".", ",") for value in values],
        label_type="center",
        color="white",
        fontweight="bold",
        fontsize=12,
    )
    ax.set(ylim=(0, 100), ylabel="Score (%)", title=title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_train_validation(folds: pd.DataFrame, path: Path, threshold: float) -> None:
    """Mostra estabilidade e diferença treino/validação do mesmo RF nos três folds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, label), color in zip(axes, SCORES, COLORS):
        ax.plot(folds.fold, 100 * folds[f"train_{metric}"], "--o", label="Treino", color="gray")
        ax.plot(folds.fold, 100 * folds[metric], "-o", label="Validação", color=color)
        ax.set(xticks=folds.fold, xlabel="Fold temporal", ylabel=f"{label} (%)", ylim=(0, 100))
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(f"Random Forest + SMOTE — threshold {threshold:g}; períodos e treino variam")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_thresholds(table: pd.DataFrame, threshold: float, path: Path) -> None:
    """Repete o diagnóstico de limiares só para RF, marcando a escolha do usuário."""
    table = table.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(table))
    selected = np.flatnonzero(np.isclose(table.threshold, threshold))
    if len(selected):
        ax.axvspan(selected[0] - 0.45, selected[0] + 0.45, color="#e5e5e5", zorder=0)
    for i, ((metric, label), color) in enumerate(zip(SCORES, COLORS)):
        bars = ax.bar(x + (i - 1) * 0.25, 100 * table[metric], 0.25, label=label, color=color)
        for bar in bars:
            value = bar.get_height()
            inside = value >= 15
            ax.annotate(
                f"{value:.2f}%".replace(".", ","),
                (bar.get_x() + bar.get_width() / 2, value / 2 if inside else value),
                xytext=(0, 0 if inside else 3),
                textcoords="offset points",
                ha="center",
                va="center" if inside else "bottom",
                rotation=90,
                color="white" if inside else "#222222",
                fontsize=8,
                fontweight="bold",
            )
    ax.set(
        xticks=x,
        xticklabels=[f"{value:g}".replace(".", ",") for value in table.threshold],
        xlabel="Threshold (escolhido: 0,70, faixa cinza)",
        ylabel="Score (%)",
        ylim=(0, 100),
        title="Random Forest + SMOTE — última validação TRAIN; análise após inspeção",
    )
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root: Path = ROOT, choice: dict | None = None) -> dict:
    """Reconstrói apenas folds de TRAIN e verifica o experimento antes de gerar saídas."""
    root = Path(root)
    choice = load_config(root / "configs/model_select.yaml") if choice is None else choice
    if choice["model"] != "random_forest__smote" or choice["primary_metric"] != "f1":
        raise ValueError("Este relatório descreve somente Random Forest com SMOTE e F1.")
    threshold = float(choice["threshold"])
    md, ed, fd = [
        root / folder for folder in ("reports/metrics", "reports/experiments", "reports/figures")
    ]
    manifest = json.loads((ed / "sampling_comparison_manifest.json").read_text(encoding="utf-8"))
    config, settings = manifest["config"], manifest["settings"]
    # O teste é reservado pelo protocolo original e descartado sem gerar previsões.
    train, _, _, source = read_split(config)
    if file_hash(source) != manifest["dataset_sha256"]:
        raise ValueError("Dataset diferente da execução original.")
    folds = temporal_folds(train, dict(config, sampling_strategies=["none"]))
    if len(folds) != choice["reference_fold"]:
        raise ValueError("O fold de referência deve ser o último fold original.")
    audit = pd.read_csv(md / "sampling_feature_selection.csv")
    reference = pd.read_csv(md / "sampling_folds.csv")
    reference = reference.loc[reference.model.eq(choice["model"])].set_index("fold")
    sensitivity = pd.read_csv(root / choice["reference_source"])
    sensitivity = sensitivity.loc[sensitivity.model.eq(choice["model"])].copy()
    point = sensitivity.loc[np.isclose(sensitivity.threshold, threshold)]
    if len(point) != 1:
        raise ValueError("Threshold escolhido não existe no diagnóstico original.")
    rows, probabilities, actuals = [], [], []
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
        fold_audit = audit.loc[audit.fold.eq(fold)]
        # Reutiliza seleção aprendida dentro do treino do fold, sem novas decisões.
        if fold_audit.empty or fold_audit.selection_input_time_max.max() > fit.Time.max():
            raise ValueError("Auditoria de features inválida para o período de treino.")
        selected = fold_audit.loc[fold_audit.keep.eq(True), "feature"].tolist()
        LOGGER.info("Reconstruindo somente Random Forest + SMOTE: fold %s/%s", fold, len(folds))
        pipeline = make_pipeline("random_forest", "smote", selected, config, settings)
        pipeline.fit(fit[FEATURES], fit[TARGET])
        probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
        # Confere as decisões em 0,50 antes de aplicar o limiar escolhido de 0,70.
        verify_reference(
            complete_metrics(validation[TARGET], probability, 0.5), reference.loc[fold].to_dict()
        )
        score = complete_metrics(validation[TARGET], probability, threshold)
        train_score = complete_metrics(
            fit[TARGET], pipeline.predict_proba(fit[FEATURES])[:, 1], threshold
        )
        rows.append(
            {
                "fold": fold,
                "source": "Validação temporal TRAIN; threshold escolhido após inspeção",
                "n_features": len(selected),
                **{k: v for k, v in score.items() if k != "f2"},
                **{
                    f"train_{k}": train_score[k]
                    for k in ("f1", "recall", "precision", "average_precision")
                },
            }
        )
        probabilities.append(probability)
        actuals.append(validation[TARGET].to_numpy())
    # O último fold deve reproduzir exatamente os 89,19% / 75% já observados.
    verify_reference(score, point.iloc[0].to_dict())
    fold_scores = pd.DataFrame(rows)
    oof = complete_metrics(np.concatenate(actuals), np.concatenate(probabilities), threshold)
    definitions = metric_dictionary(score).loc[lambda table: table.metric.ne("f2")]
    calibration = calibration_table(validation[TARGET], probability)
    segments = error_rates_by_bins(
        fit, validation, probability, threshold, "Última validação TRAIN"
    )
    segments["source"] = "Última validação TRAIN; threshold 0,70; bins aprendidos no treino do fold"
    association = pd.read_csv(md / "smote_association_importance.csv")
    association = association.loc[association.model.eq(choice["model"])]
    counts = pd.read_csv(md / "sampling_counts.csv")
    counts = counts.loc[counts.model.eq(choice["model"])]
    result = {
        **{k: v for k, v in score.items() if k != "f2"},
        "model": choice["model"],
        "source": "Última validação temporal do TRAIN (fold 3), não TEST nem média CV",
        "selected_features": selected,
        "primary_metric": "f1",
        "independent_validation": False,
    }
    # Todos os relatórios abaixo descrevem apenas o RF escolhido; nenhuma tabela de candidatos.
    fold_scores.to_csv(md / "model_select_folds.csv", index=False)
    definitions.to_csv(md / "model_select_metric_definitions.csv", index=False)
    pd.DataFrame([{k: v for k, v in oof.items() if k != "f2"}]).to_csv(
        md / "model_select_oof.csv", index=False
    )
    calibration.to_csv(md / "model_select_calibration.csv", index=False)
    segments.to_csv(md / "model_select_category_errors.csv", index=False)
    sensitivity.to_csv(md / "model_select_thresholds.csv", index=False)
    association.to_csv(md / "model_select_association_importance.csv", index=False)
    counts.to_csv(md / "model_select_sampling_counts.csv", index=False)
    report = classification_report(
        validation[TARGET],
        probability >= threshold,
        labels=[0, 1],
        target_names=["Legítima (0)", "Fraude (1)"],
        output_dict=True,
        zero_division=0,
    )
    (md / "model_select_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_metrics(
        result,
        fd / "model_select_f1_recall_precision.png",
        "Random Forest + SMOTE — threshold 0,70\nÚltima validação temporal TRAIN (fold 3)",
    )
    plot_metrics(
        oof,
        fd / "model_select_oof_f1_recall_precision.png",
        "Random Forest + SMOTE — threshold 0,70\nOOF temporal TRAIN agregado; limiar escolhido após inspeção",
    )
    plot_train_validation(fold_scores, fd / "model_select_train_validation.png", threshold)
    plot_thresholds(sensitivity, threshold, fd / "model_select_thresholds.png")
    plot_association(
        association, fd / "model_select_association_importance.png", "Random Forest + SMOTE"
    )
    save_figures(
        validation[TARGET].to_numpy(),
        probability,
        threshold,
        fd,
        calibration,
        prefix="model_select",
        label="Última validação TRAIN — RF + SMOTE",
    )
    confirmation = {
        "generated_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
        "choice": choice,
        "dataset_sha256": file_hash(source),
        "source_manifest_sha256": file_hash(ed / "sampling_comparison_manifest.json"),
        "source_metrics_sha256": file_hash(root / choice["reference_source"]),
        "selected_features": selected,
        "model_parameters": pipeline.named_steps["model"].get_params(),
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "validation_frauds": int(validation[TARGET].sum()),
        "validation_time_min": float(validation.Time.min()),
        "validation_time_max": float(validation.Time.max()),
        "reconstructed_folds": len(folds),
        "original_metrics_reproduced": True,
        "test_evaluated": False,
        "historical_model_replaced": False,
        "final_refit_performed": False,
        "oof_rows": len(np.concatenate(actuals)),
        "versions": manifest["versions"],
        "limitations": [
            "89,19% de Precision e 75% de Recall pertencem somente ao último fold TRAIN.",
            "Modelo e threshold escolhidos após observar os folds; resultado exploratório, sem validação externa à escolha.",
            "OOF agregado difere da média entre folds; o prefixo inicial só treina.",
            "Treino cresce e períodos mudam; gráfico por fold não é learning curve controlada.",
            "Importâncias e erros por faixas são diagnósticos, sem causalidade ou novo feature engineering.",
            "Dados cobrem dois dias; scores não foram calibrados; confirmação independente exige dados novos.",
        ],
    }
    (ed / "model_select_manifest.json").write_text(
        json.dumps(confirmation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (md / "model_select_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info(
        "Modelo escolhido: Precision %.2f%%, Recall %.2f%%, F1 %.2f%% (última validação TRAIN)",
        100 * result["precision"],
        100 * result["recall"],
        100 * result["f1"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.root)


if __name__ == "__main__":
    main()
