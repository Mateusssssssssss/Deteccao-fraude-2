"""Avalia uma vez o pipeline congelado no TEST histórico, sem novo fit ou tuning."""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
)

from fraud_detection.data import ROOT, TARGET, read_split
from fraud_detection.segments import error_rates_by_bins
from fraud_detection.threshold import threshold_metrics

LOGGER = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    """Identifica exatamente os bytes avaliados sem carregar a base inteira na memória."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def ratio(numerator, denominator) -> float:
    """Denominador vazio é não definido, evitando confundir ausência de eventos com zero."""
    return float(numerator / denominator) if denominator else float("nan")


def complete_metrics(y, probability, threshold: float) -> dict:
    """Reúne decisões, ordenação e qualidade probabilística sob o mesmo limiar congelado."""
    result = threshold_metrics(y, probability, threshold)
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = [result[key] for key in ["tn", "fp", "fn", "tp"]]
    precision, recall, _ = precision_recall_curve(y, probability)
    result.update(
        {
            "accuracy": float(accuracy_score(y, predicted)),
            "negative_predictive_value": ratio(tn, tn + fn),
            "false_positive_rate": ratio(fp, fp + tn),
            "false_negative_rate": ratio(fn, fn + tp),
            "false_discovery_rate": ratio(fp, fp + tp),
            "false_omission_rate": ratio(fn, fn + tn),
            "prevalence": float(np.mean(y)),
            "mcc": float(matthews_corrcoef(y, predicted)),
            "cohen_kappa": float(cohen_kappa_score(y, predicted)),
            "pr_auc_trapezoid": float(auc(recall, precision)),
            "rows": len(y),
            "positives": int(np.sum(y)),
            "negatives": int(len(y) - np.sum(y)),
        }
    )
    return result


def metric_dictionary(result: dict) -> pd.DataFrame:
    """Tabela autocontida: cada valor informa nome, unidade, fórmula e direção desejável."""
    descriptions = [
        ("precision", "Precision", "TP/(TP+FP): fração dos alertas que era fraude", "%", "Maior"),
        (
            "recall",
            "Recall / sensibilidade",
            "TP/(TP+FN): fração das fraudes detectada",
            "%",
            "Maior",
        ),
        ("f1", "F1", "2PR/(P+R): média harmônica simétrica de Precision e Recall", "%", "Maior"),
        ("f2", "F2", "5PR/(4P+R): favorece Recall; objetivo usado no TRAIN", "%", "Maior"),
        (
            "specificity",
            "Especificidade",
            "TN/(TN+FP): legítimas corretamente liberadas",
            "%",
            "Maior",
        ),
        (
            "accuracy",
            "Acurácia",
            "(TP+TN)/n: acertos totais; pode enganar em classe rara",
            "%",
            "Maior",
        ),
        ("balanced_accuracy", "Balanced Accuracy", "(Recall+Especificidade)/2", "%", "Maior"),
        (
            "negative_predictive_value",
            "Valor preditivo negativo",
            "TN/(TN+FN): legítimas entre não alertadas",
            "%",
            "Maior",
        ),
        (
            "false_positive_rate",
            "Taxa de falsos positivos (FPR)",
            "FP/(FP+TN): legítimas alertadas",
            "%",
            "Menor",
        ),
        (
            "false_negative_rate",
            "Taxa de falsos negativos (FNR)",
            "FN/(FN+TP): fraudes não detectadas",
            "%",
            "Menor",
        ),
        (
            "false_discovery_rate",
            "Parcela de alertas falsos (FDR)",
            "FP/(FP+TP): alertas que eram legítimos",
            "%",
            "Menor",
        ),
        (
            "false_omission_rate",
            "Parcela de não alertadas que era fraude",
            "FN/(FN+TN): fraudes entre liberadas",
            "%",
            "Menor",
        ),
        (
            "alert_rate",
            "Taxa de alertas",
            "(TP+FP)/n: volume relativo de revisão/bloqueio",
            "%",
            "Depende do custo/capacidade",
        ),
        (
            "prevalence",
            "Prevalência de fraude",
            "(TP+FN)/n: frequência real de fraude na amostra avaliada",
            "%",
            "Descritiva",
        ),
        (
            "average_precision",
            "Average Precision (AP)",
            "Resumo de Precision ponderado pelos ganhos de Recall; usa scores",
            "score 0–1",
            "Maior",
        ),
        (
            "pr_auc_trapezoid",
            "Área trapezoidal PR",
            "Integral trapezoidal da curva Precision–Recall; difere de AP",
            "score 0–1",
            "Maior",
        ),
        (
            "roc_auc",
            "ROC-AUC",
            "Ordenação fraude vs. legítima; complemento à AP",
            "score 0–1",
            "Maior",
        ),
        (
            "log_loss",
            "Log Loss",
            "Perda probabilística; penaliza erros confiantes",
            "perda",
            "Menor",
        ),
        ("brier", "Brier score", "Média de (probabilidade−rótulo)²", "erro quadrático", "Menor"),
        (
            "mcc",
            "Coeficiente de Matthews (MCC)",
            "Associação entre decisões e rótulos usando TN/FP/FN/TP",
            "score −1–1",
            "Maior",
        ),
        (
            "cohen_kappa",
            "Kappa de Cohen",
            "Concordância corrigida pela concordância esperada ao acaso",
            "score −1–1",
            "Maior",
        ),
    ]
    rows = []
    for key, name, definition, unit, direction in descriptions:
        value = result[key] * (100 if unit == "%" else 1)
        rows.append(
            {
                "metric": key,
                "name": name,
                "value": value,
                "unit": unit,
                "definition": definition,
                "preferred": direction,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y, probability) -> pd.DataFrame:
    """Agrupa scores sem reajustar probabilidades; é diagnóstico posterior à avaliação."""
    frame = pd.DataFrame({"probability": probability, "fraud": np.asarray(y)})
    groups = pd.qcut(frame.probability, q=10, duplicates="drop")
    table = (
        frame.groupby(groups, observed=True)
        .agg(
            count=("fraud", "size"),
            fraud_count=("fraud", "sum"),
            predicted_probability=("probability", "mean"),
            observed_fraud_rate=("fraud", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
        .reset_index(names="score_bin")
    )
    table["score_bin"] = table.score_bin.astype(str)
    return table


def save_figures(
    y,
    probability,
    threshold: float,
    figures: Path,
    calibration,
    *,
    prefix: str = "final_test",
    label: str = "TEST histórico — Random Forest filtrada",
) -> None:
    """Curvas e distribuições só descrevem o modelo pronto; não escolhem outro threshold."""
    predicted = probability >= threshold
    tn, fp, fn, tp = np.bincount(2 * np.asarray(y, dtype=int) + predicted.astype(int), minlength=4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, normalize in zip(axes, [None, "true"]):
        ConfusionMatrixDisplay.from_predictions(
            y,
            predicted,
            normalize=normalize,
            ax=ax,
            colorbar=False,
            display_labels=["Legítima", "Fraude"],
            values_format="d" if normalize is None else ".2%",
        )
        ax.set(
            title="Contagens" if normalize is None else "Proporção por classe real",
            xlabel="Classe prevista",
            ylabel="Classe real",
        )
    fig.suptitle(f"{label}; threshold {threshold:g}")
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}_confusion.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[0], name=label)
    axes[0].axhline(np.mean(y), linestyle="--", color="gray", label="Prevalência")
    axes[0].scatter(
        ratio(tp, tp + fn),
        ratio(tp, tp + fp),
        color="red",
        zorder=5,
        label=f"Ponto do threshold {threshold:g}",
    )
    axes[0].set(
        xlabel="Recall", ylabel="Precision", title="Precision–Recall (resumo principal: AP)"
    )
    axes[0].legend(fontsize=8)
    RocCurveDisplay.from_predictions(y, probability, ax=axes[1], name=label)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].scatter(ratio(fp, fp + tn), ratio(tp, tp + fn), color="red", zorder=5)
    axes[1].set(xlabel="Taxa de falsos positivos (FPR)", ylabel="Recall (TPR)", title="ROC")
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}_pr_roc.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(calibration.predicted_probability, calibration.observed_fraud_rate, "o-")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Calibração ideal")
    axes[0].set(
        xlabel="Probabilidade prevista média",
        ylabel="Fração observada de fraude",
        title="Confiabilidade por faixas de score (sem recalibrar)",
    )
    axes[0].legend()
    for value, class_label in [(0, "Legítima"), (1, "Fraude")]:
        values = probability[np.asarray(y) == value]
        axes[1].hist(
            values,
            bins=np.linspace(0, 1, 31),
            weights=np.ones(len(values)) / len(values),
            alpha=0.6,
            label=class_label,
        )
    axes[1].axvline(threshold, linestyle="--", color="black", label="Threshold congelado")
    axes[1].set(
        xlabel="Score previsto de fraude",
        ylabel="Fração dentro de cada classe",
        title="Distribuição dos scores — normalização por classe",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / f"{prefix}_calibration_scores.png", dpi=160)
    plt.close(fig)


def run(
    artifact_path=ROOT / "models/standardized_selected_pipeline.joblib",
    selection_path=ROOT / "reports/experiments/ablation_selection.json",
    output=ROOT,
) -> dict:
    """Carrega o artefato congelado e preserva os resultados de avaliações anteriores."""
    artifact_path, selection_path, output = map(Path, [artifact_path, selection_path, output])
    md, ed, fd = [output / p for p in ["reports/metrics", "reports/experiments", "reports/figures"]]
    for folder in [md, ed, fd]:
        folder.mkdir(parents=True, exist_ok=True)
    result_path = md / "final_test_metrics.json"
    if result_path.exists():
        raise FileExistsError("Esta avaliação já existe; use --output para reprodução separada.")

    # O manifesto de seleção é anterior ao TEST: fixa modelo, threshold e features.
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    artifact = joblib.load(artifact_path)
    if (
        artifact["model"] != selection["model"]
        or artifact["threshold"] != selection["threshold"]
        or artifact["selected_features"] != selection["selected_features"]
    ):
        raise ValueError("Artefato e manifesto de seleção não correspondem.")
    train, test, split, source = read_split(selection["config"])
    if file_hash(source) != selection["dataset_sha256"]:
        raise ValueError("O dataset mudou desde a seleção; a avaliação foi interrompida.")
    threshold = float(artifact["threshold"])

    # Única chamada de predição no TEST; não há fit, SMOTE nem seleção de limiar aqui.
    probability = artifact["pipeline"].predict_proba(test[artifact["features"]])[:, 1]
    result = complete_metrics(test[TARGET], probability, threshold)
    calibration = calibration_table(test[TARGET], probability)
    save_figures(test[TARGET], probability, threshold, fd, calibration)
    metric_dictionary(result).to_csv(md / "final_test_metric_definitions.csv", index=False)
    calibration.to_csv(md / "final_test_calibration.csv", index=False)

    # Métricas por classe incluem suporte e médias macro/ponderada em arquivo próprio.
    predicted = probability >= threshold
    report = classification_report(
        test[TARGET],
        predicted,
        labels=[0, 1],
        target_names=["Legítima (0)", "Fraude (1)"],
        output_dict=True,
        zero_division=0,
    )
    (md / "final_test_classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Faixas das features são aprendidas no TRAIN e aplicadas ao TEST, sem vazamento.
    segments = error_rates_by_bins(train, test, probability, threshold, "TEST histórico")
    segments["source"] = "TEST histórico reavaliado; bins aprendidos no TRAIN"
    segments.to_csv(md / "final_test_feature_segments.csv", index=False)

    # Janelas temporais do próprio TEST servem só à descrição posterior dos erros.
    windows = pd.cut(test.Time, bins=4)
    temporal = []
    for window in windows.cat.categories:
        mask = (windows == window).to_numpy()
        actual, pred = test.loc[mask, TARGET].to_numpy(), predicted[mask]
        tn, fp, fn, tp = np.bincount(2 * actual.astype(int) + pred.astype(int), minlength=4)
        temporal.append(
            {
                "time_interval": str(window),
                "count": int(mask.sum()),
                "fraud_count": int(actual.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "recall_pct": 100 * ratio(tp, tp + fn),
                "precision_pct": 100 * ratio(tp, tp + fp),
                "false_positive_rate_pct": 100 * ratio(fp, fp + tn),
                "fraud_rate_pct": 100 * ratio(tp + fn, mask.sum()),
            }
        )
    pd.DataFrame(temporal).to_csv(md / "final_test_time_segments.csv", index=False)

    result.update(
        {
            "model": artifact["model"],
            "selected_features": artifact["selected_features"],
            "source": "TEST histórico — reavaliação retrospectiva após iterações",
            "independent_test": False,
        }
    )
    manifest = {
        "evaluated_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
        "artifact_file": artifact_path.name,
        "artifact_sha256": file_hash(artifact_path),
        "dataset_sha256": file_hash(source),
        "selection_sha256": file_hash(selection_path),
        "model": artifact["model"],
        "threshold": threshold,
        "selected_features": artifact["selected_features"],
        "split": split,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_time_min": float(test.Time.min()),
        "test_time_max": float(test.Time.max()),
        "independent_test": False,
        "training_performed": False,
        "threshold_tuning_performed": False,
        "limitations": [
            "TEST já avaliado na primeira rodada; resultado retrospectivo, não confirmação independente.",
            "Nenhum ajuste foi realizado em resposta a este resultado.",
            "Poucas fraudes e apenas dois dias; sem intervalos de confiança temporais robustos.",
            "SMOTE altera a distribuição de treino; scores não são probabilidades calibradas demonstradas.",
            "Subgrupos e faixas são diagnósticos posteriores, não motivos para retuning neste TEST.",
        ],
    }
    (ed / "final_test_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Grava o resultado só após completar todos os relatórios, evitando saída parcial travada.
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(result, output)
    LOGGER.info("TEST histórico: %s", result)
    return result


def write_report(result: dict, output: Path) -> None:
    lines = [
        "# TEST histórico — pipeline final filtrado",
        "",
        (
            f"Receita: **{result['model']}**. Threshold congelado: **{result['threshold']}**. "
            f"Features: **{len(result['selected_features'])}**."
        ),
        "",
        (
            "**Reavaliação retrospectiva:** esse período já foi visto na primeira rodada. "
            "O modelo foi avaliado sem refit, seleção de features ou ajuste de threshold."
        ),
        "",
        (
            f"Transações: {result['rows']}; fraudes: {result['positives']}; "
            f"legítimas: {result['negatives']}."
        ),
        "",
        "| Recall | Precision | F1 | F2 | AP | ROC-AUC |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {result['recall']:.2%} | {result['precision']:.2%} | {result['f1']:.2%} | "
            f"{result['f2']:.2%} | {result['average_precision']:.4f} | {result['roc_auc']:.4f} |"
        ),
        "",
        "| TN | FP | FN | TP |",
        "|---:|---:|---:|---:|",
        f"| {result['tn']} | {result['fp']} | {result['fn']} | {result['tp']} |",
        "",
        (
            "Definições, unidades e todas as métricas: reports/metrics/final_test_metric_definitions.csv. "
            "Notebook dedicado: notebooks/04_final_test_metrics.ipynb."
        ),
        "",
        "![Matriz](figures/final_test_confusion.png)",
        "",
        "![PR e ROC](figures/final_test_pr_roc.png)",
        "",
        (
            "Não há evidência de generalização prospectiva, calibração ou custo ótimo. "
            "É necessário um período externo novo para avaliação independente após as iterações."
        ),
        "",
    ]
    (output / "reports/final_test_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=ROOT / "models/standardized_selected_pipeline.joblib"
    )
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "reports/experiments/ablation_selection.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.artifact, args.selection, args.output)


if __name__ == "__main__":
    main()
