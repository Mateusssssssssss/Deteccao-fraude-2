"""Gera um resumo Markdown a partir dos artefatos; não avalia modelos."""

import json

import pandas as pd

from fraud_detection.data import ROOT


def main() -> None:
    metrics = ROOT / "reports/metrics"
    comparison = pd.read_csv(metrics / "model_comparison.csv")
    test = json.loads((metrics / "test_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "reports/experiments/run_manifest.json").read_text(encoding="utf-8")
    )
    lines = [
        "# Resultados da reformulação",
        "",
        (
            "Resultados calculados com o protocolo de configs/experiment.yaml. "
            "Não comparar diretamente com os números históricos com leakage."
        ),
        "",
        (
            f"TRAIN: {manifest['train_rows']:,} transações, {manifest['train_positives']} fraudes. "
            f"TEST: {manifest['test_rows']:,} transações, {manifest['test_positives']} fraudes. "
            f"Cópias exatas removidas: {manifest['duplicates_removed']}."
        ),
        "",
        "## Seleção — CV temporal somente no TRAIN",
        "",
        (
            "Classe positiva: fraude (1). Threshold 0,5, três janelas expansivas. "
            "Valores: média ± desvio-padrão entre folds. Ordenação por AP."
        ),
        "",
        "| Receita | AP | F1 | Recall | Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        scores = [
            f"{getattr(row, metric):.4f} ± {getattr(row, metric + '_std'):.4f}"
            for metric in ["average_precision", "f1", "recall", "precision"]
        ]
        lines.append("| " + row.model + " | " + " | ".join(scores) + " |")
    lines += [
        "",
        (
            f"Selecionado: **{test['model']}** pela AP média de CV. "
            "A diferença para Random Forest sem reamostragem é pequena em relação à "
            "dispersão entre folds; não foi demonstrada superioridade estatística."
        ),
        "",
        (
            "Nesta rodada, under+SMOTE aumentou Recall médio, mas reduziu Precision "
            "e F1 e não superou a melhor AP sem reamostragem. O balanceamento não "
            "é obrigatório para estes dados/modelos. RF sem reamostragem usa pesos "
            "de classe, de modo que esta é uma comparação entre receitas completas."
        ),
        "",
        "![Comparação CV](figures/model_comparison_f1_recall_precision.png)",
        "",
        "## Avaliação final — TEST temporal, somente o vencedor",
        "",
        f"Receita congelada: **{test['model']}**. Threshold **{test['threshold']}**.",
        "",
        "| AP | F1 | Recall | Precision | ROC-AUC |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {test['average_precision']:.4f} | {test['f1']:.4f} | "
            f"{test['recall']:.4f} | {test['precision']:.4f} | {test['roc_auc']:.4f} |"
        ),
        "",
        "| TN | FP | FN | TP |",
        "|---:|---:|---:|---:|",
        f"| {test['tn']} | {test['fp']} | {test['fn']} | {test['tp']} |",
        "",
        (
            f"Foram detectadas {test['tp']} de {test['positives']} fraudes. "
            f"As {test['fn']} fraudes não detectadas mostram que alta Precision "
            "não equivale a capturar todas as fraudes."
        ),
        "",
        "![Diagnóstico TEST](figures/TEST_final_diagnostics.png)",
        "",
        "## Limitações",
        "",
        (
            "A base tem só dois dias e já foi explorada no projeto antigo. O resultado é "
            "retrospectivo, não prospectivo. Não há intervalos formais de incerteza nem "
            "custos de erro informados; probabilidades não foram calibradas. PCA original "
            "e independência por cartão não são auditáveis. Desvio entre folds não é "
            "intervalo de confiança do TEST. Não há prontidão de produção demonstrada."
        ),
        "",
        (
            "Detalhes: ../docs/methodology.md. Recrie este resumo com "
            "python scripts/summarize_results.py após gerar os artefatos."
        ),
        "",
    ]
    (ROOT / "reports/results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
