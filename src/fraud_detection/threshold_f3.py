"""Repondera a grade temporal já calculada para F3, sem consultar novamente o TEST."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.ablation import variant_pipeline
from fraud_detection.data import FEATURES, ROOT, TARGET, read_split
from fraud_detection.final_test import file_hash
from fraud_detection.threshold import threshold_metrics
from fraud_detection.validation import temporal_folds

LOGGER = logging.getLogger(__name__)


def add_f3(table: pd.DataFrame) -> pd.DataFrame:
    """F3 = 10TP/(10TP+9FN+FP); usa as contagens da grade, não requer novo fit."""
    required = {"threshold", "tp", "fp", "fn", "recall", "precision"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError("Grade sem contagens e métricas necessárias para calcular F3.")
    result = table.copy()
    denominator = 10 * result.tp + 9 * result.fn + result.fp
    result["f3"] = np.divide(
        10 * result.tp,
        denominator,
        out=np.zeros(len(result), dtype=float),
        where=denominator.to_numpy() != 0,
    )
    return result


def best_f3(table: pd.DataFrame) -> pd.Series:
    """Mesmos desempates anteriores: Recall, Precision e maior threshold."""
    return (
        add_f3(table)
        .sort_values(["f3", "recall", "precision", "threshold"], ascending=False, kind="stable")
        .iloc[0]
    )


def run(source_root: str | Path = ROOT, output: str | Path = ROOT) -> dict:
    source_root, output = Path(source_root), Path(output)
    source_metrics = source_root / "reports/metrics"
    source_selection = source_root / "reports/experiments/ablation_selection.json"
    source_artifact = source_root / "models/standardized_selected_pipeline.joblib"
    md, ed, fd = [output / p for p in ["reports/metrics", "reports/experiments", "reports/figures"]]
    for directory in [md, ed, fd, output / "models"]:
        directory.mkdir(parents=True, exist_ok=True)
    if (ed / "f3_selection.json").exists():
        raise FileExistsError("F3 já executado nesta saída; use --output para reprodução.")

    # A família, o preprocessing e a regra de features permanecem congelados.
    selection = json.loads(source_selection.read_text(encoding="utf-8"))
    artifact = joblib.load(source_artifact)
    if (
        artifact["model"] != selection["model"]
        or artifact["threshold"] != selection["threshold"]
        or artifact["selected_features"] != selection["selected_features"]
    ):
        raise ValueError("Artefato F2 não corresponde ao manifesto de seleção.")
    config, settings = selection["config"], selection["settings"]
    train, _, _, dataset = read_split(config)
    if file_hash(dataset) != selection["dataset_sha256"]:
        raise ValueError("Dataset diferente do usado na grade anterior.")
    family, recipe = selection["model"].split("__", 1)
    fold_indices = temporal_folds(train, dict(config, sampling_strategies=["none"]))

    # F3 usa as mesmas contagens da grade GridSearchCV já avaliada em janelas internas.
    # Alterar a função de resumo não exige recalcular as mesmas previsões internas.
    grids = add_f3(pd.read_csv(source_metrics / "ablation_threshold_grids.csv"))
    final_grid = add_f3(pd.read_csv(source_metrics / "ablation_threshold_final_grid.csv"))
    decisions = pd.read_csv(source_metrics / "feature_selection_folds.csv")
    historical = pd.read_csv(source_metrics / "ablation_threshold_folds.csv")
    baseline = historical.loc[historical.policy.eq("gridsearch_f2")].copy()
    baseline = add_f3(baseline)
    all_rows = baseline.to_dict("records")
    audit = []

    # Reavalia só janelas externas em que o threshold F3 difere do threshold F2.
    # Refit externo é necessário para reconstruir scores não persistidos por transação.
    for fold, (fit_idx, val_idx) in enumerate(fold_indices, start=1):
        row = best_f3(grids.loc[grids.fold.eq(fold)])
        threshold = float(row.threshold)
        old = baseline.loc[baseline.fold.eq(fold)].iloc[0]
        fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
        LOGGER.info("Fold %s: threshold F2 %.4f -> F3 %.4f", fold, old.threshold, threshold)
        if threshold == float(old.threshold):
            scores = old.to_dict()
            method = "Reuso de métricas externas: mesmo modelo e mesmo threshold"
        else:
            # Estas decisões foram aprendidas só no treino externo daquela rodada.
            selected = decisions.loc[decisions.fold.eq(fold) & decisions.keep, "feature"].tolist()
            pipeline = variant_pipeline(family, recipe, selected, config, settings)
            pipeline.fit(fit[FEATURES], fit[TARGET])
            probability = pipeline.predict_proba(validation[FEATURES])[:, 1]
            reproduced = threshold_metrics(validation[TARGET], probability, float(old.threshold))
            # Interrompe se a reconstrução não reproduzir as contagens F2 originais.
            for key in ["tp", "fp", "fn", "tn"]:
                if reproduced[key] != int(old[key]):
                    raise ValueError(f"Fold {fold}: reprodução de {key} divergiu do original.")
            scores = threshold_metrics(validation[TARGET], probability, threshold)
            denominator = 10 * scores["tp"] + 9 * scores["fn"] + scores["fp"]
            scores["f3"] = 10 * scores["tp"] / denominator if denominator else 0.0
            method = "Refit no treino externo; contagens F2 reproduzidas antes de avaliar F3"
        scores.update(
            {
                "fold": fold,
                "policy": "gridsearch_f3",
                "model": selection["model"],
                "source": "CV temporal externa TRAIN; receita fixa, threshold interno F3",
                "threshold": threshold,
            }
        )
        all_rows.append(scores)
        audit.append(
            {
                "fold": fold,
                "threshold_f2": float(old.threshold),
                "threshold_f3": threshold,
                "fit_time_max": float(fit.Time.max()),
                "validation_time_min": float(validation.Time.min()),
                "validation_time_max": float(validation.Time.max()),
                "method": method,
            }
        )
        pd.DataFrame(all_rows).to_csv(md / "f3_outer_folds.csv", index=False)

    scores = pd.DataFrame(all_rows)
    summary = []
    for policy, part in scores.groupby("policy", sort=False):
        row = {"policy": policy, "source": "CV temporal externa TRAIN, mesmos folds"}
        for metric in [
            "f3",
            "f2",
            "f1",
            "recall",
            "precision",
            "average_precision",
            "fp",
            "fn",
            "alert_rate",
        ]:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(md / "f3_comparison.csv", index=False)
    grids.to_csv(md / "f3_inner_grids.csv", index=False)
    final_grid.to_csv(md / "f3_final_grid.csv", index=False)
    pd.DataFrame(audit).to_csv(ed / "f3_fold_manifest.csv", index=False)
    chosen = best_f3(final_grid)
    threshold = float(chosen.threshold)

    # O pipeline final treinado é idêntico; só a política de decisão recebe novo limiar.
    # O artefato F2 e todas as avaliações TEST anteriores ficam preservados.
    artifact.update(
        {
            "threshold": threshold,
            "threshold_objective": "F3",
            "test_evaluated": False,
            "source_pipeline_previously_evaluated_on_historical_test": True,
        }
    )
    joblib.dump(artifact, output / "models/recall_f3_pipeline.joblib")
    result = {
        "model": selection["model"],
        "objective": "F3",
        "beta": 3,
        "threshold": threshold,
        "previous_f2_threshold": selection["threshold"],
        "selected_features": selection["selected_features"],
        "selected_inner_scores": chosen.to_dict(),
        "dataset_sha256": file_hash(dataset),
        "source_f2_artifact_sha256": file_hash(source_artifact),
        "source_selection_sha256": file_hash(source_selection),
        "source_inner_grids_sha256": file_hash(source_metrics / "ablation_threshold_grids.csv"),
        "source_final_grid_sha256": file_hash(source_metrics / "ablation_threshold_final_grid.csv"),
        "test_evaluated": False,
        "limitations": [
            "F3 escolhido após observar TEST anterior; este não serve como confirmação independente.",
            "Thresholds selecionados só nas janelas internas TRAIN; avaliação externa condicionada à receita.",
            "Média externa avalia a política de ajuste, não o threshold final isolado.",
            "Reuso de TRAIN ao longo das rodadas pode favorecer conclusões otimistas.",
            "F3 favorece Recall sem garantir aumento em dados novos ou custo ótimo.",
        ],
    }
    (ed / "f3_selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_report(result, summary, output)
    LOGGER.info("F3 concluído: threshold final %.4f", threshold)
    return result


def save_report(result, summary, output) -> None:
    """Registra a troca entre Recall e Precision, sem misturar com resultados de TEST."""
    import matplotlib.pyplot as plt

    lines = [
        "# F3 na seleção de threshold — somente TRAIN",
        "",
        (
            f"Receita fixa: {result['model']}. Threshold final F2: "
            f"{result['previous_f2_threshold']}; threshold final F3: **{result['threshold']}**."
        ),
        "",
        (
            "A mesma grade GridSearchCV foi resumida por F3 = 10TP/(10TP+9FN+FP). "
            "Isso equivale ao scorer F-beta com beta=3 sobre as mesmas previsões internas. "
            "Não houve seleção de features nem threshold no TEST."
        ),
        "",
        "| Política | Recall médio | Precision média | F2 médio | F3 médio | FP médios/fold | FN médios/fold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.policy} | {row.recall:.2%} | {row.precision:.2%} | "
            f"{row.f2:.4f} | {row.f3:.4f} | {row.fp:.1f} | {row.fn:.1f} |"
        )
    old = summary.set_index("policy").loc["gridsearch_f2"]
    new = summary.set_index("policy").loc["gridsearch_f3"]
    lines.extend(
        [
            "",
            f"Na CV externa, o Recall mudou em {(new.recall - old.recall) * 100:+.2f} p.p.",
            f"A Precision mudou em {(new.precision - old.precision) * 100:+.2f} p.p.",
            (
                "F3 não melhorou nem seu próprio score externo nesta rodada. "
                "O pequeno ganho de Recall veio com mais falsos alertas; "
                "a variante não deve ser descrita como superior ao F2."
                if new.f3 <= old.f3
                else "F3 aumentou seu score externo nesta rodada; isso não demonstra ganho em um TEST novo."
            ),
        ]
    )
    lines += [
        "",
        (
            "Cada fold escolhe seu threshold em janela interna anterior à validação externa. "
            "O threshold final é escolhido novamente na janela interna do TRAIN completo. "
            "A família e receita foram escolhidas anteriormente com os mesmos dados: "
            "a estimativa é condicionada àquela seleção."
        ),
        "",
        (
            "O modelo treinado não mudou no artefato final; só o threshold foi atualizado. "
            "Novo arquivo local: models/recall_f3_pipeline.joblib."
        ),
        "",
        (
            "**TEST não foi reavaliado.** O notebook 04 continua descrevendo o pipeline F2 "
            "com threshold 0,70. F3 foi escolhido após esse resultado e precisa de outro "
            "período externo para confirmação independente."
        ),
        "",
        "![F2 versus F3](figures/f3_comparison.png)",
        "",
    ]
    (output / "reports/f3_results.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary))
    for i, metric in enumerate(["recall", "precision", "f3"]):
        bars = ax.bar(
            x + (i - 1) * 0.25,
            summary[metric],
            0.25,
            yerr=summary[f"{metric}_std"],
            capsize=3,
            label=metric,
        )
        ax.bar_label(bars, fmt="%.2f", label_type="center")
    ax.set(
        xticks=x,
        xticklabels=summary.policy,
        ylim=(0, 1),
        ylabel="Score",
        title="F2 × F3 — CV temporal externa TRAIN; média ± desvio",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reports/figures/f3_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.source_root, args.output)


if __name__ == "__main__":
    main()
