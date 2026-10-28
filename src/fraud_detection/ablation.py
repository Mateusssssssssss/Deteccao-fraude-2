"""Ablações de scaling, SMOTE e regra conjunta de seleção, somente no TRAIN."""

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
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from fraud_detection.data import FEATURES, ROOT, TARGET, load_config, read_split, temporal_split
from fraud_detection.features import audit_features
from fraud_detection.model import candidates
from fraud_detection.preprocess import build_pipeline
from fraud_detection.segments import error_rates_by_bins
from fraud_detection.threshold import search_threshold, threshold_metrics
from fraud_detection.validation import temporal_folds

LOGGER = logging.getLogger(__name__)


def variant_pipeline(family: str, recipe: str, selected: list[str], config: dict, settings: dict):
    """Cada receita aprende imputação, scaling e sampling no próprio fit."""
    estimator = clone(candidates(config)[f"{family}__none"].named_steps["model"])
    sampling = "smote" if "smote" in recipe else "none"
    if family == "random_forest" and sampling == "smote":
        estimator.set_params(class_weight=None)
    return build_pipeline(
        estimator,
        dict(config, smote_ratio=settings["smote_ratio"]),
        sampling,
        standardize=recipe != "raw_all",
        features=selected if "selected" in recipe else FEATURES,
    )


def aggregate(table, group: str) -> pd.DataFrame:
    rows = []
    for name, part in table.groupby(group, sort=False):
        row = {group: name, "source": "CV temporal externa TRAIN"}
        for metric in [
            "f2",
            "f1",
            "recall",
            "precision",
            "average_precision",
            "alert_rate",
            "fp",
            "fn",
        ]:
            row[metric] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
        for metric in ["f2", "f1", "recall", "precision", "alert_rate"]:
            row[f"{metric}_pct"] = 100 * row[metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f2", ascending=False)


def run(config: dict, settings: dict, threshold_settings: dict, output: str | Path = ROOT):
    output = Path(output)
    md, ed, fd = [output / p for p in ["reports/metrics", "reports/experiments", "reports/figures"]]
    for folder in [md, ed, fd, output / "models"]:
        folder.mkdir(parents=True, exist_ok=True)
    if (ed / "ablation_selection.json").exists():
        raise FileExistsError("Rodada já executada; use --output para reprodução separada.")
    # Etapa 1: reutiliza só TRAIN; TEST histórico não é pontuado novamente.
    train, _, split_metadata, source = read_split(config)
    folds = temporal_folds(train, dict(config, sampling_strategies=["none"]))
    # Etapa 2: compara mudanças isoladas com threshold comum 0,5.
    rows, audits, fitted, probabilities, selection_by_fold = [], [], {}, {}, {}
    families = ["xgboost", "random_forest"]
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
        LOGGER.info("Fold %s: aprendendo a regra conjunta apenas no treino", fold)
        # Reaprende a seleção no treino deste fold, nunca na validação externa.
        audit = audit_features(fit, config, settings)
        selected = audit.loc[audit.keep, "feature"].tolist()
        selection_by_fold[fold] = selected
        audits.append(audit.assign(fold=fold))
        for family in families:
            for recipe in settings["recipes"]:
                name = f"{family}__{recipe}"
                LOGGER.info(
                    "Fold %s: %s (%s features)",
                    fold,
                    name,
                    len(selected) if "selected" in recipe else len(FEATURES),
                )
                estimator = variant_pipeline(family, recipe, selected, config, settings)
                estimator.fit(fit[FEATURES], fit[TARGET])
                probability = estimator.predict_proba(validation[FEATURES])[:, 1]
                rows.append(
                    {
                        "model": name,
                        "family": family,
                        "recipe": recipe,
                        "fold": fold,
                        "n_features": len(selected) if "selected" in recipe else len(FEATURES),
                        "source": "CV temporal TRAIN; threshold 0.5",
                        **threshold_metrics(validation[TARGET], probability, 0.5),
                    }
                )
                fitted[(name, fold)] = estimator
                probabilities[(name, fold)] = probability
                pd.DataFrame(rows).to_csv(md / "ablation_folds.csv", index=False)
    scores = pd.DataFrame(rows)
    comparison = aggregate(scores, "model")
    comparison.to_csv(md / "ablation_comparison.csv", index=False)
    pd.concat(audits).to_csv(md / "feature_selection_folds.csv", index=False)
    # Respeita a versão com retirada solicitada; a ablação mantém os controles completos.
    # A versão final respeita a retirada solicitada; controles completos seguem no relatório.
    selected_candidates = comparison[comparison.model.str.contains("selected")]
    winner = selected_candidates.iloc[0].model
    family, recipe = winner.split("__", 1)
    LOGGER.info("Melhor receita filtrada por F2 em CV: %s", winner)
    # Etapa 3: a receita filtrada recebe threshold escolhido em uma janela interna anterior.
    nested_rows, grids, error_segments, nested_audits = [], [], [], []
    oof_y, oof_probability, oof_labels = [], [], []
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        fit, validation = train.iloc[fit_idx], train.iloc[val_idx]
        # A seleção também precisa ser refeita no subtreino do threshold para evitar leakage.
        prefix, _, _ = temporal_split(fit, threshold_settings["inner_validation_time_fraction"])
        inner_audit = audit_features(prefix, config, settings)
        inner_selected = inner_audit.loc[inner_audit.keep, "feature"].tolist()
        nested_audits.append(inner_audit.assign(fold=fold, stage="subtreino do threshold"))
        inner_pipeline = variant_pipeline(family, recipe, inner_selected, config, settings)
        search, grid, meta = search_threshold(inner_pipeline, fit, threshold_settings)
        threshold = float(search.best_params_["threshold"])
        probability = probabilities[(winner, fold)]
        for policy, value in [("fixed_0.5", 0.5), ("gridsearch_f2", threshold)]:
            nested_rows.append(
                {
                    "fold": fold,
                    "policy": policy,
                    "model": winner,
                    "source": "CV temporal externa TRAIN; receita condicionada à seleção",
                    **threshold_metrics(validation[TARGET], probability, value),
                }
            )
        grids.append(grid.assign(fold=fold))
        error_segments.append(error_rates_by_bins(fit, validation, probability, threshold, fold))
        oof_y.extend(validation[TARGET].to_numpy())
        oof_probability.extend(probability)
        oof_labels.extend(probability >= threshold)
        LOGGER.info("Fold %s: threshold interno %.4f", fold, threshold)
    pd.DataFrame(nested_rows).to_csv(md / "ablation_threshold_folds.csv", index=False)
    nested_summary = aggregate(pd.DataFrame(nested_rows), "policy")
    nested_summary.to_csv(md / "ablation_threshold_comparison.csv", index=False)
    pd.concat(grids).to_csv(md / "ablation_threshold_grids.csv", index=False)
    pd.concat(nested_audits).to_csv(md / "feature_selection_threshold_folds.csv", index=False)
    pd.concat(error_segments).to_csv(md / "category_error_metrics.csv", index=False)
    LOGGER.info("Regra final no TRAIN completo, sem acessar TEST para decisões")
    # Etapa 4: reaprende a regra no TRAIN completo para o refit final.
    final_audit = audit_features(train, config, settings)
    final_audit.to_csv(md / "feature_selection_final.csv", index=False)
    final_selected = final_audit.loc[final_audit.keep, "feature"].tolist()
    prefix, _, _ = temporal_split(train, threshold_settings["inner_validation_time_fraction"])
    prefix_audit = audit_features(prefix, config, settings)
    prefix_selected = prefix_audit.loc[prefix_audit.keep, "feature"].tolist()
    pipeline = variant_pipeline(family, recipe, prefix_selected, config, settings)
    search, grid, meta = search_threshold(pipeline, train, threshold_settings)
    threshold = float(search.best_params_["threshold"])
    grid.to_csv(md / "ablation_threshold_final_grid.csv", index=False)
    final = variant_pipeline(family, recipe, final_selected, config, settings)
    final.fit(train[FEATURES], train[TARGET])
    # Artefato separado preserva os modelos/resultados das rodadas anteriores.
    joblib.dump(
        {
            "pipeline": final,
            "features": FEATURES,
            "selected_features": final_selected,
            "model": winner,
            "threshold": threshold,
            "threshold_objective": "F2",
            "test_evaluated": False,
        },
        output / "models/standardized_selected_pipeline.joblib",
    )
    selection = {
        "model": winner,
        "best_overall_cv": comparison.iloc[0].model,
        "selection": "Maior F2 CV entre receitas com regra de retirada solicitada",
        "threshold": threshold,
        "selected_features": final_selected,
        "removed_features": final_audit.loc[final_audit.remove, "feature"].tolist(),
        "rule": "remove = abs(Pearson) < 0.090 AND permutation_AP < 0.100",
        "config": config,
        "settings": settings,
        "threshold_settings": threshold_settings,
        "final_threshold_inner_split": meta,
        "split": split_metadata,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "test_evaluated": False,
        "limitations": [
            "Ablações reutilizam TRAIN de rodadas anteriores; novo TEST externo necessário.",
            "Threshold avaliado em janela externa ao ajuste, condicionado à receita selecionada.",
            "Importância de referência XGBoost em validação interna; não é porcentagem.",
            "Features podem variar entre folds; regra final reaprendida no TRAIN completo.",
            "CV externa da receita vencedora também participou da seleção; pode ser otimista.",
            "Refit e seleção de features podem alterar distribuição dos scores; sem calibração.",
        ],
    }
    (ed / "ablation_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_comparison(comparison, fd / "ablation_f1_recall_precision.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(oof_y, oof_labels, ax=axes[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(oof_y, oof_probability, ax=axes[1], name="OOF TRAIN")
    axes[1].axhline(np.mean(oof_y), color="gray", linestyle="--", label="Prevalência")
    axes[1].legend()
    fig.suptitle("Receita filtrada — OOF temporal TRAIN; thresholds internos por fold")
    fig.tight_layout()
    fig.savefig(fd / "ablation_oof_pr_confusion.png", dpi=160)
    plt.close(fig)
    write_report(selection, comparison, nested_summary, output)
    LOGGER.info(
        "Concluído: %s, threshold %.4f, %s features", winner, threshold, len(final_selected)
    )
    return selection


def plot_comparison(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(len(table))
    for i, metric in enumerate(["f1", "recall", "precision"]):
        bars = ax.bar(
            x + (i - 1) * 0.25,
            table[metric],
            0.25,
            yerr=table[f"{metric}_std"],
            capsize=2,
            label=metric,
        )
        ax.bar_label(bars, fmt="%.2f", label_type="center", fontsize=7)
    ax.set(
        xticks=x,
        xticklabels=table.model,
        ylim=(0, 1),
        ylabel="Score",
        title="Ablações — média ± desvio de CV temporal TRAIN; threshold 0,5; ordem por F2",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(selection, comparison, nested, output) -> None:
    lines = [
        "# StandardScaler, SMOTE e seleção conjunta de features",
        "",
        (
            "CV temporal somente TRAIN, mesmos três folds para dez receitas. "
            "A tabela principal usa threshold fixo 0,5 para separar mudanças de preprocessing "
            "do ajuste de threshold. RF sem SMOTE usa balanced_subsample; com SMOTE usa "
            "pesos uniformes. Essa mudança faz parte da receita."
        ),
        "",
        "| Receita | F2 | Recall (%) | Precision (%) | F1 (%) | AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.model} | {row.f2:.4f} | {row.recall_pct:.2f} | "
            f"{row.precision_pct:.2f} | {row.f1_pct:.2f} | {row.average_precision:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Melhor receita geral por F2: **{selection['best_overall_cv']}**.",
            f"Receita com retirada de features solicitada: **{selection['model']}**.",
            "",
            (
                "**Retirar somente se |Pearson| < 0,090 E queda de AP por permutação < 0,100.** "
                "Correlação negativa forte é preservada. Os cortes não são porcentagens. "
                "Importâncias são recalculadas por referência XGBoost dentro de cada treino."
            ),
            "",
            "Mantidas no refit final: " + ", ".join(selection["selected_features"]) + ".",
            "Retiradas no refit final: " + ", ".join(selection["removed_features"]) + ".",
            "",
            "## Threshold e erros por faixa",
            "",
            f"Threshold final por GridSearchCV/F2 no TRAIN: **{selection['threshold']:g}**.",
            "",
            "| Política da receita filtrada | Recall (%) | Precision (%) | F2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in nested.itertuples():
        lines.append(
            f"| {row.policy} | {row.recall_pct:.2f} | {row.precision_pct:.2f} | {row.f2:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "Thresholds são ajustados internamente por fold; scores acima avaliam a política, "
                "não o threshold final isoladamente. Seleção da receita usou os mesmos folds: "
                "estes resultados são condicionados à seleção e podem ser otimistas."
            ),
            "",
            (
                "A base não tem features categóricas; SMOTE foi aplicado às numéricas padronizadas. "
                "SMOTENC é para entradas mistas numéricas/categóricas, não porque o target é uma classe. "
                "SMOTE não é garantidamente melhor. A razão minoritária/majoritária pesquisada aqui "
                "é fixa em 0,1, mantendo todas as legítimas; não houve undersampling nesta rodada."
            ),
            "",
            (
                "Os nomes raw_all, scaled_all, scaled_smote_all, scaled_selected e "
                "scaled_smote_selected significam, respectivamente: todas sem scaling; todas "
                "com StandardScaler; todas com StandardScaler+SMOTE; regra conjunta+scaling; "
                "regra conjunta+scaling+SMOTE."
            ),
            "",
            (
                "Taxas de fraude observada por faixa: reports/metrics/eda_feature_bins.csv. "
                "Erros do modelo por faixa e fold: reports/metrics/category_error_metrics.csv. "
                "Cada percentual tem seu denominador explicado em docs/metrics_guide.md. "
                "As faixas são segmentos de features numéricas, não categorias de fraude reais."
            ),
            "",
            (
                "**TEST não foi reavaliado.** Novo artefato local: "
                "models/standardized_selected_pipeline.joblib. Mais períodos independentes são "
                "necessários para validar o resultado; não houve calibração nem custo financeiro definido."
            ),
            "",
            "![Ablações](figures/ablation_f1_recall_precision.png)",
            "",
            "![PR e matriz OOF](figures/ablation_oof_pr_confusion.png)",
            "",
        ]
    )
    (output / "reports/ablation_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--ablation-config", type=Path, default=ROOT / "configs/ablation.yaml")
    parser.add_argument("--threshold-config", type=Path, default=ROOT / "configs/threshold.yaml")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(
        load_config(args.config),
        load_config(args.ablation_config),
        load_config(args.threshold_config),
        args.output,
    )


if __name__ == "__main__":
    main()
