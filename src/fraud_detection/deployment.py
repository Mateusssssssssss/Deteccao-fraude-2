"""Empacota a receita RF + SMOTE escolhida, sem refit nem ajuste de threshold."""

import argparse
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from fraud_detection.data import ROOT
from fraud_detection.final_test import file_hash
from fraud_detection.final_test import run as evaluate_test
from fraud_detection.model_select import plot_metrics

DEFAULT_MODEL = ROOT / "best_model/best_model.joblib"


def export_best_model(root: Path = ROOT) -> Path:
    """Valida a equivalência com metricas_smotec e persiste o pipeline completo."""
    root = Path(root)
    destination = root / "best_model"
    destination.mkdir(parents=True, exist_ok=True)
    source = root / "models/standardized_selected_pipeline.joblib"
    selection_path = root / "reports/experiments/ablation_selection.json"
    choice_path = root / "reports/experiments/model_select_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    original_test = json.loads(
        (root / "reports/experiments/final_test_manifest.json").read_text(encoding="utf-8")
    )
    # Este SHA identifica o ajuste no TRAIN completo; o último fold é outro ajuste.
    if file_hash(source) != original_test["artifact_sha256"]:
        raise ValueError("O artefato original mudou; confira sua proveniência antes de exportar.")
    artifact = joblib.load(source)
    if choice["choice"]["model"] != "random_forest__smote":
        raise ValueError("A escolha deve corresponder ao RF de metricas_smotec.")
    if artifact["selected_features"] != choice["selected_features"]:
        raise ValueError("Features diferentes da receita escolhida.")
    if artifact["threshold"] != 0.7 or choice["choice"]["threshold"] != 0.7:
        raise ValueError("O threshold escolhido deve permanecer 0,70.")
    model = artifact["pipeline"].named_steps["model"]
    if model.get_params() != choice["model_parameters"]:
        raise ValueError("Parâmetros diferentes do RF escolhido em metricas_smotec.")
    if selection["dataset_sha256"] != choice["dataset_sha256"]:
        raise ValueError("As receitas não pertencem à mesma versão dos dados.")
    sampler = artifact["pipeline"].named_steps["smote"]
    if sampler.sampling_strategy != 0.1 or sampler.k_neighbors != 3:
        raise ValueError("SMOTE diferente do experimento escolhido.")
    # O metadata antigo descreve a origem; a escolha atual foi manual e usa F1.
    artifact.update(
        schema_version=1,
        positive_class=1,
        selection_metric="f1",
        threshold_selection="Escolhido pelo usuário a partir de metricas_smotec: 0,70",
        source_artifact_sha256=file_hash(source),
        test_evaluated=True,
        score_calibrated=False,
    )
    path = destination / "best_model.joblib"
    joblib.dump(artifact, path, compress=3)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "joblib"]
    }
    metadata = {
        "schema_version": 1,
        "model": "Random Forest + StandardScaler + SMOTE",
        "threshold": 0.7,
        "positive_class": 1,
        "selection_metric": "f1",
        "source_notebook": "notebooks/metricas_smotec.ipynb",
        "source_artifact": str(source.relative_to(root)),
        "source_artifact_sha256": file_hash(source),
        "artifact_sha256": file_hash(path),
        "dataset_sha256": selection["dataset_sha256"],
        "selection_manifest_sha256": file_hash(selection_path),
        "choice_manifest_sha256": file_hash(choice_path),
        "features": artifact["features"],
        "selected_features": artifact["selected_features"],
        "parameters": model.get_params(),
        "smote": {"sampling_strategy": 0.1, "k_neighbors": 3, "random_state": 42},
        "fit_scope": "TRAIN completo; ajuste já existente, sem novo fit na exportação",
        "validation_reference": {"precision": 33 / 37, "recall": 33 / 44, "f1": 66 / 81},
        "independent_test": False,
        "score_calibrated": False,
        "versions": versions,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Exemplo sintético: não publica nenhuma transação da base.
    example = {feature: 0.0 for feature in artifact["features"]}
    (destination / "request_example.json").write_text(
        json.dumps({"transactions": [example]}, indent=2), encoding="utf-8"
    )
    (destination / "requirements.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    # A serialização não pode alterar nem score nem decisão.
    restored = joblib.load(path)
    import pandas as pd

    frame = pd.DataFrame([example])
    np.testing.assert_array_equal(
        artifact["pipeline"].predict_proba(frame), restored["pipeline"].predict_proba(frame)
    )
    return path


def evaluate_best_model(root: Path = ROOT, output: Path | None = None) -> dict:
    """Avalia o TEST solicitado com pipeline/threshold congelados e relatórios próprios."""
    root = Path(root)
    path = root / "best_model/best_model.joblib"
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    if file_hash(path) != metadata["artifact_sha256"]:
        raise ValueError("Checksum inválido no pacote best_model.")
    output = root if output is None else Path(output)
    result = evaluate_test(
        path,
        root / "reports/experiments/ablation_selection.json",
        output,
        prefix="best_model",
    )
    plot_metrics(
        result,
        output / "reports/figures/best_model_f1_recall_precision.png",
        "Random Forest + SMOTE — TEST; threshold 0,70",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Reexecuta a avaliação no TEST")
    parser.add_argument("--output", type=Path, help="Diretório separado para reavaliação")
    args = parser.parse_args()
    print(export_best_model())
    if args.evaluate:
        print(json.dumps(evaluate_best_model(output=args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
