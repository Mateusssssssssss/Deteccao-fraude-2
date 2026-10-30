"""Registra o modelo escolhido e a avaliação existente no MLflow local."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

# Tracking local: não há autolog de requisições nem envio das transações para terceiros.
os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from fraud_detection.data import ROOT
from fraud_detection.final_test import file_hash
from fraud_detection.predict import predict_with_artifact


class FraudModel(mlflow.pyfunc.PythonModel):
    """Flavor pyfunc mantém o threshold 0,70, além do pipeline sklearn/SMOTE."""

    def load_context(self, context):
        self.artifact = joblib.load(context.artifacts["pipeline"])

    def predict(self, context, model_input, params=None):
        return predict_with_artifact(model_input, self.artifact)


def log_best_model(root: Path = ROOT, tracking_uri: str | None = None) -> dict:
    """Loga parâmetros, métricas rotuladas e modelo executável, sem treinar ou pontuar TEST."""
    root = Path(root).resolve()
    package = root / "best_model"
    path = package / "best_model.joblib"
    metadata = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    if file_hash(path) != metadata["artifact_sha256"]:
        raise ValueError("O modelo mudou desde a serialização.")
    artifact = joblib.load(path)
    metrics_path = root / "reports/metrics/best_model_metrics.json"
    evaluation_path = root / "reports/experiments/best_model_manifest.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation["artifact_sha256"] != metadata["artifact_sha256"]:
        raise ValueError("As métricas não correspondem ao modelo serializado.")
    if metrics["threshold"] != artifact["threshold"]:
        raise ValueError("As métricas usam outro threshold.")
    tracking_dir = root / "mlflow"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri or f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    name = "fraud-detection-best-model"
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=(tracking_dir / "artifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    example = pd.DataFrame([{feature: 0.0 for feature in artifact["features"]}])
    expected = predict_with_artifact(example, artifact)
    requirements = [
        f"{name}=={importlib.metadata.version(name)}"
        for name in [
            "numpy",
            "pandas",
            "scikit-learn",
            "imbalanced-learn",
            "joblib",
            "matplotlib",
            "PyYAML",
            "xgboost",
        ]
    ]
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="random-forest-smote-threshold-0.70"
    ):
        mlflow.set_tags(
            {
                "selection_source": "metricas_smotec",
                "evaluation_scope": "reavaliação do TEST já consultado",
                "independent_test": "false",
                "training_performed_in_run": "false",
                "threshold_tuning_performed_in_run": "false",
                "artifact_sha256": metadata["artifact_sha256"],
                "dataset_sha256": metadata["dataset_sha256"],
            }
        )
        mlflow.log_params(
            {
                **metadata["parameters"],
                "threshold": 0.7,
                "smote_ratio": 0.1,
                "positive_class": 1,
                "selected_features_count": len(metadata["selected_features"]),
                "selection_metric": "f1",
            }
        )
        # Nomes distintos impedem confundir validação com o TEST reexecutado.
        mlflow.log_metrics(
            {
                f"test_{key}": float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            }
        )
        mlflow.log_metrics(
            {
                f"validation_reference_{key}": value
                for key, value in metadata["validation_reference"].items()
            }
        )
        for artifact_path in [
            package / "metadata.json",
            evaluation_path,
            metrics_path,
            package / "requirements.txt",
        ]:
            mlflow.log_artifact(str(artifact_path), artifact_path="provenance")
        for figure in (root / "reports/figures").glob("best_model_*.png"):
            mlflow.log_artifact(str(figure), artifact_path="figures")
        info = mlflow.pyfunc.log_model(
            name="best_model",
            python_model=FraudModel(),
            artifacts={"pipeline": str(path)},
            code_paths=[str(ROOT / "src/fraud_detection")],
            signature=infer_signature(example, expected),
            input_example=example,
            pip_requirements=requirements,
        )
        # Teste de round-trip: carrega o modelo realmente armazenado pelo MLflow.
        restored = mlflow.pyfunc.load_model(info.model_uri)
        actual = restored.predict(example)
        np.testing.assert_allclose(actual.fraud_probability, expected.fraud_probability, atol=0)
        np.testing.assert_array_equal(actual.prediction, expected.prediction)
        run_id = mlflow.active_run().info.run_id
        result = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "tracking_uri": uri,
            "model_uri": info.model_uri,
            "artifact_sha256": metadata["artifact_sha256"],
            "prediction_round_trip_passed": True,
        }
    (package / "mlflow_run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", help="Padrão: SQLite local em mlflow/mlflow.db")
    args = parser.parse_args()
    print(json.dumps(log_best_model(tracking_uri=args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
