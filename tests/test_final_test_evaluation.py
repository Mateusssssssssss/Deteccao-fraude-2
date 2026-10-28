import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from fraud_detection.data import FEATURES, read_split
from fraud_detection.final_test import complete_metrics, file_hash, run
from fraud_detection.model import candidates


def test_final_metric_denominators():
    result = complete_metrics(np.array([0, 0, 1, 1]), np.array([0.8, 0.1, 0.8, 0.8]), 0.7)
    assert result["tp"] == 2 and result["fp"] == 1 and result["fn"] == 0 and result["tn"] == 1
    assert result["false_positive_rate"] == 0.5
    assert result["false_discovery_rate"] == pytest.approx(1 / 3)
    assert result["negative_predictive_value"] == 1
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 0.75


def test_evaluation_never_refits_and_preserves_history(frame, config, tmp_path, monkeypatch):
    data = tmp_path / "original.csv"
    frame.to_csv(data, index=False)
    config["data_path"] = str(data)
    train, _, _, _ = read_split(config)
    pipeline = candidates(config)["random_forest__none"].fit(train[FEATURES], train.Class)
    path = tmp_path / "frozen.joblib"
    joblib.dump(
        {
            "model": "rf_fixture",
            "pipeline": pipeline,
            "features": FEATURES,
            "selected_features": FEATURES,
            "threshold": 0.7,
        },
        path,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "model": "rf_fixture",
                "selected_features": FEATURES,
                "threshold": 0.7,
                "config": config,
                "dataset_sha256": file_hash(data),
            }
        ),
        encoding="utf-8",
    )
    historical = tmp_path / "reports/metrics/test_metrics.json"
    historical.parent.mkdir(parents=True)
    historical.write_text('{"historical": true}', encoding="utf-8")
    before = file_hash(path)

    def reject_fit(*args, **kwargs):
        raise AssertionError("Avaliar TEST não pode chamar fit.")

    monkeypatch.setattr(RandomForestClassifier, "fit", reject_fit)
    result = run(path, selection, tmp_path)
    assert not result["independent_test"]
    assert result["threshold"] == 0.7
    assert file_hash(path) == before
    assert historical.read_text(encoding="utf-8") == '{"historical": true}'
    assert (tmp_path / "reports/metrics/final_test_metric_definitions.csv").is_file()
    with pytest.raises(FileExistsError):
        run(path, selection, tmp_path)


def test_frozen_manifest_mismatch_stops_evaluation(tmp_path):
    artifact = tmp_path / "artifact.joblib"
    joblib.dump({"model": "a", "threshold": 0.7, "selected_features": ["V1"]}, artifact)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "model": "a",
                "threshold": 0.5,
                "selected_features": ["V1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="não correspondem"):
        run(artifact, selection, tmp_path)
