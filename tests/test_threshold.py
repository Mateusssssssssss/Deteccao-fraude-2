import json

import joblib
import numpy as np
import pytest

from fraud_detection.data import FEATURES, ROOT, load_config
from fraud_detection.model import candidates
from fraud_detection.predict import predict
from fraud_detection.threshold import run, search_threshold, select_f2_index


def test_grid_is_temporal_and_keeps_estimator_frozen(frame, config):
    settings = load_config(ROOT / "configs/threshold.yaml")
    settings["thresholds"] = [0, 0.05, 0.5, 1]
    pipeline = candidates(config)["xgboost__none"]
    search, grid, meta = search_threshold(pipeline, frame, settings)
    assert meta["inner_fit_time_max"] < meta["inner_tune_time_min"]
    assert grid.loc[search.best_index_, "f2"] == grid.f2.max()
    assert search.best_params_["threshold"] in settings["thresholds"]
    assert grid.iloc[0].recall == 1
    fitted = search.best_estimator_.estimator_
    imputer = fitted.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    prefix = frame.loc[frame.Time < meta["cutoff_time"]]
    np.testing.assert_allclose(imputer.statistics_, prefix[FEATURES].median().to_numpy())
    assert len(search.best_estimator_.predict(frame[FEATURES].iloc[:13])) == 13


def test_deterministic_tie_breaking():
    results = {
        "param_threshold": [0.01, 0.1, 0.2],
        "mean_test_f2": [0.9, 0.9, 0.9],
        "mean_test_recall": [0.8, 0.9, 0.9],
        "mean_test_precision": [0.8, 0.8, 0.8],
    }
    assert select_f2_index(results) == 2


def test_nested_run_preserves_test_and_inference(frame, config, tmp_path):
    settings = load_config(ROOT / "configs/threshold.yaml")
    settings["thresholds"] = [0, 0.05, 0.5, 1]
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    config["data_path"] = str(source)
    metric_dir = tmp_path / "reports/metrics"
    metric_dir.mkdir(parents=True)
    historical = metric_dir / "test_metrics.json"
    historical.write_text('{"historical": true}', encoding="utf-8")
    before = historical.read_bytes()
    result = run(config, settings, tmp_path)
    assert result["test_evaluated"] is False
    assert historical.read_bytes() == before
    path = tmp_path / "models/recall_pipeline.joblib"
    artifact = joblib.load(path)
    predictions = predict(frame[FEATURES].iloc[:25], path)
    np.testing.assert_array_equal(
        predictions.prediction,
        artifact["pipeline"].predict_proba(frame[FEATURES].iloc[:25])[:, 1] >= result["threshold"],
    )
    folds = json.loads(
        (tmp_path / "reports/experiments/threshold_selection.json").read_text(encoding="utf-8")
    )
    assert folds["objective"] == "F2"
    with pytest.raises(FileExistsError):
        run(config, settings, tmp_path)


def test_reject_invalid_grid(frame, config):
    settings = load_config(ROOT / "configs/threshold.yaml")
    settings["thresholds"] = [-0.1]
    with pytest.raises(ValueError, match="entre 0 e 1"):
        search_threshold(candidates(config)["xgboost__none"], frame, settings)
