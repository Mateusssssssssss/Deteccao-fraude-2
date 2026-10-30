import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from fraud_detection.data import FEATURES, ROOT, load_config
from fraud_detection.model import candidates
from fraud_detection.xgboost_ap_threshold import run, search_threshold, select_parameters


def test_ap_selection_ignores_fold_reserved_for_threshold(monkeypatch):
    rows = []
    for depth, values in [(2, [0.8, 0.8, 0.01]), (3, [0.7, 0.7, 0.99])]:
        row = {"params": {"model__max_depth": depth}}
        for metric in ["average_precision", "roc_auc"]:
            for phase in ["train", "test"]:
                row.update({f"split{i}_{phase}_{metric}": value for i, value in enumerate(values)})
        rows.append(row)
    monkeypatch.setattr(
        "fraud_detection.classification_search.load_reusable_results",
        lambda *args, **kwargs: (pd.DataFrame(rows), {}),
    )
    settings = {"param_grid": {"max_depth": [2, 3]}}
    table, params, _, fits = select_parameters(
        None, {}, {}, settings, None, [None, None, None], None, "hash"
    )
    assert params == {"model__max_depth": 2}
    assert table.mean_test_average_precision.max() == 0.8
    assert not any(column.startswith("split2_") for column in table)
    assert fits == 0


def test_threshold_grid_never_refits_and_ap_is_constant(frame, config, monkeypatch):
    fit, tune = frame.iloc[:1200], frame.iloc[1200:1800]
    model = candidates(config)["xgboost__none"].fit(fit[FEATURES], fit.Class)

    def reject(*args, **kwargs):
        raise AssertionError("Grid de threshold não pode ajustar o estimador congelado.")

    monkeypatch.setattr(XGBClassifier, "fit", reject)
    settings = {"threshold_objective": "f1", "thresholds": [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1]}
    threshold, table = search_threshold(model, fit, tune, settings)
    assert threshold in settings["thresholds"]
    assert table.loc[table.threshold.eq(threshold), "f1"].iloc[0] == table.f1.max()
    np.testing.assert_allclose(table.average_precision, table.average_precision.iloc[0])


def test_ap_threshold_workflow_uses_three_ordered_windows(frame, config, tmp_path):
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    config["data_path"] = str(source)
    settings = load_config(ROOT / "configs/xgboost_ap_threshold.yaml")
    settings.update(
        experiment_name="ap_fixture",
        reuse_from=str(tmp_path),
        model_n_jobs=1,
        search_n_jobs=1,
        permutation_repeats=1,
        iteration_points=[2, 5],
    )
    settings["param_grid"] = {
        "max_depth": [2, 4],
        "n_estimators": [5],
        "reg_alpha": [0.0],
        "reg_lambda": [1.0],
        "learning_rate": [0.05],
    }
    features = load_config(ROOT / "configs/ablation.yaml")
    features["importance_repeats"] = 1
    result = run(config, features, settings, tmp_path)
    assert result["primary_metric"] == "average_precision"
    assert result["threshold_objective"] == "f1"
    assert result["search_fit_time_max"] < result["threshold_time_min"]
    assert result["threshold_time_max"] < result["validation_time_min"]
    assert not result["test_evaluated"] and not result["refit_after_threshold"]
    assert result["inner_folds"] == 2 and result["executed_grid_fits"] == 4
    artifact = joblib.load(tmp_path / "models/ap_fixture_pipeline.joblib")
    assert artifact["threshold"] == result["threshold"]
    assert artifact["pipeline"].named_steps["select"].fit_time_max_ == result["search_fit_time_max"]
    with pytest.raises(FileExistsError):
        run(config, features, settings, tmp_path)


def test_f1_selects_balanced_decisions_not_highest_ap(monkeypatch):
    rows = []
    for depth, f1, ap in [(2, 0.70, 0.99), (3, 0.80, 0.60)]:
        row = {"params": {"model__max_depth": depth}}
        for key, value in [
            ("f1", f1),
            ("recall", f1),
            ("precision", f1),
            ("average_precision", ap),
            ("roc_auc", ap),
        ]:
            for phase in ["train", "test"]:
                row.update({f"split{i}_{phase}_{key}": value for i in range(3)})
        rows.append(row)
    monkeypatch.setattr(
        "fraud_detection.classification_search.load_reusable_results",
        lambda *args, **kwargs: (pd.DataFrame(rows), {}),
    )
    table, params, _, _ = select_parameters(
        None,
        {},
        {},
        {"primary_metric": "f1", "param_grid": {"max_depth": [2, 3]}},
        None,
        [None, None, None],
        None,
        "hash",
    )
    assert params == {"model__max_depth": 3}
    assert table.mean_test_f1.max() == 0.80


def test_random_forest_uses_same_ordered_protocol_and_smote(frame, config, tmp_path):
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    config["data_path"] = str(source)
    settings = load_config(ROOT / "configs/random_forest_gridsearch.yaml")
    settings.pop("reuse_from", None)
    settings.pop("reuse_prefix", None)
    settings.update(
        experiment_name="rf_fixture", model_n_jobs=1, search_n_jobs=1, permutation_repeats=1
    )
    settings["param_grid"] = {
        "max_depth": [2, 4],
        "n_estimators": [5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "criterion": ["gini"],
    }
    features = load_config(ROOT / "configs/ablation.yaml")
    features["importance_repeats"] = 1
    result = run(config, features, settings, tmp_path)
    assert result["family"] == "random_forest" and result["primary_metric"] == "f1"
    assert result["search_fit_time_max"] < result["threshold_time_min"]
    assert result["threshold_time_max"] < result["validation_time_min"]
    assert not result["test_evaluated"]
    artifact = joblib.load(tmp_path / "models/rf_fixture_pipeline.joblib")
    assert artifact["pipeline"].named_steps["model"].class_weight is None
    assert "smote" in artifact["pipeline"].named_steps
    assert result["threshold"] in settings["thresholds"]
    old_scores = (tmp_path / "reports/metrics/rf_fixture_results.csv").read_bytes()
    expanded = dict(
        settings, experiment_name="rf_expanded", reuse_from=str(tmp_path), reuse_prefix="rf_fixture"
    )
    expanded["param_grid"] = {**settings["param_grid"], "min_samples_leaf": [1, 2, 3]}
    new = run(config, features, expanded, tmp_path)
    assert new["combinations"] == 6
    assert new["reused_grid_fits"] == 8 and new["executed_grid_fits"] == 4
    assert (tmp_path / "reports/metrics/rf_fixture_results.csv").read_bytes() == old_scores
