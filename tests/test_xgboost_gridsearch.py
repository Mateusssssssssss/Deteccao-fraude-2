import json
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline

from fraud_detection.data import FEATURES, ROOT, load_config, read_split
from fraud_detection.features import TrainFeatureSelector
from fraud_detection.xgboost_gridsearch import (
    build_search_pipeline,
    load_reusable_results,
    make_search,
    run,
)


def test_selector_scaler_and_smote_only_learn_from_fit(frame, config):
    features = load_config(ROOT / "configs/ablation.yaml")
    settings = load_config(ROOT / "configs/xgboost_gridsearch.yaml")
    settings["model_n_jobs"] = 1
    fit, validation = frame.iloc[:1600], frame.iloc[1600:].copy()
    validation["Amount"] += 1e6
    pipeline = build_search_pipeline(config, features, settings)
    pipeline.fit(fit[FEATURES], fit.Class)
    selector = pipeline.named_steps["select"]
    selected = selector.selected_features_
    assert selector.fit_time_max_ == fit.Time.max()
    assert selector.audit_.selection_input_time_max.max() < validation.Time.min()
    scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, fit[selected].mean().to_numpy())
    before = scaler.mean_.copy()
    assert pipeline.predict_proba(validation[FEATURES]).shape == (len(validation), 2)
    np.testing.assert_array_equal(scaler.mean_, before)
    assert pipeline.named_steps["smote"].sampling_strategy_[1] > 0


def test_gridsearch_excludes_reserved_test_and_keeps_artifact_loadable(
    frame, config, tmp_path, monkeypatch
):
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    config["data_path"] = str(source)
    settings = load_config(ROOT / "configs/xgboost_gridsearch.yaml")
    settings.update(model_n_jobs=1, permutation_repeats=1, iteration_points=[2, 5])
    settings["param_grid"] = {
        "max_depth": [2, 4],
        "n_estimators": [5],
        "reg_alpha": [0, 1],
        "reg_lambda": [1],
    }
    features = load_config(ROOT / "configs/ablation.yaml")
    features["importance_repeats"] = 1
    _, test, _, _ = read_split(config)
    original_predict = Pipeline.predict_proba
    original_fit = TrainFeatureSelector.fit
    selector_fit_maxima = []

    @wraps(original_predict)
    def guarded_predict(self, X, **kwargs):
        if isinstance(X, pd.DataFrame) and "Time" in X:
            assert X.Time.max() < test.Time.min(), "O TEST não pode receber predict."
        return original_predict(self, X, **kwargs)

    def audited_fit(self, X, y):
        selector_fit_maxima.append(X.Time.max())
        return original_fit(self, X, y)

    monkeypatch.setattr(Pipeline, "predict_proba", guarded_predict)
    monkeypatch.setattr(TrainFeatureSelector, "fit", audited_fit)
    historical = tmp_path / "reports/metrics/final_test_metrics.json"
    historical.parent.mkdir(parents=True)
    historical.write_text('{"preserve": true}', encoding="utf-8")
    manifest = run(config, features, settings, tmp_path)
    assert manifest["combinations"] == 4 and manifest["grid_fits"] == 12
    assert not manifest["test_evaluated"]
    assert not manifest["validation_used_for_selection"]
    assert not manifest["selected_random_forest_replaced"]
    assert max(selector_fit_maxima) < manifest["validation_time_min"]
    folds = pd.read_csv(tmp_path / "reports/experiments/xgboost_gridsearch_folds.csv")
    assert (folds.fit_time_max < folds.validation_time_min).all()
    assert folds.validation_time_max.max() < manifest["validation_time_min"]
    artifact = joblib.load(tmp_path / "models/xgboost_smote_gridsearch.joblib")
    assert isinstance(artifact["pipeline"].named_steps["select"], TrainFeatureSelector)
    assert artifact["pipeline"].memory is None
    assert historical.read_text(encoding="utf-8") == '{"preserve": true}'
    selection = json.loads(
        (tmp_path / "reports/experiments/xgboost_gridsearch_selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["best_params"] == manifest["best_params"]
    with pytest.raises(FileExistsError):
        run(config, features, settings, tmp_path)
    # Uma grade ampliada reutiliza os mesmos scores de 0,05 e executa só a taxa nova.
    expanded = dict(
        settings, experiment_name="xgboost_learning_rate_fixture", reuse_from=str(tmp_path)
    )
    expanded["param_grid"] = {**settings["param_grid"], "learning_rate": [0.05, 0.1]}
    old_scores = (tmp_path / "reports/metrics/xgboost_gridsearch_results.csv").read_bytes()
    historical_plots = {
        suffix: (tmp_path / f"reports/figures/xgboost_gridsearch_{suffix}.png").read_bytes()
        for suffix in ["cv_f1_recall_precision", "validation_f1_recall_precision"]
    }
    second = run(config, features, expanded, tmp_path)
    assert second["combinations"] == 8
    assert second["reused_grid_fits"] == 12 and second["executed_grid_fits"] == 12
    combined = pd.read_csv(tmp_path / "reports/metrics/xgboost_learning_rate_fixture_results.csv")
    assert combined.reused_from_previous.sum() == 4
    for suffix, content in historical_plots.items():
        assert (
            tmp_path / f"reports/figures/xgboost_gridsearch_{suffix}.png"
        ).read_bytes() == content
        assert (tmp_path / f"reports/figures/xgboost_learning_rate_fixture_{suffix}.png").is_file()
    assert set(combined.param_model__learning_rate) == {0.05, 0.1}
    assert (tmp_path / "reports/metrics/xgboost_gridsearch_results.csv").read_bytes() == old_scores
    # Reutilização deixa de ser válida ao mudar seed/configuração ou dataset.
    with pytest.raises(ValueError, match="Reutilização recusada"):
        load_reusable_results(
            dict(config, seed=0), features, expanded, folds, manifest["dataset_sha256"]
        )
    with pytest.raises(ValueError, match="Reutilização recusada"):
        load_reusable_results(config, features, expanded, folds, "dataset_alterado")


def test_grid_rejects_threshold_tuning(config):
    features = load_config(ROOT / "configs/ablation.yaml")
    settings = load_config(ROOT / "configs/xgboost_gridsearch.yaml")
    settings["threshold"] = 0.7
    with pytest.raises(ValueError, match="threshold fixo"):
        make_search(build_search_pipeline(config, features, settings), [], settings)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, 1.5])
def test_grid_rejects_invalid_learning_rate(config, learning_rate):
    features = load_config(ROOT / "configs/ablation.yaml")
    settings = load_config(ROOT / "configs/xgboost_gridsearch.yaml")
    settings["param_grid"]["learning_rate"] = [learning_rate, 0.05]
    with pytest.raises(ValueError, match="learning_rate"):
        make_search(build_search_pipeline(config, features, settings), [], settings)
