import numpy as np
import pandas as pd

from fraud_detection.ablation import run, variant_pipeline
from fraud_detection.data import FEATURES, ROOT, load_config
from fraud_detection.features import audit_features, removal_mask
from fraud_detection.segments import error_rates_by_bins, fraud_rates_by_bins


def test_removal_requires_both_conditions_and_uses_absolute_correlation():
    correlation = np.array([0.01, -0.3, 0.01, 0.09, 0.01])
    importance = np.array([0.01, 0.01, 0.2, 0.01, 0.1])
    np.testing.assert_array_equal(
        removal_mask(correlation, importance), [True, False, False, False, False]
    )


def test_audit_and_scaling_are_fitted_inside_training_only(frame, config):
    settings = load_config(ROOT / "configs/ablation.yaml")
    audit = audit_features(frame.iloc[:1600], config, settings)
    assert audit.reference_fit_time_max.max() < audit.reference_validation_time_min.min()
    assert audit.reference_validation_time_max.max() <= frame.iloc[:1600].Time.max()
    assert audit.keep.any()
    selected = audit.loc[audit.keep, "feature"].tolist()
    pipeline = variant_pipeline("xgboost", "scaled_smote_selected", selected, config, settings)
    pipeline.fit(frame[FEATURES].iloc[:1600], frame.Class.iloc[:1600])
    preprocessor = pipeline.named_steps["preprocess"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, frame[selected].iloc[:1600].mean().to_numpy())
    assert preprocessor.transform(frame[FEATURES].iloc[1600:]).shape[1] == len(selected)
    assert len(pipeline.predict_proba(frame[FEATURES].iloc[1600:])) == len(frame) - 1600


def test_fraud_percentages_and_denominators(frame):
    table = fraud_rates_by_bins(frame, ["V10"])
    assert table["size"].sum() == len(frame)
    assert table.fraud_count.sum() == frame.Class.sum()
    np.testing.assert_allclose(table.fraud_rate_pct, 100 * table.fraud_count / table["size"])
    np.testing.assert_allclose(table.fraud_rate_pct, 100 * table["mean"])
    assert np.isclose(table.share_of_train_frauds_pct.sum(), 100)


def test_error_rates_distinguish_false_positives_from_false_alert_share():
    fit = pd.DataFrame({"V1": [0.0, 1.0, 2.0, 3.0]})
    validation = pd.DataFrame({"V1": [1.0, 1.0, 1.0, 1.0], "Class": [0, 0, 1, 1]})
    table = error_rates_by_bins(
        fit, validation, np.array([0.8, 0.1, 0.8, 0.8]), 0.5, fold=1, features=["V1"]
    )
    row = table.iloc[0]
    assert row.tp == 2 and row.fp == 1 and row.tn == 1 and row.fn == 0
    assert row.false_positive_rate_pct == 50
    assert np.isclose(row.false_alert_share_pct, 100 / 3)
    assert row.recall_pct == 100


def test_ablation_workflow_preserves_historical_test(frame, config, tmp_path):
    source = tmp_path / "original.csv"
    frame.to_csv(source, index=False)
    config["data_path"] = str(source)
    settings = load_config(ROOT / "configs/ablation.yaml")
    settings["recipes"] = ["raw_all", "scaled_selected"]
    threshold_settings = load_config(ROOT / "configs/threshold.yaml")
    threshold_settings["thresholds"] = [0, 0.1, 0.5, 1]
    historical = tmp_path / "reports/metrics/test_metrics.json"
    historical.parent.mkdir(parents=True)
    historical.write_text('{"historical": true}', encoding="utf-8")
    result = run(config, settings, threshold_settings, tmp_path)
    assert not result["test_evaluated"]
    assert historical.read_text(encoding="utf-8") == '{"historical": true}'
    assert (tmp_path / "models/standardized_selected_pipeline.joblib").is_file()
    decisions = pd.read_csv(tmp_path / "reports/metrics/feature_selection_final.csv")
    np.testing.assert_array_equal(
        decisions.remove, (decisions.abs_pearson < 0.09) & (decisions.permutation_ap < 0.1)
    )
