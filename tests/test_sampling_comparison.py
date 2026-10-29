import json

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import FEATURES, ROOT, load_config
from fraud_detection.sampling_comparison import (
    make_pipeline,
    refresh_reports,
    run,
    sampling_counts,
    sampling_metrics,
    summarize,
)


@pytest.mark.parametrize("strategy", ["smote", "under", "none"])
def test_samplers_are_isolated_and_scaler_uses_only_fit(frame, config, strategy):
    settings = load_config(ROOT / "configs/sampling_comparison.yaml")
    selected = ["V1", "V2", "Amount"]
    fit = frame.iloc[:1600].copy()
    fit.loc[0, "Amount"] = np.nan
    pipeline = make_pipeline("random_forest", strategy, selected, config, settings)
    pipeline.fit(fit[FEATURES], fit.Class)
    assert ("smote" in pipeline.named_steps) == (strategy == "smote")
    assert ("under" in pipeline.named_steps) == (strategy == "under")
    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    expected = fit[selected].fillna(fit[selected].median()).mean().to_numpy()
    np.testing.assert_allclose(numeric.named_steps["scale"].mean_, expected)
    counts = sampling_counts(pipeline, fit.Class)
    if strategy == "smote":
        assert counts["legitimate_after"] == counts["legitimate_before"]
        assert counts["synthetic_frauds"] > 0 and counts["legitimate_removed"] == 0
    elif strategy == "under":
        assert counts["fraud_after"] == counts["fraud_before"]
        assert counts["legitimate_removed"] > 0 and counts["synthetic_frauds"] == 0
        indices = pipeline.named_steps["under"].sample_indices_
        assert indices.max() < len(fit)
    else:
        assert counts["legitimate_removed"] == counts["synthetic_frauds"] == 0
    assert len(pipeline.predict_proba(frame[FEATURES].iloc[1600:])) == len(frame) - 1600


def test_comparison_workflow_preserves_test_and_main_model(frame, config, tmp_path, monkeypatch):
    data = tmp_path / "original.csv"
    frame.to_csv(data, index=False)
    config["data_path"] = str(data)
    feature_settings = load_config(ROOT / "configs/ablation.yaml")
    feature_settings["importance_repeats"] = 1
    settings = load_config(ROOT / "configs/sampling_comparison.yaml")
    settings["permutation_repeats"] = 1
    settings["iteration_points"] = [1, 5]
    settings["threshold_diagnostic_grid"] = [0.1, 0.5, 0.9]
    historical = tmp_path / "reports/metrics/final_test_metrics.json"
    historical.parent.mkdir(parents=True)
    historical.write_text('{"preserved":true}', encoding="utf-8")
    primary = tmp_path / "models/standardized_selected_pipeline.joblib"
    primary.parent.mkdir()
    primary.write_bytes(b"primary model must stay untouched")
    result = run(config, feature_settings, settings, tmp_path)
    assert not result["test_evaluated"] and not result["primary_model_replaced"]
    assert historical.read_text(encoding="utf-8") == '{"preserved":true}'
    assert primary.read_bytes() == b"primary model must stay untouched"
    comparison = pd.read_csv(tmp_path / "reports/metrics/sampling_comparison.csv")
    assert len(comparison) == 7 and comparison.threshold.eq(0.5).all()

    assert "f2" not in comparison.columns
    original_f1 = comparison.set_index("model").f1.sort_index()
    # Simula os relatórios antigos: a atualização remove a coluna antiga sem mudar decisões.
    folds_path = tmp_path / "reports/metrics/sampling_folds.csv"
    folds = pd.read_csv(folds_path)
    folds["f2"] = 0.0
    folds["train_f2"] = 0.0
    folds.to_csv(folds_path, index=False)
    manifest_path = tmp_path / "reports/experiments/sampling_comparison_manifest.json"
    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached["settings"]["primary_metric"] = "f2"
    manifest_path.write_text(json.dumps(cached), encoding="utf-8")

    def reject_fit(*args, **kwargs):
        raise AssertionError("Atualizar relatório não pode retreinar.")

    monkeypatch.setattr("imblearn.pipeline.Pipeline.fit", reject_fit)
    refreshed = refresh_reports(tmp_path, settings)
    assert refreshed["selection_metric"] == "F1 média na CV"
    assert refreshed["metrics_recomputed_without_refit"]
    assert "f2" not in pd.read_csv(folds_path).columns
    updated = pd.read_csv(tmp_path / "reports/metrics/sampling_comparison.csv")
    pd.testing.assert_series_equal(original_f1, updated.set_index("model").f1.sort_index())
    assert updated.iloc[0].f1 == updated.f1.max()
    assert historical.read_text(encoding="utf-8") == '{"preserved":true}'
    assert primary.read_bytes() == b"primary model must stay untouched"

    counts = pd.read_csv(tmp_path / "reports/metrics/sampling_counts.csv")
    assert counts.loc[counts.strategy.eq("under"), "synthetic_frauds"].eq(0).all()
    assert counts.loc[counts.strategy.eq("smote"), "legitimate_removed"].eq(0).all()
    assert counts.loc[counts.strategy.eq("none"), "synthetic_frauds"].eq(0).all()
    for strategy in ["smote", "under"]:
        assert (
            tmp_path / f"reports/figures/{strategy}_comparison_f1_recall_precision.png"
        ).is_file()
        assert (tmp_path / f"reports/metrics/{strategy}_association_importance.csv").is_file()
        assert (tmp_path / f"reports/metrics/{strategy}_category_errors.csv").is_file()
    manifest = json.loads(
        (tmp_path / "reports/experiments/sampling_comparison_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["oof_coverage"] < 1
    with pytest.raises(FileExistsError):
        run(config, feature_settings, settings, tmp_path)


def test_ranking_prioritizes_f1_not_recall():
    y = np.r_[np.ones(30, dtype=int), np.zeros(70, dtype=int)]
    precise = np.full(100, 0.1)
    precise[:18] = 0.9
    precise[30:32] = 0.9
    sensitive = np.full(100, 0.1)
    sensitive[:24] = 0.9
    sensitive[30:46] = 0.9
    rows = []
    for name, probability in [("precise", precise), ("sensitive", sensitive)]:
        for fold in [1, 2]:
            rows.append(
                {
                    "strategy": "smote",
                    "model": name,
                    "fold": fold,
                    "fit_seconds": 0.0,
                    **sampling_metrics(y, probability, 0.5),
                }
            )
    summary = summarize(pd.DataFrame(rows))
    assert summary.iloc[0].model == "precise"
    assert "f2" not in summary.columns
    assert summary.iloc[0].f1 == pytest.approx(0.72)


def test_refresh_rejects_changed_sampling_protocol(tmp_path):
    settings = load_config(ROOT / "configs/sampling_comparison.yaml")
    manifest = tmp_path / "reports/experiments/sampling_comparison_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"settings": settings}), encoding="utf-8")
    changed = dict(settings, sampling_ratio=0.2)
    with pytest.raises(ValueError, match="além da métrica"):
        refresh_reports(tmp_path, changed)
