import json

import pytest

from fraud_detection.train import run


def test_full_workflow_and_test_lock(frame, config, tmp_path):
    data = tmp_path / "original.csv"
    frame.to_csv(data, index=False)
    config["data_path"] = str(data)
    comparison, final = run(config, tmp_path)
    assert len(comparison) == 5
    assert 0 <= final["average_precision"] <= 1
    selection = json.loads(
        (tmp_path / "reports/experiments/selection.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (tmp_path / "reports/metrics/test_metrics.json").read_text(encoding="utf-8")
    )
    assert result["model"] == selection["model"]
    assert comparison.iloc[0].model == selection["model"]
    with pytest.raises(FileExistsError, match="TEST já avaliado"):
        run(config, tmp_path)


@pytest.mark.parametrize("key,value", [("threshold", 0.2), ("primary_metric", "recall")])
def test_selection_policy_is_explicit(config, tmp_path, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        run(config, tmp_path)
