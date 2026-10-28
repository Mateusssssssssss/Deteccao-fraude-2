import numpy as np
import pandas as pd

from fraud_detection.evaluation import comparison_plot, metrics


def test_ap_uses_probabilities_not_labels():
    y = np.array([0, 1, 0, 1])
    result = metrics(y, np.array([0.1, 0.4, 0.2, 0.3]))
    assert result["average_precision"] == 1
    assert result["recall"] == 0
    assert result["fn"] == 2


def test_grouped_comparison_plot(tmp_path):
    table = pd.DataFrame(
        [
            {
                "model": "rf",
                "average_precision": 0.7,
                "f1": 0.6,
                "recall": 0.5,
                "precision": 0.8,
                "f1_std": 0.1,
                "recall_std": 0.1,
                "precision_std": 0.1,
            }
        ]
    )
    path = tmp_path / "comparison.png"
    comparison_plot(table, path)
    assert path.stat().st_size > 1000
