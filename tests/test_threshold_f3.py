import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from fraud_detection.threshold_f3 import add_f3, best_f3


def test_f3_matches_beta_three_score():
    table = pd.DataFrame(
        {
            "threshold": [0.5],
            "tp": [2],
            "fp": [1],
            "fn": [0],
            "recall": [1.0],
            "precision": [2 / 3],
        }
    )
    expected = fbeta_score([0, 0, 1, 1], [0, 1, 1, 1], beta=3)
    assert add_f3(table).iloc[0].f3 == pytest.approx(expected)


def test_f3_prefers_recall_in_the_observed_tradeoff():
    # Estas contagens ilustram a decisão na janela interna, sem consultar TEST.
    table = pd.DataFrame(
        {
            "threshold": [0.5, 0.7],
            "tp": [30, 29],
            "fp": [12, 5],
            "fn": [9, 10],
            "recall": [30 / 39, 29 / 39],
            "precision": [30 / 42, 29 / 34],
        }
    )
    chosen = best_f3(table)
    assert chosen.threshold == 0.5
    assert chosen.recall > table.iloc[1].recall
    assert chosen.precision < table.iloc[1].precision


def test_f3_zero_and_missing_inputs():
    table = pd.DataFrame(
        {
            "threshold": [1.0],
            "tp": [0],
            "fp": [0],
            "fn": [0],
            "recall": [0.0],
            "precision": [0.0],
        }
    )
    assert np.isfinite(add_f3(table).iloc[0].f3)
    assert add_f3(table).iloc[0].f3 == 0
    with pytest.raises(ValueError):
        add_f3(pd.DataFrame({"threshold": [0.5]}))
