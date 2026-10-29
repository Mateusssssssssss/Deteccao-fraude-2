import numpy as np
import pytest

from fraud_detection.final_test import complete_metrics
from fraud_detection.model_select import verify_reference


def test_selected_metrics_reference_distinguishes_validation_from_test():
    # Reproduz as contagens da janela escolhida sem depender do CSV ou treinar.
    y = np.r_[np.zeros(54115), np.ones(44)]
    probability = np.r_[np.full(4, 0.8), np.full(54111, 0.1), np.full(33, 0.8), np.full(11, 0.1)]
    score = complete_metrics(y, probability, 0.7)
    reference = {
        "threshold": 0.7,
        "tn": 54111,
        "fp": 4,
        "fn": 11,
        "tp": 33,
        "precision": 33 / 37,
        "recall": 0.75,
        "f1": 66 / 81,
    }
    verify_reference(score, reference)
    # Métricas do TEST histórico não podem ser apresentadas como este recorte.
    with pytest.raises(ValueError, match="precision"):
        verify_reference(score, dict(reference, precision=1.0))


@pytest.mark.parametrize("key,value", [("threshold", 0.5), ("tp", 34), ("fn", 10)])
def test_reconstruction_refuses_changed_threshold_or_counts(key, value):
    original = {"threshold": 0.7, "tp": 33, "fn": 11}
    with pytest.raises(ValueError, match=key):
        verify_reference(dict(original, **{key: value}), original)
