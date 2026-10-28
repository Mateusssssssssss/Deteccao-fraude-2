import numpy as np

from fraud_detection.data import FEATURES, temporal_split
from fraud_detection.model import candidates
from fraud_detection.validation import evaluate_cv, temporal_folds


def test_temporal_folds_are_disjoint_and_never_see_test(frame, config):
    train, test, _ = temporal_split(frame)
    folds = temporal_folds(train, config)
    validations = []
    for fit_idx, val_idx in folds:
        assert not set(fit_idx) & set(val_idx)
        assert train.iloc[fit_idx].Time.max() < train.iloc[val_idx].Time.min()
        assert train.iloc[val_idx].Time.max() < test.Time.min()
        validations.extend(val_idx)
    assert len(validations) == len(set(validations))
    result = evaluate_cv(
        candidates(config)["xgboost__under_smote"], train[FEATURES], train.Class, folds
    )
    assert np.isfinite(result["test_average_precision"]).all()
