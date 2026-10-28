import numpy as np
from sklearn.base import clone

from fraud_detection.data import FEATURES, temporal_split
from fraud_detection.model import candidates
from fraud_detection.validation import temporal_folds


def test_resampling_only_fit_fold_and_imputer_is_local(frame, config):
    train, _, _ = temporal_split(frame)
    fit_idx, val_idx = temporal_folds(train, config)[0]
    X = train[FEATURES].copy()
    X.loc[fit_idx[0], "Amount"] = np.nan
    pipeline = candidates(config)["random_forest__under_smote"]
    pipeline.fit(X.iloc[fit_idx], train.Class.iloc[fit_idx])
    imputer = (
        pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    )
    assert imputer.statistics_[-1] == np.nanmedian(X.iloc[fit_idx].Amount)
    sampler = pipeline.named_steps["under"]
    assert sampler.sample_indices_.max() < len(fit_idx)
    indices = sampler.sample_indices_.copy()
    assert len(pipeline.predict_proba(X.iloc[val_idx])) == len(val_idx)
    np.testing.assert_array_equal(indices, sampler.sample_indices_)
    # Validação extrema não participa do ajuste.
    changed = X.copy()
    changed.loc[val_idx, "Amount"] = 1e12
    other = clone(pipeline).fit(changed.iloc[fit_idx], train.Class.iloc[fit_idx])
    np.testing.assert_allclose(
        pipeline.predict_proba(X.iloc[val_idx]), other.predict_proba(X.iloc[val_idx])
    )
