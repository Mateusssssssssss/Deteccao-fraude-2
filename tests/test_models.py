import joblib
import numpy as np
import pytest

from fraud_detection.data import FEATURES
from fraud_detection.model import candidates
from fraud_detection.predict import predict


@pytest.mark.parametrize(
    "name",
    [
        "dummy",
        "random_forest__none",
        "xgboost__none",
        "random_forest__under_smote",
        "xgboost__under_smote",
    ],
)
def test_pipeline_and_persisted_inference(frame, config, tmp_path, name):
    pipeline = candidates(config)[name].fit(frame[FEATURES], frame.Class)
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "threshold": 0.5}, path)
    result = predict(frame[FEATURES].iloc[:17], path)
    assert len(result) == 17
    np.testing.assert_allclose(
        result.fraud_probability, pipeline.predict_proba(frame[FEATURES].iloc[:17])[:, 1]
    )
    with pytest.raises(ValueError, match="ausentes"):
        predict(frame.drop(columns="V1"), path)
