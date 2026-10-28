import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import FEATURES, load_config


@pytest.fixture
def frame():
    rng = np.random.default_rng(42)
    n = 2400
    data = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    data["Time"] = np.arange(n) // 2
    data["Amount"] = rng.exponential(20, n)
    data["Class"] = (np.arange(n) % 50 == 0).astype(int)
    data["V1"] += data.Class * 5
    return data


@pytest.fixture
def config():
    result = load_config()
    result["models"]["random_forest"]["n_estimators"] = 5
    result["models"]["xgboost"]["n_estimators"] = 5
    result["n_jobs"] = 1
    return result
