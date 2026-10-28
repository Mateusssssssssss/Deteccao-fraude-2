import pandas as pd
import pytest

from fraud_detection.data import load_data, temporal_split, validate_schema


def test_chronology_duplicates_and_ties(frame):
    train, test, meta = temporal_split(pd.concat([frame, frame.iloc[[0]]]))
    assert meta["duplicates_removed"] == 1
    assert train.Time.max() < test.Time.min()
    assert len(train) + len(test) == len(frame)
    assert not set(train.Time) & set(test.Time)


def test_schema_failures(frame, tmp_path):
    with pytest.raises(ValueError, match="ausentes"):
        validate_schema(frame.drop(columns="V1"))
    with pytest.raises(FileNotFoundError, match="original ausente"):
        load_data(tmp_path / "missing.csv")
    invalid = frame.copy()
    invalid.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="duas classes"):
        validate_schema(invalid)
    invalid = frame.copy()
    invalid.loc[0, "Amount"] = -1
    with pytest.raises(ValueError, match="negativos"):
        validate_schema(invalid)
