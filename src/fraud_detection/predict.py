"""Inferência com pipeline e threshold persistidos."""

import joblib
import numpy as np
import pandas as pd


def predict(frame: pd.DataFrame, artifact_path) -> pd.DataFrame:
    # O artefato reúne o pré-processamento ajustado, o estimador e o threshold.
    artifact = joblib.load(artifact_path)
    features = artifact["features"]
    # Rejeita schema incompatível em vez de prever com colunas trocadas ou ausentes.
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Features ausentes: {sorted(missing)}")
    X = frame[features]
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in features):
        raise ValueError("Features devem ser numéricas.")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Features infinitas não são aceitas.")
    # A inferência usa o mesmo pipeline do treino, sem ajustar scaler nem aplicar SMOTE.
    probability = artifact["pipeline"].predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "fraud_probability": probability,
            "prediction": (probability >= artifact["threshold"]).astype(int),
        },
        index=frame.index,
    )
