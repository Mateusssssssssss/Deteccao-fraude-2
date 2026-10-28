"""Schema e separação temporal antes de qualquer ajuste."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"


def load_config(path: str | Path = ROOT / "configs/experiment.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def validate_schema(frame: pd.DataFrame) -> None:
    # Valida o contrato de entrada antes de qualquer transformação ou treino.
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES + [TARGET]):
        raise ValueError("Todas as colunas devem ser numéricas.")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Class deve conter as duas classes 0 e 1 e nenhum nulo.")
    if frame.Time.isna().any() or not np.isfinite(frame.Time).all():
        raise ValueError("Time deve ser finito e não ausente.")
    if np.isinf(frame[FEATURES].to_numpy()).any():
        raise ValueError("Features não podem conter infinitos.")
    if (frame.Time < 0).any() or (frame.Amount.dropna() < 0).any():
        raise ValueError("Time e Amount não podem ser negativos.")


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Base original ausente: {path}. Coloque creditcard.csv em data/raw/."
        )
    # Apenas leitura: nenhuma estatística de imputação/scaling é aprendida aqui.
    frame = pd.read_csv(path)
    validate_schema(frame)
    return frame[FEATURES + [TARGET]]


def temporal_split(
    frame: pd.DataFrame, test_time_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove cópias exatas; empates temporais permanecem do mesmo lado."""
    validate_schema(frame)
    if not 0 < test_time_fraction < 1:
        raise ValueError("test_time_fraction deve estar entre 0 e 1.")
    # Remoção determinística de cópias exatas: o CSV original nunca é modificado.
    clean = frame.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    # O corte representa uma fração da duração, e não uma fração das linhas.
    cutoff = clean.Time.min() + (1 - test_time_fraction) * (clean.Time.max() - clean.Time.min())
    # Usar < e >= mantém transações com o mesmo instante no mesmo conjunto.
    train, test = clean.loc[clean.Time < cutoff].copy(), clean.loc[clean.Time >= cutoff].copy()
    for name, part in [("TRAIN", train), ("TEST", test)]:
        if part.empty or part[TARGET].nunique() != 2:
            raise ValueError(f"{name} precisa conter as duas classes; reveja corte temporal.")
    return (
        train,
        test,
        {"cutoff_time": float(cutoff), "duplicates_removed": len(frame) - len(clean)},
    )


def read_split(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    path = Path(config["data_path"])
    if not path.is_absolute():
        path = ROOT / path
    frame = load_data(path)
    train, test, metadata = temporal_split(frame, config["test_time_fraction"])
    return train, test, metadata, path
