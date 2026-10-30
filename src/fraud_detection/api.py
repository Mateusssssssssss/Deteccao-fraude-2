"""API de inferência do RF + SMOTE; carrega um único pipeline na inicialização."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud_detection.data import FEATURES
from fraud_detection.deployment import DEFAULT_MODEL
from fraud_detection.final_test import file_hash
from fraud_detection.predict import predict_with_artifact

# Mesmo contrato da base: 30 medidas numéricas, sem target ou campos adicionais.
# A escala/PCA deve ser a mesma da origem: V1–V28 já são componentes fornecidos pela base.
Numeric = Annotated[float, Field(strict=True, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0)]
Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid"),
    **{
        feature: (NonNegative if feature in {"Time", "Amount"} else Numeric, ...)
        for feature in FEATURES
    },
)


class PredictionRequest(BaseModel):
    """Lotes pequenos limitam o consumo de memória e tempo por requisição."""

    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    fraud_probability: float
    prediction: int


class PredictionResponse(BaseModel):
    model_sha256: str
    threshold: float
    positive_class: int
    score_calibrated: bool
    predictions: list[Prediction]


def create_app(artifact_path: str | Path | None = None) -> FastAPI:
    """Factory facilita testes isolados; importar este módulo não carrega nem treina."""
    path = Path(artifact_path or os.environ.get("FRAUD_MODEL_PATH", DEFAULT_MODEL)).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Falha na inicialização se o pacote está ausente ou foi alterado.
        metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
        checksum = file_hash(path)
        if checksum != metadata["artifact_sha256"]:
            raise ValueError("Checksum do modelo diverge de metadata.json.")
        artifact = joblib.load(path)
        if (
            artifact["features"] != FEATURES
            or artifact["threshold"] != 0.7
            or metadata["threshold"] != artifact["threshold"]
            or artifact["positive_class"] != 1
            or metadata["selected_features"] != artifact["selected_features"]
        ):
            raise ValueError("Schema, features ou threshold incompatível com best_model.")
        app.state.artifact = artifact
        app.state.metadata = metadata
        app.state.checksum = checksum
        yield
        del app.state.artifact

    app = FastAPI(
        title="Detecção de fraude — Random Forest + SMOTE",
        version="1.0.0",
        description=(
            "Class=1 representa fraude. Threshold fixo: 0,70. "
            "Scores não possuem calibração demonstrada; os mesmos componentes V1–V28 "
            "da origem são necessários. Sem treino ou SMOTE durante a inferência."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        # Não devolve valores da transação; também evita serializar NaN/Infinity no erro.
        details = [{key: item[key] for key in ["type", "loc", "msg"]} for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "model_loaded": hasattr(request.app.state, "artifact")}

    @app.get("/model")
    def model_info(request: Request) -> dict:
        metadata = request.app.state.metadata
        return {
            key: metadata[key]
            for key in [
                "model",
                "threshold",
                "positive_class",
                "features",
                "selected_features",
                "artifact_sha256",
                "score_calibrated",
            ]
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> dict:
        # DataFrame impõe a ordem; nenhum scaler, seletor ou sampler é ajustado aqui.
        frame = pd.DataFrame([row.model_dump() for row in body.transactions], columns=FEATURES)
        scores = predict_with_artifact(frame, request.app.state.artifact)
        return {
            "model_sha256": request.app.state.checksum,
            "threshold": 0.7,
            "positive_class": 1,
            "score_calibrated": False,
            "predictions": scores.to_dict(orient="records"),
        }

    return app


app = create_app()
