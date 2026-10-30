"""Verifica serialização, contrato HTTP e consistência de threshold sem usar TEST."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import anyio.to_thread
import joblib
import numpy as np
import pytest

from fraud_detection.api import create_app
from fraud_detection.data import FEATURES
from fraud_detection.deployment import export_best_model
from fraud_detection.final_test import file_hash
from fraud_detection.predict import predict_with_artifact
from fraud_detection.sampling_comparison import make_pipeline


def finish(coroutine):
    """Executa uma chamada ASGI cujas operações de I/O foram substituídas por memória."""
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError("O teste ASGI tentou aguardar I/O externo.")


class MemoryClient:
    """Exercita roteamento, validação e resposta HTTP sem sockets ou event loop TCP."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, json=None):
        body = b"" if json is None else __import__("json").dumps(json).encode()
        messages = []

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        finish(self.app(scope, receive, send))
        status = next(item["status"] for item in messages if item["type"] == "http.response.start")
        content = b"".join(
            item.get("body", b"") for item in messages if item["type"] == "http.response.body"
        )
        return SimpleNamespace(status_code=status, json=lambda: __import__("json").loads(content))

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, json):
        return self.request("POST", path, json)


@contextmanager
def memory_client(app):
    """Substitui só o executor de threads no teste; a aplicação ASGI real é preservada."""

    # O sandbox Windows bloqueia socketpair do event loop; não abrimos conexão de rede.
    async def direct(function, *args, **kwargs):
        return function(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(anyio.to_thread, "run_sync", direct)
        lifespan = app.router.lifespan_context(app)
        finish(lifespan.__aenter__())
        try:
            yield MemoryClient(app)
        finally:
            finish(lifespan.__aexit__(None, None, None))


@pytest.fixture
def package(tmp_path, frame, config):
    # Um pipeline pequeno e realmente ajustado exercita scaler e SMOTE de ponta a ponta.
    models = tmp_path / "models"
    experiments = tmp_path / "reports/experiments"
    models.mkdir()
    experiments.mkdir(parents=True)
    selected = ["V1", "V2", "V3", "V4"]
    pipeline = make_pipeline(
        "random_forest", "smote", selected, config, {"sampling_ratio": 0.1}
    ).fit(frame[FEATURES], frame.Class)
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "selected_features": selected,
        "model": "random_forest__scaled_smote_selected",
        "threshold": 0.7,
    }
    source = models / "standardized_selected_pipeline.joblib"
    joblib.dump(artifact, source)
    selection = {"dataset_sha256": "synthetic", "selected_features": selected}
    choice = {
        "choice": {"model": "random_forest__smote", "threshold": 0.7},
        "selected_features": selected,
        "model_parameters": pipeline.named_steps["model"].get_params(),
        "dataset_sha256": "synthetic",
    }
    for name, value in [
        ("ablation_selection", selection),
        ("model_select_manifest", choice),
        ("final_test_manifest", {"artifact_sha256": file_hash(source)}),
    ]:
        (experiments / f"{name}.json").write_text(json.dumps(value), encoding="utf-8")
    path = export_best_model(tmp_path)
    return path, artifact, frame


def test_export_preserves_full_pipeline_and_probabilities(package):
    path, original, frame = package
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        original["pipeline"].predict_proba(frame[FEATURES]),
        restored["pipeline"].predict_proba(frame[FEATURES]),
    )
    assert restored["threshold"] == 0.7
    assert restored["positive_class"] == 1
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_sha256"] == file_hash(path)


def test_export_rejects_modified_source(package):
    path, _, _ = package
    root = path.parent.parent
    source = root / "models/standardized_selected_pipeline.joblib"
    with source.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="original mudou"):
        export_best_model(root)


def test_api_uses_loaded_pipeline_once_without_fit(package, monkeypatch):
    path, artifact, frame = package
    from fraud_detection import api

    calls = []
    load = joblib.load

    def counted_load(model_path):
        calls.append(model_path)
        return load(model_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Inferência não pode ajustar ou reamostrar.")

    monkeypatch.setattr(api.joblib, "load", counted_load)
    monkeypatch.setattr(type(artifact["pipeline"]), "fit", forbidden)
    monkeypatch.setattr(type(artifact["pipeline"].named_steps["smote"]), "fit_resample", forbidden)
    batch = frame.iloc[:100][FEATURES]
    expected = predict_with_artifact(batch, artifact)
    payload = {"transactions": batch.to_dict(orient="records")}
    with memory_client(create_app(path)) as client:
        assert client.get("/health").json()["model_loaded"]
        assert client.get("/model").json()["threshold"] == 0.7
        for _ in range(2):
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert body["threshold"] == 0.7
            np.testing.assert_array_equal(
                [row["prediction"] for row in body["predictions"]], expected.prediction
            )
            np.testing.assert_allclose(
                [row["fraud_probability"] for row in body["predictions"]],
                expected.fraud_probability,
            )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "string", "negative", "null", "empty", "nan", "infinite", "large"],
)
def test_api_rejects_invalid_input(package, change):
    path, _, frame = package
    transaction = frame.iloc[0][FEATURES].to_dict()
    if change == "missing":
        transaction.pop("V2")
    elif change == "extra":
        transaction["Class"] = 1
    elif change == "string":
        transaction["V1"] = "0.4"
    elif change == "negative":
        transaction["Amount"] = -1
    elif change == "null":
        transaction["V1"] = None
    elif change == "nan":
        transaction["V1"] = float("nan")
    elif change == "infinite":
        transaction["V1"] = float("inf")
    transactions = [] if change == "empty" else [transaction] * (1001 if change == "large" else 1)
    payload = {"transactions": transactions}
    with memory_client(create_app(path)) as client:
        assert client.post("/predict", json=payload).status_code == 422


def test_api_rejects_tampered_artifact(package):
    path, _, _ = package
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="Checksum"), memory_client(create_app(path)):
        pass


def test_mlflow_wrapper_keeps_threshold_and_scores(package):
    pytest.importorskip("mlflow")
    from fraud_detection.tracking import FraudModel

    path, artifact, frame = package
    wrapper = FraudModel()
    wrapper.load_context(SimpleNamespace(artifacts={"pipeline": str(path)}))
    actual = wrapper.predict(None, frame[FEATURES])
    expected = predict_with_artifact(frame[FEATURES], artifact)
    np.testing.assert_array_equal(actual.prediction, expected.prediction)
    np.testing.assert_array_equal(actual.fraud_probability, expected.fraud_probability)
