import pytest
from fastapi.testclient import TestClient

from machinewatch import api
from machinewatch.data import SENSORS
from machinewatch.registry import ModelMeta


class FakePredictor:
    """Cycles left = 200 minus the cycle number: enough to test the flow without a trained model."""

    meta = ModelMeta("test-version", "fake", "2026-01-01", {"rmse": 1.0}, "x")

    def predict(self, recent):
        return float(200 - recent["cycle"].iloc[-1])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("MACHINEWATCH_INGEST_KEY", raising=False)
    monkeypatch.delenv("MACHINEWATCH_ALLOW_RESET", raising=False)
    store = api.Store(str(tmp_path / "test.db"))
    api.app.dependency_overrides[api.get_store] = lambda: store
    api.app.dependency_overrides[api.get_predictor] = FakePredictor
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def reading(unit=1, cycle=1, **override):
    body = {"unit": unit, "cycle": cycle, "settings": [0, 0, 100], "sensors": {s: 1.0 for s in SENSORS}}
    return body | override


def test_ingest_predicts_and_stores(client):
    response = client.post("/api/readings", json=reading(cycle=150))
    assert response.status_code == 200
    assert response.json() | {"model_version": None} == {
        "unit": 1, "cycle": 150, "rul": 50.0, "status": "watch", "duplicate": False, "model_version": None
    }
    assert client.get("/api/engines/1").json()[0]["rul"] == 50.0


def test_a_retried_reading_is_not_counted_twice(client):
    client.post("/api/readings", json=reading())
    assert client.post("/api/readings", json=reading()).json()["duplicate"] is True
    assert len(client.get("/api/engines/1").json()) == 1


def test_fleet_lists_most_urgent_engine_first(client):
    client.post("/api/readings", json=reading(unit=1, cycle=20))   # 180 left
    client.post("/api/readings", json=reading(unit=2, cycle=190))  # 10 left
    fleet = client.get("/api/fleet").json()
    assert [e["unit"] for e in fleet] == [2, 1]
    assert fleet[0]["status"] == "critical"


def test_a_reading_must_carry_all_21_sensors(client):
    body = reading()
    del body["sensors"]["s7"]
    assert client.post("/api/readings", json=body).status_code == 422


def test_ingest_requires_the_key_when_one_is_configured(client, monkeypatch):
    monkeypatch.setenv("MACHINEWATCH_INGEST_KEY", "secret")
    assert client.post("/api/readings", json=reading()).status_code == 401
    assert client.post("/api/readings", json=reading(), headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/readings", json=reading(), headers={"X-API-Key": "secret"}).status_code == 200


def test_reset_is_off_unless_enabled(client, monkeypatch):
    client.post("/api/readings", json=reading())
    assert client.post("/api/reset").status_code == 403
    monkeypatch.setenv("MACHINEWATCH_ALLOW_RESET", "1")
    assert client.post("/api/reset").status_code == 200
    assert client.get("/api/fleet").json() == []


def test_unknown_engine_is_a_404(client):
    assert client.get("/api/engines/999").status_code == 404
