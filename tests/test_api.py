from fastapi.testclient import TestClient
from moneyscam.serving import api


class DummyModel:
    classes_ = ["non", "scam"]

    def predict_proba(self, X):
        return [[0.7, 0.3] for _ in range(len(X))]


def test_predict_endpoint(monkeypatch):
    api.model = DummyModel()
    client = TestClient(api.app)
    resp = client.post("/predict", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "probs" in data
