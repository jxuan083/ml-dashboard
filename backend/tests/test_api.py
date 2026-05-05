import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── Health ──

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "datasets" in body
    assert "jobs" in body


# ── Dataset push + preview ──

def test_push_and_preview():
    payload = {
        "name": "test_ds",
        "data": [
            {"a": 1, "b": 2, "c": 0},
            {"a": 3, "b": 4, "c": 1},
            {"a": 5, "b": 6, "c": 0},
        ],
    }
    r = client.post("/api/datasets/push", json=payload)
    assert r.status_code == 200
    ds_id = r.json()["id"]

    r = client.get(f"/api/datasets/{ds_id}/preview?rows=2")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 2


def test_push_returns_metadata():
    payload = {"name": "meta_test", "data": [{"x": 1, "y": 2}]}
    r = client.post("/api/datasets/push", json=payload)
    body = r.json()
    assert body["rows"] == 1
    assert body["cols"] == 2


def test_get_dataset_not_found():
    r = client.get("/api/datasets/nonexistent")
    assert r.status_code == 404


# ── List datasets ──

def test_list_datasets():
    r = client.get("/api/datasets")
    assert r.status_code == 200
    assert "datasets" in r.json()


# ── Analyze: cluster ──

def _push_dataset(rows=50):
    import random
    data = [{"f1": random.random() * 10, "f2": random.random() * 10} for _ in range(rows)]
    r = client.post("/api/datasets/push", json={"name": "cluster_test", "data": data})
    return r.json()["id"]


def test_cluster_kmeans():
    ds_id = _push_dataset()
    r = client.post("/api/analyze/cluster", json={
        "dataset_id": ds_id, "model_id": "kmeans", "params": {"k": 3}
    })
    assert r.status_code == 200
    result = r.json()["result"]
    assert "groups" in result


def test_cluster_dbscan():
    ds_id = _push_dataset()
    r = client.post("/api/analyze/cluster", json={
        "dataset_id": ds_id, "model_id": "dbscan", "params": {}
    })
    assert r.status_code == 200


# ── Analyze: predict ──

def _push_classification_dataset(rows=80):
    import random
    data = [{"f1": random.random(), "f2": random.random(), "label": random.choice([0, 1])} for _ in range(rows)]
    r = client.post("/api/datasets/push", json={"name": "pred_test", "data": data})
    return r.json()["id"]


def test_predict_random_forest():
    ds_id = _push_classification_dataset()
    r = client.post("/api/analyze/predict", json={
        "dataset_id": ds_id, "model_id": "random_forest", "params": {"test_size": 0.2}
    })
    assert r.status_code == 200
    result = r.json()["result"]
    assert "accuracy" in result
    assert "f1" in result


def test_predict_logistic():
    ds_id = _push_classification_dataset()
    r = client.post("/api/analyze/predict", json={
        "dataset_id": ds_id, "model_id": "logistic", "params": {"test_size": 0.2}
    })
    assert r.status_code == 200


def test_predict_compare():
    ds_id = _push_classification_dataset()
    r = client.post("/api/analyze/predict/compare", json={
        "dataset_id": ds_id, "test_size": 0.2,
        "models": {"random_forest": {}, "logistic": {}},
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["rank"] == 1


# ── Analyze: recommend ──

def test_recommend_collab():
    ds_id = _push_dataset()
    r = client.post("/api/analyze/recommend", json={
        "dataset_id": ds_id, "model_id": "collab", "params": {}
    })
    assert r.status_code == 200


# ── Code execution ──

def test_code_run_basic():
    ds_id = _push_dataset()
    r = client.post("/api/code/run", json={
        "code": "result = df.shape[0]",
        "dataset_id": ds_id,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["data"] == 50


def test_code_run_stdout():
    ds_id = _push_dataset()
    r = client.post("/api/code/run", json={
        "code": "print('hello')\nresult = 42",
        "dataset_id": ds_id,
    })
    body = r.json()
    assert body["ok"] is True
    assert "hello" in body["stdout"]
    assert body["result"]["data"] == 42


def test_code_run_dataframe_result():
    ds_id = _push_dataset()
    r = client.post("/api/code/run", json={
        "code": "result = df.head(3)",
        "dataset_id": ds_id,
    })
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["type"] == "dataframe"
    assert len(body["result"]["rows"]) == 3


def test_code_run_error():
    ds_id = _push_dataset()
    r = client.post("/api/code/run", json={
        "code": "1/0",
        "dataset_id": ds_id,
    })
    body = r.json()
    assert body["ok"] is False
    assert "ZeroDivisionError" in body["error"]


def test_code_run_no_dataset():
    r = client.post("/api/code/run", json={
        "code": "result = 1 + 1",
    })
    body = r.json()
    assert body["ok"] is True
    assert body["result"]["data"] == 2


def test_code_too_long():
    r = client.post("/api/code/run", json={"code": "x" * 10001})
    assert r.status_code == 400


# ── Error cases ──

def test_analyze_unknown_type():
    ds_id = _push_dataset()
    r = client.post("/api/analyze/badtype", json={
        "dataset_id": ds_id, "model_id": "kmeans", "params": {}
    })
    assert r.status_code in (400, 422)


def test_analyze_missing_dataset():
    r = client.post("/api/analyze/cluster", json={
        "dataset_id": "nope", "model_id": "kmeans", "params": {}
    })
    assert r.status_code == 404
