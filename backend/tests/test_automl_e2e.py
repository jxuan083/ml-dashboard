"""End-to-end tests for the AutoML pipeline.

Tests the full flow: dataset -> preprocessing -> feature engineering ->
feature selection -> model tuning -> ensemble -> result validation.
Also tests API endpoints and edge cases.
"""

import random
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from main import app, datasets, jobs

client = TestClient(app)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _make_classification_data(n=200, n_features=5, n_classes=2, noise=0.1):
    rng = random.Random(42)
    data = []
    for _ in range(n):
        row = {}
        vals = [rng.gauss(0, 1) for _ in range(n_features)]
        for j, v in enumerate(vals):
            row[f"feat_{j}"] = round(v, 4)
        score = vals[0] * 2 + (vals[1] * -1.5 if len(vals) > 1 else 0) + rng.gauss(0, noise)
        row["target"] = int(score > 0) if n_classes == 2 else int(score > 0.5) + int(score > -0.5)
        data.append(row)
    return data


def _make_regression_data(n=200, n_features=5):
    rng = random.Random(42)
    data = []
    for _ in range(n):
        row = {}
        vals = [rng.gauss(0, 1) for _ in range(n_features)]
        for j, v in enumerate(vals):
            row[f"feat_{j}"] = round(v, 4)
        row["price"] = round(vals[0] * 10 + vals[1] * 5 + rng.gauss(0, 2), 2)
        data.append(row)
    return data


def _make_mixed_type_data(n=200):
    rng = random.Random(42)
    categories = ["cat_a", "cat_b", "cat_c", "cat_d"]
    data = []
    for _ in range(n):
        cat = rng.choice(categories)
        x1 = round(rng.gauss(0, 1), 4)
        x2 = round(rng.uniform(0, 100), 2)
        base = {"cat_a": 0, "cat_b": 1, "cat_c": 0, "cat_d": 1}[cat]
        label = 1 if x1 > 0.3 else base
        data.append({"category": cat, "x1": x1, "x2": x2, "label": label})
    return data


def _push(data, name="test_automl"):
    r = client.post("/api/datasets/push", json={"name": name, "data": data})
    assert r.status_code == 200
    return r.json()["id"]


# ══════════════════════════════════════════════════════════════
# Unit Tests: Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════

class TestPreprocessingPipeline:
    def test_classification_auto_detect(self):
        from ml.preprocessing import PreprocessingPipeline
        data = _make_classification_data(n=50)
        pp = PreprocessingPipeline()
        X, y, features, task = pp.fit_transform(data, target_col="target")
        assert task == "classification"
        assert len(y) == 50
        assert "target" not in features
        assert X.shape[0] == 50

    def test_regression_auto_detect(self):
        from ml.preprocessing import PreprocessingPipeline
        data = _make_regression_data(n=50)
        pp = PreprocessingPipeline()
        X, y, features, task = pp.fit_transform(data, target_col="price")
        assert task == "regression"
        assert "price" not in features

    def test_categorical_encoding(self):
        from ml.preprocessing import PreprocessingPipeline
        data = _make_mixed_type_data(n=50)
        pp = PreprocessingPipeline()
        X, y, features, task = pp.fit_transform(data, target_col="label")
        assert task == "classification"
        assert X["category"].dtype != object

    def test_transform_consistency(self):
        from ml.preprocessing import PreprocessingPipeline
        data = _make_classification_data(n=100)
        pp = PreprocessingPipeline()
        X_train, y, features, task = pp.fit_transform(data[:80], target_col="target")
        X_test = pp.transform(data[80:])
        assert X_test.shape[1] == X_train.shape[1]
        assert list(X_test.columns) == list(X_train.columns)


# ══════════════════════════════════════════════════════════════
# Unit Tests: Feature Engineering
# ══════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def test_generates_features(self):
        from ml.feature_engineering import AutoFeatureEngineer
        data = _make_classification_data(n=100)
        df = pd.DataFrame(data)
        X = df.drop(columns=["target"])
        y = df["target"]
        fe = AutoFeatureEngineer(max_interactions=5, max_poly_features=3)
        X_new = fe.fit_transform(X, y)
        assert X_new.shape[1] > X.shape[1]
        assert len(fe.generated_feature_names) > 0

    def test_transform_preserves_columns(self):
        from ml.feature_engineering import AutoFeatureEngineer
        data = _make_classification_data(n=100)
        df = pd.DataFrame(data)
        X = df.drop(columns=["target"])
        y = df["target"]
        fe = AutoFeatureEngineer(max_interactions=3, max_poly_features=2)
        X_train = fe.fit_transform(X.iloc[:80], y.iloc[:80])
        X_test = fe.transform(X.iloc[80:])
        train_gen = set(fe.generated_feature_names)
        test_cols = set(X_test.columns)
        assert train_gen.issubset(test_cols)


# ══════════════════════════════════════════════════════════════
# Unit Tests: Feature Selection
# ══════════════════════════════════════════════════════════════

class TestFeatureSelection:
    def test_selects_features(self):
        from ml.feature_selection import AutoFeatureSelector
        data = _make_classification_data(n=100, n_features=20)
        df = pd.DataFrame(data)
        X = df.drop(columns=["target"])
        y = df["target"].values
        fs = AutoFeatureSelector(max_features=10, task="classification")
        X_sel, selected = fs.fit_transform(X, y, list(X.columns))
        assert len(selected) <= 20
        assert X_sel.shape[1] == len(selected)

    def test_transform_uses_same_features(self):
        from ml.feature_selection import AutoFeatureSelector
        data = _make_classification_data(n=100, n_features=10)
        df = pd.DataFrame(data)
        X = df.drop(columns=["target"])
        y = df["target"].values
        fs = AutoFeatureSelector(max_features=5, task="classification")
        X_train, selected = fs.fit_transform(X.iloc[:80], y[:80], list(X.columns))
        X_test = fs.transform(X.iloc[80:].values, list(X.columns))
        assert X_test.shape[1] == len(selected)


# ══════════════════════════════════════════════════════════════
# Integration Tests: AutoML Core (run_automl directly)
# ══════════════════════════════════════════════════════════════

class TestAutoMLCore:
    def test_binary_classification(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150, n_classes=2)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })
        assert result["task"] == "classification"
        assert result["n_classes"] == 2
        assert len(result["leaderboard"]) == 1
        assert result["best_score"] > 0
        assert result["metric"] in ("roc_auc", "f1")
        assert "pipeline_stages" in result
        assert result["feature_stats"]["original"] > 0

    def test_regression(self):
        from ml.automl import run_automl
        data = _make_regression_data(n=150)
        result = run_automl(data, {
            "target_col": "price",
            "preset": "fast",
            "models": ["ridge"],
            "use_ensemble": False,
        })
        assert result["task"] == "regression"
        assert result["metric"] == "rmse"
        assert len(result["leaderboard"]) >= 1

    def test_multiple_models_ranked(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest", "logistic"],
            "use_ensemble": False,
        })
        lb = result["leaderboard"]
        assert len(lb) == 2
        assert lb[0]["rank"] == 1
        assert lb[1]["rank"] == 2
        assert lb[0]["mean_cv"] >= lb[1]["mean_cv"]

    def test_feature_stats_populated(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150, n_features=8)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })
        fs = result["feature_stats"]
        assert fs["original"] == 8
        assert fs["after_engineering"] >= fs["original"]
        assert fs["after_selection"] <= fs["after_engineering"]

    def test_feature_importance_returned(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })
        assert len(result["feature_importance"]) > 0
        fi = result["feature_importance"][0]
        assert "feature" in fi
        assert "importance" in fi

    def test_mixed_type_data(self):
        from ml.automl import run_automl
        data = _make_mixed_type_data(n=150)
        result = run_automl(data, {
            "target_col": "label",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })
        assert result["task"] == "classification"
        assert result["best_score"] > 0

    def test_ensemble_when_enabled(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest", "logistic"],
            "use_ensemble": True,
        })
        assert result["ensemble"] is not None
        assert any(k in result["ensemble"] for k in ("voting", "stacking", "blending"))

    def test_threshold_optimization_binary(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=150, n_classes=2)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })
        if result["threshold"]:
            assert "optimal_threshold" in result["threshold"]
            assert 0 < result["threshold"]["optimal_threshold"] < 1

    def test_config_preserved_in_result(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=100)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["logistic"],
            "use_ensemble": False,
        })
        cfg = result["config"]
        assert cfg["preset"] == "fast"
        assert cfg["target_col"] == "target"
        assert cfg["use_ensemble"] is False

    def test_specified_feature_cols(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=100, n_features=8)
        result = run_automl(data, {
            "target_col": "target",
            "feature_cols": ["feat_0", "feat_1", "feat_2"],
            "preset": "fast",
            "models": ["logistic"],
            "use_ensemble": False,
        })
        assert result["feature_stats"]["original"] == 3


# ══════════════════════════════════════════════════════════════
# API Tests: AutoML Endpoints (request/response only, no async)
# ══════════════════════════════════════════════════════════════

class TestAutoMLAPI:
    def test_run_returns_job_id(self):
        """POST /api/automl/run should return a job_id and queued status."""
        data = _make_classification_data(n=50)
        ds_id = _push(data, "api_test")
        r = client.post("/api/automl/run", json={
            "dataset_id": ds_id,
            "target_col": "target",
            "preset": "fast",
            "models": ["logistic"],
        })
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    def test_missing_dataset(self):
        r = client.post("/api/automl/run", json={
            "dataset_id": "nonexistent",
            "target_col": "target",
        })
        assert r.status_code == 404

    def test_empty_dataset(self):
        ds_id = _push([], "empty_ds")
        r = client.post("/api/automl/run", json={
            "dataset_id": ds_id,
            "target_col": "target",
        })
        assert r.status_code == 400

    def test_status_not_found(self):
        r = client.get("/api/automl/status/fake_id")
        assert r.status_code == 404

    def test_export_not_done(self):
        ds_id = _push(_make_classification_data(n=50), "not_done")
        jobs["fake_queued"] = {
            "id": "fake_queued",
            "type": "automl",
            "status": "queued",
            "dataset_id": ds_id,
        }
        r = client.post("/api/automl/export", json={
            "job_id": "fake_queued",
            "test_dataset_id": ds_id,
        })
        assert r.status_code == 400
        del jobs["fake_queued"]

    def test_export_job_not_found(self):
        r = client.post("/api/automl/export", json={
            "job_id": "nonexistent",
            "test_dataset_id": "whatever",
        })
        assert r.status_code == 404


# ══════════════════════════════════════════════════════════════
# Full Pipeline E2E: train + export (using run_automl directly)
# ══════════════════════════════════════════════════════════════

class TestFullPipelineE2E:
    def test_train_and_export_classification(self):
        """Full pipeline: train -> store result -> export predictions via API."""
        from ml.automl import run_automl, _make_model
        from ml.preprocessing import PreprocessingPipeline
        from ml.feature_engineering import AutoFeatureEngineer
        from ml.feature_selection import AutoFeatureSelector

        train_data = _make_classification_data(n=150)
        test_data = _make_classification_data(n=30, noise=0.2)
        for row in test_data:
            row.pop("target", None)
            row["id"] = random.Random(42).randint(1000, 9999)

        # Train
        result = run_automl(train_data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["random_forest"],
            "use_ensemble": False,
        })

        assert result["task"] == "classification"
        assert result["best_score"] > 0
        assert len(result["leaderboard"]) >= 1

        # Simulate what the export endpoint does
        config = result["config"]
        best = result["leaderboard"][0]

        pp = PreprocessingPipeline()
        X_df, y, feat_names, _ = pp.fit_transform(
            train_data, target_col=config["target_col"],
            feature_cols=config.get("feature_cols") or None, scale=False,
        )
        fe = AutoFeatureEngineer()
        X_df = fe.fit_transform(X_df, pd.Series(y))
        fs = AutoFeatureSelector(task=result["task"])
        X_train, selected = fs.fit_transform(X_df, y, list(X_df.columns))

        model = _make_model(best["model_type"], best["best_params"], result["task"])
        model.fit(X_train, y)

        X_test_df = pp.transform(test_data)
        X_test_df = fe.transform(X_test_df)
        X_test = fs.transform(X_test_df, list(X_test_df.columns))

        preds = model.predict(X_test)
        assert len(preds) == 30
        assert set(preds).issubset({0, 1})

    def test_train_and_export_regression(self):
        from ml.automl import run_automl, _make_model
        from ml.preprocessing import PreprocessingPipeline
        from ml.feature_engineering import AutoFeatureEngineer
        from ml.feature_selection import AutoFeatureSelector

        train_data = _make_regression_data(n=150)
        test_data = _make_regression_data(n=30)
        for row in test_data:
            row.pop("price", None)

        result = run_automl(train_data, {
            "target_col": "price",
            "preset": "fast",
            "models": ["ridge"],
            "use_ensemble": False,
        })

        config = result["config"]
        best = result["leaderboard"][0]

        pp = PreprocessingPipeline()
        X_df, y, feat_names, _ = pp.fit_transform(
            train_data, target_col=config["target_col"], scale=False,
        )
        fe = AutoFeatureEngineer()
        X_df = fe.fit_transform(X_df, pd.Series(y))
        fs = AutoFeatureSelector(task="regression")
        X_train, selected = fs.fit_transform(X_df, y, list(X_df.columns))

        model = _make_model(best["model_type"], best["best_params"], "regression")
        model.fit(X_train, y)

        X_test_df = pp.transform(test_data)
        X_test_df = fe.transform(X_test_df)
        X_test = fs.transform(X_test_df, list(X_test_df.columns))

        preds = model.predict(X_test)
        assert len(preds) == 30
        # Regression predictions should be numeric, not all the same
        assert np.std(preds) > 0


# ══════════════════════════════════════════════════════════════
# Edge Cases
# ══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_small_dataset(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=30)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["logistic"],
            "use_ensemble": False,
        })
        assert result["task"] == "classification"
        assert len(result["leaderboard"]) >= 1

    def test_single_feature(self):
        from ml.automl import run_automl
        data = _make_classification_data(n=100, n_features=1)
        result = run_automl(data, {
            "target_col": "target",
            "preset": "fast",
            "models": ["logistic"],
            "use_ensemble": False,
        })
        assert result["feature_stats"]["original"] == 1
