import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from ml.preprocessing import prepare_data

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def _prepare(data: list[dict], target_col: str | None = None, feature_cols: list[str] | None = None):
    return prepare_data(data, target_col, feature_cols, encoding="label", scale=True)


def _feature_importance(model, feature_cols: list[str]) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        imps = np.ones(len(feature_cols))
    imps = imps / (imps.sum() + 1e-9)
    pairs = sorted(zip(feature_cols, imps), key=lambda x: -x[1])[:6]
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


def _run(model, model_name: str, data: list[dict],
         target_col=None, test_size: float = 0.2, feature_cols=None) -> dict:
    X, y, feat_cols = _prepare(data, target_col, feature_cols)

    # Train / Test split
    can_split = len(X) >= 20 and test_size > 0
    if can_split:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "macro"

    acc       = round(float(accuracy_score(y_te, preds)), 4)
    precision = round(float(precision_score(y_te, preds, average=avg, zero_division=0)), 4)
    recall    = round(float(recall_score(y_te, preds, average=avg, zero_division=0)), 4)
    f1        = round(float(f1_score(y_te, preds, average=avg, zero_division=0)), 4)

    roc_auc = None
    sample_proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_te)
            sample_proba = [round(float(v), 4) for v in model.predict_proba(X_te[:1])[0]]
            if n_classes == 2:
                roc_auc = round(float(roc_auc_score(y_te, proba[:, 1])), 4)
            else:
                roc_auc = round(float(roc_auc_score(y_te, proba, multi_class="ovr", average="macro")), 4)
        except Exception:
            pass

    split_label = f"test {int(test_size*100)}% / train {int((1-test_size)*100)}%"

    metrics = [
        {"label": "準確率",  "score": f"{acc*100:.1f}%",       "value": acc,       "trend": "up",     "delta": split_label},
        {"label": "Precision","score": f"{precision*100:.1f}%","value": precision, "trend": "neutral", "delta": avg},
        {"label": "Recall",  "score": f"{recall*100:.1f}%",    "value": recall,    "trend": "neutral", "delta": avg},
        {"label": "F1 Score","score": f"{f1*100:.1f}%",        "value": f1,        "trend": "up",      "delta": avg},
    ]
    if roc_auc is not None:
        metrics.append({"label": "ROC-AUC", "score": f"{roc_auc:.3f}", "value": roc_auc, "trend": "up", "delta": "macro OvR"})

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "cv_accuracy": f1,  # use F1 as primary quality signal
        "feature_importance": _feature_importance(model, feat_cols),
        "sample_prediction": {
            "probabilities": sample_proba,
            "predicted_class": int(preds[0]) if len(preds) > 0 else None,
        },
        "metrics": metrics,
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
    }


def run_xgboost(data, target_col=None, n_estimators=100, max_depth=4, learning_rate=0.1,
                test_size=0.2, feature_cols=None, **kw):
    if HAS_XGB:
        m = XGBClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth),
                          learning_rate=float(learning_rate), random_state=42,
                          eval_metric="logloss", verbosity=0)
    else:
        m = GradientBoostingClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth),
                                       learning_rate=float(learning_rate), random_state=42)
    return _run(m, "XGBoost" if HAS_XGB else "GradientBoosting", data, target_col, test_size, feature_cols)


def run_random_forest(data, target_col=None, n_estimators=100,
                      test_size=0.2, feature_cols=None, **kw):
    m = RandomForestClassifier(n_estimators=int(n_estimators), random_state=42)
    return _run(m, "Random Forest", data, target_col, test_size, feature_cols)


def run_logistic(data, target_col=None, max_iter=500,
                 test_size=0.2, feature_cols=None, **kw):
    m = LogisticRegression(max_iter=int(max_iter), random_state=42)
    return _run(m, "Logistic Regression", data, target_col, test_size, feature_cols)


def run_lightgbm(data, target_col=None, n_estimators=100,
                 test_size=0.2, feature_cols=None, **kw):
    if HAS_LGB:
        m = LGBMClassifier(n_estimators=int(n_estimators), random_state=42, verbose=-1)
    else:
        m = RandomForestClassifier(n_estimators=int(n_estimators), random_state=42)
    return _run(m, "LightGBM" if HAS_LGB else "Random Forest", data, target_col, test_size, feature_cols)


RUNNERS = {
    "xgboost":       run_xgboost,
    "random_forest": run_random_forest,
    "logistic":      run_logistic,
    "lightgbm":      run_lightgbm,
}


def run_predict(data: list[dict], model_id: str, params: dict = {}) -> dict:
    fn = RUNNERS.get(model_id, run_xgboost)
    return fn(data, **params)
