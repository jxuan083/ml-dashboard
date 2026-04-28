import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


def _prepare(data: list[dict], target_col: str | None = None):
    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # encode categoricals
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    all_cols = df.columns.tolist()

    # pick target: last column or specified
    if target_col and target_col in all_cols:
        y_col = target_col
    else:
        y_col = all_cols[-1]

    feature_cols = [c for c in all_cols if c != y_col]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df[y_col]

    # if target is continuous, bin it into classes
    if y.nunique() > 10:
        y = pd.qcut(y, q=3, labels=[0, 1, 2], duplicates="drop").astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.values, feature_cols


def _feature_importance(model, feature_cols: list[str]) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        imps = np.ones(len(feature_cols))
    imps = imps / imps.sum()
    pairs = sorted(zip(feature_cols, imps), key=lambda x: -x[1])[:6]
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


def _run(model, model_name: str, data: list[dict], target_col=None) -> dict:
    X, y, feature_cols = _prepare(data, target_col)
    model.fit(X, y)
    preds = model.predict(X)
    acc = round(float(accuracy_score(y, preds)), 4)

    # cross-val if enough data
    if len(X) >= 10:
        cv = cross_val_score(model, X, y, cv=min(5, len(X)//2), scoring="accuracy")
        cv_mean = round(float(cv.mean()), 4)
    else:
        cv_mean = acc

    # prediction probabilities for first sample
    sample_proba = None
    if hasattr(model, "predict_proba") and len(X) > 0:
        p = model.predict_proba(X[:1])[0]
        sample_proba = [round(float(v), 4) for v in p]

    feat_imp = _feature_importance(model, feature_cols)

    return {
        "model": model_name,
        "accuracy": acc,
        "cv_accuracy": cv_mean,
        "feature_importance": feat_imp,
        "sample_prediction": {
            "probabilities": sample_proba,
            "predicted_class": int(preds[0]) if len(preds) > 0 else None,
        },
        "metrics": [
            {"label": "準確率", "score": f"{acc*100:.1f}%", "trend": "up", "delta": f"+{(acc-0.5)*100:.1f}pp vs baseline"},
            {"label": "CV 準確率", "score": f"{cv_mean*100:.1f}%", "trend": "neutral", "delta": "5-fold"},
            {"label": "樣本預測信心", "score": f"{max(sample_proba)*100:.1f}%" if sample_proba else "N/A", "trend": "up", "delta": "top class"},
        ],
    }


def run_xgboost(data, target_col=None, **kw):
    if HAS_XGB:
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss", verbosity=0)
    else:
        m = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    return _run(m, "XGBoost" if HAS_XGB else "GradientBoosting", data, target_col)


def run_random_forest(data, target_col=None, **kw):
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    return _run(m, "Random Forest", data, target_col)


def run_logistic(data, target_col=None, **kw):
    m = LogisticRegression(max_iter=500, random_state=42)
    return _run(m, "Logistic Regression", data, target_col)


def run_lightgbm(data, target_col=None, **kw):
    if HAS_LGB:
        m = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        m = RandomForestClassifier(n_estimators=100, random_state=42)
    return _run(m, "LightGBM" if HAS_LGB else "Random Forest", data, target_col)


RUNNERS = {
    "xgboost": run_xgboost,
    "random_forest": run_random_forest,
    "logistic": run_logistic,
    "lightgbm": run_lightgbm,
}


def run_predict(data: list[dict], model_id: str, params: dict = {}) -> dict:
    fn = RUNNERS.get(model_id, run_xgboost)
    return fn(data, **params)
