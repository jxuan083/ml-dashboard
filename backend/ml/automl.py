"""AutoML Pipeline: fully automated, Kaggle-competitive ML system.

Pipeline stages:
1. Preprocessing    — auto-detect task, encode categoricals
2. Feature Eng.     — auto-generate interactions, polynomials, stats, freq encoding
3. Feature Select.  — remove noise via permutation importance + correlation filter
4. Model Tuning     — Optuna hyperparameter search for multiple model types
5. Ensemble         — Blending via out-of-fold predictions + stacking + voting
6. Post-processing  — threshold optimization (classification) / identity (regression)

Usage: result = run_automl(data, config)
"""

import time
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    make_scorer, f1_score, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone

from ml.preprocessing import PreprocessingPipeline
from ml.feature_engineering import AutoFeatureEngineer
from ml.feature_selection import AutoFeatureSelector

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════
# Presets
# ══════════════════════════════════════════════════════════════

PRESETS = {
    "fast":     {"n_trials": 30,  "n_folds": 3, "max_fe_interactions": 20, "max_features": 80},
    "balanced": {"n_trials": 60,  "n_folds": 5, "max_fe_interactions": 50, "max_features": 150},
    "thorough": {"n_trials": 200, "n_folds": 5, "max_fe_interactions": 80, "max_features": 250},
    "extreme":  {"n_trials": 400, "n_folds": 10, "max_fe_interactions": 100, "max_features": 400},
}


# ══════════════════════════════════════════════════════════════
# Search Spaces
# ══════════════════════════════════════════════════════════════

def _xgb_clf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 10.0),
    }

def _xgb_reg_params(trial):
    return _xgb_clf_params(trial)  # same space works

def _lgb_clf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

def _lgb_reg_params(trial):
    return _lgb_clf_params(trial)

def _rf_clf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }

def _rf_reg_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

def _logistic_params(trial):
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
    params = {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "l1_ratio": l1_ratio,
        "max_iter": 2000,
        "solver": "saga",
    }
    return params

def _ridge_params(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1000.0, log=True)}

def _lasso_params(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True), "max_iter": 5000}


SEARCH_SPACES = {
    "classification": {
        "xgboost": _xgb_clf_params,
        "lightgbm": _lgb_clf_params,
        "random_forest": _rf_clf_params,
        "logistic": _logistic_params,
    },
    "regression": {
        "xgboost": _xgb_reg_params,
        "lightgbm": _lgb_reg_params,
        "random_forest": _rf_reg_params,
        "ridge": _ridge_params,
        "lasso": _lasso_params,
    },
}


# ══════════════════════════════════════════════════════════════
# Model Factories
# ══════════════════════════════════════════════════════════════

def _make_model(model_type: str, params: dict, task: str = "classification"):
    if task == "classification":
        return _make_clf(model_type, params)
    return _make_reg(model_type, params)


def _make_clf(model_type: str, params: dict):
    if model_type == "xgboost":
        if HAS_XGB:
            return XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1)
        return GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
        )
    elif model_type == "lightgbm":
        if HAS_LGB:
            return LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 200), random_state=42, n_jobs=-1)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif model_type == "logistic":
        p = {**params}
        p.setdefault("solver", "saga")
        p.setdefault("max_iter", 2000)
        return LogisticRegression(**p, random_state=42)
    raise ValueError(f"Unknown clf model: {model_type}")


def _make_reg(model_type: str, params: dict):
    if model_type == "xgboost":
        if HAS_XGB:
            return XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
        )
    elif model_type == "lightgbm":
        if HAS_LGB:
            return LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 200), random_state=42, n_jobs=-1)
    elif model_type == "random_forest":
        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == "ridge":
        return Ridge(**params)
    elif model_type == "lasso":
        return Lasso(**params)
    raise ValueError(f"Unknown reg model: {model_type}")


MODEL_LABELS = {
    "xgboost": "XGBoost" if HAS_XGB else "GradientBoosting",
    "lightgbm": "LightGBM" if HAS_LGB else "RandomForest (fallback)",
    "random_forest": "Random Forest",
    "logistic": "Logistic Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
}


# ══════════════════════════════════════════════════════════════
# CV Scoring
# ══════════════════════════════════════════════════════════════

def _cv_score(model, X, y, n_folds, metric, task, handle_imbalance=False):
    """K-Fold CV with optional SMOTE, returns list of fold scores."""
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    use_smote = handle_imbalance and HAS_SMOTE and task == "classification"
    scores = []

    for train_idx, val_idx in kf.split(X, y if task == "classification" else None):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if use_smote:
            try:
                X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            except ValueError:
                pass

        m = clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        score = _compute_metric(y_val, preds, m, X_val, metric, task, y)
        scores.append(float(score))

    return scores


def _compute_metric(y_true, preds, model, X_val, metric, task, y_full):
    """Compute a single metric value."""
    if task == "regression":
        if metric == "rmse":
            return -np.sqrt(mean_squared_error(y_true, preds))  # negative so higher=better
        elif metric == "mae":
            return -mean_absolute_error(y_true, preds)
        else:  # r2
            return r2_score(y_true, preds)

    # Classification
    n_classes = len(np.unique(y_full))
    if metric == "roc_auc" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_val)
            if n_classes == 2:
                return roc_auc_score(y_true, proba[:, 1])
            return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except Exception:
            pass
    if metric == "accuracy":
        return accuracy_score(y_true, preds)
    # Default: f1
    avg = "binary" if n_classes == 2 else "macro"
    return f1_score(y_true, preds, average=avg, zero_division=0)


# ══════════════════════════════════════════════════════════════
# Out-of-Fold Predictions for Blending
# ══════════════════════════════════════════════════════════════

def _oof_predictions(model, X, y, n_folds, task, handle_imbalance=False):
    """Generate out-of-fold predictions for blending."""
    if task == "classification":
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    use_smote = handle_imbalance and HAS_SMOTE and task == "classification"
    n_classes = len(np.unique(y))
    use_proba = task == "classification" and hasattr(model, "predict_proba") and n_classes <= 20

    if use_proba:
        oof = np.zeros((len(X), n_classes))
    else:
        oof = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X, y if task == "classification" else None):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr = y[train_idx]

        if use_smote:
            try:
                X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            except ValueError:
                pass

        m = clone(model)
        m.fit(X_tr, y_tr)

        if use_proba:
            proba = m.predict_proba(X_val)
            # Handle case where not all classes appear in fold
            if proba.shape[1] == n_classes:
                oof[val_idx] = proba
            else:
                oof[val_idx, :proba.shape[1]] = proba
        else:
            oof[val_idx] = m.predict(X_val)

    return oof


# ══════════════════════════════════════════════════════════════
# Optuna Tuning
# ══════════════════════════════════════════════════════════════

def _tune_model(model_type, X, y, n_trials, n_folds, metric, task, handle_imbalance=False, progress_cb=None):
    """Tune one model type. Returns result dict."""
    spaces = SEARCH_SPACES.get(task, SEARCH_SPACES["classification"])
    if model_type not in spaces:
        return None

    def objective(trial):
        params = spaces[model_type](trial)
        model = _make_model(model_type, params, task)
        scores = _cv_score(model, X, y, n_folds, metric, task, handle_imbalance)
        mean_score = float(np.mean(scores))
        if progress_cb and (trial.number + 1) % 5 == 0:
            progress_cb(model_type, trial.number + 1, n_trials, mean_score)
        return mean_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception:
        pass

    if not study.trials:
        return {
            "model_type": model_type, "label": MODEL_LABELS.get(model_type, model_type),
            "best_params": {}, "cv_scores": [], "mean_cv": 0.0, "std_cv": 0.0,
            "error": "All trials failed",
        }

    best_params = study.best_trial.params
    model = _make_model(model_type, best_params, task)
    cv_scores = _cv_score(model, X, y, n_folds, metric, task, handle_imbalance)
    model.fit(X, y)

    return {
        "model_type": model_type,
        "label": MODEL_LABELS.get(model_type, model_type),
        "best_params": best_params,
        "cv_scores": cv_scores,
        "mean_cv": round(float(np.mean(cv_scores)), 5),
        "std_cv": round(float(np.std(cv_scores)), 5),
        "n_trials": n_trials,
        "best_trial_value": round(float(study.best_value), 5),
        "_fitted_model": model,  # internal, stripped before return
    }


# ══════════════════════════════════════════════════════════════
# Ensemble Builder
# ══════════════════════════════════════════════════════════════

def _build_ensemble(tuned_results, X, y, n_folds, metric, task, handle_imbalance=False):
    """Build voting, stacking, and blending ensembles."""
    valid = [r for r in tuned_results if "_fitted_model" in r and r["mean_cv"] > 0]
    if len(valid) < 2:
        return None

    top = sorted(valid, key=lambda r: r["mean_cv"], reverse=True)[:4]
    estimators = [(r["model_type"], r["_fitted_model"]) for r in top]
    results = {}

    # 1. Voting
    try:
        if task == "classification":
            voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        else:
            voting = VotingRegressor(estimators=estimators, n_jobs=-1)
        v_scores = _cv_score(voting, X, y, n_folds, metric, task, handle_imbalance)
        results["voting"] = {
            "mean_cv": round(float(np.mean(v_scores)), 5),
            "std_cv": round(float(np.std(v_scores)), 5),
            "cv_scores": v_scores,
            "models_used": [r["label"] for r in top],
        }
    except Exception as e:
        results["voting"] = {"error": str(e), "mean_cv": 0.0}

    # 2. Stacking
    try:
        if task == "classification":
            meta = LogisticRegression(max_iter=2000, random_state=42)
            stacking = StackingClassifier(estimators=estimators, final_estimator=meta, cv=n_folds, passthrough=False, n_jobs=-1)
        else:
            meta = Ridge(alpha=1.0)
            stacking = StackingRegressor(estimators=estimators, final_estimator=meta, cv=n_folds, passthrough=False, n_jobs=-1)
        s_scores = _cv_score(stacking, X, y, n_folds, metric, task, handle_imbalance)
        results["stacking"] = {
            "mean_cv": round(float(np.mean(s_scores)), 5),
            "std_cv": round(float(np.std(s_scores)), 5),
            "cv_scores": s_scores,
            "models_used": [r["label"] for r in top],
        }
    except Exception as e:
        results["stacking"] = {"error": str(e), "mean_cv": 0.0}

    # 3. Blending (OOF-based meta-learner)
    try:
        oof_stack = []
        for r in top:
            oof = _oof_predictions(r["_fitted_model"], X, y, n_folds, task, handle_imbalance)
            if oof.ndim == 1:
                oof_stack.append(oof.reshape(-1, 1))
            else:
                oof_stack.append(oof)

        meta_X = np.hstack(oof_stack)
        if task == "classification":
            meta_model = LogisticRegression(max_iter=2000, random_state=42)
        else:
            meta_model = Ridge(alpha=1.0)

        b_scores = _cv_score(meta_model, meta_X, y, n_folds, metric, task, False)
        results["blending"] = {
            "mean_cv": round(float(np.mean(b_scores)), 5),
            "std_cv": round(float(np.std(b_scores)), 5),
            "cv_scores": b_scores,
            "models_used": [r["label"] for r in top],
        }
    except Exception as e:
        results["blending"] = {"error": str(e), "mean_cv": 0.0}

    return results


# ══════════════════════════════════════════════════════════════
# Feature Importance
# ══════════════════════════════════════════════════════════════

def _feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
    elif hasattr(model, "coef_"):
        imps = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return []

    if len(imps) != len(feature_names):
        return []

    imps = imps / (imps.sum() + 1e-9)
    pairs = sorted(zip(feature_names, imps), key=lambda x: -x[1])[:20]
    return [{"feature": f, "importance": round(float(v), 5)} for f, v in pairs]


# ══════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════

def run_automl(data: list[dict], config: dict, progress_cb=None) -> dict:
    """Run the full AutoML pipeline.

    Args:
        data: List of row dicts (the dataset).
        config: {
            target_col, feature_cols, metric, preset,
            models, use_ensemble, handle_imbalance
        }
        progress_cb: Optional callback(stage, detail, pct).

    Returns:
        Full results with leaderboard, ensemble, feature importance, pipeline stages.
    """
    t0 = time.time()
    stages = []

    def _stage(name, detail=""):
        stages.append({"stage": name, "detail": detail, "time": round(time.time() - t0, 1)})
        if progress_cb:
            progress_cb(name, detail, 0.0)

    preset_name = config.get("preset", "balanced")
    preset = PRESETS.get(preset_name, PRESETS["balanced"])
    n_trials = preset["n_trials"]
    n_folds = preset["n_folds"]
    handle_imbalance = config.get("handle_imbalance", False)

    # ── Stage 1: Preprocessing ──
    _stage("preprocessing", "Encoding and preparing data")
    pp = PreprocessingPipeline()
    X_df, y, feature_names, task = pp.fit_transform(
        data,
        target_col=config.get("target_col"),
        feature_cols=config.get("feature_cols") or None,
        scale=False,
    )

    _stage("preprocessing_done", f"Task: {task}, features: {len(feature_names)}, samples: {len(y)}")

    # ── Stage 2: Feature Engineering ──
    _stage("feature_engineering", "Generating interaction, polynomial, and statistical features")
    fe = AutoFeatureEngineer(
        max_interactions=preset["max_fe_interactions"],
        max_poly_features=min(20, len(feature_names)),
    )
    X_df = fe.fit_transform(X_df, pd.Series(y))
    all_features = list(X_df.columns)
    _stage("feature_engineering_done", f"Generated {len(fe.generated_feature_names)} new features, total: {len(all_features)}")

    # ── Stage 3: Feature Selection ──
    _stage("feature_selection", "Ranking and filtering features")
    fs = AutoFeatureSelector(
        max_features=preset["max_features"],
        corr_threshold=0.98,
        task=task,
    )
    X, selected_features = fs.fit_transform(X_df, y, all_features)
    _stage("feature_selection_done", f"Selected {len(selected_features)} from {len(all_features)} features")

    # ── Stage 4: Auto-detect metric ──
    n_classes = len(np.unique(y)) if task == "classification" else 0
    metric = config.get("metric")
    if not metric:
        if task == "classification":
            metric = "roc_auc" if n_classes == 2 else "f1"
        else:
            metric = "rmse"

    # ── Stage 5: Model Tuning ──
    default_models = {
        "classification": ["xgboost", "lightgbm", "random_forest", "logistic"],
        "regression": ["xgboost", "lightgbm", "random_forest", "ridge", "lasso"],
    }
    model_types = config.get("models") or default_models.get(task, default_models["classification"])

    tuned = []
    for i, model_type in enumerate(model_types):
        _stage("tuning", f"Tuning {MODEL_LABELS.get(model_type, model_type)} ({i+1}/{len(model_types)})")

        def model_progress(mt, trial, total, score):
            if progress_cb:
                overall_pct = (i / len(model_types) + trial / total / len(model_types)) * 100
                progress_cb("tuning", f"{MODEL_LABELS.get(mt, mt)}: trial {trial}/{total}, best={score:.4f}", overall_pct)

        result = _tune_model(model_type, X, y, n_trials, n_folds, metric, task, handle_imbalance, model_progress)
        if result:
            tuned.append(result)

    _stage("tuning_done", f"Tuned {len(tuned)} models")

    # ── Stage 6: Ensemble ──
    ensemble = None
    if config.get("use_ensemble", True) and len(tuned) >= 2:
        _stage("ensemble", "Building voting, stacking, and blending ensembles")
        ensemble = _build_ensemble(tuned, X, y, n_folds, metric, task, handle_imbalance)
        _stage("ensemble_done", "Ensemble complete")

    # ── Stage 7: Post-processing (threshold optimization for binary classification) ──
    threshold_info = None
    if task == "classification" and n_classes == 2:
        best_model = max(tuned, key=lambda r: r["mean_cv"]) if tuned else None
        if best_model and "_fitted_model" in best_model and hasattr(best_model["_fitted_model"], "predict_proba"):
            _stage("post_processing", "Optimizing classification threshold")
            threshold_info = _optimize_threshold(best_model["_fitted_model"], X, y, n_folds)

    # ── Build results ──
    leaderboard = sorted(tuned, key=lambda r: r["mean_cv"], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    best = max(tuned, key=lambda r: r["mean_cv"]) if tuned else None
    feat_imp = _feature_importance(best["_fitted_model"], selected_features) if best and "_fitted_model" in best else []

    best_score = best["mean_cv"] if best else 0.0
    best_model_type = best["model_type"] if best else None

    if ensemble:
        for ens_type, ens_result in ensemble.items():
            if ens_result.get("mean_cv", 0) > best_score:
                best_score = ens_result["mean_cv"]
                best_model_type = f"ensemble_{ens_type}"

    # Clean fitted models from output
    serializable_leaderboard = []
    for entry in leaderboard:
        clean = {k: v for k, v in entry.items() if k != "_fitted_model"}
        serializable_leaderboard.append(clean)

    _stage("done", f"Best: {best_model_type} = {best_score:.5f}")

    return {
        "leaderboard": serializable_leaderboard,
        "ensemble": ensemble,
        "feature_importance": feat_imp,
        "best_model_type": best_model_type,
        "best_score": round(best_score, 5),
        "metric": metric,
        "task": task,
        "n_classes": n_classes,
        "total_trials": n_trials * len(model_types),
        "duration_seconds": round(time.time() - t0, 1),
        "threshold": threshold_info,
        "pipeline_stages": stages,
        "feature_stats": {
            "original": len(feature_names),
            "after_engineering": len(all_features),
            "after_selection": len(selected_features),
            "generated": len(fe.generated_feature_names),
        },
        "config": {
            "preset": preset_name,
            "models": model_types,
            "n_trials": n_trials,
            "n_folds": n_folds,
            "metric": metric,
            "handle_imbalance": handle_imbalance,
            "use_ensemble": config.get("use_ensemble", True),
            "target_col": config.get("target_col"),
        },
    }


def _optimize_threshold(model, X, y, n_folds):
    """Find optimal classification threshold for binary problems."""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_proba = np.zeros(len(y))

    for train_idx, val_idx in kf.split(X, y):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        all_proba[val_idx] = m.predict_proba(X[val_idx])[:, 1]

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (all_proba >= thresh).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh

    return {
        "optimal_threshold": round(float(best_thresh), 3),
        "f1_at_optimal": round(float(best_f1), 5),
        "f1_at_default": round(float(f1_score(y, (all_proba >= 0.5).astype(int), zero_division=0)), 5),
    }
