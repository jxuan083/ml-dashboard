"""Automatic feature selection to remove noise and redundancy.

Uses permutation importance and correlation-based filtering to keep
only the most predictive features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold


class AutoFeatureSelector:
    """Fit on training data to identify top features, then filter train+test."""

    def __init__(
        self,
        max_features: int = 150,
        corr_threshold: float = 0.98,
        importance_threshold: float = 0.0,
        task: str = "classification",
    ):
        self.max_features = max_features
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.task = task
        self._fitted = False
        self._selected_features: list[str] = []
        self._dropped_corr: list[str] = []
        self._feature_scores: dict[str, float] = {}

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray, feature_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Select best features from training data."""

        # Step 1: Remove near-constant features
        variances = X.var(axis=0) if isinstance(X, pd.DataFrame) else np.var(X, axis=0)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names)

        low_var = [col for col, v in zip(feature_names, variances) if v < 1e-10]
        if low_var:
            X = X.drop(columns=low_var, errors="ignore")
            feature_names = [f for f in feature_names if f not in low_var]

        # Step 2: Remove highly correlated features
        self._dropped_corr = self._find_correlated(X, feature_names)
        if self._dropped_corr:
            X = X.drop(columns=self._dropped_corr, errors="ignore")
            feature_names = [f for f in feature_names if f not in self._dropped_corr]

        # Step 3: Permutation importance ranking
        if len(feature_names) > self.max_features:
            scores = self._compute_importance(X.values, y, feature_names)
            self._feature_scores = scores

            # Keep top N by importance
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            keep = [name for name, score in ranked[:self.max_features] if score > self.importance_threshold]
            if len(keep) < 10:
                keep = [name for name, _ in ranked[:max(10, self.max_features)]]

            self._selected_features = keep
        else:
            self._selected_features = feature_names
            self._feature_scores = {f: 1.0 for f in feature_names}

        self._fitted = True
        X_out = X[self._selected_features].values
        return X_out, self._selected_features

    def transform(self, X: np.ndarray | pd.DataFrame, feature_names: list[str]) -> np.ndarray:
        """Apply the same feature selection to test data."""
        if not self._fitted:
            raise RuntimeError("Must call fit_transform first")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_names)

        # Use only selected features, fill missing with 0
        for col in self._selected_features:
            if col not in X.columns:
                X[col] = 0

        return X[self._selected_features].values

    @property
    def selected_features(self) -> list[str]:
        return list(self._selected_features)

    @property
    def feature_importance_scores(self) -> dict[str, float]:
        return dict(self._feature_scores)

    def _find_correlated(self, X: pd.DataFrame, features: list[str]) -> list[str]:
        """Find features to drop due to high pairwise correlation."""
        if len(features) < 2:
            return []

        try:
            corr_matrix = X[features].corr().abs()
        except Exception:
            return []

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        return to_drop

    def _compute_importance(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> dict[str, float]:
        """Quick permutation importance using a fast RF."""
        n_samples = min(5000, len(X))  # subsample for speed
        if n_samples < len(X):
            idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        if self.task == "classification":
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            )

        try:
            model.fit(X_sub, y_sub)
            result = permutation_importance(
                model, X_sub, y_sub, n_repeats=5, random_state=42, n_jobs=-1
            )
            scores = {name: float(imp) for name, imp in zip(feature_names, result.importances_mean)}
        except Exception:
            # Fallback to built-in feature importance
            try:
                model.fit(X_sub, y_sub)
                imps = model.feature_importances_
                scores = {name: float(imp) for name, imp in zip(feature_names, imps)}
            except Exception:
                scores = {name: 1.0 for name in feature_names}

        return scores
