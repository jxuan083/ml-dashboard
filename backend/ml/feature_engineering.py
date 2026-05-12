"""Automatic feature engineering for Kaggle-competitive pipelines.

Generates interaction, polynomial, statistical, and frequency-based features
from raw data. Designed to be called once on training data, then applied
identically to test data via the returned transformer state.
"""

import itertools
import numpy as np
import pandas as pd
from typing import Any


class AutoFeatureEngineer:
    """Stateful feature engineer: fit on train, transform train+test identically."""

    def __init__(self, max_interactions: int = 50, max_poly_features: int = 20):
        self.max_interactions = max_interactions
        self.max_poly_features = max_poly_features
        self._fitted = False
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._freq_maps: dict[str, dict] = {}
        self._interaction_pairs: list[tuple[str, str]] = []
        self._poly_cols: list[str] = []
        self._agg_stats: dict[str, dict[str, float]] = {}
        self._generated_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit on training data and generate features."""
        df = df.copy()
        self._num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._generated_cols = []

        # 1. Frequency encoding for categoricals
        df = self._fit_frequency_encoding(df)

        # 2. Interaction features (top numeric pairs by correlation with target)
        df = self._fit_interactions(df, y)

        # 3. Polynomial features (squares, sqrt, log1p for top numerics)
        df = self._fit_polynomial(df, y)

        # 4. Statistical aggregation features (per-row stats across numerics)
        df = self._fit_row_stats(df)

        # 5. Null indicator features
        df = self._fit_null_indicators(df)

        # 6. Binned features for high-cardinality numerics
        df = self._fit_bins(df)

        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to new data (test set)."""
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform")
        df = df.copy()

        # Frequency encoding
        for col, freq_map in self._freq_maps.items():
            if col in df.columns:
                col_name = f"{col}_freq"
                df[col_name] = df[col].map(freq_map).fillna(0.0)

        # Interactions
        for c1, c2 in self._interaction_pairs:
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_x_{c2}"] = df[c1].fillna(0) * df[c2].fillna(0)
                denom = df[c2].fillna(0).replace(0, np.nan)
                df[f"{c1}_div_{c2}"] = df[c1].fillna(0) / denom
                df[f"{c1}_div_{c2}"] = df[f"{c1}_div_{c2}"].fillna(0).replace([np.inf, -np.inf], 0)

        # Polynomial
        for col in self._poly_cols:
            if col in df.columns:
                vals = df[col].fillna(0)
                df[f"{col}_sq"] = vals ** 2
                df[f"{col}_sqrt"] = np.sqrt(np.abs(vals))
                df[f"{col}_log1p"] = np.log1p(np.abs(vals))

        # Row stats
        num_cols_present = [c for c in self._num_cols if c in df.columns]
        if num_cols_present:
            num_data = df[num_cols_present].fillna(0)
            df["_row_mean"] = num_data.mean(axis=1)
            df["_row_std"] = num_data.std(axis=1).fillna(0)
            df["_row_max"] = num_data.max(axis=1)
            df["_row_min"] = num_data.min(axis=1)
            df["_row_range"] = df["_row_max"] - df["_row_min"]
            df["_row_null_count"] = df[num_cols_present].isnull().sum(axis=1)

        # Null indicators
        for col in self._num_cols:
            if col in df.columns:
                col_name = f"{col}_is_null"
                if col_name in self._generated_cols:
                    df[col_name] = df[col].isnull().astype(int)

        # Bins
        for col in self._num_cols:
            col_name = f"{col}_bin"
            if col_name in self._generated_cols and col in df.columns:
                df[col_name] = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
                df[col_name] = df[col_name].fillna(-1).astype(int)

        return df

    @property
    def generated_feature_names(self) -> list[str]:
        return list(self._generated_cols)

    # ── Internal fit methods ──

    def _fit_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        self._freq_maps = {}
        for col in self._cat_cols:
            freq = df[col].value_counts(normalize=True).to_dict()
            self._freq_maps[col] = freq
            col_name = f"{col}_freq"
            df[col_name] = df[col].map(freq).fillna(0.0)
            self._generated_cols.append(col_name)
        return df

    def _fit_interactions(self, df: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        if len(self._num_cols) < 2:
            return df

        # Rank numeric columns by correlation with target
        if y is not None:
            correlations = {}
            for col in self._num_cols:
                try:
                    correlations[col] = abs(df[col].fillna(0).corr(y.astype(float)))
                except Exception:
                    correlations[col] = 0.0
            ranked = sorted(self._num_cols, key=lambda c: correlations.get(c, 0), reverse=True)
        else:
            ranked = self._num_cols

        # Take top N columns for interactions
        top_n = min(8, len(ranked))
        top_cols = ranked[:top_n]

        pairs = list(itertools.combinations(top_cols, 2))[:self.max_interactions]
        self._interaction_pairs = pairs

        for c1, c2 in pairs:
            mult_name = f"{c1}_x_{c2}"
            div_name = f"{c1}_div_{c2}"
            df[mult_name] = df[c1].fillna(0) * df[c2].fillna(0)
            denom = df[c2].fillna(0).replace(0, np.nan)
            df[div_name] = df[c1].fillna(0) / denom
            df[div_name] = df[div_name].fillna(0).replace([np.inf, -np.inf], 0)
            self._generated_cols.extend([mult_name, div_name])

        return df

    def _fit_polynomial(self, df: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
        if not self._num_cols:
            return df

        # Pick top columns by target correlation
        if y is not None:
            correlations = {}
            for col in self._num_cols:
                try:
                    correlations[col] = abs(df[col].fillna(0).corr(y.astype(float)))
                except Exception:
                    correlations[col] = 0.0
            ranked = sorted(self._num_cols, key=lambda c: correlations.get(c, 0), reverse=True)
        else:
            ranked = self._num_cols

        self._poly_cols = ranked[:self.max_poly_features]

        for col in self._poly_cols:
            vals = df[col].fillna(0)
            sq_name = f"{col}_sq"
            sqrt_name = f"{col}_sqrt"
            log_name = f"{col}_log1p"
            df[sq_name] = vals ** 2
            df[sqrt_name] = np.sqrt(np.abs(vals))
            df[log_name] = np.log1p(np.abs(vals))
            self._generated_cols.extend([sq_name, sqrt_name, log_name])

        return df

    def _fit_row_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._num_cols:
            return df
        num_data = df[self._num_cols].fillna(0)
        df["_row_mean"] = num_data.mean(axis=1)
        df["_row_std"] = num_data.std(axis=1).fillna(0)
        df["_row_max"] = num_data.max(axis=1)
        df["_row_min"] = num_data.min(axis=1)
        df["_row_range"] = df["_row_max"] - df["_row_min"]
        df["_row_null_count"] = df[self._num_cols].isnull().sum(axis=1)
        self._generated_cols.extend(["_row_mean", "_row_std", "_row_max", "_row_min", "_row_range", "_row_null_count"])
        return df

    def _fit_null_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._num_cols:
            null_rate = df[col].isnull().mean()
            if 0.01 < null_rate < 0.95:  # only useful if some but not all are null
                col_name = f"{col}_is_null"
                df[col_name] = df[col].isnull().astype(int)
                self._generated_cols.append(col_name)
        return df

    def _fit_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._num_cols:
            if df[col].nunique() > 20:  # only bin high-cardinality numerics
                col_name = f"{col}_bin"
                try:
                    df[col_name] = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
                    df[col_name] = df[col_name].fillna(-1).astype(int)
                    self._generated_cols.append(col_name)
                except Exception:
                    pass
        return df
