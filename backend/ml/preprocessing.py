"""Shared data preparation utilities for ML modules.

Provides both legacy prepare_data() for existing predict/cluster modules
and a new PreprocessingPipeline class for the AutoML pipeline that maintains
state for consistent train/test transformation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder


# ── Legacy functions (used by predict.py, cluster.py) ──

def _infer_target(df: pd.DataFrame, target_col: str | None) -> str:
    if target_col and target_col in df.columns:
        return target_col
    return df.columns[-1]


def _infer_features(df: pd.DataFrame, target: str, feature_cols: list[str] | None) -> list[str]:
    if feature_cols:
        valid = [c for c in feature_cols if c in df.columns and c != target]
        if valid:
            return valid
    return [c for c in df.columns if c != target]


def _encode_label(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    le = LabelEncoder()
    for col in cols:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def _encode_smart(df: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.DataFrame:
    low_card = [c for c in cols if df[c].dtype == object and df[c].nunique() <= 10]
    high_card = [c for c in cols if df[c].dtype == object and df[c].nunique() > 10]

    if low_card:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[low_card] = oe.fit_transform(df[low_card].astype(str))

    if high_card:
        try:
            from sklearn.preprocessing import TargetEncoder
            te = TargetEncoder(smooth="auto")
            df[high_card] = te.fit_transform(df[high_card].astype(str), y)
        except ImportError:
            le = LabelEncoder()
            for col in high_card:
                df[col] = le.fit_transform(df[col].astype(str))

    return df


def _bin_target(y: pd.Series) -> pd.Series:
    if y.nunique() > 10:
        return pd.qcut(y, q=3, labels=[0, 1, 2], duplicates="drop").astype(int)
    return y


def prepare_data(
    data: list[dict],
    target_col: str | None = None,
    feature_cols: list[str] | None = None,
    encoding: str = "label",
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Legacy prepare_data for backward compatibility."""
    df = pd.DataFrame(data)
    target = _infer_target(df, target_col)
    features = _infer_features(df, target, feature_cols)

    y = _bin_target(df[target])

    if encoding == "smart":
        df = _encode_smart(df, y, features)
    else:
        df = _encode_label(df, features)

    X = df[features].fillna(df[features].median(numeric_only=True))

    if scale:
        X_out = StandardScaler().fit_transform(X)
    else:
        X_out = X.values

    return X_out, y.values, features


# ── New: Stateful Preprocessing Pipeline for AutoML ──

class PreprocessingPipeline:
    """Stateful preprocessor that remembers encoders for consistent train/test transforms."""

    def __init__(self):
        self._fitted = False
        self._target_col: str = ""
        self._feature_cols: list[str] = []
        self._task: str = "classification"  # or "regression"
        self._label_encoders: dict = {}
        self._ordinal_encoder = None
        self._target_encoder = None
        self._low_card_cols: list[str] = []
        self._high_card_cols: list[str] = []
        self._num_cols: list[str] = []
        self._medians: dict[str, float] = {}
        self._scaler: StandardScaler | None = None

    def fit_transform(
        self,
        data: list[dict],
        target_col: str | None = None,
        feature_cols: list[str] | None = None,
        scale: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
        """Fit preprocessing on training data.

        Returns: (X_df, y, feature_names, task_type)
        """
        df = pd.DataFrame(data)
        self._target_col = _infer_target(df, target_col)
        self._feature_cols = _infer_features(df, self._target_col, feature_cols)

        y_raw = df[self._target_col]

        # Detect task type
        if y_raw.dtype == object or y_raw.nunique() <= 15:
            self._task = "classification"
            if y_raw.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
                self._label_encoders["__target__"] = le
            else:
                y = y_raw.values
        else:
            # Continuous target with many unique values -> regression
            self._task = "regression"
            y = y_raw.astype(float).values

        # Identify column types
        feat_df = df[self._feature_cols].copy()
        self._num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = feat_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._low_card_cols = [c for c in cat_cols if feat_df[c].nunique() <= 10]
        self._high_card_cols = [c for c in cat_cols if feat_df[c].nunique() > 10]

        # Encode categoricals
        if self._low_card_cols:
            self._ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            feat_df[self._low_card_cols] = self._ordinal_encoder.fit_transform(
                feat_df[self._low_card_cols].astype(str)
            )

        if self._high_card_cols:
            try:
                from sklearn.preprocessing import TargetEncoder
                self._target_encoder = TargetEncoder(smooth="auto")
                feat_df[self._high_card_cols] = self._target_encoder.fit_transform(
                    feat_df[self._high_card_cols].astype(str), y
                )
            except ImportError:
                for col in self._high_card_cols:
                    le = LabelEncoder()
                    feat_df[col] = le.fit_transform(feat_df[col].astype(str))
                    self._label_encoders[col] = le

        # Store medians for imputation
        for col in self._feature_cols:
            if col in feat_df.columns and feat_df[col].dtype in [np.float64, np.int64, float, int]:
                self._medians[col] = float(feat_df[col].median())

        feat_df = feat_df.fillna(self._medians)

        if scale:
            self._scaler = StandardScaler()
            X_out = pd.DataFrame(
                self._scaler.fit_transform(feat_df),
                columns=self._feature_cols,
                index=feat_df.index,
            )
        else:
            X_out = feat_df

        self._fitted = True
        return X_out, y, self._feature_cols, self._task

    def transform(self, data: list[dict]) -> pd.DataFrame:
        """Transform test data using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("Must call fit_transform first")

        df = pd.DataFrame(data)
        feat_df = df[self._feature_cols].copy() if all(c in df.columns for c in self._feature_cols) else df.copy()

        # Ensure all feature columns exist
        for col in self._feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0

        feat_df = feat_df[self._feature_cols]

        if self._low_card_cols and self._ordinal_encoder:
            present = [c for c in self._low_card_cols if c in feat_df.columns]
            if present:
                feat_df[present] = self._ordinal_encoder.transform(feat_df[present].astype(str))

        if self._high_card_cols:
            if self._target_encoder is not None:
                present = [c for c in self._high_card_cols if c in feat_df.columns]
                if present:
                    feat_df[present] = self._target_encoder.transform(feat_df[present].astype(str))
            else:
                for col in self._high_card_cols:
                    if col in self._label_encoders and col in feat_df.columns:
                        le = self._label_encoders[col]
                        feat_df[col] = feat_df[col].astype(str).map(
                            lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
                        )

        feat_df = feat_df.fillna(self._medians)

        if self._scaler:
            feat_df = pd.DataFrame(
                self._scaler.transform(feat_df),
                columns=self._feature_cols,
                index=feat_df.index,
            )

        return feat_df

    @property
    def task_type(self) -> str:
        return self._task

    @property
    def target_col(self) -> str:
        return self._target_col

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)
