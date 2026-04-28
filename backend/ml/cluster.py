import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def _prepare(data: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("沒有數值欄位可以分群")
    X = df[num_cols].fillna(df[num_cols].median())
    X_scaled = StandardScaler().fit_transform(X)
    return df, X_scaled


def _cluster_stats(labels: np.ndarray, X: np.ndarray, colors: list[str]):
    unique = sorted(set(labels[labels >= 0]))
    total = len(labels)
    groups = []
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        count = int(mask.sum())
        groups.append({
            "label": f"Cluster {chr(65+i)}",
            "count": count,
            "pct": round(count / total, 4),
            "color": colors[i % len(colors)],
        })
    sil = float(silhouette_score(X, labels)) if len(unique) > 1 else 0.0
    return groups, round(sil, 4)


COLORS = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7"]


def run_kmeans(data: list[dict], k: int = 4) -> dict:
    df, X = _prepare(data)
    k = min(k, len(data) - 1)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    groups, sil = _cluster_stats(labels, X, COLORS)
    return {
        "groups": groups,
        "silhouette": sil,
        "inertia": round(float(model.inertia_), 2),
        "k": k,
        "model": "K-Means",
    }


def run_dbscan(data: list[dict], eps: float = 0.5, min_samples: int = 5) -> dict:
    df, X = _prepare(data)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    valid = labels[labels >= 0]
    k = len(set(valid)) if len(valid) > 0 else 1
    groups, sil = _cluster_stats(labels, X, COLORS)
    noise = int((labels == -1).sum())
    return {
        "groups": groups,
        "silhouette": sil,
        "inertia": 0,
        "k": k,
        "model": "DBSCAN",
        "noise_points": noise,
    }


def run_hierarchical(data: list[dict], k: int = 4) -> dict:
    df, X = _prepare(data)
    k = min(k, len(data) - 1)
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    groups, sil = _cluster_stats(labels, X, COLORS)
    return {
        "groups": groups,
        "silhouette": sil,
        "inertia": 0,
        "k": k,
        "model": "Hierarchical",
    }


def run_gmm(data: list[dict], k: int = 3) -> dict:
    df, X = _prepare(data)
    k = min(k, len(data) - 1)
    model = GaussianMixture(n_components=k, random_state=42)
    model.fit(X)
    labels = model.predict(X)
    groups, sil = _cluster_stats(labels, X, COLORS)
    bic = round(float(model.bic(X)), 2)
    return {
        "groups": groups,
        "silhouette": sil,
        "inertia": bic,
        "k": k,
        "model": "GMM",
        "bic": bic,
    }


RUNNERS = {
    "kmeans": run_kmeans,
    "dbscan": run_dbscan,
    "hierarchical": run_hierarchical,
    "gmm": run_gmm,
}


def run_cluster(data: list[dict], model_id: str, params: dict = {}) -> dict:
    fn = RUNNERS.get(model_id, run_kmeans)
    return fn(data, **params)
