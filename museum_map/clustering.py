from __future__ import annotations

import hdbscan
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler


def combine_features(
    clip_embs: np.ndarray,
    palette_feats: np.ndarray,
    comp_feats: np.ndarray,
    weight_clip: float = 1.0,
    weight_palette: float = 0.5,
    weight_composition: float = 0.7,
) -> np.ndarray:
    x_clip = StandardScaler().fit_transform(clip_embs) * weight_clip
    x_palette = StandardScaler().fit_transform(palette_feats) * weight_palette
    x_comp = StandardScaler().fit_transform(comp_feats) * weight_composition

    return np.hstack([x_clip, x_palette, x_comp])


def project_umap(
    features: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


def cluster_hdbscan(
    projection: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 10,
) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(projection)
