from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


def extract_palette_features(
    img_path: Path,
    n_colors: int = 5,
    resize: int = 256,
    random_state: int = 42,
) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize))

    arr = img.reshape(-1, 3).astype(np.float32)

    km = KMeans(
        n_clusters=n_colors,
        n_init=10,
        random_state=random_state,
    )
    labels = km.fit_predict(arr)
    centers = km.cluster_centers_ / 255.0

    counts = np.bincount(labels, minlength=n_colors).astype(np.float32)
    weights = counts / counts.sum()

    feat: list[float] = []
    order = np.argsort(-weights)

    for idx in order:
        feat.extend(centers[idx].tolist())
        feat.append(float(weights[idx]))

    mean_rgb = arr.mean(axis=0) / 255.0
    std_rgb = arr.std(axis=0) / 255.0

    feat.extend(mean_rgb.tolist())
    feat.extend(std_rgb.tolist())

    return np.array(feat, dtype=np.float32)


def extract_composition_features(img_path: Path, resize: int = 256) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0

    ys, xs = np.indices(gray.shape)
    weights = gray.astype(np.float32) + 1e-6

    x_center = (xs * weights).sum() / weights.sum() / resize
    y_center = (ys * weights).sum() / weights.sum() / resize

    left = gray[:, :resize // 2]
    right = gray[:, resize - resize // 2:]
    right_flip = np.fliplr(right)

    symmetry = (
        1.0
        - np.mean(np.abs(left.astype(np.float32) - right_flip.astype(np.float32)))
        / 255.0
    )

    thirds = np.array(
        [
            [1 / 3, 1 / 3],
            [2 / 3, 1 / 3],
            [1 / 3, 2 / 3],
            [2 / 3, 2 / 3],
        ],
        dtype=np.float32,
    )

    center = np.array([x_center, y_center], dtype=np.float32)
    thirds_dist = np.min(np.linalg.norm(thirds - center, axis=1))

    contrast = gray.std() / 255.0

    return np.array(
        [
            edge_density,
            x_center,
            y_center,
            symmetry,
            thirds_dist,
            contrast,
        ],
        dtype=np.float32,
    )


def build_palette_matrix(
    filepaths: list[Path],
    n_colors: int = 5,
    resize: int = 256,
) -> np.ndarray:
    return np.vstack(
        [
            extract_palette_features(path, n_colors=n_colors, resize=resize)
            for path in tqdm(filepaths, desc="Palette features")
        ]
    )


def build_composition_matrix(filepaths: list[Path], resize: int = 256) -> np.ndarray:
    return np.vstack(
        [
            extract_composition_features(path, resize=resize)
            for path in tqdm(filepaths, desc="Composition features")
        ]
    )
