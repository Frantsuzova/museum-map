from __future__ import annotations

from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pyvis.network import Network
from sklearn.metrics.pairwise import cosine_similarity


PALETTE = [
    "#6C5CE7",
    "#00B894",
    "#E17055",
    "#0984E3",
    "#E84393",
    "#FDCB6E",
    "#00CEC9",
    "#D63031",
    "#A29BFE",
    "#55EFC4",
    "#FD79A8",
    "#74B9FF",
    "#FAB1A0",
    "#81ECEC",
    "#636E72",
]


def cluster_colors(cluster_ids: list[int]) -> dict[int, str]:
    return {
        int(cluster_id): PALETTE[i % len(PALETTE)]
        for i, cluster_id in enumerate(sorted(cluster_ids))
    }


def representative_subset(
    df_plot: pd.DataFrame,
    max_per_cluster: int = 35,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[int]]:
    parts = []

    for cluster_id, sub in df_plot.groupby("cluster"):
        if int(cluster_id) == -1:
            continue

        parts.append(
            sub.sample(
                min(max_per_cluster, len(sub)),
                random_state=random_state,
            )
        )

    if not parts:
        raise ValueError("No non-noise clusters found for graph export.")

    graph_df = pd.concat(parts).copy()
    selected_idx = graph_df.index.tolist()
    graph_df = graph_df.reset_index(drop=False)

    return graph_df, selected_idx


def knn_edges(features: np.ndarray, k_neighbors: int = 4) -> set[tuple[int, int, float]]:
    sim = cosine_similarity(features)
    edges: set[tuple[int, int, float]] = set()

    for i in range(len(features)):
        nn_idx = np.argsort(sim[i])[::-1][1:k_neighbors + 1]

        for j in nn_idx:
            a, b = sorted((i, int(j)))
            edges.add((a, b, float(sim[i, j])))

    return edges


def _make_bordered_thumbnail(
    src_path: Path,
    dst_path: Path,
    border_color: str,
    size: int = 96,
    border_px: int = 6,
) -> None:
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((size, size))
    img = ImageOps.expand(img, border=border_px, fill=border_color)
    img.save(dst_path, format="JPEG", quality=90)


def build_similarity_graph_html(
    graph_df: pd.DataFrame,
    graph_features: np.ndarray,
    output_dir: str | Path,
    k_neighbors: int = 4,
    thumb_size: int = 96,
    thumb_border_px: int = 6,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    thumbs_dir = output_path / "thumbs"
    if thumbs_dir.exists():
        shutil.rmtree(thumbs_dir)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    colors = cluster_colors(graph_df["cluster"].astype(int).unique().tolist())
    edges = knn_edges(graph_features, k_neighbors=k_neighbors)

    thumb_paths: dict[int, Path] = {}

    for i, row in graph_df.iterrows():
        cluster_id = int(row["cluster"])
        dst = thumbs_dir / f"thumb_{i:04d}.jpg"

        _make_bordered_thumbnail(
            Path(row["filepath"]),
            dst,
            border_color=colors[cluster_id],
            size=thumb_size,
            border_px=thumb_border_px,
        )
        thumb_paths[i] = dst

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut()

    for i, row in graph_df.iterrows():
        cluster_id = int(row["cluster"])

        title_html = (
            f"<b>Artist:</b> {row['artist']}<br>"
            f"<b>Style:</b> {row['style']}<br>"
            f"<b>Genre:</b> {row['genre']}<br>"
            f"<b>Cluster:</b> {cluster_id}<br>"
            f"<b>File:</b> {os.path.basename(str(row['filepath']))}"
        )

        net.add_node(
            n_id=i,
            label="",
            title=title_html,
            shape="image",
            image=f"thumbs/{thumb_paths[i].name}",
            size=34,
        )

    for a, b, w in edges:
        net.add_edge(
            a,
            b,
            value=w,
            color="rgba(70, 90, 140, 0.28)",
        )

    net.set_options(
        """
        var options = {
          "interaction": {
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -5000,
              "centralGravity": 0.15,
              "springLength": 120,
              "springConstant": 0.03,
              "damping": 0.9
            },
            "minVelocity": 0.75
          }
        }
        """
    )

    html_path = output_path / "similarity_graph.html"
    net.save_graph(str(html_path))

    return html_path
