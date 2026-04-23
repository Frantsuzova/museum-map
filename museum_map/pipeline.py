from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import cluster_hdbscan, combine_features, project_umap
from .config import MuseumMapConfig
from .embeddings import ClipImageEmbedder
from .export import make_export_zip
from .features import build_composition_matrix, build_palette_matrix
from .graph import build_similarity_graph_html, representative_subset
from .utils import list_images


class MuseumMapPipeline:
    def __init__(self, config: MuseumMapConfig) -> None:
        self.config = config
        self.records_: pd.DataFrame | None = None
        self.clip_embs_: np.ndarray | None = None
        self.palette_feats_: np.ndarray | None = None
        self.comp_feats_: np.ndarray | None = None
        self.features_: np.ndarray | None = None
        self.umap_2d_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.df_plot_: pd.DataFrame | None = None
        self.graph_html_path_: Path | None = None
        self.export_zip_path_: Path | None = None

    def fit(self) -> "MuseumMapPipeline":
        cfg = self.config
        cfg.output_path.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(cfg.input_path)
        self.records_ = pd.DataFrame(
            {
                "id": list(range(len(image_paths))),
                "filepath": [str(p) for p in image_paths],
                "artist": ["unknown"] * len(image_paths),
                "style": ["unknown"] * len(image_paths),
                "genre": ["unknown"] * len(image_paths),
            }
        )

        embedder = ClipImageEmbedder(model_name=cfg.clip_model_name)
        self.clip_embs_ = embedder.transform(image_paths, batch_size=cfg.batch_size)
        self.palette_feats_ = build_palette_matrix(
            image_paths,
            n_colors=cfg.n_palette_colors,
            resize=cfg.image_size,
        )
        self.comp_feats_ = build_composition_matrix(
            image_paths,
            resize=cfg.image_size,
        )

        self.features_ = combine_features(
            self.clip_embs_,
            self.palette_feats_,
            self.comp_feats_,
            weight_clip=cfg.weight_clip,
            weight_palette=cfg.weight_palette,
            weight_composition=cfg.weight_composition,
        )

        self.umap_2d_ = project_umap(
            self.features_,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=cfg.random_state,
        )

        self.labels_ = cluster_hdbscan(
            self.umap_2d_,
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
        )

        self.df_plot_ = self.records_.copy()
        self.df_plot_["cluster"] = self.labels_.astype(int)
        self.df_plot_["x"] = self.umap_2d_[:, 0]
        self.df_plot_["y"] = self.umap_2d_[:, 1]
        return self

    def build_graph(self) -> Path:
        if self.df_plot_ is None or self.features_ is None:
            raise RuntimeError("Call fit() before build_graph().")

        graph_df, selected_idx = representative_subset(
            self.df_plot_,
            max_per_cluster=self.config.max_per_cluster_for_graph,
            random_state=self.config.random_state,
        )
        graph_features = self.features_[selected_idx]

        self.graph_html_path_ = build_similarity_graph_html(
            graph_df,
            graph_features,
            output_dir=self.config.output_path,
            k_neighbors=self.config.graph_k_neighbors,
            thumb_size=self.config.thumb_size,
            thumb_border_px=self.config.thumb_border_px,
        )
        return self.graph_html_path_

    def save_artifacts(self) -> None:
        if self.df_plot_ is None:
            raise RuntimeError("Call fit() before save_artifacts().")

        out = self.config.output_path
        np.save(out / "clip_embeddings.npy", self.clip_embs_)
        np.save(out / "palette_features.npy", self.palette_feats_)
        np.save(out / "composition_features.npy", self.comp_feats_)
        np.save(out / "feature_matrix.npy", self.features_)
        np.save(out / "umap_2d.npy", self.umap_2d_)
        np.save(out / "cluster_labels.npy", self.labels_)
        self.df_plot_.to_csv(out / "df_plot.csv", index=False)
        pd.DataFrame([asdict(self.config)]).to_csv(out / "config.csv", index=False)

    def export(self) -> Path:
        self.save_artifacts()
        if self.graph_html_path_ is None:
            self.build_graph()
        self.export_zip_path_ = make_export_zip(self.config.output_path)
        return self.export_zip_path_


def build_museum_map(input_dir: str | Path, output_dir: str | Path, **kwargs) -> MuseumMapPipeline:
    config = MuseumMapConfig(input_dir=input_dir, output_dir=output_dir, **kwargs)
    pipeline = MuseumMapPipeline(config)
    pipeline.fit()
    pipeline.build_graph()
    pipeline.export()
    return pipeline
