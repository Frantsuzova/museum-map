from __future__ import annotations
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
