from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MuseumMapConfig:
    input_dir: str | Path
    output_dir: str | Path
    metadata_csv: str | Path | None = None

    image_size: int = 256
    batch_size: int = 32
    n_palette_colors: int = 5

    clip_model_name: str = "openai/clip-vit-base-patch32"

    weight_clip: float = 1.0
    weight_palette: float = 0.5
    weight_composition: float = 0.7

    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.05
    umap_metric: str = "cosine"

    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 10

    graph_k_neighbors: int = 4
    max_per_cluster_for_graph: int = 35
    thumb_size: int = 96
    thumb_border_px: int = 6

    random_state: int = 42

    @property
    def input_path(self) -> Path:
        return Path(self.input_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def metadata_path(self) -> Path | None:
        if self.metadata_csv is None:
            return None
        return Path(self.metadata_csv)
