# museum-map

`museum-map` is a Python package for interpretable clustering and graph-based visualization of painting collections.

The package is designed for exploratory work with art collections, especially in contexts where users need not only a clustering result, but also a visual and inspectable representation of relationships between artworks. The current implementation combines CLIP-based image embeddings with explicitly defined visual descriptors for color distribution and spatial composition, projects the resulting feature space, detects clusters, and exports an interactive similarity graph with painting thumbnails.

The intended audience includes researchers in digital humanities, museum professionals, curators, collection managers, and computational analysts working with visual collections.

## Installation

Install from PyPI:

```bash
pip install museum-map
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/Frantsuzova/museum-map.git
```

For local development:

```bash
git clone https://github.com/Frantsuzova/museum-map.git
cd museum-map
pip install -e .
```

## What the package does

Given a folder of painting images, `museum-map`:

- computes semantic image embeddings using a CLIP image encoder
- extracts palette-based descriptors
- extracts composition-based descriptors
- combines these signals into a shared representation
- detects clusters in the collection
- selects representative paintings from each cluster
- builds an interactive HTML graph where:
  - each node is a painting
  - node image = painting thumbnail
  - node border color = cluster membership
  - edges = local similarity relationships
- exports intermediate artifacts and a ready-to-share zip archive

This workflow is intended to support exploratory analysis of collections and to help identify non-obvious relationships between paintings.

## Quick start

### Minimal Python usage

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="/path/to/paintings",
    output_dir="/path/to/output",
)

print(pipeline.graph_html_path_)
print(pipeline.export_zip_path_)
```

This is the simplest one-line workflow: point the package to a folder with images and receive an exported interactive result.

## What input data can be used

The package currently expects a **local folder of images**.

Supported image extensions are:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`
- `.tif`
- `.tiff`

### Typical input scenarios

You can use:

- a subset of a public art dataset such as WikiArt
- a digitized museum collection exported as image files
- a folder with paintings gathered for a pilot experiment
- a thematic subcollection, for example portraits, landscapes, or one author’s works

### Current assumptions about metadata

The current pipeline is image-first. It does **not require metadata** to run.

This means you can start with a plain folder of images:

```text
my_collection/
├── painting_001.jpg
├── painting_002.jpg
├── painting_003.png
└── ...
```

If metadata such as artist, style, genre, accession number, or inventory ID is available, it can be integrated in future versions. In the current scaffold, unknown values are filled with placeholder labels.

## How to prepare data for the module

The simplest workflow is:

1. Create a folder containing only painting images.
2. Pass the folder path as `input_dir`.
3. Specify an `output_dir` where results should be written.

Example:

```python
from museum_map import build_museum_map

build_museum_map(
    input_dir="./data/paintings",
    output_dir="./out/museum_map_run_01",
)
```

The package recursively scans the input directory and collects all supported images.

## Example directory structure

```text
project/
├── data/
│   └── paintings/
│       ├── aivazovsky_001.jpg
│       ├── monet_014.jpg
│       ├── shishkin_003.jpg
│       └── ...
└── out/
```

Then run:

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="project/data/paintings",
    output_dir="project/out/run_01",
)
```

## Main output files

After execution, the output directory contains the computational artifacts and the interactive graph.

Typical output:

```text
output_dir/
├── clip_embeddings.npy
├── palette_features.npy
├── composition_features.npy
├── feature_matrix.npy
├── umap_2d.npy
├── cluster_labels.npy
├── df_plot.csv
├── config.csv
├── similarity_graph.html
├── thumbs/
│   ├── thumb_0000.jpg
│   ├── thumb_0001.jpg
│   └── ...
└── museum_map_export.zip
```

### What these files mean

- `clip_embeddings.npy` — semantic image embeddings
- `palette_features.npy` — color-based descriptors
- `composition_features.npy` — composition descriptors
- `feature_matrix.npy` — combined representation used for clustering
- `umap_2d.npy` — low-dimensional projection of the collection
- `cluster_labels.npy` — cluster assignments
- `df_plot.csv` — metadata and coordinates per image
- `config.csv` — run configuration
- `similarity_graph.html` — interactive graph visualization
- `thumbs/` — thumbnails used inside the HTML graph
- `museum_map_export.zip` — packaged result for local sharing or archiving

## What the HTML output contains

The main visual artifact is `similarity_graph.html`.

It represents the collection as a graph:

- nodes correspond to paintings
- each node contains a painting thumbnail
- node border color indicates cluster membership
- edges connect paintings with strong local similarity
- hover displays metadata such as artist, style, genre, cluster, and filename

This file can be opened locally in a browser. If some browsers restrict local file access for thumbnails, it can also be served from a lightweight local server.

Example:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/similarity_graph.html
```

## Configuration

The main entry point accepts optional keyword parameters that control the pipeline.

Example:

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="./data/paintings",
    output_dir="./out/run_02",
    batch_size=16,
    n_palette_colors=6,
    weight_clip=1.0,
    weight_palette=0.4,
    weight_composition=0.8,
    graph_k_neighbors=3,
    max_per_cluster_for_graph=25,
)
```

Important parameters include:

- `batch_size` — embedding batch size
- `n_palette_colors` — number of dominant palette colors
- `weight_clip` — weight of CLIP embeddings in the combined representation
- `weight_palette` — weight of palette descriptors
- `weight_composition` — weight of composition descriptors
- `graph_k_neighbors` — number of neighbors in the similarity graph
- `max_per_cluster_for_graph` — representative sample size per cluster
- `thumb_size` — thumbnail size used in the HTML graph

## Current limitations

At the current stage, the package has several deliberate limitations:

- it expects local image folders rather than remote datasets
- it does not yet ingest structured metadata tables automatically
- it is optimized for exploratory work rather than industrial-scale deployment
- it does not yet include optional graph-embedding extensions such as Node2Vec
- it currently exports the graph view as the main interactive artifact

These points are expected development directions rather than defects.

## Roadmap

Planned next steps include:

- command-line interface
- optional metadata ingestion from CSV/JSON
- optional museum map scatter export
- cluster summaries and automatic cluster labels
- graph-aware extensions such as Node2Vec
- richer support for digital humanities and museum collection workflows
- automated release workflow for PyPI

## Minimal example

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="./paintings",
    output_dir="./museum_map_output",
)

print("Graph saved to:", pipeline.graph_html_path_)
print("ZIP saved to:", pipeline.export_zip_path_)
```

## Citation

If you use this package in academic work, please cite the corresponding paper or software record. A suggested citation file is included as `CITATION.cff`.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
