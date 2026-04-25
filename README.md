# museum-map

`museum-map` is a Python package for interpretable clustering and graph-based visualization of painting collections.

The package is designed for exploratory work with art collections. It is useful when users need not only a clustering result, but also a visual and inspectable representation of relationships between artworks.

The current implementation combines:

- CLIP-based image embeddings
- palette-based visual descriptors
- composition-based visual descriptors
- UMAP projection
- HDBSCAN clustering
- k-nearest-neighbor similarity graph export
- optional metadata ingestion from CSV

The intended audience includes researchers in digital humanities, museum professionals, curators, collection managers, and computational analysts working with visual collections.

[demo-results](https://frantsuzova.github.io/museum-map-demo/)

## Installation

Install from PyPI:

```bash
pip install museum-map
```

Install a specific version:

```bash
pip install museum-map==0.1.3
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
  - node image is a painting thumbnail
  - node border color indicates cluster membership
  - edges indicate local visual similarity
  - metadata can be shown for each painting
- exports intermediate artifacts and a lightweight ready-to-share archive

The workflow is intended to support exploratory analysis of collections and help identify non-obvious relationships between paintings.

## Quick start

### Minimal usage without metadata

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="/path/to/paintings",
    output_dir="/path/to/output",
)

print(pipeline.graph_html_path_)
print(pipeline.export_zip_path_)
```

This is the simplest workflow: point the package to a folder with images and receive an exported interactive result.

### Usage with metadata

If metadata is available, pass it as a CSV file:

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="/path/to/paintings",
    output_dir="/path/to/output",
    metadata_csv="/path/to/metadata.csv",
)

print(pipeline.graph_html_path_)
print(pipeline.export_zip_path_)
```

The metadata is preserved in `df_plot.csv` and used in the HTML graph.

## Input data

The package expects a local folder of images.

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`
- `.tif`
- `.tiff`

You can use:

- a subset of a public art dataset such as WikiArt
- a digitized museum collection exported as image files
- a folder with paintings gathered for a pilot experiment
- a thematic subcollection, for example portraits, landscapes, or one artist's works

## Metadata format

Metadata is optional. The pipeline can run without it.

If you provide metadata, use a CSV file with a required `filepath` column.

Recommended columns:

```text
filepath,artist,style,genre
```

Example:

```csv
filepath,artist,style,genre
data/paintings/img_00001.jpg,vincent-van-gogh,Post_Impressionism,landscape
data/paintings/img_00002.jpg,claude-monet,Impressionism,landscape
data/paintings/img_00003.jpg,Unknown Artist,Early_Renaissance,religious_painting
```

Only `filepath` is required. If `artist`, `style`, or `genre` are missing, they are filled with `unknown`.

Metadata matching is performed in two steps:

1. exact normalized filepath matching
2. filename fallback

This means metadata can still be matched if images were moved to another folder but filenames stayed the same.

## How to prepare data

The simplest folder structure is:

```text
project/
├── data/
│   └── paintings/
│       ├── img_00001.jpg
│       ├── img_00002.jpg
│       ├── img_00003.png
│       └── ...
├── metadata.csv
└── out/
```

Then run:

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="project/data/paintings",
    output_dir="project/out/run_01",
    metadata_csv="project/metadata.csv",
)
```

If you do not have metadata:

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="project/data/paintings",
    output_dir="project/out/run_01",
)
```

## Main output files

After execution, the output directory contains computational artifacts and the interactive graph.

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
- `df_plot.csv` — image paths, metadata, cluster labels, and projection coordinates
- `config.csv` — run configuration
- `similarity_graph.html` — interactive graph visualization
- `thumbs/` — thumbnails used inside the HTML graph
- `museum_map_export.zip` — lightweight packaged result for local sharing or archiving

## Lightweight export

The package creates a lightweight export zip for sharing. It contains the visual outputs rather than all heavy numerical arrays.

The export is intended to include:

```text
similarity_graph.html
thumbs/
df_plot.csv
config.csv
```

Large files such as embeddings and feature matrices remain in the output directory but are not needed for the interactive demo.

## What the HTML output contains

The main visual artifact is `similarity_graph.html`.

It represents the collection as a graph:

- nodes correspond to paintings
- each node contains a painting thumbnail
- node border color indicates cluster membership
- edges connect paintings with strong local similarity
- metadata such as artist, style, genre, cluster, and filename can be displayed

This file can be opened locally in a browser. If a browser restricts local file access for thumbnails, serve the output directory with a lightweight local server.

Example:

```bash
cd output_dir
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
    metadata_csv="./metadata.csv",
    batch_size=16,
    n_palette_colors=6,
    weight_clip=1.0,
    weight_palette=0.4,
    weight_composition=0.8,
    graph_k_neighbors=3,
    max_per_cluster_for_graph=25,
    thumb_size=96,
)
```

Important parameters:

- `metadata_csv` — optional CSV file with metadata
- `batch_size` — embedding batch size
- `n_palette_colors` — number of dominant palette colors
- `weight_clip` — weight of CLIP embeddings in the combined representation
- `weight_palette` — weight of palette descriptors
- `weight_composition` — weight of composition descriptors
- `graph_k_neighbors` — number of neighbors in the similarity graph
- `max_per_cluster_for_graph` — representative sample size per cluster
- `thumb_size` — thumbnail size used in the HTML graph

## Example with WikiArt-style data

```python
from museum_map import build_museum_map

pipeline = build_museum_map(
    input_dir="./wikiart_subset/images",
    output_dir="./wikiart_subset/output",
    metadata_csv="./wikiart_subset/metadata.csv",
    batch_size=16,
    n_palette_colors=5,
    weight_clip=1.0,
    weight_palette=0.45,
    weight_composition=0.70,
    graph_k_neighbors=3,
    max_per_cluster_for_graph=30,
    thumb_size=96,
)

print("Graph:", pipeline.graph_html_path_)
print("Export:", pipeline.export_zip_path_)
```

## Current limitations

At the current stage, the package has several deliberate limitations:

- it expects local image folders rather than remote datasets
- it is optimized for exploratory work rather than industrial-scale deployment
- metadata support is currently CSV-based
- it exports the graph view as the main interactive artifact
- optional graph-embedding extensions such as Node2Vec are not part of the package API yet

These points are expected development directions rather than defects.

## Roadmap

Planned next steps include:

- command-line interface
- richer metadata formats such as JSON
- optional museum map scatter export
- cluster summaries and automatic cluster labels
- graph-aware extensions such as Node2Vec
- richer support for digital humanities and museum collection workflows
- automated release workflow for PyPI

## Citation

If you use this package in academic work, please cite the corresponding paper or software record. A suggested citation file is included as `CITATION.cff`.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
