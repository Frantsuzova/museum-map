from __future__ import annotations
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
        _make_bordered_thumbnail(Path(row["filepath"]), dst, border_color=colors[cluster_id], size=thumb_size, border_px=thumb_border_px)
        thumb_paths[i] = dst

    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, cdn_resources="in_line")
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
        net.add_edge(a, b, value=w, color="rgba(70, 90, 140, 0.28)")

    net.set_options("""
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
    """)

    html_path = output_path / "similarity_graph.html"
    net.save_graph(str(html_path))
    return html_path
