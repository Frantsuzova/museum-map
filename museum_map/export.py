from __future__ import annotations

from pathlib import Path
import shutil


DEFAULT_EXPORT_ITEMS = [
    "similarity_graph.html",
    "thumbs",
    "df_plot.csv",
    "config.csv",
]


def make_export_zip(
    output_dir: str | Path,
    zip_name: str = "museum_map_export",
    items: list[str] | None = None,
) -> Path:
    output_path = Path(output_dir)
    items = items or DEFAULT_EXPORT_ITEMS

    export_dir = output_path / "_export_light"

    if export_dir.exists():
        shutil.rmtree(export_dir)

    export_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        src = output_path / item
        dst = export_dir / item

        if not src.exists():
            continue

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    zip_path = shutil.make_archive(
        str(output_path / zip_name),
        "zip",
        root_dir=export_dir,
    )

    return Path(zip_path)
