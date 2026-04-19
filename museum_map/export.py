from __future__ import annotations

from pathlib import Path
import shutil


def make_export_zip(output_dir: str | Path, zip_name: str = "museum_map_export") -> Path:
    output_path = Path(output_dir)
    zip_path = shutil.make_archive(str(output_path / zip_name), "zip", root_dir=output_path)
    return Path(zip_path)
