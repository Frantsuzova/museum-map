from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    files = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not files:
        raise ValueError(f"No supported images found in: {input_path}")

    return files
