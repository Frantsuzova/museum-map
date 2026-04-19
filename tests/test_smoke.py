from pathlib import Path

import pytest


def test_package_import():
    import museum_map  # noqa: F401


def test_readme_exists():
    assert Path("README.md").exists()
