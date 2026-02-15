"""Shared fixtures for the streamlined first-principles workflow."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def box_mesh_file() -> str:
    mesh = trimesh.creation.box(extents=[200, 140, 120])
    mesh.apply_translation([0.0, 0.0, 60.0])

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        mesh.export(f.name)
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.fixture
def cylinder_mesh_file() -> str:
    mesh = trimesh.creation.cylinder(radius=55, height=180)
    mesh.apply_translation([0.0, 0.0, 90.0])

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        mesh.export(f.name)
        path = f.name
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)
