"""Shared mesh/material context for fabrication strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh

from materials import MATERIALS
from openscad_step1.audit import sha256_file

from fabrication.contracts import FabricationConfig, Vec3


@dataclass
class FabricationContext:
    """Reusable geometry/material state consumed by every strategy."""

    config: FabricationConfig
    source_mesh_path: Path
    mesh_hash_sha256: str
    mesh: trimesh.Trimesh
    scale_factor: float
    mesh_bounds_mm: Vec3
    mesh_volume_mm3: float
    material_name: str
    material_key: str
    material_thickness_mm: float
    material_density_kg_m3: float
    max_sheet_size_mm: Tuple[float, float]

    def summary_payload(self) -> Dict[str, object]:
        return {
            "mesh_path": str(self.source_mesh_path),
            "mesh_hash_sha256": self.mesh_hash_sha256,
            "scale_factor": self.scale_factor,
            "mesh_bounds_mm": [float(v) for v in self.mesh_bounds_mm],
            "mesh_volume_mm3": float(self.mesh_volume_mm3),
            "material": {
                "key": self.material_key,
                "name": self.material_name,
                "thickness_mm": float(self.material_thickness_mm),
                "density_kg_m3": float(self.material_density_kg_m3),
                "max_sheet_size_mm": [float(v) for v in self.max_sheet_size_mm],
            },
        }


def build_fabrication_context(config: FabricationConfig) -> FabricationContext:
    mesh_path = Path(config.mesh_path)
    mesh_hash = sha256_file(mesh_path)
    mesh = _load_mesh(mesh_path)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    mesh.remove_unreferenced_vertices()

    mesh_norm, scale_factor = _normalize_mesh(mesh, config)
    material, thickness_mm = _resolve_material(config)

    return FabricationContext(
        config=config,
        source_mesh_path=mesh_path,
        mesh_hash_sha256=mesh_hash,
        mesh=mesh_norm,
        scale_factor=scale_factor,
        mesh_bounds_mm=_to_vec3(mesh_norm.extents),
        mesh_volume_mm3=float(abs(mesh_norm.volume)),
        material_name=material.name,
        material_key=config.material_key,
        material_thickness_mm=thickness_mm,
        material_density_kg_m3=float(material.density),
        max_sheet_size_mm=tuple(float(v) for v in material.max_size_mm),
    )


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Scene has no mesh geometry: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {mesh_path}")
    return loaded


def _normalize_mesh(
    mesh: trimesh.Trimesh, config: FabricationConfig
) -> Tuple[trimesh.Trimesh, float]:
    mesh_out = mesh.copy()
    scale_factor = 1.0
    if config.auto_scale and mesh_out.extents[2] > 1e-6:
        scale_factor = float(config.target_height_mm / mesh_out.extents[2])
        mesh_out.apply_scale(scale_factor)

    center = mesh_out.bounding_box.centroid
    mesh_out.apply_translation(-center)
    return mesh_out, scale_factor


def _resolve_material(config: FabricationConfig):
    if config.material_key not in MATERIALS:
        raise ValueError(f"Unknown material key: {config.material_key}")
    material = MATERIALS[config.material_key]
    thicknesses = sorted(float(t) for t in material.thicknesses_mm)
    if not thicknesses:
        raise ValueError(f"Material has no thickness values: {config.material_key}")

    if config.preferred_thickness_mm is None:
        selected = thicknesses[0] if len(thicknesses) == 1 else thicknesses[1]
    else:
        target = float(config.preferred_thickness_mm)
        selected = min(thicknesses, key=lambda value: abs(value - target))
    return material, float(selected)


def _to_vec3(values) -> Vec3:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))
