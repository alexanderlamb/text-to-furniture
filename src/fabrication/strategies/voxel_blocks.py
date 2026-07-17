"""Voxel/block-fill fabrication strategy prototype.

This strategy approximates a normalized mesh as a coarse set of occupied voxels,
then greedily merges adjacent occupied cells into rectangular block parts. It is
intended as a deterministic tournament baseline rather than a final fabrication
method.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from fabrication.context import FabricationContext
from fabrication.contracts import FabricationPlan, Operation, Part, Vec3
from fabrication.scoring import add_basic_score

GridIndex = Tuple[int, int, int]
GridBounds = Tuple[GridIndex, GridIndex]


@dataclass(frozen=True)
class _OccupancyGrid:
    centers_by_index: Dict[GridIndex, np.ndarray]
    pitch_mm: float
    shape: GridIndex
    method: str

    @property
    def occupied_indices(self) -> List[GridIndex]:
        return sorted(self.centers_by_index)

    @property
    def occupied_count(self) -> int:
        return len(self.centers_by_index)


class VoxelBlocksStrategy:
    """Approximate the mesh as merged axis-aligned stock blocks."""

    strategy_id = "voxel_blocks"

    def generate(
        self, context: FabricationContext, artifacts_dir: Path | None = None
    ) -> FabricationPlan:
        warnings: List[str] = []
        mesh = context.mesh.copy()
        mesh.remove_unreferenced_vertices()

        pitch_mm = _choose_pitch(context)
        occupancy = _build_occupancy_grid(
            mesh=mesh,
            pitch_mm=pitch_mm,
            max_voxels_per_axis=int(context.config.max_voxels_per_axis),
            warnings=warnings,
        )

        if occupancy.occupied_count == 0:
            plan = FabricationPlan(
                strategy_id=self.strategy_id,
                status="error",
                warnings=warnings + ["Voxel strategy produced no occupied cells."],
                scores={"overall": 0.0},
                debug={
                    "pitch_mm": float(pitch_mm),
                    "occupancy_method": occupancy.method,
                    "grid_shape": list(occupancy.shape),
                    "occupied_voxel_count": 0,
                },
            )
            return plan

        block_bounds = _greedy_merge_blocks(occupancy.occupied_indices)
        parts = [
            _part_from_block(
                strategy_id=self.strategy_id,
                block_number=index,
                bounds=bounds,
                occupancy=occupancy,
                context=context,
            )
            for index, bounds in enumerate(block_bounds)
        ]

        if len(parts) > context.config.part_budget_max:
            warnings.append(
                "Voxel block count exceeds part budget: "
                f"{len(parts)} > {context.config.part_budget_max}"
            )

        material_estimate = _material_estimate(parts, context)
        scores = _score_components(parts, context, warnings)
        status = "ok" if not warnings else "warning"
        plan = FabricationPlan(
            strategy_id=self.strategy_id,
            status=status,
            parts=parts,
            operations=[
                Operation(
                    operation_id="voxel_blocks_occupancy_merge",
                    strategy_id=self.strategy_id,
                    kind="voxelize_and_greedy_merge_blocks",
                    part_ids=[part.part_id for part in parts],
                    metadata={
                        "pitch_mm": float(occupancy.pitch_mm),
                        "occupied_voxel_count": int(occupancy.occupied_count),
                        "block_count": int(len(parts)),
                        "merge_algorithm": "seeded_rectangular_prisms_axis_order_v0",
                    },
                )
            ],
            scores=scores,
            warnings=warnings,
            artifacts={},
            debug={
                "mesh": {
                    "bounds_mm": [float(v) for v in context.mesh_bounds_mm],
                    "volume_mm3": float(context.mesh_volume_mm3),
                    "watertight": bool(mesh.is_watertight),
                },
                "occupancy": {
                    "method": occupancy.method,
                    "pitch_mm": float(occupancy.pitch_mm),
                    "grid_shape": list(occupancy.shape),
                    "occupied_voxel_count": int(occupancy.occupied_count),
                    "occupied_volume_mm3": float(
                        occupancy.occupied_count * occupancy.pitch_mm**3
                    ),
                    "voxel_pitch_multiplier": float(
                        context.config.voxel_pitch_multiplier
                    ),
                    "max_voxels_per_axis": int(context.config.max_voxels_per_axis),
                },
                "merge": {
                    "algorithm": "greedy_seeded_max_prism",
                    "axis_orders_considered": [
                        list(order) for order in permutations((0, 1, 2))
                    ],
                    "block_count": int(len(parts)),
                    "block_index_bounds_preview": [
                        {
                            "min": list(bounds[0]),
                            "max": list(bounds[1]),
                        }
                        for bounds in block_bounds[:50]
                    ],
                },
                "material_estimate": material_estimate,
            },
        )
        add_basic_score(plan, context)

        if artifacts_dir is not None:
            strategy_dir = Path(artifacts_dir) / self.strategy_id
            strategy_dir.mkdir(parents=True, exist_ok=True)
            debug_path = strategy_dir / "voxel_blocks_debug.json"
            _write_json(
                debug_path,
                {
                    "strategy_id": self.strategy_id,
                    "status": plan.status,
                    "scores": plan.scores,
                    "warnings": plan.warnings,
                    "debug": plan.debug,
                },
            )
            plan.artifacts = {"debug_json": str(debug_path)}

        return plan


# Backwards-friendly singular alias for callers that guess the class name.
VoxelBlockStrategy = VoxelBlocksStrategy


def _choose_pitch(context: FabricationContext) -> float:
    cfg = context.config
    extents = np.asarray(context.mesh_bounds_mm, dtype=float)
    max_extent = float(np.max(extents)) if extents.size else 0.0
    axis_budget = max(1, int(cfg.max_voxels_per_axis) - 2)
    pitch_from_axis_cap = max_extent / axis_budget if max_extent > 0.0 else 0.0
    pitch_from_material = float(context.material_thickness_mm) * float(
        cfg.voxel_pitch_multiplier
    )
    return float(
        max(
            pitch_from_material,
            pitch_from_axis_cap,
            float(context.material_thickness_mm),
            float(cfg.min_feature_mm),
            1e-6,
        )
    )


def _build_occupancy_grid(
    *,
    mesh,
    pitch_mm: float,
    max_voxels_per_axis: int,
    warnings: List[str],
) -> _OccupancyGrid:
    surface_points = np.empty((0, 3), dtype=float)

    try:
        voxel_grid = mesh.voxelized(pitch=pitch_mm)
        surface_points = np.asarray(voxel_grid.points, dtype=float)
        try:
            filled = voxel_grid.fill()
            if filled is not None:
                voxel_grid = filled
            occupancy = _occupancy_from_trimesh_grid(
                voxel_grid=voxel_grid,
                pitch_mm=pitch_mm,
                method="trimesh.voxelized.fill",
            )
            if occupancy.occupied_count > 0:
                return occupancy
        except Exception as exc:  # pragma: no cover - depends on optional deps
            warnings.append(
                "trimesh voxel fill failed; falling back to sampled occupancy: "
                f"{type(exc).__name__}: {exc}"
            )
    except Exception as exc:  # pragma: no cover - defensive for odd meshes
        warnings.append(f"trimesh voxelization failed: {type(exc).__name__}: {exc}")

    sampled = _sample_occupancy_grid(
        mesh=mesh,
        pitch_mm=pitch_mm,
        surface_points=surface_points,
        max_voxels_per_axis=max_voxels_per_axis,
        warnings=warnings,
    )
    if sampled.occupied_count > 0:
        return sampled

    if surface_points.size:
        return _occupancy_from_points(
            points=surface_points,
            pitch_mm=pitch_mm,
            method="trimesh.voxelized.surface",
        )

    return _OccupancyGrid(
        centers_by_index={},
        pitch_mm=float(pitch_mm),
        shape=(0, 0, 0),
        method="empty",
    )


def _occupancy_from_trimesh_grid(
    voxel_grid, pitch_mm: float, method: str
) -> _OccupancyGrid:
    indices = np.asarray(voxel_grid.sparse_indices, dtype=int)
    points = np.asarray(voxel_grid.points, dtype=float)
    if indices.size == 0 or points.size == 0:
        shape = tuple(int(v) for v in getattr(voxel_grid, "shape", (0, 0, 0)))
        return _OccupancyGrid({}, float(pitch_mm), shape, method)

    centers_by_index: Dict[GridIndex, np.ndarray] = {}
    for raw_index, point in zip(indices, points):
        index = (int(raw_index[0]), int(raw_index[1]), int(raw_index[2]))
        centers_by_index[index] = np.asarray(point, dtype=float)

    shape_raw = getattr(voxel_grid, "shape", None)
    if shape_raw is None:
        max_index = np.max(indices, axis=0)
        shape = tuple(int(v) + 1 for v in max_index)
    else:
        shape = tuple(int(v) for v in shape_raw)
    return _OccupancyGrid(centers_by_index, float(pitch_mm), shape, method)


def _sample_occupancy_grid(
    *,
    mesh,
    pitch_mm: float,
    surface_points: np.ndarray,
    max_voxels_per_axis: int,
    warnings: List[str],
) -> _OccupancyGrid:
    bounds = np.asarray(mesh.bounds, dtype=float)
    if surface_points.size:
        ref_min = np.minimum(bounds[0], np.min(surface_points, axis=0))
        ref_max = np.maximum(bounds[1], np.max(surface_points, axis=0))
    else:
        ref_min = bounds[0]
        ref_max = bounds[1]

    grid_min = np.floor((ref_min - pitch_mm * 0.5) / pitch_mm) * pitch_mm
    grid_max = np.ceil((ref_max + pitch_mm * 0.5) / pitch_mm) * pitch_mm
    shape_arr = np.maximum(np.ceil((grid_max - grid_min) / pitch_mm).astype(int), 1)

    if int(np.max(shape_arr)) > max_voxels_per_axis + 4:
        warnings.append(
            "Sampled occupancy grid exceeded requested axis cap after padding: "
            f"shape={tuple(int(v) for v in shape_arr)} cap={max_voxels_per_axis}"
        )

    center_origin = grid_min + pitch_mm * 0.5
    surface_occupied: Dict[GridIndex, np.ndarray] = {}

    if surface_points.size:
        surface_indices = np.rint((surface_points - center_origin) / pitch_mm).astype(
            int
        )
        valid = np.all((surface_indices >= 0) & (surface_indices < shape_arr), axis=1)
        for raw_index in surface_indices[valid]:
            index = (int(raw_index[0]), int(raw_index[1]), int(raw_index[2]))
            surface_occupied[index] = center_origin + pitch_mm * np.asarray(
                index, dtype=float
            )

    ranges = [np.arange(int(shape_arr[axis]), dtype=int) for axis in range(3)]
    grid_indices = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    grid_centers = center_origin + pitch_mm * grid_indices.astype(float)

    try:
        inside = np.asarray(mesh.contains(grid_centers), dtype=bool)
        inside_occupied: Dict[GridIndex, np.ndarray] = {}
        for raw_index, center in zip(grid_indices[inside], grid_centers[inside]):
            index = (int(raw_index[0]), int(raw_index[1]), int(raw_index[2]))
            inside_occupied[index] = np.asarray(center, dtype=float)
        if inside_occupied:
            occupied = inside_occupied
            method = "trimesh.contains.centers"
        else:
            occupied = surface_occupied
            method = "trimesh.surface_voxels_no_contained_centers"
    except Exception as exc:  # pragma: no cover - depends on rtree/mesh quality
        warnings.append(
            "trimesh contains sampling failed; using surface voxels only: "
            f"{type(exc).__name__}: {exc}"
        )
        occupied = surface_occupied
        method = "surface_voxels_only"

    return _OccupancyGrid(
        centers_by_index=occupied,
        pitch_mm=float(pitch_mm),
        shape=tuple(int(v) for v in shape_arr),
        method=method,
    )


def _occupancy_from_points(
    *, points: np.ndarray, pitch_mm: float, method: str
) -> _OccupancyGrid:
    ref_min = np.min(points, axis=0)
    grid_min = np.floor((ref_min - pitch_mm * 0.5) / pitch_mm) * pitch_mm
    center_origin = grid_min + pitch_mm * 0.5
    raw_indices = np.rint((points - center_origin) / pitch_mm).astype(int)

    occupied: Dict[GridIndex, np.ndarray] = {}
    for raw_index in raw_indices:
        index = (int(raw_index[0]), int(raw_index[1]), int(raw_index[2]))
        occupied[index] = center_origin + pitch_mm * np.asarray(index, dtype=float)

    max_index = (
        np.max(raw_indices, axis=0) if len(raw_indices) else np.zeros(3, dtype=int)
    )
    return _OccupancyGrid(
        centers_by_index=occupied,
        pitch_mm=float(pitch_mm),
        shape=tuple(int(v) + 1 for v in max_index),
        method=method,
    )


def _greedy_merge_blocks(indices: Sequence[GridIndex]) -> List[GridBounds]:
    remaining = set(indices)
    blocks: List[GridBounds] = []

    while remaining:
        seed = min(remaining)
        candidates = []
        for axis_order in permutations((0, 1, 2)):
            bounds = _expand_seed(seed, remaining, axis_order)
            candidates.append((bounds, axis_order))

        bounds, _axis_order = min(
            candidates,
            key=lambda item: (
                -_block_voxel_count(item[0]),
                item[0][0],
                item[0][1],
                item[1],
            ),
        )
        blocks.append(bounds)
        remaining.difference_update(_iter_block_indices(bounds))

    return blocks


def _expand_seed(
    seed: GridIndex, occupied: set[GridIndex], axis_order: Tuple[int, int, int]
) -> GridBounds:
    mins = [int(seed[0]), int(seed[1]), int(seed[2])]
    maxs = [int(seed[0]), int(seed[1]), int(seed[2])]

    for axis in axis_order:
        while True:
            candidate_maxs = list(maxs)
            candidate_maxs[axis] += 1
            candidate_bounds = (
                (mins[0], mins[1], mins[2]),
                (candidate_maxs[0], candidate_maxs[1], candidate_maxs[2]),
            )
            if all(
                index in occupied for index in _iter_block_indices(candidate_bounds)
            ):
                maxs = candidate_maxs
            else:
                break

    return ((mins[0], mins[1], mins[2]), (maxs[0], maxs[1], maxs[2]))


def _iter_block_indices(bounds: GridBounds) -> Iterable[GridIndex]:
    mins, maxs = bounds
    for ix in range(mins[0], maxs[0] + 1):
        for iy in range(mins[1], maxs[1] + 1):
            for iz in range(mins[2], maxs[2] + 1):
                yield (ix, iy, iz)


def _block_voxel_count(bounds: GridBounds) -> int:
    mins, maxs = bounds
    return int(
        (maxs[0] - mins[0] + 1) * (maxs[1] - mins[1] + 1) * (maxs[2] - mins[2] + 1)
    )


def _part_from_block(
    *,
    strategy_id: str,
    block_number: int,
    bounds: GridBounds,
    occupancy: _OccupancyGrid,
    context: FabricationContext,
) -> Part:
    mins, maxs = bounds
    min_center = occupancy.centers_by_index[mins]
    max_center = occupancy.centers_by_index[maxs]
    half_pitch = occupancy.pitch_mm * 0.5
    aabb_min_arr = np.minimum(min_center, max_center) - half_pitch
    aabb_max_arr = np.maximum(min_center, max_center) + half_pitch
    dimensions = np.maximum(aabb_max_arr - aabb_min_arr, 0.0)
    volume_mm3 = float(np.prod(dimensions))
    surface_area_mm2 = float(
        2.0
        * (
            dimensions[0] * dimensions[1]
            + dimensions[0] * dimensions[2]
            + dimensions[1] * dimensions[2]
        )
    )
    mass_kg = _mass_kg(volume_mm3, context)

    return Part(
        part_id=f"voxel_block_{block_number:04d}",
        strategy_id=strategy_id,
        kind="voxel_block",
        quantity=1,
        material_thickness_mm=float(context.material_thickness_mm),
        area_mm2=surface_area_mm2,
        volume_mm3=volume_mm3,
        aabb_min=_to_vec3(aabb_min_arr),
        aabb_max=_to_vec3(aabb_max_arr),
        metadata={
            "grid_index_min": list(mins),
            "grid_index_max": list(maxs),
            "voxel_count": _block_voxel_count(bounds),
            "pitch_mm": float(occupancy.pitch_mm),
            "dimensions_mm": [float(v) for v in dimensions],
            "center_3d": [float(v) for v in (aabb_min_arr + dimensions * 0.5)],
            "material_volume_mm3": volume_mm3,
            "estimated_mass_kg": mass_kg,
        },
    )


def _material_estimate(
    parts: Sequence[Part], context: FabricationContext
) -> Dict[str, object]:
    total_volume = float(sum(part.volume_mm3 for part in parts))
    return {
        "material_key": context.material_key,
        "material_name": context.material_name,
        "part_count": int(len(parts)),
        "total_volume_mm3": total_volume,
        "total_volume_cm3": total_volume / 1000.0,
        "estimated_mass_kg": _mass_kg(total_volume, context),
    }


def _score_components(
    parts: Sequence[Part], context: FabricationContext, warnings: Sequence[str]
) -> Dict[str, float]:
    total_volume = float(sum(part.volume_mm3 for part in parts))
    mesh_volume = max(float(context.mesh_volume_mm3), 1.0)
    volume_error = abs(total_volume - mesh_volume) / mesh_volume
    overfill = max(0.0, total_volume / mesh_volume - 1.0)
    part_count = len(parts)

    return {
        "fidelity": _clamp01(1.0 - min(volume_error, 1.0) * 0.75),
        "material_efficiency": _clamp01(1.0 / (1.0 + overfill)),
        "assembly_simplicity": _clamp01(1.0 / (1.0 + part_count / 18.0)),
        "strength_proxy": 0.85,
        "part_count": _clamp01(1.0 / (1.0 + part_count / 36.0)),
        "risk": 0.70 if not warnings else 0.55,
    }


def _mass_kg(volume_mm3: float, context: FabricationContext) -> float:
    return float(volume_mm3) / 1_000_000_000.0 * float(context.material_density_kg_m3)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_vec3(values: Sequence[float]) -> Vec3:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
