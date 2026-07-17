"""Prototype waffle/rib fabrication strategy.

The strategy samples a mesh with coarse axis-aligned bands and emits two
perpendicular sets of vertical sheet ribs. The geometry is intentionally AABB
based: each occupied band becomes a rectangular rib envelope with enough
metadata to prototype slot layout and assembly order without promising exact
cut-ready outlines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from fabrication.context import FabricationContext
from fabrication.contracts import FabricationPlan, Joint, Operation, Part, Vec3
from fabrication.scoring import add_basic_score

_EPS = 1e-7
_AXIS_NAMES = ("x", "y", "z")
_VERTICAL_AXIS = 2


@dataclass(frozen=True)
class _RibSetSpec:
    """A vertical rib family running along one horizontal world axis."""

    rib_set: str
    run_axis: int
    station_axis: int
    count: int

    @property
    def run_axis_name(self) -> str:
        return _AXIS_NAMES[self.run_axis]

    @property
    def station_axis_name(self) -> str:
        return _AXIS_NAMES[self.station_axis]


@dataclass(frozen=True)
class _BandSample:
    station_mm: float
    band_min_mm: float
    band_max_mm: float
    aabb_min: Vec3
    aabb_max: Vec3
    triangle_count: int
    method: str


class WaffleRibsStrategy:
    """Approximate a mesh as two slotted perpendicular rib sets."""

    strategy_id = "waffle_ribs"

    def generate(
        self, context: FabricationContext, artifacts_dir: Path | None = None
    ) -> FabricationPlan:
        warnings: list[str] = []
        mesh = context.mesh.copy()
        mesh.remove_unreferenced_vertices()

        bounds = _mesh_bounds(mesh)
        extents = np.maximum(bounds[1] - bounds[0], 0.0)
        if not _has_usable_horizontal_extents(extents):
            return FabricationPlan(
                strategy_id=self.strategy_id,
                status="error",
                warnings=["Mesh has no usable horizontal extent for waffle ribs."],
                scores={"overall": 0.0},
                debug={"mesh_bounds_mm": [_vec3(bounds[0]), _vec3(bounds[1])]},
            )

        specs, allocation_debug, allocation_warnings = _rib_set_specs(
            context=context,
            extents=extents,
        )
        warnings.extend(allocation_warnings)

        triangles = np.asarray(mesh.triangles, dtype=float)
        triangle_bounds = _triangle_bounds(triangles)
        parts: list[Part] = []
        band_debug: list[dict[str, object]] = []

        for spec in specs:
            set_parts, set_debug, set_warnings = _build_rib_set(
                strategy_id=self.strategy_id,
                spec=spec,
                bounds=bounds,
                extents=extents,
                triangles=triangles,
                triangle_bounds=triangle_bounds,
                context=context,
            )
            parts.extend(set_parts)
            band_debug.extend(set_debug)
            warnings.extend(set_warnings)

        _renumber_parts(parts)
        joints = _build_slot_joints(parts, context)
        _annotate_slot_counts(parts, joints)
        operations = _build_operations(parts, joints, specs, context, allocation_debug)
        material_estimate = _material_estimate(parts, context)

        has_x_set = any(part.metadata.get("rib_set") == "x" for part in parts)
        has_y_set = any(part.metadata.get("rib_set") == "y" for part in parts)
        status = "ok" if parts and has_x_set and has_y_set else "error"
        if status == "ok" and warnings:
            status = "warning"
        elif not parts:
            warnings.append("Waffle rib strategy produced no rib parts.")
        elif not (has_x_set and has_y_set):
            warnings.append("Waffle rib strategy did not produce both rib sets.")

        plan = FabricationPlan(
            strategy_id=self.strategy_id,
            status=status,
            parts=parts,
            joints=joints,
            operations=operations,
            scores=_score_components(parts, joints, context, warnings),
            warnings=warnings,
            artifacts={},
            debug={
                "mesh": {
                    "bounds_min_mm": _vec3(bounds[0]),
                    "bounds_max_mm": _vec3(bounds[1]),
                    "bounds_extents_mm": _vec3(extents),
                    "volume_mm3": float(context.mesh_volume_mm3),
                    "watertight": bool(mesh.is_watertight),
                },
                "allocation": allocation_debug,
                "rib_sets": [
                    {
                        "rib_set": spec.rib_set,
                        "run_axis": spec.run_axis_name,
                        "station_axis": spec.station_axis_name,
                        "requested_count": int(spec.count),
                    }
                    for spec in specs
                ],
                "bands": band_debug,
                "slot_count": len(joints),
                "material_estimate": material_estimate,
            },
        )
        add_basic_score(plan, context)

        strategy_artifacts = _write_artifacts(plan, artifacts_dir)
        if strategy_artifacts:
            plan.artifacts = strategy_artifacts

        return plan


# Backwards-friendly singular alias for callers that guess the class name.
WaffleRibStrategy = WaffleRibsStrategy


def _mesh_bounds(mesh) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)):
        return bounds

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return np.zeros((2, 3), dtype=float)
    return np.vstack((np.min(vertices, axis=0), np.max(vertices, axis=0)))


def _has_usable_horizontal_extents(extents: Sequence[float]) -> bool:
    return float(extents[0]) > _EPS and float(extents[1]) > _EPS


def _rib_set_specs(
    *, context: FabricationContext, extents: np.ndarray
) -> tuple[list[_RibSetSpec], dict[str, object], list[str]]:
    warnings: list[str] = []
    thickness = max(
        float(context.material_thickness_mm), float(context.config.min_feature_mm), _EPS
    )
    target_spacing = max(thickness * 8.0, float(context.config.min_feature_mm) * 4.0)

    raw_x_running = _raw_count(float(extents[1]), target_spacing)
    raw_y_running = _raw_count(float(extents[0]), target_spacing)
    budget = int(context.config.part_budget_max)
    effective_budget = max(2, budget)
    if budget < 2:
        warnings.append(
            "part_budget_max is below the two-rib minimum for perpendicular "
            f"waffle sets: {budget}; emitting 2 ribs."
        )

    x_count, y_count = _fit_counts_to_budget(
        raw_x_running,
        raw_y_running,
        effective_budget,
        extents,
    )

    debug = {
        "target_spacing_mm": round(float(target_spacing), 6),
        "raw_counts": {
            "x_running_ribs": int(raw_x_running),
            "y_running_ribs": int(raw_y_running),
        },
        "part_budget_max": int(budget),
        "effective_budget": int(effective_budget),
        "selected_counts": {
            "x_running_ribs": int(x_count),
            "y_running_ribs": int(y_count),
        },
    }
    return (
        [
            _RibSetSpec(
                rib_set="x",
                run_axis=0,
                station_axis=1,
                count=x_count,
            ),
            _RibSetSpec(
                rib_set="y",
                run_axis=1,
                station_axis=0,
                count=y_count,
            ),
        ],
        debug,
        warnings,
    )


def _raw_count(extent_mm: float, target_spacing_mm: float) -> int:
    if extent_mm <= _EPS:
        return 1
    return max(1, int(math.ceil(extent_mm / max(target_spacing_mm, _EPS))))


def _fit_counts_to_budget(
    x_count: int, y_count: int, budget: int, extents: Sequence[float]
) -> tuple[int, int]:
    x_count = max(1, int(x_count))
    y_count = max(1, int(y_count))
    budget = max(2, int(budget))

    while x_count + y_count > budget:
        if x_count <= 1 and y_count <= 1:
            break
        x_spacing = float(extents[1]) / max(x_count, 1)
        y_spacing = float(extents[0]) / max(y_count, 1)
        if x_count > 1 and (y_count <= 1 or x_spacing <= y_spacing):
            x_count -= 1
        else:
            y_count -= 1
    return x_count, y_count


def _build_rib_set(
    *,
    strategy_id: str,
    spec: _RibSetSpec,
    bounds: np.ndarray,
    extents: np.ndarray,
    triangles: np.ndarray,
    triangle_bounds: tuple[np.ndarray, np.ndarray],
    context: FabricationContext,
) -> tuple[list[Part], list[dict[str, object]], list[str]]:
    parts: list[Part] = []
    debug: list[dict[str, object]] = []
    warnings: list[str] = []
    thickness = max(float(context.material_thickness_mm), _EPS)
    stations = _stations_for_extent(
        float(bounds[0, spec.station_axis]),
        float(bounds[1, spec.station_axis]),
        int(spec.count),
    )
    spacing = _effective_spacing(
        float(extents[spec.station_axis]),
        len(stations),
    )
    sampling_band_width = max(
        thickness,
        float(context.config.min_feature_mm),
        spacing * 0.50,
    )

    for station_index, station in enumerate(stations):
        sample = _sample_band(
            bounds=bounds,
            triangles=triangles,
            triangle_bounds=triangle_bounds,
            station_axis=spec.station_axis,
            run_axis=spec.run_axis,
            station_mm=float(station),
            sampling_band_width_mm=sampling_band_width,
            material_thickness_mm=thickness,
        )
        part = _part_from_sample(
            strategy_id=strategy_id,
            spec=spec,
            station_index=station_index,
            sample=sample,
            context=context,
        )
        if part is None:
            debug.append(
                {
                    "rib_set": spec.rib_set,
                    "station_index": int(station_index),
                    "station_mm": round(float(station), 6),
                    "status": "empty",
                    "sampling_band_width_mm": round(float(sampling_band_width), 6),
                    "triangle_count": int(sample.triangle_count),
                }
            )
            continue

        parts.append(part)
        debug.append(
            {
                "rib_set": spec.rib_set,
                "station_index": int(station_index),
                "part_id": part.part_id,
                "station_mm": round(float(station), 6),
                "status": "ok",
                "sampling_band_width_mm": round(float(sampling_band_width), 6),
                "triangle_count": int(sample.triangle_count),
                "method": sample.method,
                "aabb_min": part.aabb_min,
                "aabb_max": part.aabb_max,
                "area_mm2": round(float(part.area_mm2), 6),
            }
        )

    if not parts:
        fallback = _fallback_rib_part(
            strategy_id=strategy_id,
            spec=spec,
            bounds=bounds,
            context=context,
        )
        parts.append(fallback)
        warnings.append(
            f"No occupied bands found for rib set {spec.rib_set}; emitted one "
            "global AABB fallback rib."
        )
        debug.append(
            {
                "rib_set": spec.rib_set,
                "station_index": 0,
                "part_id": fallback.part_id,
                "status": "fallback",
                "method": "global_mesh_aabb",
            }
        )

    return parts, debug, warnings


def _stations_for_extent(min_value: float, max_value: float, count: int) -> np.ndarray:
    count = max(1, int(count))
    if count == 1:
        return np.asarray([(min_value + max_value) * 0.5], dtype=float)

    extent = max_value - min_value
    margin = min(extent * 0.10, extent / (count + 1.0))
    start = min_value + margin
    end = max_value - margin
    if end <= start:
        return np.linspace(min_value, max_value, count, dtype=float)
    return np.linspace(start, end, count, dtype=float)


def _effective_spacing(extent_mm: float, count: int) -> float:
    if count <= 1:
        return max(float(extent_mm), _EPS)
    return max(float(extent_mm) / float(count - 1), _EPS)


def _triangle_bounds(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if triangles.size == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty
    return np.min(triangles, axis=1), np.max(triangles, axis=1)


def _sample_band(
    *,
    bounds: np.ndarray,
    triangles: np.ndarray,
    triangle_bounds: tuple[np.ndarray, np.ndarray],
    station_axis: int,
    run_axis: int,
    station_mm: float,
    sampling_band_width_mm: float,
    material_thickness_mm: float,
) -> _BandSample:
    half_band = max(float(sampling_band_width_mm), _EPS) * 0.5
    band_min = float(station_mm) - half_band
    band_max = float(station_mm) + half_band
    tri_min, tri_max = triangle_bounds
    mask = np.zeros(0, dtype=bool)
    if tri_min.size and tri_max.size:
        mask = (tri_min[:, station_axis] <= band_max + _EPS) & (
            tri_max[:, station_axis] >= band_min - _EPS
        )

    if bool(np.any(mask)):
        points = triangles[mask].reshape(-1, 3)
        aabb_min_arr = np.min(points, axis=0)
        aabb_max_arr = np.max(points, axis=0)
        method = "triangle_aabb_band_overlap"
        triangle_count = int(np.count_nonzero(mask))
    else:
        aabb_min_arr = np.array(bounds[0], dtype=float)
        aabb_max_arr = np.array(bounds[1], dtype=float)
        method = "global_mesh_aabb_fallback"
        triangle_count = 0

    aabb_min_arr = np.maximum(aabb_min_arr, bounds[0])
    aabb_max_arr = np.minimum(aabb_max_arr, bounds[1])
    _ensure_axis_span(aabb_min_arr, aabb_max_arr, run_axis, bounds)
    _ensure_axis_span(aabb_min_arr, aabb_max_arr, _VERTICAL_AXIS, bounds)

    half_thickness = max(float(material_thickness_mm), _EPS) * 0.5
    aabb_min_arr[station_axis] = max(
        float(bounds[0, station_axis]), float(station_mm) - half_thickness
    )
    aabb_max_arr[station_axis] = min(
        float(bounds[1, station_axis]), float(station_mm) + half_thickness
    )
    if aabb_max_arr[station_axis] - aabb_min_arr[station_axis] <= _EPS:
        aabb_min_arr[station_axis] = float(station_mm) - half_thickness
        aabb_max_arr[station_axis] = float(station_mm) + half_thickness

    return _BandSample(
        station_mm=float(station_mm),
        band_min_mm=float(band_min),
        band_max_mm=float(band_max),
        aabb_min=_vec3(aabb_min_arr),
        aabb_max=_vec3(aabb_max_arr),
        triangle_count=triangle_count,
        method=method,
    )


def _ensure_axis_span(
    aabb_min: np.ndarray, aabb_max: np.ndarray, axis: int, bounds: np.ndarray
) -> None:
    if aabb_max[axis] - aabb_min[axis] > _EPS:
        return

    center = float((aabb_min[axis] + aabb_max[axis]) * 0.5)
    span = max(float(bounds[1, axis] - bounds[0, axis]), _EPS)
    half_span = span * 0.5
    aabb_min[axis] = max(float(bounds[0, axis]), center - half_span)
    aabb_max[axis] = min(float(bounds[1, axis]), center + half_span)


def _part_from_sample(
    *,
    strategy_id: str,
    spec: _RibSetSpec,
    station_index: int,
    sample: _BandSample,
    context: FabricationContext,
) -> Part | None:
    min_arr = np.asarray(sample.aabb_min, dtype=float)
    max_arr = np.asarray(sample.aabb_max, dtype=float)
    dimensions = np.maximum(max_arr - min_arr, 0.0)
    run_length = float(dimensions[spec.run_axis])
    height = float(dimensions[_VERTICAL_AXIS])
    if run_length <= _EPS or height <= _EPS:
        return None

    area_mm2 = run_length * height
    thickness = float(context.material_thickness_mm)
    volume_mm3 = area_mm2 * thickness
    center = min_arr + dimensions * 0.5

    return Part(
        part_id=f"waffle_rib_{spec.rib_set}_{station_index:03d}",
        strategy_id=strategy_id,
        kind="waffle_rib",
        quantity=1,
        material_thickness_mm=thickness,
        area_mm2=area_mm2,
        volume_mm3=volume_mm3,
        aabb_min=_vec3(min_arr),
        aabb_max=_vec3(max_arr),
        metadata={
            "rib_set": spec.rib_set,
            "rib_index": int(station_index),
            "run_axis": spec.run_axis_name,
            "station_axis": spec.station_axis_name,
            "vertical_axis": _AXIS_NAMES[_VERTICAL_AXIS],
            "station_offset_mm": round(float(sample.station_mm), 6),
            "sampling_band_min_mm": round(float(sample.band_min_mm), 6),
            "sampling_band_max_mm": round(float(sample.band_max_mm), 6),
            "sampling_band_width_mm": round(
                float(sample.band_max_mm - sample.band_min_mm), 6
            ),
            "source_triangle_count": int(sample.triangle_count),
            "aabb_method": sample.method,
            "sheet_profile_width_mm": run_length,
            "sheet_profile_height_mm": height,
            "dimensions_mm": [float(value) for value in dimensions],
            "center_3d": [float(value) for value in center],
            "slot_count_estimate": 0,
        },
    )


def _fallback_rib_part(
    *,
    strategy_id: str,
    spec: _RibSetSpec,
    bounds: np.ndarray,
    context: FabricationContext,
) -> Part:
    station = float((bounds[0, spec.station_axis] + bounds[1, spec.station_axis]) * 0.5)
    sample = _sample_band(
        bounds=bounds,
        triangles=np.empty((0, 3, 3), dtype=float),
        triangle_bounds=_triangle_bounds(np.empty((0, 3, 3), dtype=float)),
        station_axis=spec.station_axis,
        run_axis=spec.run_axis,
        station_mm=station,
        sampling_band_width_mm=float(context.material_thickness_mm),
        material_thickness_mm=float(context.material_thickness_mm),
    )
    part = _part_from_sample(
        strategy_id=strategy_id,
        spec=spec,
        station_index=0,
        sample=sample,
        context=context,
    )
    assert part is not None
    part.metadata["aabb_method"] = "global_mesh_aabb_fallback"
    return part


def _renumber_parts(parts: Sequence[Part]) -> None:
    counters: Dict[str, int] = {"x": 0, "y": 0}
    for part in sorted(
        parts,
        key=lambda item: (
            str(item.metadata.get("rib_set", "")),
            float(item.metadata.get("station_offset_mm", 0.0)),
            item.part_id,
        ),
    ):
        rib_set = str(part.metadata.get("rib_set", "rib"))
        index = counters.get(rib_set, 0)
        counters[rib_set] = index + 1
        part.part_id = f"waffle_rib_{rib_set}_{index:03d}"
        part.metadata["rib_index"] = index


def _build_slot_joints(
    parts: Sequence[Part], context: FabricationContext
) -> list[Joint]:
    x_ribs = [part for part in parts if part.metadata.get("rib_set") == "x"]
    y_ribs = [part for part in parts if part.metadata.get("rib_set") == "y"]
    x_ribs.sort(
        key=lambda part: (float(part.metadata["station_offset_mm"]), part.part_id)
    )
    y_ribs.sort(
        key=lambda part: (float(part.metadata["station_offset_mm"]), part.part_id)
    )

    joints: list[Joint] = []
    thickness = float(context.material_thickness_mm)
    for x_rib in x_ribs:
        y_station = float(x_rib.metadata["station_offset_mm"])
        for y_rib in y_ribs:
            x_station = float(y_rib.metadata["station_offset_mm"])
            if not _stations_intersect_ribs(x_rib, y_rib, x_station, y_station):
                continue

            z_min = max(float(x_rib.aabb_min[2]), float(y_rib.aabb_min[2]))
            z_max = min(float(x_rib.aabb_max[2]), float(y_rib.aabb_max[2]))
            overlap_height = max(0.0, z_max - z_min)
            if overlap_height <= _EPS:
                continue

            slot_depth = overlap_height * 0.5
            joint_index = len(joints)
            joints.append(
                Joint(
                    joint_id=f"waffle_slot_{joint_index:04d}",
                    strategy_id=WaffleRibsStrategy.strategy_id,
                    part_ids=[x_rib.part_id, y_rib.part_id],
                    kind="half_lap_slot",
                    metadata={
                        "x_rib_id": x_rib.part_id,
                        "y_rib_id": y_rib.part_id,
                        "intersection_3d": [
                            round(float(x_station), 6),
                            round(float(y_station), 6),
                            round(float((z_min + z_max) * 0.5), 6),
                        ],
                        "slot_width_mm": round(thickness, 6),
                        "slot_depth_mm": round(float(slot_depth), 6),
                        "slot_overlap_height_mm": round(float(overlap_height), 6),
                        "fit": "prototype_half_lap_centered",
                    },
                )
            )
    return joints


def _stations_intersect_ribs(
    x_rib: Part, y_rib: Part, x_station: float, y_station: float
) -> bool:
    return (
        float(x_rib.aabb_min[0]) - _EPS
        <= float(x_station)
        <= float(x_rib.aabb_max[0]) + _EPS
        and float(y_rib.aabb_min[1]) - _EPS
        <= float(y_station)
        <= float(y_rib.aabb_max[1]) + _EPS
    )


def _annotate_slot_counts(parts: Sequence[Part], joints: Sequence[Joint]) -> None:
    counts: Dict[str, int] = {part.part_id: 0 for part in parts}
    for joint in joints:
        for part_id in joint.part_ids:
            counts[part_id] = counts.get(part_id, 0) + 1

    for part in parts:
        part.metadata["slot_count_estimate"] = int(counts.get(part.part_id, 0))


def _build_operations(
    parts: Sequence[Part],
    joints: Sequence[Joint],
    specs: Sequence[_RibSetSpec],
    context: FabricationContext,
    allocation_debug: dict[str, object],
) -> list[Operation]:
    part_ids = [part.part_id for part in parts]
    slot_preview = [
        {
            "joint_id": joint.joint_id,
            "part_ids": list(joint.part_ids),
            "intersection_3d": joint.metadata["intersection_3d"],
            "slot_width_mm": joint.metadata["slot_width_mm"],
            "slot_depth_mm": joint.metadata["slot_depth_mm"],
        }
        for joint in joints[:100]
    ]
    return [
        Operation(
            operation_id="waffle_ribs_sample_bands",
            strategy_id=WaffleRibsStrategy.strategy_id,
            kind="coarse_cross_section_banding",
            part_ids=part_ids,
            metadata={
                "band_source": "triangle_aabb_overlap",
                "vertical_axis": _AXIS_NAMES[_VERTICAL_AXIS],
                "target_spacing_mm": allocation_debug["target_spacing_mm"],
                "rib_sets": [
                    {
                        "rib_set": spec.rib_set,
                        "run_axis": spec.run_axis_name,
                        "station_axis": spec.station_axis_name,
                        "requested_count": int(spec.count),
                    }
                    for spec in specs
                ],
            },
        ),
        Operation(
            operation_id="waffle_ribs_cut_profiles",
            strategy_id=WaffleRibsStrategy.strategy_id,
            kind="rib_profile_cutting",
            part_ids=part_ids,
            metadata={
                "material_key": context.material_key,
                "material_name": context.material_name,
                "material_thickness_mm": float(context.material_thickness_mm),
                "rib_count": int(len(parts)),
                "prototype_profile": "rectangular_aabb_envelope",
            },
        ),
        Operation(
            operation_id="waffle_ribs_cut_slots",
            strategy_id=WaffleRibsStrategy.strategy_id,
            kind="half_lap_slot_cutting",
            part_ids=part_ids,
            metadata={
                "slot_count": int(len(joints)),
                "slot_width_mm": float(context.material_thickness_mm),
                "slot_depth_rule": "half_of_vertical_overlap",
                "slot_layout_preview": slot_preview,
                "slot_layout_truncated": len(joints) > len(slot_preview),
            },
        ),
        Operation(
            operation_id="waffle_ribs_assemble_grid",
            strategy_id=WaffleRibsStrategy.strategy_id,
            kind="waffle_grid_assembly",
            part_ids=part_ids,
            metadata={
                "assembly_method": "perpendicular_half_lap_interlock",
                "joint_count": int(len(joints)),
                "recommended_order": [
                    "lay_out_x_running_ribs",
                    "drop_in_y_running_ribs",
                    "square_and_clamp_grid",
                ],
            },
        ),
    ]


def _material_estimate(
    parts: Sequence[Part], context: FabricationContext
) -> Dict[str, object]:
    total_area = float(sum(part.area_mm2 for part in parts))
    total_volume = float(sum(part.volume_mm3 for part in parts))
    return {
        "material_key": context.material_key,
        "material_name": context.material_name,
        "part_count": int(len(parts)),
        "total_sheet_area_mm2": total_area,
        "total_volume_mm3": total_volume,
        "total_volume_cm3": total_volume / 1000.0,
        "estimated_mass_kg": _mass_kg(total_volume, context),
    }


def _score_components(
    parts: Sequence[Part],
    joints: Sequence[Joint],
    context: FabricationContext,
    warnings: Sequence[str],
) -> Dict[str, float]:
    part_count = len(parts)
    total_volume = float(sum(part.volume_mm3 for part in parts))
    mesh_volume = max(float(context.mesh_volume_mm3), 1.0)
    volume_ratio = total_volume / mesh_volume
    x_count = sum(1 for part in parts if part.metadata.get("rib_set") == "x")
    y_count = sum(1 for part in parts if part.metadata.get("rib_set") == "y")
    full_grid_slots = x_count * y_count
    slot_fill = len(joints) / max(float(full_grid_slots), 1.0)

    return {
        "fidelity": _clamp01(
            0.30 + 0.28 * min(1.0, part_count / 12.0) + 0.22 * min(1.0, slot_fill)
        ),
        "material_efficiency": _clamp01(1.0 / (1.0 + max(0.0, volume_ratio - 0.35))),
        "assembly_simplicity": _clamp01(
            1.0 / (1.0 + part_count / 18.0 + len(joints) / 72.0)
        ),
        "strength_proxy": _clamp01(
            0.45
            + (0.15 if x_count > 0 and y_count > 0 else 0.0)
            + 0.25 * min(1.0, len(joints) / 12.0)
        ),
        "part_count": _clamp01(1.0 / (1.0 + part_count / 36.0)),
        "risk": 0.70 if not warnings else 0.55,
    }


def _mass_kg(volume_mm3: float, context: FabricationContext) -> float:
    return float(volume_mm3) / 1_000_000_000.0 * float(context.material_density_kg_m3)


def _write_artifacts(
    plan: FabricationPlan, artifacts_dir: Path | None
) -> dict[str, str]:
    if artifacts_dir is None:
        return {}

    strategy_dir = Path(artifacts_dir) / WaffleRibsStrategy.strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)
    debug_path = strategy_dir / "waffle_ribs_debug.json"
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy_id": plan.strategy_id,
                "status": plan.status,
                "warnings": plan.warnings,
                "scores": plan.scores,
                "debug": plan.debug,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return {"debug_json": str(debug_path)}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _vec3(values: Sequence[float]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))
