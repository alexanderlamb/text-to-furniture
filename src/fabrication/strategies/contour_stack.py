"""Contour-stack fabrication strategy.

This prototype samples a normalized mesh with parallel section planes and emits
one cuttable layer part per non-empty section. It keeps the output contract small:
parts carry outline summaries, world AABBs, area/volume estimates, and enough
metadata to debug the selected slicing axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import polygonize
import trimesh

from fabrication.context import FabricationContext
from fabrication.contracts import FabricationPlan, Joint, Operation, Part, Vec3
from fabrication.scoring import add_basic_score

_EPS = 1e-7


@dataclass(frozen=True)
class _AxisCandidate:
    name: str
    vector: np.ndarray
    min_offset: float
    max_offset: float
    extent_mm: float
    layer_count: int
    effective_spacing_mm: float
    spacing_quality: float
    selection_score: float

    def to_debug(self) -> dict[str, object]:
        return {
            "axis_name": self.name,
            "axis_vector": _vec3(self.vector),
            "min_offset_mm": round(float(self.min_offset), 6),
            "max_offset_mm": round(float(self.max_offset), 6),
            "extent_mm": round(float(self.extent_mm), 6),
            "layer_count": int(self.layer_count),
            "effective_spacing_mm": round(float(self.effective_spacing_mm), 6),
            "spacing_quality": round(float(self.spacing_quality), 6),
            "selection_score": round(float(self.selection_score), 6),
        }


class ContourStackStrategy:
    """Approximate a mesh as stacked 2D contour layers."""

    strategy_id = "contour_stack"

    def generate(
        self, context: FabricationContext, artifacts_dir: Path | None = None
    ) -> FabricationPlan:
        warnings: list[str] = []
        candidates = _axis_candidates(context)
        if not candidates:
            plan = FabricationPlan(
                strategy_id=self.strategy_id,
                status="error",
                warnings=["Mesh has no usable slicing extent."],
                scores={"overall": 0.0},
                debug={"axis_candidates": []},
            )
            return plan

        selected = candidates[0]
        if selected.effective_spacing_mm > context.material_thickness_mm * 1.25:
            warnings.append(
                "Layer spacing was increased above material thickness to respect "
                f"part_budget_max={context.config.part_budget_max}."
            )

        parts, slice_debug, slice_warnings = _slice_mesh(context, selected)
        warnings.extend(slice_warnings)

        joints = _build_joints(parts, self.strategy_id)
        operations = _build_operations(parts, selected, self.strategy_id, context)
        status = "ok" if parts else "error"
        if not parts:
            warnings.append("No non-empty contour sections were generated.")

        total_part_volume = sum(float(part.volume_mm3) for part in parts)
        volume_ratio = _safe_ratio(total_part_volume, max(context.mesh_volume_mm3, 1.0))
        spacing_quality = selected.spacing_quality
        non_empty_ratio = _safe_ratio(len(parts), max(selected.layer_count, 1))

        plan = FabricationPlan(
            strategy_id=self.strategy_id,
            status=status,
            parts=parts,
            joints=joints,
            operations=operations,
            scores={
                "fidelity": _clamp01(0.35 + 0.50 * spacing_quality * non_empty_ratio),
                "material_efficiency": _clamp01(
                    min(volume_ratio, 1.0 / max(volume_ratio, _EPS))
                ),
                "assembly_simplicity": _clamp01(1.0 / (1.0 + len(parts) / 30.0)),
                "strength_proxy": _clamp01(0.45 + 0.25 * min(1.0, len(parts) / 24.0)),
                "risk": 0.70 if status == "ok" and not warnings else 0.45,
            },
            warnings=warnings,
            artifacts={},
            debug={
                "selected_axis": selected.to_debug(),
                "axis_candidates": [candidate.to_debug() for candidate in candidates],
                "requested_slice_count": int(selected.layer_count),
                "non_empty_slice_count": len(parts),
                "empty_slice_count": int(selected.layer_count - len(parts)),
                "total_contour_area_mm2": round(
                    sum(float(part.area_mm2) for part in parts), 6
                ),
                "estimated_material_volume_mm3": round(total_part_volume, 6),
                "mesh_volume_mm3": round(float(context.mesh_volume_mm3), 6),
                "volume_ratio": round(float(volume_ratio), 6),
                "slices": slice_debug,
            },
        )

        strategy_artifacts = _write_artifacts(plan, artifacts_dir)
        if strategy_artifacts:
            plan.artifacts = strategy_artifacts

        add_basic_score(plan, context)
        return plan


def _axis_candidates(context: FabricationContext) -> list[_AxisCandidate]:
    mesh = context.mesh
    budget = max(1, int(context.config.part_budget_max))
    thickness = max(float(context.material_thickness_mm), _EPS)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return []

    candidates: list[_AxisCandidate] = []
    for name, axis in _candidate_axis_vectors(mesh):
        projections = vertices @ axis
        min_offset = float(np.min(projections))
        max_offset = float(np.max(projections))
        extent = max_offset - min_offset
        if extent <= _EPS:
            continue

        material_layer_count = max(1, int(math.ceil(extent / thickness)))
        layer_count = min(material_layer_count, budget)
        effective_spacing = extent / max(layer_count, 1)
        spacing_quality = _clamp01(thickness / max(effective_spacing, thickness))
        enough_samples = _clamp01(layer_count / max(6.0, min(float(budget), 24.0)))
        assembly_simplicity = 1.0 / (1.0 + layer_count / max(float(budget), 1.0))
        world_axis_bias = {"z": 0.03, "y": 0.02, "x": 0.01}.get(name, 0.0)
        selection_score = (
            0.62 * spacing_quality
            + 0.18 * enough_samples
            + 0.17 * assembly_simplicity
            + world_axis_bias
        )
        candidates.append(
            _AxisCandidate(
                name=name,
                vector=axis,
                min_offset=min_offset,
                max_offset=max_offset,
                extent_mm=extent,
                layer_count=layer_count,
                effective_spacing_mm=effective_spacing,
                spacing_quality=spacing_quality,
                selection_score=selection_score,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -round(float(candidate.selection_score), 12),
            int(candidate.layer_count),
            candidate.name,
        )
    )
    return candidates


def _candidate_axis_vectors(mesh: trimesh.Trimesh) -> list[tuple[str, np.ndarray]]:
    axes: list[tuple[str, np.ndarray]] = [
        ("x", np.asarray([1.0, 0.0, 0.0], dtype=float)),
        ("y", np.asarray([0.0, 1.0, 0.0], dtype=float)),
        ("z", np.asarray([0.0, 0.0, 1.0], dtype=float)),
    ]

    try:
        principal = np.asarray(mesh.principal_inertia_vectors, dtype=float)
    except Exception:
        principal = np.empty((0, 3), dtype=float)

    if principal.shape == (3, 3):
        for index, vector in enumerate(principal):
            unit = _unit(vector)
            if unit is None:
                continue
            unit = _canonical_axis(unit)
            if _is_duplicate_axis(unit, (axis for _, axis in axes)):
                continue
            axes.append((f"principal_{index}", unit))

    return axes


def _slice_mesh(
    context: FabricationContext, axis: _AxisCandidate
) -> tuple[list[Part], list[dict[str, object]], list[str]]:
    parts: list[Part] = []
    debug: list[dict[str, object]] = []
    warnings: list[str] = []
    half_thickness = float(context.material_thickness_mm) / 2.0

    for layer_index in range(axis.layer_count):
        offset = axis.min_offset + (layer_index + 0.5) * axis.effective_spacing_mm
        plane_origin = axis.vector * offset
        section = None
        aabb_min: Vec3 | None = None
        aabb_max: Vec3 | None = None
        polygons: list[Polygon] = []
        section_warning = None
        try:
            section = context.mesh.section(
                plane_origin=plane_origin,
                plane_normal=axis.vector,
            )
            polygons = _section_polygons(section)
        except Exception as exc:
            try:
                polygons, aabb_min, aabb_max = _manual_section_polygons_and_aabb(
                    mesh=context.mesh,
                    axis=axis.vector,
                    offset=offset,
                    half_thickness=half_thickness,
                )
                section_warning = (
                    "trimesh section fallback used: " f"{type(exc).__name__}: {exc}"
                )
            except Exception as fallback_exc:
                section_warning = (
                    f"slice {layer_index} failed: {type(exc).__name__}: {exc}; "
                    f"fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                )
                warnings.append(section_warning)

        area_mm2 = float(sum(poly.area for poly in polygons))
        if area_mm2 <= _EPS:
            debug.append(
                {
                    "layer_index": layer_index,
                    "offset_mm": round(float(offset), 6),
                    "status": "empty" if section_warning is None else "failed",
                    "warning": section_warning,
                }
            )
            continue

        if aabb_min is None or aabb_max is None:
            aabb_min, aabb_max = _section_aabb(section, axis.vector, half_thickness)
        outline_summaries = _outline_summaries(polygons)
        part = Part(
            part_id=f"contour_stack_{axis.name}_{len(parts):03d}",
            strategy_id=ContourStackStrategy.strategy_id,
            kind="contour_layer",
            quantity=1,
            material_thickness_mm=float(context.material_thickness_mm),
            area_mm2=area_mm2,
            volume_mm3=area_mm2 * float(context.material_thickness_mm),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            metadata={
                "axis_name": axis.name,
                "axis_vector": _vec3(axis.vector),
                "layer_index": layer_index,
                "stack_index": len(parts),
                "slice_offset_mm": round(float(offset), 6),
                "effective_spacing_mm": round(float(axis.effective_spacing_mm), 6),
                "target_spacing_mm": round(float(context.material_thickness_mm), 6),
                "outline_count": len(outline_summaries),
                "outline_summaries": outline_summaries,
                "area_estimate_method": "trimesh_section_to_shapely_polygon",
            },
        )
        parts.append(part)
        debug.append(
            {
                "layer_index": layer_index,
                "part_id": part.part_id,
                "offset_mm": round(float(offset), 6),
                "status": "ok",
                "area_mm2": round(area_mm2, 6),
                "outline_count": len(outline_summaries),
            }
        )

    return parts, debug, warnings


def _section_polygons(section) -> list[Polygon]:
    if section is None:
        return []
    if len(getattr(section, "entities", [])) == 0:
        return []

    path_2d = _section_to_2d_path(section)
    raw_polygons: list[object] = []

    for attr in ("polygons_full", "polygons_closed"):
        try:
            raw = getattr(path_2d, attr)
            raw_polygons = list(raw)
            if raw_polygons:
                break
        except Exception:
            raw_polygons = []

    if not raw_polygons:
        try:
            raw_polygons = [Polygon(points) for points in path_2d.discrete]
        except Exception:
            raw_polygons = []

    polygons: list[Polygon] = []
    for raw in raw_polygons:
        polygons.extend(_clean_polygon(raw))

    polygons.sort(
        key=lambda polygon: (
            -round(float(polygon.area), 9),
            tuple(round(float(v), 9) for v in polygon.bounds),
        )
    )
    return polygons


def _section_to_2d_path(section):
    if hasattr(section, "to_2D"):
        result = section.to_2D()
    else:
        result = section.to_planar()
    if isinstance(result, tuple):
        return result[0]
    return result


def _clean_polygon(raw: object) -> list[Polygon]:
    if not isinstance(raw, (Polygon, MultiPolygon)):
        try:
            raw = Polygon(raw)  # type: ignore[arg-type]
        except Exception:
            return []

    if raw.is_empty:
        return []
    if not raw.is_valid:
        raw = raw.buffer(0)
    if raw.is_empty:
        return []

    if isinstance(raw, MultiPolygon):
        geoms: Iterable[Polygon] = raw.geoms
    else:
        geoms = [raw]

    polygons: list[Polygon] = []
    for polygon in geoms:
        if polygon.is_empty or polygon.area <= _EPS:
            continue
        polygons.append(polygon)
    return polygons


def _section_aabb(
    section, axis: np.ndarray, half_thickness: float
) -> tuple[Vec3, Vec3]:
    vertices = np.asarray(getattr(section, "vertices", []), dtype=float)
    if vertices.size == 0:
        point = np.zeros(3, dtype=float)
        return _vec3(point), _vec3(point)

    extrusion = axis * half_thickness
    swept = np.vstack((vertices - extrusion, vertices + extrusion))
    return _vec3(np.min(swept, axis=0)), _vec3(np.max(swept, axis=0))


def _manual_section_polygons_and_aabb(
    *,
    mesh: trimesh.Trimesh,
    axis: np.ndarray,
    offset: float,
    half_thickness: float,
) -> tuple[list[Polygon], Vec3, Vec3]:
    """Intersect triangles with a plane without trimesh path graph helpers."""

    normal = np.asarray(axis, dtype=float)
    basis_u, basis_v = _basis_from_axis(normal)
    origin = normal * float(offset)
    lines: list[LineString] = []
    world_points: list[np.ndarray] = []

    for face in np.asarray(mesh.faces, dtype=int):
        triangle = np.asarray(mesh.vertices[face], dtype=float)
        signed = triangle @ normal - float(offset)
        points: list[np.ndarray] = []

        for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
            d0 = float(signed[edge_start])
            d1 = float(signed[edge_end])
            p0 = triangle[edge_start]
            p1 = triangle[edge_end]

            if abs(d0) <= _EPS:
                points.append(p0)
            if d0 * d1 < -_EPS:
                t = d0 / (d0 - d1)
                points.append(p0 + t * (p1 - p0))
            if abs(d1) <= _EPS:
                points.append(p1)

        unique = _unique_points(points)
        if len(unique) < 2:
            continue
        if len(unique) > 2:
            unique = _farthest_pair(unique)

        p0, p1 = unique[0], unique[1]
        if np.linalg.norm(p1 - p0) <= _EPS:
            continue
        world_points.extend([p0, p1])
        lines.append(
            LineString(
                [
                    _project_manual_point(p0, origin, basis_u, basis_v),
                    _project_manual_point(p1, origin, basis_u, basis_v),
                ]
            )
        )

    polygons = _polygons_from_lines(lines)
    if not polygons and world_points:
        projected = [
            _project_manual_point(point, origin, basis_u, basis_v)
            for point in world_points
        ]
        hull = MultiPoint(projected).convex_hull
        polygons = _clean_polygon(hull)

    if world_points:
        points_3d = np.asarray(world_points, dtype=float)
        swept = np.vstack(
            (points_3d - normal * half_thickness, points_3d + normal * half_thickness)
        )
        aabb_min = _vec3(np.min(swept, axis=0))
        aabb_max = _vec3(np.max(swept, axis=0))
    else:
        point = normal * float(offset)
        aabb_min = _vec3(point)
        aabb_max = _vec3(point)
    return polygons, aabb_min, aabb_max


def _basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, ref))) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
    basis_u = np.cross(ref, axis)
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    basis_v = basis_v / np.linalg.norm(basis_v)
    return basis_u, basis_v


def _project_manual_point(
    point: np.ndarray, origin: np.ndarray, basis_u: np.ndarray, basis_v: np.ndarray
) -> tuple[float, float]:
    delta = point - origin
    return (
        round(float(np.dot(delta, basis_u)), 9),
        round(float(np.dot(delta, basis_v)), 9),
    )


def _unique_points(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for point in points:
        key = tuple(round(float(value), 9) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.asarray(point, dtype=float))
    return unique


def _farthest_pair(points: Sequence[np.ndarray]) -> list[np.ndarray]:
    best = (points[0], points[1])
    best_dist = -1.0
    for i, point_a in enumerate(points):
        for point_b in points[i + 1 :]:
            dist = float(np.linalg.norm(point_b - point_a))
            if dist > best_dist:
                best = (point_a, point_b)
                best_dist = dist
    return [best[0], best[1]]


def _polygons_from_lines(lines: Sequence[LineString]) -> list[Polygon]:
    polygons: list[Polygon] = []
    for polygon in polygonize(lines):
        polygons.extend(_clean_polygon(polygon))
    polygons.sort(
        key=lambda polygon: (
            -round(float(polygon.area), 9),
            tuple(round(float(v), 9) for v in polygon.bounds),
        )
    )
    return polygons


def _outline_summaries(polygons: Sequence[Polygon]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for index, polygon in enumerate(polygons):
        minx, miny, maxx, maxy = polygon.bounds
        exterior = list(polygon.exterior.coords)
        summaries.append(
            {
                "outline_index": index,
                "area_mm2": round(float(polygon.area), 6),
                "bounds_2d": [
                    round(float(minx), 6),
                    round(float(miny), 6),
                    round(float(maxx), 6),
                    round(float(maxy), 6),
                ],
                "vertex_count": max(0, len(exterior) - 1),
                "hole_count": len(polygon.interiors),
            }
        )
    return summaries


def _build_joints(parts: Sequence[Part], strategy_id: str) -> list[Joint]:
    joints: list[Joint] = []
    for index, (lower, upper) in enumerate(zip(parts, parts[1:])):
        joints.append(
            Joint(
                joint_id=f"contour_stack_bond_{index:03d}",
                strategy_id=strategy_id,
                part_ids=[lower.part_id, upper.part_id],
                kind="laminated_face_bond",
                metadata={
                    "lower_stack_index": lower.metadata.get("stack_index"),
                    "upper_stack_index": upper.metadata.get("stack_index"),
                    "axis_name": lower.metadata.get("axis_name"),
                },
            )
        )
    return joints


def _build_operations(
    parts: Sequence[Part],
    axis: _AxisCandidate,
    strategy_id: str,
    context: FabricationContext,
) -> list[Operation]:
    part_ids = [part.part_id for part in parts]
    return [
        Operation(
            operation_id="contour_stack_section_mesh",
            strategy_id=strategy_id,
            kind="mesh_sectioning",
            part_ids=part_ids,
            metadata={
                "axis_name": axis.name,
                "axis_vector": _vec3(axis.vector),
                "slice_count": axis.layer_count,
                "effective_spacing_mm": round(float(axis.effective_spacing_mm), 6),
            },
        ),
        Operation(
            operation_id="contour_stack_cut_layers",
            strategy_id=strategy_id,
            kind="profile_cutting",
            part_ids=part_ids,
            metadata={
                "material_key": context.material_key,
                "material_thickness_mm": float(context.material_thickness_mm),
            },
        ),
        Operation(
            operation_id="contour_stack_laminate_layers",
            strategy_id=strategy_id,
            kind="stack_assembly",
            part_ids=part_ids,
            metadata={"joint_count": max(0, len(parts) - 1)},
        ),
    ]


def _write_artifacts(
    plan: FabricationPlan, artifacts_dir: Path | None
) -> dict[str, str]:
    if artifacts_dir is None:
        return {}

    strategy_dir = Path(artifacts_dir) / ContourStackStrategy.strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)
    debug_path = strategy_dir / "contour_stack_debug.json"
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strategy_id": plan.strategy_id,
                "status": plan.status,
                "warnings": plan.warnings,
                "debug": plan.debug,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return {"debug_json": str(debug_path)}


def _unit(vector: Sequence[float]) -> np.ndarray | None:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS:
        return None
    return arr / norm


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    return axis


def _is_duplicate_axis(axis: np.ndarray, existing_axes: Iterable[np.ndarray]) -> bool:
    return any(abs(float(np.dot(axis, other))) > 0.999 for other in existing_axes)


def _vec3(values: Sequence[float]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= _EPS:
        return 0.0
    return float(numerator) / float(denominator)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
