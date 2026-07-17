"""Mesh regioning helpers for hybrid fabrication composition.

The hybrid compositor needs coarse, deterministic chunks of the source mesh
before it can choose which fabrication strategy should own each chunk. This
module intentionally stays prototype-simple: it first looks for meaningful
connected components and otherwise slices the mesh AABB into broad axis-aligned
bands, annotating each region with lightweight occupancy signals.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fabrication.context import FabricationContext
from fabrication.contracts import HybridRegion, Vec3

_AXIS_NAMES = ("x", "y", "z")
_EPSILON = 1e-9


def regionize_mesh(
    context: FabricationContext, max_regions: int = 8
) -> List[HybridRegion]:
    """Return deterministic coarse regions for a mesh context.

    Multi-component meshes become component regions when there are at least two
    meaningful connected pieces. Single-component meshes fall back to observed
    longest-axis bands. Each region carries debug metadata describing the split,
    extents, and occupancy proxies used for downstream hybrid strategy
    assignment.
    """

    if max_regions < 1:
        raise ValueError("max_regions must be at least 1")

    bounds = np.asarray(context.mesh.bounds, dtype=float)
    if bounds.shape != (2, 3):
        return []

    mesh_min = bounds[0]
    mesh_max = bounds[1]
    mesh_extents = np.maximum(mesh_max - mesh_min, 0.0)
    if not np.all(np.isfinite(mesh_extents)) or float(np.max(mesh_extents)) <= _EPSILON:
        return []

    component_regions = _component_regions(
        context=context,
        mesh_min=mesh_min,
        mesh_max=mesh_max,
        mesh_extents=mesh_extents,
        max_regions=max_regions,
    )
    if component_regions:
        return component_regions

    split_axis = _choose_split_axis(mesh_extents)
    band_count = _choose_band_count(context, mesh_extents, max_regions)
    coarse_bounds = _axis_band_bounds(mesh_min, mesh_max, split_axis, band_count)
    candidate_bounds = []
    raw_stats = []
    for coarse_min, coarse_max in coarse_bounds:
        observed_points = _observed_points_in_aabb(context, coarse_min, coarse_max)
        aabb_min, aabb_max, bounds_metadata = _tighten_transverse_bounds(
            context=context,
            mesh_min=mesh_min,
            mesh_max=mesh_max,
            coarse_min=coarse_min,
            coarse_max=coarse_max,
            split_axis=split_axis,
            observed_points=observed_points,
        )
        stats = _region_raw_stats(
            context=context,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        stats.update(bounds_metadata)
        candidate_bounds.append((aabb_min, aabb_max))
        raw_stats.append(stats)

    raw_volume_total = sum(stats["raw_volume_mm3"] for stats in raw_stats)
    mesh_volume = max(float(context.mesh_volume_mm3), 0.0)
    if raw_volume_total > _EPSILON and mesh_volume > _EPSILON:
        volume_scale = mesh_volume / raw_volume_total
    else:
        volume_scale = 1.0

    regions: List[HybridRegion] = []
    for index, ((aabb_min, aabb_max), stats) in enumerate(
        zip(candidate_bounds, raw_stats)
    ):
        region_extents = np.maximum(aabb_max - aabb_min, 0.0)
        volume_mm3 = float(max(stats["raw_volume_mm3"] * volume_scale, 0.0))
        surface_area_mm2 = _estimate_region_surface_area(context, stats, raw_stats)
        kind, reasons = _classify_region(
            context=context,
            mesh_extents=mesh_extents,
            region_extents=region_extents,
            fill_ratio=float(stats["fill_ratio"]),
        )

        regions.append(
            HybridRegion(
                region_id=f"region_{_AXIS_NAMES[split_axis]}_band_{index:02d}",
                kind=kind,
                aabb_min=_to_vec3(aabb_min),
                aabb_max=_to_vec3(aabb_max),
                volume_mm3=volume_mm3,
                surface_area_mm2=surface_area_mm2,
                metadata={
                    "algorithm": "aabb_observed_axis_bands_v1",
                    "band_index": int(index),
                    "band_count": int(band_count),
                    "split_axis": _AXIS_NAMES[split_axis],
                    "split_axis_index": int(split_axis),
                    "classification_reasons": reasons,
                    "bounds_method": stats["bounds_method"],
                    "coarse_aabb_min": stats["coarse_aabb_min"],
                    "coarse_aabb_max": stats["coarse_aabb_max"],
                    "bounds_padding_mm": float(stats["bounds_padding_mm"]),
                    "observed_point_count": int(stats["observed_point_count"]),
                    "transverse_shrink_ratio": float(stats["transverse_shrink_ratio"]),
                    "debug": {
                        "region_extents_mm": _to_float_list(region_extents),
                        "region_centroid_mm": _to_float_list(
                            (aabb_min + aabb_max) / 2.0
                        ),
                        "mesh_extents_mm": _to_float_list(mesh_extents),
                        "mesh_watertight": bool(context.mesh.is_watertight),
                    },
                    "occupancy_proxy": {
                        "method": stats["method"],
                        "fill_ratio": float(stats["fill_ratio"]),
                        "raw_volume_mm3": float(stats["raw_volume_mm3"]),
                        "aabb_volume_mm3": float(stats["aabb_volume_mm3"]),
                        "sample_inside_count": int(stats["sample_inside_count"]),
                        "sample_count": int(stats["sample_count"]),
                        "vertex_count": int(stats["vertex_count"]),
                        "face_centroid_count": int(stats["face_centroid_count"]),
                        "face_area_mm2": float(stats["face_area_mm2"]),
                    },
                },
            )
        )

    return regions


def region_payload(region: HybridRegion) -> Dict[str, object]:
    """Serialize one region with derived values useful for debug artifacts."""

    aabb_min = np.asarray(region.aabb_min, dtype=float)
    aabb_max = np.asarray(region.aabb_max, dtype=float)
    extents = np.maximum(aabb_max - aabb_min, 0.0)
    return {
        "region_id": region.region_id,
        "kind": region.kind,
        "aabb_min": _to_float_list(aabb_min),
        "aabb_max": _to_float_list(aabb_max),
        "extents_mm": _to_float_list(extents),
        "centroid_mm": _to_float_list((aabb_min + aabb_max) / 2.0),
        "volume_mm3": float(region.volume_mm3),
        "surface_area_mm2": float(region.surface_area_mm2),
        "metadata": region.metadata,
    }


def regioning_debug_payload(
    context: FabricationContext, regions: Sequence[HybridRegion]
) -> Dict[str, object]:
    """Return a compact debug payload for regioning artifacts or plan metadata."""

    return {
        "algorithm": _regions_algorithm(regions),
        "region_count": int(len(regions)),
        "mesh": {
            "bounds_mm": _to_float_list(context.mesh_bounds_mm),
            "volume_mm3": float(context.mesh_volume_mm3),
            "watertight": bool(context.mesh.is_watertight),
            "source_mesh_path": str(context.source_mesh_path),
        },
        "regions": [region_payload(region) for region in regions],
    }


def _regions_algorithm(regions: Sequence[HybridRegion]) -> str:
    algorithms = {
        str(region.metadata.get("algorithm", "unknown")) for region in regions
    }
    if not algorithms:
        return "none"
    if len(algorithms) == 1:
        return next(iter(algorithms))
    return "mixed_regioning"


def _component_regions(
    *,
    context: FabricationContext,
    mesh_min: np.ndarray,
    mesh_max: np.ndarray,
    mesh_extents: np.ndarray,
    max_regions: int,
) -> List[HybridRegion]:
    try:
        split_components = list(context.mesh.split(only_watertight=False))
    except Exception:
        return []

    components = [
        (index, component)
        for index, component in enumerate(split_components)
        if _is_meaningful_component(component)
    ]
    if len(components) < 2:
        return []

    split_axis = _choose_split_axis(mesh_extents)
    groups = _component_groups(
        components, split_axis=split_axis, max_regions=max_regions
    )
    regions: List[HybridRegion] = []
    total_component_count = len(components)
    for group_index, group in enumerate(groups):
        if not group:
            continue
        component_indices = [int(index) for index, _component in group]
        component_bounds = [
            np.asarray(component.bounds, dtype=float) for _index, component in group
        ]
        aabb_min = np.min([bounds[0] for bounds in component_bounds], axis=0)
        aabb_max = np.max([bounds[1] for bounds in component_bounds], axis=0)
        padding = max(float(context.config.min_feature_mm), _EPSILON)
        aabb_min, aabb_max = _pad_bounds(
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            mesh_min=mesh_min,
            mesh_max=mesh_max,
            padding=padding,
        )
        stats = _region_raw_stats(
            context=context,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
        )
        region_extents = np.maximum(aabb_max - aabb_min, 0.0)
        aabb_volume = float(np.prod(region_extents))
        component_volume = sum(
            _component_volume(component) for _index, component in group
        )
        component_area = sum(
            float(getattr(component, "area", 0.0)) for _index, component in group
        )
        component_face_count = sum(
            int(len(component.faces)) for _index, component in group
        )
        component_fill = (
            float(np.clip(component_volume / aabb_volume, 0.0, 1.0))
            if aabb_volume > _EPSILON and component_volume > _EPSILON
            else float(stats["fill_ratio"])
        )
        kind, reasons = _classify_region(
            context=context,
            mesh_extents=region_extents,
            region_extents=region_extents,
            fill_ratio=component_fill,
            global_fill=component_fill,
        )
        raw_volume = (
            component_volume
            if component_volume > _EPSILON
            else float(stats["raw_volume_mm3"])
        )
        region_id = (
            f"region_component_{group_index:02d}"
            if len(group) == 1
            else f"region_component_group_{group_index:02d}"
        )
        regions.append(
            HybridRegion(
                region_id=region_id,
                kind=kind,
                aabb_min=_to_vec3(aabb_min),
                aabb_max=_to_vec3(aabb_max),
                volume_mm3=float(max(raw_volume, 0.0)),
                surface_area_mm2=float(max(component_area, stats["face_area_mm2"])),
                metadata={
                    "algorithm": "connected_component_regions_v0",
                    "component_indices": component_indices,
                    "component_count": int(len(group)),
                    "total_component_count": int(total_component_count),
                    "component_face_count": int(component_face_count),
                    "component_area_mm2": float(component_area),
                    "component_volume_mm3": float(component_volume),
                    "group_index": int(group_index),
                    "group_count": int(len(groups)),
                    "split_axis": _AXIS_NAMES[split_axis],
                    "split_axis_index": int(split_axis),
                    "classification_reasons": reasons,
                    "bounds_method": "connected_component_aabb",
                    "bounds_padding_mm": float(padding),
                    "observed_point_count": int(stats["vertex_count"])
                    + int(stats["face_centroid_count"]),
                    "transverse_shrink_ratio": 1.0,
                    "debug": {
                        "region_extents_mm": _to_float_list(region_extents),
                        "region_centroid_mm": _to_float_list(
                            (aabb_min + aabb_max) / 2.0
                        ),
                        "mesh_extents_mm": _to_float_list(mesh_extents),
                        "mesh_watertight": bool(context.mesh.is_watertight),
                        "component_watertight_all": all(
                            bool(component.is_watertight) for _index, component in group
                        ),
                    },
                    "occupancy_proxy": {
                        "method": "component_aabb_volume_ratio",
                        "fill_ratio": float(component_fill),
                        "raw_volume_mm3": float(raw_volume),
                        "aabb_volume_mm3": float(aabb_volume),
                        "sample_inside_count": int(stats["sample_inside_count"]),
                        "sample_count": int(stats["sample_count"]),
                        "vertex_count": int(stats["vertex_count"]),
                        "face_centroid_count": int(stats["face_centroid_count"]),
                        "face_area_mm2": float(stats["face_area_mm2"]),
                    },
                },
            )
        )

    return regions


def _is_meaningful_component(component) -> bool:
    face_count = int(len(getattr(component, "faces", [])))
    if face_count < 4:
        return False
    area = float(getattr(component, "area", 0.0))
    if area <= _EPSILON:
        return False
    extents = np.asarray(getattr(component, "extents", []), dtype=float)
    return bool(extents.size == 3 and float(np.max(extents)) > _EPSILON)


def _component_groups(
    components: Sequence[Tuple[int, object]], *, split_axis: int, max_regions: int
) -> List[List[Tuple[int, object]]]:
    sorted_components = sorted(
        components,
        key=lambda item: (
            float(np.asarray(item[1].bounding_box.centroid, dtype=float)[split_axis]),
            int(item[0]),
        ),
    )
    if len(sorted_components) <= max_regions:
        return [[item] for item in sorted_components]

    groups: List[List[Tuple[int, object]]] = []
    for chunk in np.array_split(np.arange(len(sorted_components)), max_regions):
        groups.append([sorted_components[int(index)] for index in chunk])
    return groups


def _component_volume(component) -> float:
    try:
        volume = float(abs(component.volume))
    except Exception:
        return 0.0
    return volume if np.isfinite(volume) else 0.0


def _pad_bounds(
    *,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    mesh_min: np.ndarray,
    mesh_max: np.ndarray,
    padding: float,
) -> Tuple[np.ndarray, np.ndarray]:
    padded_min = np.maximum(aabb_min - float(padding), mesh_min)
    padded_max = np.minimum(aabb_max + float(padding), mesh_max)
    return padded_min, padded_max


def _choose_split_axis(mesh_extents: np.ndarray) -> int:
    """Choose the longest axis, relying on numpy's stable first-max tie break."""

    return int(np.argmax(mesh_extents))


def _choose_band_count(
    context: FabricationContext, mesh_extents: np.ndarray, max_regions: int
) -> int:
    longest_extent = float(np.max(mesh_extents))
    if max_regions == 1 or longest_extent <= _EPSILON:
        return 1

    min_band_width = max(
        longest_extent / float(max_regions),
        float(context.material_thickness_mm) * 3.0,
        float(context.config.min_feature_mm) * 3.0,
        _EPSILON,
    )
    width_limited_count = int(np.ceil(longest_extent / min_band_width))
    return max(1, min(int(max_regions), width_limited_count))


def _axis_band_bounds(
    mesh_min: np.ndarray, mesh_max: np.ndarray, split_axis: int, band_count: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cuts = np.linspace(mesh_min[split_axis], mesh_max[split_axis], band_count + 1)
    bounds: List[Tuple[np.ndarray, np.ndarray]] = []
    for index in range(band_count):
        aabb_min = mesh_min.copy()
        aabb_max = mesh_max.copy()
        aabb_min[split_axis] = cuts[index]
        aabb_max[split_axis] = cuts[index + 1]
        bounds.append((aabb_min, aabb_max))
    return bounds


def _observed_points_in_aabb(
    context: FabricationContext, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> np.ndarray:
    """Return deterministic geometry evidence touching an AABB band."""

    point_sets = []
    vertices = np.asarray(context.mesh.vertices, dtype=float)
    if vertices.size:
        point_sets.append(vertices[_points_in_aabb_mask(vertices, aabb_min, aabb_max)])

    face_centroids = np.asarray(context.mesh.triangles_center, dtype=float)
    if face_centroids.size:
        point_sets.append(
            face_centroids[_points_in_aabb_mask(face_centroids, aabb_min, aabb_max)]
        )

    triangles = np.asarray(context.mesh.triangles, dtype=float)
    if triangles.size:
        tri_min = np.min(triangles, axis=1)
        tri_max = np.max(triangles, axis=1)
        overlap = np.all(tri_max >= (aabb_min - _EPSILON), axis=1) & np.all(
            tri_min <= (aabb_max + _EPSILON), axis=1
        )
        if np.any(overlap):
            point_sets.append(triangles[overlap].reshape(-1, 3))

    non_empty = [points for points in point_sets if len(points)]
    if not non_empty:
        return np.empty((0, 3), dtype=float)
    return np.vstack(non_empty)


def _tighten_transverse_bounds(
    *,
    context: FabricationContext,
    mesh_min: np.ndarray,
    mesh_max: np.ndarray,
    coarse_min: np.ndarray,
    coarse_max: np.ndarray,
    split_axis: int,
    observed_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Shrink non-split axes to observed geometry while preserving band seams."""

    aabb_min = coarse_min.copy()
    aabb_max = coarse_max.copy()
    coarse_volume = float(np.prod(np.maximum(coarse_max - coarse_min, 0.0)))
    padding = max(
        float(context.material_thickness_mm),
        float(context.config.min_feature_mm),
        _EPSILON,
    )
    min_extent = max(padding * 2.0, float(context.config.min_feature_mm), _EPSILON)

    if len(observed_points) == 0:
        return (
            aabb_min,
            aabb_max,
            _bounds_metadata(
                method="coarse_aabb_no_observed_geometry",
                coarse_min=coarse_min,
                coarse_max=coarse_max,
                refined_min=aabb_min,
                refined_max=aabb_max,
                padding=padding,
                observed_point_count=0,
                coarse_volume=coarse_volume,
            ),
        )

    observed_min = np.min(observed_points, axis=0)
    observed_max = np.max(observed_points, axis=0)
    for axis in range(3):
        if axis == split_axis:
            continue
        low = max(float(mesh_min[axis]), float(observed_min[axis] - padding))
        high = min(float(mesh_max[axis]), float(observed_max[axis] + padding))
        if high - low < min_extent:
            center = (high + low) * 0.5
            low = max(float(mesh_min[axis]), center - min_extent * 0.5)
            high = min(float(mesh_max[axis]), center + min_extent * 0.5)
        if high > low:
            aabb_min[axis] = low
            aabb_max[axis] = high

    return (
        aabb_min,
        aabb_max,
        _bounds_metadata(
            method="observed_geometry_transverse_aabb",
            coarse_min=coarse_min,
            coarse_max=coarse_max,
            refined_min=aabb_min,
            refined_max=aabb_max,
            padding=padding,
            observed_point_count=len(observed_points),
            coarse_volume=coarse_volume,
        ),
    )


def _bounds_metadata(
    *,
    method: str,
    coarse_min: np.ndarray,
    coarse_max: np.ndarray,
    refined_min: np.ndarray,
    refined_max: np.ndarray,
    padding: float,
    observed_point_count: int,
    coarse_volume: float,
) -> Dict[str, object]:
    refined_volume = float(np.prod(np.maximum(refined_max - refined_min, 0.0)))
    shrink_ratio = (
        float(refined_volume / coarse_volume) if coarse_volume > _EPSILON else 1.0
    )
    return {
        "bounds_method": method,
        "coarse_aabb_min": _to_float_list(coarse_min),
        "coarse_aabb_max": _to_float_list(coarse_max),
        "bounds_padding_mm": float(padding),
        "observed_point_count": int(observed_point_count),
        "transverse_shrink_ratio": float(np.clip(shrink_ratio, 0.0, 1.0)),
    }


def _region_raw_stats(
    *, context: FabricationContext, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> Dict[str, object]:
    aabb_extents = np.maximum(aabb_max - aabb_min, 0.0)
    aabb_volume = float(np.prod(aabb_extents))
    fill_ratio, sample_inside_count, sample_count, method = _sample_fill_ratio(
        context=context, aabb_min=aabb_min, aabb_max=aabb_max
    )
    vertex_count = _count_points_in_aabb(
        np.asarray(context.mesh.vertices, dtype=float), aabb_min, aabb_max
    )
    face_centroids = np.asarray(context.mesh.triangles_center, dtype=float)
    face_centroid_mask = _points_in_aabb_mask(face_centroids, aabb_min, aabb_max)
    face_centroid_count = int(np.count_nonzero(face_centroid_mask))
    face_areas = np.asarray(getattr(context.mesh, "area_faces", []), dtype=float)
    if len(face_areas) == len(face_centroid_mask):
        face_area = float(np.sum(face_areas[face_centroid_mask]))
    else:
        face_area = 0.0

    return {
        "method": method,
        "fill_ratio": float(fill_ratio),
        "raw_volume_mm3": float(aabb_volume * fill_ratio),
        "aabb_volume_mm3": aabb_volume,
        "sample_inside_count": int(sample_inside_count),
        "sample_count": int(sample_count),
        "vertex_count": int(vertex_count),
        "face_centroid_count": int(face_centroid_count),
        "face_area_mm2": face_area,
    }


def _sample_fill_ratio(
    *, context: FabricationContext, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> Tuple[float, int, int, str]:
    points = _interior_sample_points(aabb_min, aabb_max)
    if len(points) == 0:
        return 0.0, 0, 0, "empty_region"

    try:
        contained = np.asarray(context.mesh.contains(points), dtype=bool)
        inside_count = int(np.count_nonzero(contained))
        if len(contained) == len(points):
            return (
                float(inside_count / len(points)),
                inside_count,
                len(points),
                "mesh_contains_grid",
            )
    except Exception:
        pass

    mesh_bounds = np.asarray(context.mesh.bounds, dtype=float)
    mesh_aabb_volume = float(np.prod(np.maximum(mesh_bounds[1] - mesh_bounds[0], 0.0)))
    if mesh_aabb_volume <= _EPSILON:
        fallback_fill = 0.0
    else:
        fallback_fill = float(
            np.clip(context.mesh_volume_mm3 / mesh_aabb_volume, 0.0, 1.0)
        )
    return fallback_fill, 0, len(points), "mesh_aabb_volume_ratio"


def _interior_sample_points(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    extents = np.maximum(aabb_max - aabb_min, 0.0)
    if float(np.max(extents)) <= _EPSILON:
        return np.empty((0, 3), dtype=float)

    axis_values = []
    for axis in range(3):
        if extents[axis] <= _EPSILON:
            axis_values.append([float(aabb_min[axis])])
        else:
            axis_values.append(
                [
                    float(aabb_min[axis] + extents[axis] * 0.25),
                    float(aabb_min[axis] + extents[axis] * 0.75),
                ]
            )
    return np.asarray(list(product(*axis_values)), dtype=float)


def _count_points_in_aabb(
    points: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> int:
    if points.size == 0:
        return 0

    return int(np.count_nonzero(_points_in_aabb_mask(points, aabb_min, aabb_max)))


def _points_in_aabb_mask(
    points: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    lower = points >= (aabb_min - _EPSILON)
    upper = points <= (aabb_max + _EPSILON)
    return np.all(lower & upper, axis=1)


def _estimate_region_surface_area(
    context: FabricationContext,
    stats: Dict[str, object],
    all_stats: Sequence[Dict[str, object]],
) -> float:
    mesh_area = float(getattr(context.mesh, "area", 0.0))
    if mesh_area <= _EPSILON:
        return 0.0

    face_area = float(stats.get("face_area_mm2", 0.0))
    total_face_area = sum(float(item.get("face_area_mm2", 0.0)) for item in all_stats)
    if face_area > _EPSILON and total_face_area > _EPSILON:
        return float(face_area)

    total_aabb_volume = sum(float(item["aabb_volume_mm3"]) for item in all_stats)
    if total_aabb_volume <= _EPSILON:
        return 0.0

    aabb_share = float(stats["aabb_volume_mm3"]) / total_aabb_volume
    return float(max(mesh_area * aabb_share, 0.0))


def _classify_region(
    *,
    context: FabricationContext,
    mesh_extents: np.ndarray,
    region_extents: np.ndarray,
    fill_ratio: float,
    global_fill: float | None = None,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    mesh_longest = float(np.max(mesh_extents))
    mesh_shortest = float(np.min(mesh_extents))
    mesh_flatness = mesh_shortest / mesh_longest if mesh_longest > _EPSILON else 0.0
    thin_limit = max(
        float(context.material_thickness_mm) * 2.5,
        float(context.config.min_feature_mm) * 2.5,
    )
    mesh_bounds = np.asarray(context.mesh.bounds, dtype=float)
    mesh_aabb_volume = float(np.prod(np.maximum(mesh_bounds[1] - mesh_bounds[0], 0.0)))
    if global_fill is None:
        global_fill = (
            float(np.clip(context.mesh_volume_mm3 / mesh_aabb_volume, 0.0, 1.0))
            if mesh_aabb_volume > _EPSILON
            else 0.0
        )

    if float(context.mesh_volume_mm3) <= _EPSILON:
        reasons.append("zero_or_unknown_mesh_volume")
        return "shell_band", reasons

    if mesh_flatness <= 0.18 or mesh_shortest <= thin_limit:
        reasons.append("mesh_has_thin_axis")
        return "flat_band", reasons

    if not bool(context.mesh.is_watertight):
        reasons.append("mesh_not_watertight")
        return "shell_band", reasons

    if fill_ratio < 0.30 or global_fill < 0.30:
        reasons.append("low_occupancy_fill_ratio")
        return "shell_band", reasons

    region_shortest = float(np.min(region_extents))
    region_longest = float(np.max(region_extents))
    if region_longest > _EPSILON and region_shortest / region_longest <= 0.12:
        reasons.append("region_band_is_thin")
        return "flat_band", reasons

    reasons.append("solid_aabb_occupancy")
    return "blocky_band", reasons


def _to_vec3(values: Sequence[float]) -> Vec3:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _to_float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]
