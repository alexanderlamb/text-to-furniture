"""
Mesh cleanup helpers for flattening near-planar regions and reducing noise.

This module provides an optional preprocessing pass used before decomposition.
It can:
1. Snap nearly planar regions to fitted planes.
2. Align near-parallel region families to shared normals.
3. Quantize planar-region vertices to reduce unnecessary triangulation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import logging
import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class MeshCleanupConfig:
    """Configuration for mesh geometry cleanup."""

    enabled: bool = False
    planar_angle_threshold_deg: float = 8.0
    planar_distance_threshold_mm: float = 1.5
    parallel_angle_threshold_deg: float = 5.0
    boundary_simplify_tolerance_mm: float = 2.0
    min_region_area_mm2: float = 400.0
    max_iterations: int = 2
    simplify_enabled: bool = True
    simplify_target_reduction: float = 0.50
    simplify_planar_boost: float = 1.25
    simplify_nonplanar_scale: float = 0.60
    simplify_min_faces: int = 3000
    simplify_max_normal_change_deg: float = 12.0
    simplify_max_bbox_drift_ratio: float = 0.02


@dataclass
class _PlanarRegion:
    """Internal planar-region container."""

    face_indices: np.ndarray
    vertex_indices: np.ndarray
    normal: np.ndarray
    offset: float
    centroid: np.ndarray
    area_mm2: float


def clean_mesh_geometry(
    mesh: trimesh.Trimesh,
    config: Optional[MeshCleanupConfig] = None,
) -> trimesh.Trimesh:
    """Return a cleaned copy of *mesh*.

    The cleanup pass is intentionally conservative and may skip regions if
    they do not meet area/planarity thresholds.
    """
    if config is None:
        config = MeshCleanupConfig()
    if not config.enabled or len(mesh.faces) == 0:
        return mesh

    cleaned = mesh.copy()

    for _ in range(max(1, config.max_iterations)):
        regions = _extract_planar_regions(cleaned, config)
        if not regions:
            break

        changed = False
        changed |= _snap_regions_to_planes(cleaned, regions)
        changed |= _enforce_parallel_families(cleaned, regions, config)
        changed |= _quantize_region_vertices(cleaned, regions, config)

        before_faces = len(cleaned.faces)
        cleaned.merge_vertices(digits_vertex=7)
        cleaned.update_faces(cleaned.unique_faces())
        cleaned.update_faces(cleaned.nondegenerate_faces())
        cleaned.remove_unreferenced_vertices()
        cleaned.fix_normals()
        changed |= len(cleaned.faces) != before_faces

        if not changed:
            break

    return cleaned


def simplify_mesh_geometry(
    mesh: trimesh.Trimesh,
    config: Optional[MeshCleanupConfig] = None,
) -> Tuple[trimesh.Trimesh, dict]:
    """Optionally simplify mesh faces with quality guards.

    Returns:
        (mesh_after_simplification, stats_dict)
    """
    if config is None:
        config = MeshCleanupConfig()

    before_faces = int(len(mesh.faces))
    stats = {
        "enabled": bool(config.simplify_enabled),
        "before_faces": before_faces,
        "after_faces": before_faces,
        "mode": "disabled",
        "guard_reverted": False,
        "bbox_drift_ratio": 0.0,
        "normal_change_deg": 0.0,
    }

    if not config.simplify_enabled:
        return mesh, stats

    if before_faces < int(config.simplify_min_faces):
        stats["mode"] = "skip_small"
        return mesh, stats

    target_reduction = float(np.clip(config.simplify_target_reduction, 0.0, 0.95))
    if target_reduction <= 0.0:
        stats["mode"] = "target_zero"
        return mesh, stats

    mode = "adaptive"
    try:
        simplified = _simplify_mesh_adaptive(mesh, config)
    except Exception as exc:
        logger.warning("Adaptive simplification failed: %s", exc)
        mode = "global_fallback"
        try:
            simplified = _simplify_mesh_global(mesh, target_reduction)
        except Exception as fallback_exc:
            logger.warning("Global simplification fallback failed: %s", fallback_exc)
            stats["mode"] = "unavailable"
            return mesh, stats

    if simplified is None or len(simplified.faces) == 0:
        stats["mode"] = "failed"
        return mesh, stats

    simplified = _postprocess_mesh(simplified)
    after_faces = int(len(simplified.faces))
    stats["after_faces"] = after_faces
    if after_faces >= before_faces:
        stats["mode"] = "no_gain"
        return mesh, stats

    bbox_drift = _bbox_drift_ratio(mesh, simplified)
    normal_change = _estimate_normal_change_deg(mesh, simplified)
    stats["bbox_drift_ratio"] = float(bbox_drift)
    stats["normal_change_deg"] = float(normal_change)
    stats["mode"] = mode

    if bbox_drift > float(config.simplify_max_bbox_drift_ratio):
        stats["guard_reverted"] = True
        stats["mode"] = f"{mode}_bbox_revert"
        stats["after_faces"] = before_faces
        return mesh, stats

    if normal_change > float(config.simplify_max_normal_change_deg):
        stats["guard_reverted"] = True
        stats["mode"] = f"{mode}_normal_revert"
        stats["after_faces"] = before_faces
        return mesh, stats

    return simplified, stats


def _extract_planar_regions(
    mesh: trimesh.Trimesh,
    config: MeshCleanupConfig,
) -> List[_PlanarRegion]:
    """Collect connected face regions that are approximately planar."""
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces
    n_faces = len(mesh.faces)

    if n_faces == 0:
        return []

    adjacency = _build_face_adjacency(mesh, n_faces)
    visited = np.zeros(n_faces, dtype=bool)

    cos_thresh = float(np.cos(np.radians(config.planar_angle_threshold_deg)))
    dist_thresh = float(config.planar_distance_threshold_mm)

    regions: List[_PlanarRegion] = []

    for seed in range(n_faces):
        if visited[seed]:
            continue

        queue: List[int] = [seed]
        cluster: Set[int] = set()
        candidate_marked: Set[int] = {seed}

        while queue:
            fidx = queue.pop()
            if visited[fidx]:
                continue
            cluster.add(fidx)
            for nb in adjacency[fidx]:
                if visited[nb] or nb in candidate_marked:
                    continue
                ndot = abs(float(np.dot(_unit(face_normals[fidx]), _unit(face_normals[nb]))))
                if ndot >= cos_thresh:
                    candidate_marked.add(nb)
                    queue.append(nb)

        if not cluster:
            continue

        cluster_idx = np.array(sorted(cluster), dtype=int)
        cluster_vertex_idx = np.unique(mesh.faces[cluster_idx].reshape(-1))
        cluster_points = mesh.vertices[cluster_vertex_idx]
        plane_n, plane_d = _fit_plane(cluster_points)

        centre_dists = np.abs(face_centers[cluster_idx] @ plane_n - plane_d)
        normal_align = np.abs(face_normals[cluster_idx] @ plane_n) >= cos_thresh
        inlier_mask = (centre_dists <= dist_thresh) & normal_align
        face_idx = cluster_idx[inlier_mask]

        if len(face_idx) == 0:
            visited[cluster_idx] = True
            continue

        area_mm2 = float(face_areas[face_idx].sum())
        if area_mm2 < config.min_region_area_mm2:
            visited[cluster_idx] = True
            continue

        vertex_idx = np.unique(mesh.faces[face_idx].reshape(-1))
        points = mesh.vertices[vertex_idx]
        plane_n, plane_d = _fit_plane(points)
        centroid = np.mean(points, axis=0)

        centre_dists = np.abs(face_centers[face_idx] @ plane_n - plane_d)
        if np.percentile(centre_dists, 90) > dist_thresh * 2.0:
            visited[cluster_idx] = True
            continue

        visited[cluster_idx] = True
        regions.append(
            _PlanarRegion(
                face_indices=face_idx,
                vertex_indices=vertex_idx,
                normal=plane_n,
                offset=float(plane_d),
                centroid=centroid,
                area_mm2=area_mm2,
            )
        )

    return regions


def _simplify_mesh_adaptive(
    mesh: trimesh.Trimesh,
    config: MeshCleanupConfig,
) -> trimesh.Trimesh:
    """Simplify planar and non-planar faces with different reduction targets."""
    regions = _extract_planar_regions(mesh, config)
    if not regions:
        return _simplify_mesh_global(
            mesh, float(np.clip(config.simplify_target_reduction, 0.0, 0.95))
        )

    n_faces = len(mesh.faces)
    planar_idx = np.unique(np.concatenate([r.face_indices for r in regions]))
    planar_mask = np.zeros(n_faces, dtype=bool)
    planar_mask[planar_idx] = True

    nonplanar_idx = np.where(~planar_mask)[0]
    if len(planar_idx) == 0 or len(nonplanar_idx) == 0:
        return _simplify_mesh_global(
            mesh, float(np.clip(config.simplify_target_reduction, 0.0, 0.95))
        )

    base = float(np.clip(config.simplify_target_reduction, 0.0, 0.95))
    planar_reduction = float(np.clip(base * config.simplify_planar_boost, 0.0, 0.95))
    nonplanar_reduction = float(
        np.clip(base * config.simplify_nonplanar_scale, 0.0, 0.95)
    )

    planar_mesh = mesh.submesh([planar_idx], append=True, repair=False)
    nonplanar_mesh = mesh.submesh([nonplanar_idx], append=True, repair=False)

    planar_simplified = _simplify_mesh_global(planar_mesh, planar_reduction)
    nonplanar_simplified = _simplify_mesh_global(nonplanar_mesh, nonplanar_reduction)
    combined = trimesh.util.concatenate([planar_simplified, nonplanar_simplified])
    return _postprocess_mesh(combined)


def _simplify_mesh_global(mesh: trimesh.Trimesh, target_reduction: float) -> trimesh.Trimesh:
    """Apply global face-count simplification with quadric decimation."""
    target_reduction = float(np.clip(target_reduction, 0.0, 0.95))
    if target_reduction <= 0.0 or len(mesh.faces) < 8:
        return mesh.copy()

    target_faces = max(4, int(round(len(mesh.faces) * (1.0 - target_reduction))))
    if target_faces >= len(mesh.faces):
        return mesh.copy()

    if hasattr(mesh, "simplify_quadric_decimation"):
        try:
            return mesh.simplify_quadric_decimation(face_count=target_faces)
        except ModuleNotFoundError:
            # Common case when fast_simplification is not installed.
            pass
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Quadric simplification failed, using clustering fallback: %s", exc)

    return _simplify_by_vertex_clustering(mesh, target_faces)


def _simplify_by_vertex_clustering(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> trimesh.Trimesh:
    """Fallback simplification via vertex quantization clustering."""
    if len(mesh.faces) <= target_faces:
        return mesh.copy()

    extents = mesh.bounds[1] - mesh.bounds[0]
    diag = float(np.linalg.norm(extents))
    if diag < 1e-6:
        return mesh.copy()

    best = mesh.copy()
    best_faces = len(best.faces)
    # Increasing grid steps progressively merges more vertices.
    for divisor in [2000, 1200, 800, 500, 300, 200, 120, 80, 50, 30]:
        step = diag / float(divisor)
        simplified = _quantize_mesh_vertices(mesh, step)
        fcount = len(simplified.faces)
        if fcount < best_faces:
            best = simplified
            best_faces = fcount
        if fcount <= target_faces:
            return simplified

    return best


def _quantize_mesh_vertices(mesh: trimesh.Trimesh, step: float) -> trimesh.Trimesh:
    """Quantize vertices onto a 3D grid and rebuild mesh."""
    if step <= 0.0 or len(mesh.vertices) == 0:
        return mesh.copy()

    mins = mesh.bounds[0]
    key = np.round((mesh.vertices - mins) / step).astype(np.int64)
    unique_key, inverse = np.unique(key, axis=0, return_inverse=True)
    new_vertices = mins + unique_key.astype(float) * step
    new_faces = inverse[mesh.faces]

    # Drop degenerate faces introduced by vertex merging.
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    if not np.any(keep):
        return mesh.copy()

    rebuilt = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces[keep],
        process=False,
    )
    return _postprocess_mesh(rebuilt)


def _postprocess_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh topology after simplification."""
    out = mesh.copy()
    out.merge_vertices(digits_vertex=7)
    out.update_faces(out.unique_faces())
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def _bbox_drift_ratio(before: trimesh.Trimesh, after: trimesh.Trimesh) -> float:
    """Max relative drift of axis extents between two meshes."""
    b_ext = before.bounds[1] - before.bounds[0]
    a_ext = after.bounds[1] - after.bounds[0]
    denom = np.maximum(np.abs(b_ext), 1e-6)
    rel = np.abs(a_ext - b_ext) / denom
    return float(np.max(rel))


def _canonicalize_normals(normals: np.ndarray) -> np.ndarray:
    """Flip normals to a consistent hemisphere."""
    if len(normals) == 0:
        return normals
    out = normals.copy()
    axis = np.argmax(np.abs(out), axis=1)
    signs = np.sign(out[np.arange(len(out)), axis])
    signs[signs == 0.0] = 1.0
    out *= signs[:, None]
    lens = np.linalg.norm(out, axis=1)
    valid = lens > 1e-8
    out[valid] = out[valid] / lens[valid][:, None]
    return out


def _estimate_normal_change_deg(
    before: trimesh.Trimesh,
    after: trimesh.Trimesh,
    sample_count: int = 512,
) -> float:
    """Estimate orientation drift by nearest-normal matching."""
    if len(before.faces) == 0 or len(after.faces) == 0:
        return 0.0

    rng = np.random.default_rng(1234)
    n_before = _canonicalize_normals(before.face_normals)
    n_after = _canonicalize_normals(after.face_normals)

    if len(n_before) > sample_count:
        n_before = n_before[rng.choice(len(n_before), sample_count, replace=False)]
    if len(n_after) > sample_count:
        n_after = n_after[rng.choice(len(n_after), sample_count, replace=False)]

    dots = np.abs(n_after @ n_before.T)
    best = np.max(dots, axis=1)
    best = np.clip(best, -1.0, 1.0)
    angles = np.degrees(np.arccos(best))
    return float(np.mean(angles))


def _build_face_adjacency(mesh: trimesh.Trimesh, n_faces: int) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(n_faces)]
    for a, b in mesh.face_adjacency:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _snap_regions_to_planes(mesh: trimesh.Trimesh, regions: List[_PlanarRegion]) -> bool:
    """Snap each region to its fitted plane."""
    targets: Dict[int, List[np.ndarray]] = {}
    for region in regions:
        n = region.normal
        d = region.offset
        for vidx in region.vertex_indices:
            p = mesh.vertices[vidx]
            proj = p - (np.dot(n, p) - d) * n
            targets.setdefault(int(vidx), []).append(proj)
    return _apply_vertex_targets(mesh, targets)


def _enforce_parallel_families(
    mesh: trimesh.Trimesh,
    regions: List[_PlanarRegion],
    config: MeshCleanupConfig,
) -> bool:
    """Align near-parallel region families to canonical normals."""
    if len(regions) < 2:
        return False

    cos_thresh = float(np.cos(np.radians(config.parallel_angle_threshold_deg)))

    families: List[List[int]] = []
    family_ref_normals: List[np.ndarray] = []

    sorted_idx = sorted(range(len(regions)), key=lambda i: regions[i].area_mm2, reverse=True)
    for idx in sorted_idx:
        n = _canonicalize_hemisphere(regions[idx].normal)
        assigned = False
        for fidx, ref in enumerate(family_ref_normals):
            if abs(float(np.dot(n, ref))) >= cos_thresh:
                families[fidx].append(idx)
                assigned = True
                break
        if not assigned:
            families.append([idx])
            family_ref_normals.append(n)

    targets: Dict[int, List[np.ndarray]] = {}
    for family in families:
        if len(family) < 2:
            continue

        weighted = np.zeros(3, dtype=float)
        for ridx in family:
            region = regions[ridx]
            n = _canonicalize_hemisphere(region.normal)
            weighted += n * region.area_mm2
        canonical = _unit(weighted)

        for ridx in family:
            region = regions[ridx]
            target_n = canonical.copy()
            if float(np.dot(target_n, region.normal)) < 0.0:
                target_n = -target_n
            target_d = float(np.dot(target_n, region.centroid))
            for vidx in region.vertex_indices:
                p = mesh.vertices[vidx]
                proj = p - (np.dot(target_n, p) - target_d) * target_n
                targets.setdefault(int(vidx), []).append(proj)

    return _apply_vertex_targets(mesh, targets)


def _quantize_region_vertices(
    mesh: trimesh.Trimesh,
    regions: List[_PlanarRegion],
    config: MeshCleanupConfig,
) -> bool:
    """Quantize planar-region coordinates to collapse unnecessary triangles."""
    tol = float(config.boundary_simplify_tolerance_mm)
    if tol <= 0.0:
        return False

    targets: Dict[int, List[np.ndarray]] = {}
    for region in regions:
        n = region.normal
        origin = region.centroid
        u, v = _plane_axes(n)
        d = region.offset

        buckets: Dict[Tuple[int, int], List[int]] = {}
        for vidx in region.vertex_indices:
            p = mesh.vertices[vidx]
            # Keep quantization on the fitted plane.
            p_proj = p - (np.dot(n, p) - d) * n
            rel = p_proj - origin
            x = float(np.dot(rel, u))
            y = float(np.dot(rel, v))
            key = (int(round(x / tol)), int(round(y / tol)))
            buckets.setdefault(key, []).append(int(vidx))

        for key, vids in buckets.items():
            if len(vids) < 2:
                continue
            qx = key[0] * tol
            qy = key[1] * tol
            snapped = origin + u * qx + v * qy
            for vidx in vids:
                targets.setdefault(vidx, []).append(snapped)

    return _apply_vertex_targets(mesh, targets)


def _apply_vertex_targets(
    mesh: trimesh.Trimesh, targets: Dict[int, List[np.ndarray]]
) -> bool:
    """Apply proposed vertex positions, choosing the smallest displacement."""
    changed = False
    for vidx, proposals in targets.items():
        if not proposals:
            continue
        current = mesh.vertices[vidx]
        best = min(proposals, key=lambda p: float(np.linalg.norm(p - current)))
        if float(np.linalg.norm(best - current)) > 1e-6:
            mesh.vertices[vidx] = best
            changed = True
    return changed


def _fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a plane n.x = d to points using SVD."""
    if len(points) < 3:
        n = np.array([0.0, 0.0, 1.0], dtype=float)
        d = float(np.dot(n, points[0])) if len(points) else 0.0
        return n, d

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    n = _unit(vh[-1])
    d = float(np.dot(n, centroid))
    return n, d


def _plane_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _unit(normal)
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = _unit(np.cross(n, ref))
    v = _unit(np.cross(n, u))
    return u, v


def _canonicalize_hemisphere(normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    axis = int(np.argmax(np.abs(n)))
    if n[axis] < 0:
        n = -n
    return n


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vec / norm
