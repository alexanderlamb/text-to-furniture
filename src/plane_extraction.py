"""
RANSAC + region-growing planar patch extraction from 3D meshes.

Iteratively fits planes to unassigned faces, grows each region along
mesh face adjacency, projects boundaries to 2D via Shapely, and merges
near-coplanar patches.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon

from geometry_primitives import PlanarPatch, project_faces_to_2d, project_faces_to_2d_with_basis

logger = logging.getLogger(__name__)


@dataclass
class PlaneExtractionConfig:
    """Configuration for planar patch extraction."""
    ransac_threshold_mm: float = 2.0
    ransac_iterations: int = 1000
    min_patch_area_mm2: float = 500.0
    merge_angle_threshold_deg: float = 10.0
    region_grow_angle_deg: float = 15.0
    max_patches: int = 50


def extract_planar_patches(
    mesh: trimesh.Trimesh,
    config: Optional[PlaneExtractionConfig] = None,
) -> List[PlanarPatch]:
    """Extract planar patches from a mesh using RANSAC + region growing.

    Algorithm:
    1. Iteratively RANSAC-fit best plane to unassigned faces
    2. Region-grow from inliers along mesh.face_adjacency
    3. Project boundary to 2D via Shapely
    4. Merge near-coplanar patches
    5. Stop when remaining area < min_patch_area_mm2

    Args:
        mesh: Input triangle mesh.
        config: Extraction parameters.

    Returns:
        List of PlanarPatch sorted by area descending.
    """
    if config is None:
        config = PlaneExtractionConfig()

    face_normals = mesh.face_normals  # (N, 3)
    face_centres = mesh.triangles_center  # (N, 3)
    face_areas = mesh.area_faces  # (N,)
    n_faces = len(face_normals)

    # Precompute face adjacency graph
    adjacency = _build_adjacency(mesh)

    unassigned: Set[int] = set(range(n_faces))
    patches: List[PlanarPatch] = []

    grow_cos = np.cos(np.radians(config.region_grow_angle_deg))

    for _ in range(config.max_patches):
        if not unassigned:
            break

        remaining_area = sum(face_areas[i] for i in unassigned)
        if remaining_area < config.min_patch_area_mm2:
            break

        # RANSAC: find best plane among unassigned faces
        best_plane = _ransac_plane(
            face_centres, face_normals, face_areas,
            list(unassigned), config,
        )
        if best_plane is None:
            break

        plane_n, plane_d = best_plane

        # Region grow from RANSAC inliers
        inlier_faces = _find_inliers(
            face_centres, face_normals, list(unassigned),
            plane_n, plane_d,
            config.ransac_threshold_mm, grow_cos,
        )

        grown = _region_grow(
            inlier_faces, unassigned, adjacency,
            face_normals, plane_n, grow_cos,
        )

        if not grown:
            # Remove these from unassigned to avoid infinite loop
            unassigned -= set(inlier_faces)
            continue

        patch_area = sum(face_areas[i] for i in grown)
        if patch_area < config.min_patch_area_mm2:
            unassigned -= grown
            continue

        # Project faces to 2D boundary
        face_list = sorted(grown)
        boundary, basis_u, basis_v, _ = project_faces_to_2d_with_basis(
            mesh, face_list, plane_n, plane_d,
        )

        if boundary.is_empty or boundary.area < 1.0:
            unassigned -= grown
            continue

        centroid_3d = np.mean(face_centres[face_list], axis=0)

        patches.append(PlanarPatch(
            plane_normal=plane_n.copy(),
            plane_offset=float(plane_d),
            face_indices=face_list,
            boundary_2d=boundary,
            area_mm2=float(patch_area),
            centroid_3d=centroid_3d,
            basis_u=basis_u.copy(),
            basis_v=basis_v.copy(),
        ))

        unassigned -= grown

    # Merge near-coplanar patches
    patches = _merge_coplanar(patches, config.merge_angle_threshold_deg)

    # Sort by area descending
    patches.sort(key=lambda p: p.area_mm2, reverse=True)

    logger.info(
        "Extracted %d planar patches (%.0f mm2 total)",
        len(patches),
        sum(p.area_mm2 for p in patches),
    )
    return patches


# ─── RANSAC ──────────────────────────────────────────────────────────────────

def _ransac_plane(
    centres: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    candidates: List[int],
    config: PlaneExtractionConfig,
) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC plane fitting on face centroids.

    Uses face normals as a prior: randomly sample a face, use its normal
    as the plane normal and its centroid to compute the offset. Score by
    total area of inlier faces.
    """
    if not candidates:
        return None

    rng = np.random.default_rng()
    best_score = 0.0
    best_plane = None

    cand_arr = np.array(candidates)
    n_cand = len(cand_arr)
    threshold = config.ransac_threshold_mm
    grow_cos = np.cos(np.radians(config.region_grow_angle_deg))

    for _ in range(config.ransac_iterations):
        # Sample a random face
        idx = cand_arr[rng.integers(n_cand)]
        plane_n = normals[idx].copy()
        norm = np.linalg.norm(plane_n)
        if norm < 1e-8:
            continue
        plane_n /= norm
        plane_d = float(np.dot(plane_n, centres[idx]))

        # Score: total area of inlier faces
        dists = np.abs(centres[cand_arr] @ plane_n - plane_d)
        normal_dots = np.abs(normals[cand_arr] @ plane_n)
        inlier_mask = (dists < threshold) & (normal_dots > grow_cos)
        score = float(areas[cand_arr[inlier_mask]].sum())

        if score > best_score:
            best_score = score
            best_plane = (plane_n, plane_d)

    return best_plane


def _find_inliers(
    centres: np.ndarray,
    normals: np.ndarray,
    candidates: List[int],
    plane_n: np.ndarray,
    plane_d: float,
    threshold: float,
    cos_angle: float,
) -> List[int]:
    """Find faces that are inliers to the given plane."""
    inliers = []
    for idx in candidates:
        dist = abs(float(np.dot(plane_n, centres[idx]) - plane_d))
        ndot = abs(float(np.dot(plane_n, normals[idx])))
        if dist < threshold and ndot > cos_angle:
            inliers.append(idx)
    return inliers


# ─── Region Growing ─────────────────────────────────────────────────────────

def _build_adjacency(mesh: trimesh.Trimesh) -> List[List[int]]:
    """Build face adjacency list from mesh.face_adjacency."""
    n_faces = len(mesh.faces)
    adj: List[List[int]] = [[] for _ in range(n_faces)]
    for a, b in mesh.face_adjacency:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _region_grow(
    seeds: List[int],
    unassigned: Set[int],
    adjacency: List[List[int]],
    normals: np.ndarray,
    plane_n: np.ndarray,
    cos_angle: float,
) -> Set[int]:
    """Grow region from seed faces along adjacency, constrained by normal angle."""
    region: Set[int] = set()
    queue = [s for s in seeds if s in unassigned]

    for s in queue:
        region.add(s)

    while queue:
        face = queue.pop()
        for neighbor in adjacency[face]:
            if neighbor in region or neighbor not in unassigned:
                continue
            ndot = abs(float(np.dot(normals[neighbor], plane_n)))
            if ndot > cos_angle:
                region.add(neighbor)
                queue.append(neighbor)

    return region


# ─── Merging ─────────────────────────────────────────────────────────────────

def _merge_coplanar(
    patches: List[PlanarPatch],
    angle_threshold_deg: float,
) -> List[PlanarPatch]:
    """Merge patches whose normals and offsets are close."""
    if len(patches) <= 1:
        return patches

    cos_thresh = np.cos(np.radians(angle_threshold_deg))
    merged = [patches[0]]

    for patch in patches[1:]:
        did_merge = False
        for i, existing in enumerate(merged):
            ndot = abs(float(np.dot(patch.plane_normal, existing.plane_normal)))
            if ndot < cos_thresh:
                continue
            # Check offset proximity (project patch centroid onto existing plane)
            offset_diff = abs(
                float(np.dot(existing.plane_normal, patch.centroid_3d))
                - existing.plane_offset
            )
            if offset_diff > 10.0:  # mm tolerance for "same plane"
                continue

            # Merge: combine face indices, union boundaries
            combined_faces = existing.face_indices + patch.face_indices
            try:
                combined_boundary = existing.boundary_2d.union(patch.boundary_2d)
                if hasattr(combined_boundary, 'geoms'):
                    combined_boundary = max(combined_boundary.geoms, key=lambda g: g.area)
            except Exception:
                combined_boundary = existing.boundary_2d

            merged[i] = PlanarPatch(
                plane_normal=existing.plane_normal,
                plane_offset=existing.plane_offset,
                face_indices=combined_faces,
                boundary_2d=combined_boundary,
                area_mm2=existing.area_mm2 + patch.area_mm2,
                centroid_3d=(
                    existing.centroid_3d * existing.area_mm2
                    + patch.centroid_3d * patch.area_mm2
                ) / (existing.area_mm2 + patch.area_mm2),
            )
            did_merge = True
            break

        if not did_merge:
            merged.append(patch)

    return merged
