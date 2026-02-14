"""
Candidate slab generation for v2 manufacturing pipeline.

Generates Slab3D objects from three sources:
  A. RANSAC planar patches (reuse plane_extraction.py)
  B. Axis-aligned cross-sections
  C. Surface-aligned slices (dominant normal clusters)

All geometry stays in 3D — each slab carries its own (basis_u, basis_v, origin)
frame so 2D projection happens once and is consistent.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon

from geometry_primitives import (
    PartProfile2D,
    PlanarPatch,
    _make_2d_basis,
    project_faces_to_2d_with_basis,
)
from plane_extraction import PlaneExtractionConfig, extract_planar_patches
from mesh_decomposer import find_dominant_orientations, DecompositionConfig
from materials import MATERIALS

logger = logging.getLogger(__name__)


@dataclass
class Slab3D:
    """A candidate flat slab in 3D space."""
    normal: np.ndarray          # (3,) unit normal (slab's thin axis)
    origin: np.ndarray          # (3,) point on the slab's mid-plane
    thickness_mm: float         # SCS material thickness
    width_mm: float             # extent along basis_u
    height_mm: float            # extent along basis_v
    basis_u: np.ndarray         # (3,) first in-plane axis
    basis_v: np.ndarray         # (3,) second in-plane axis
    outline_2d: Polygon         # 2D profile in (u,v) frame
    material_key: str
    source: str                 # "ransac" | "axis_slice" | "surface_aligned" | "paired_interior"

    def world_to_uv(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """Project a 3D point into this slab's local (u, v) frame."""
        d = point_3d - self.origin
        return (float(d @ self.basis_u), float(d @ self.basis_v))

    def uv_to_world(self, u: float, v: float) -> np.ndarray:
        """Convert a (u, v) coordinate back to 3D world space."""
        return self.origin + u * self.basis_u + v * self.basis_v

    def plane_distance(self, point_3d: np.ndarray) -> float:
        """Signed distance from a 3D point to this slab's mid-plane."""
        return float(np.dot(point_3d - self.origin, self.normal))


@dataclass
class Slab3DConfig:
    """Configuration for candidate slab generation."""
    material_key: str = "plywood_baltic_birch"
    target_height_mm: float = 750.0
    ransac_normal_threshold_deg: float = 15.0
    ransac_distance_mm: float = 2.0
    max_ransac_patches: int = 50
    axis_slice_spacing_mm: float = 50.0
    min_slab_area_mm2: float = 500.0
    voxel_resolution_mm: float = 5.0
    normal_cluster_threshold_deg: float = 15.0
    dedupe_angle_deg: float = 10.0
    dedupe_offset_mm: float = 5.0
    enable_paired_interior: bool = True
    paired_parallel_angle_deg: float = 10.0
    paired_min_overlap_ratio: float = 0.15
    paired_min_gap_mm: float = 25.0
    paired_dual_gap_factor: float = 4.0
    max_paired_candidates: int = 40


def generate_candidates(
    mesh: trimesh.Trimesh,
    config: Optional[Slab3DConfig] = None,
) -> List[Slab3D]:
    """Generate candidate slabs from all three sources.

    Args:
        mesh: Prepared trimesh (already scaled/centred).
        config: Generation parameters.

    Returns:
        List of Slab3D candidates from RANSAC, axis slices, and surface-aligned.
    """
    if config is None:
        config = Slab3DConfig()

    material = MATERIALS[config.material_key]
    thickness = material.thicknesses_mm[len(material.thicknesses_mm) // 2]

    candidates: List[Slab3D] = []

    # Source A: RANSAC patches
    ransac_patches = _extract_ransac_patches(mesh, config)
    ransac = _candidates_from_ransac_patches(
        ransac_patches, config, thickness,
    )
    candidates.extend(ransac)
    logger.info("RANSAC source: %d candidates", len(ransac))

    # Source A2: Paired interior slabs between opposing near-parallel faces
    if config.enable_paired_interior:
        paired = _candidates_from_paired_interior(
            ransac_patches, config, thickness,
        )
        candidates.extend(paired)
        logger.info("Paired-interior source: %d candidates", len(paired))

    # Source B: Axis-aligned slices
    axis = _candidates_from_axis_slices(mesh, config, thickness)
    candidates.extend(axis)
    logger.info("Axis-slice source: %d candidates", len(axis))

    # Source C: Surface-aligned slices
    surface = _candidates_from_surface_aligned(mesh, config, thickness)
    candidates.extend(surface)
    logger.info("Surface-aligned source: %d candidates", len(surface))

    # Source D: AABB bounding-box diversity candidates (always generated to
    # ensure candidates exist in multiple orientations, not just RANSAC's)
    bbox = _candidates_from_bounding_box(mesh, config, thickness)
    candidates.extend(bbox)
    logger.info("Bbox-diversity source: %d candidates", len(bbox))

    logger.info("Total candidates before dedupe: %d", len(candidates))
    candidates = _deduplicate_candidates(candidates, config)
    logger.info("Total candidates after dedupe: %d", len(candidates))

    # Split oversized candidates into sheet-sized pieces
    candidates = _split_oversized_candidates(candidates, config)

    return candidates


# ─── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate_candidates(
    candidates: List[Slab3D],
    config: Slab3DConfig,
) -> List[Slab3D]:
    """Merge near-duplicate candidates using union-find grouping.

    Two candidates are "similar" when:
      - Their normals are within dedupe_angle_deg
      - Their plane offsets differ by less than dedupe_offset_mm

    For each group, one representative Slab3D is produced with:
      - Area-weighted mean normal (re-normalised)
      - Group centroid projected onto representative plane as origin
      - Union of all outlines in the representative UV frame
    """
    n = len(candidates)
    if n <= 1:
        return candidates

    cos_thresh = np.cos(np.radians(config.dedupe_angle_deg))

    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ni = candidates[i].normal
            nj = candidates[j].normal
            dot = abs(float(ni @ nj))
            if dot < cos_thresh:
                continue
            # Plane offset along the shared normal direction
            offset = abs(float(ni @ (candidates[i].origin - candidates[j].origin)))
            if offset < config.dedupe_offset_mm:
                union(i, j)

    # Group by root
    groups: dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    merged: List[Slab3D] = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(candidates[indices[0]])
            continue

        group_slabs = [candidates[i] for i in indices]

        # Area-weighted mean normal (flip anti-parallel normals to match first)
        areas = np.array([s.outline_2d.area for s in group_slabs])
        total_area = areas.sum()
        if total_area < 1e-12:
            merged.append(group_slabs[0])
            continue

        ref_n = group_slabs[0].normal
        weighted_normal = sum(
            a * (s.normal if float(s.normal @ ref_n) >= 0 else -s.normal)
            for a, s in zip(areas, group_slabs)
        )
        norm_len = np.linalg.norm(weighted_normal)
        if norm_len < 1e-12:
            merged.append(group_slabs[0])
            continue
        rep_normal = weighted_normal / norm_len

        # Representative basis
        from geometry_primitives import _make_2d_basis
        rep_u, rep_v = _make_2d_basis(rep_normal)

        # Group centroid as origin, projected onto the representative plane
        centroid_3d = np.mean([s.origin for s in group_slabs], axis=0)
        # Project centroid onto the plane defined by rep_normal through first slab origin
        ref_origin = group_slabs[0].origin
        dist = float(rep_normal @ (centroid_3d - ref_origin))
        rep_origin = centroid_3d - dist * rep_normal

        # Union outlines in representative UV frame
        from shapely.ops import unary_union
        reprojected = []
        for s in group_slabs:
            # Transform each slab's outline from its own UV frame to 3D, then to rep UV
            coords = list(s.outline_2d.exterior.coords)
            new_coords = []
            for cu, cv in coords:
                pt_3d = s.origin + cu * s.basis_u + cv * s.basis_v
                d = pt_3d - rep_origin
                new_u = float(d @ rep_u)
                new_v = float(d @ rep_v)
                new_coords.append((new_u, new_v))
            try:
                p = Polygon(new_coords)
                if p.is_valid and not p.is_empty:
                    reprojected.append(p)
            except Exception:
                pass

        if not reprojected:
            merged.append(group_slabs[0])
            continue

        union_outline = unary_union(reprojected)
        if union_outline.is_empty:
            merged.append(group_slabs[0])
            continue

        # If union is a MultiPolygon, take the largest
        if union_outline.geom_type == "MultiPolygon":
            union_outline = max(union_outline.geoms, key=lambda g: g.area)

        # Build a raw slab, then canonicalize (center + recompute dims)
        bounds = union_outline.bounds
        raw = Slab3D(
            normal=rep_normal,
            origin=rep_origin,
            thickness_mm=group_slabs[0].thickness_mm,
            width_mm=bounds[2] - bounds[0],
            height_mm=bounds[3] - bounds[1],
            basis_u=rep_u,
            basis_v=rep_v,
            outline_2d=union_outline,
            material_key=group_slabs[0].material_key,
            source=group_slabs[0].source,
        )
        merged.append(_canonicalize_slab_frame(raw))

    logger.info("Deduplicated %d -> %d candidates", n, len(merged))
    return merged


# ─── Source A: RANSAC ─────────────────────────────────────────────────────────

def _extract_ransac_patches(
    mesh: trimesh.Trimesh,
    config: Slab3DConfig,
) -> List[PlanarPatch]:
    """Extract planar patches used by RANSAC-derived sources."""
    pe_config = PlaneExtractionConfig(
        ransac_threshold_mm=config.ransac_distance_mm,
        region_grow_angle_deg=config.ransac_normal_threshold_deg,
        max_patches=config.max_ransac_patches,
        min_patch_area_mm2=config.min_slab_area_mm2,
    )
    return extract_planar_patches(mesh, pe_config)


def _candidates_from_ransac_patches(
    patches: List[PlanarPatch],
    config: Slab3DConfig,
    thickness: float,
) -> List[Slab3D]:
    """Create Slab3D candidates from already-extracted planar patches."""
    slabs: List[Slab3D] = []
    for patch in patches:
        slab = _patch_to_slab(patch, thickness, config.material_key)
        if slab is not None:
            slabs.append(slab)
    return slabs


def _candidates_from_ransac(
    mesh: trimesh.Trimesh,
    config: Slab3DConfig,
    thickness: float,
) -> List[Slab3D]:
    """Create Slab3D candidates from RANSAC-extracted planar patches."""
    patches = _extract_ransac_patches(mesh, config)
    return _candidates_from_ransac_patches(patches, config, thickness)


def _patch_to_slab(
    patch: PlanarPatch,
    thickness: float,
    material_key: str,
) -> Optional[Slab3D]:
    """Convert a PlanarPatch to a Slab3D.

    The patch's boundary_2d is in a raw projection frame (u = pt @ basis_u),
    not relative to any origin. We translate it into the slab's origin-relative
    UV frame, then center at (0,0).
    """
    from shapely.affinity import translate

    if patch.boundary_2d.is_empty or patch.boundary_2d.area < 1.0:
        return None

    normal = patch.plane_normal / np.linalg.norm(patch.plane_normal)

    # Use patch basis if available, otherwise compute
    if patch.basis_u is not None and patch.basis_v is not None:
        basis_u = patch.basis_u.copy()
        basis_v = patch.basis_v.copy()
    else:
        basis_u, basis_v = _make_2d_basis(normal)

    # Origin: the centroid projected onto the plane
    origin = patch.centroid_3d.copy()

    # boundary_2d is in raw frame (u = pt_3d @ basis_u). The slab's UV
    # frame is u = (pt_3d - origin) @ basis_u = pt_3d @ basis_u - origin @ basis_u.
    # Shift boundary into origin-relative frame.
    offset_u = float(origin @ basis_u)
    offset_v = float(origin @ basis_v)
    outline = translate(patch.boundary_2d, -offset_u, -offset_v)

    # Build a raw slab, then canonicalize (center + recompute dims)
    bounds = outline.bounds
    raw = Slab3D(
        normal=normal,
        origin=origin,
        thickness_mm=thickness,
        width_mm=bounds[2] - bounds[0],
        height_mm=bounds[3] - bounds[1],
        basis_u=basis_u,
        basis_v=basis_v,
        outline_2d=outline,
        material_key=material_key,
        source="ransac",
    )
    return _canonicalize_slab_frame(raw)


def _patch_polygon_in_frame(
    patch: PlanarPatch,
    frame_origin: np.ndarray,
    frame_u: np.ndarray,
    frame_v: np.ndarray,
) -> Optional[Polygon]:
    """Reproject a patch boundary into an arbitrary UV frame."""
    if patch.boundary_2d.is_empty:
        return None
    if patch.basis_u is None or patch.basis_v is None:
        return None

    n = patch.plane_normal / np.linalg.norm(patch.plane_normal)
    pu = patch.basis_u / np.linalg.norm(patch.basis_u)
    pv = patch.basis_v / np.linalg.norm(patch.basis_v)

    try:
        coords = list(patch.boundary_2d.exterior.coords)
    except Exception:
        return None
    if len(coords) < 4:
        return None

    reproj = []
    for raw_u, raw_v in coords:
        # Raw boundary_2d coordinates are pt@basis_u / pt@basis_v.
        pt_3d = pu * raw_u + pv * raw_v + n * patch.plane_offset
        d = pt_3d - frame_origin
        reproj.append((float(d @ frame_u), float(d @ frame_v)))

    try:
        poly = Polygon(reproj)
    except Exception:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.area < 1.0:
        return None
    return poly


def _build_slab_from_outline(
    normal: np.ndarray,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    outline: Polygon,
    thickness: float,
    material_key: str,
    source: str,
) -> Optional[Slab3D]:
    """Create a centered slab from a frame-aligned outline polygon."""
    if outline.is_empty or outline.area < 1.0:
        return None
    if isinstance(outline, MultiPolygon):
        outline = max(outline.geoms, key=lambda g: g.area)
    if not outline.is_valid:
        outline = outline.buffer(0)
    if outline.is_empty or outline.area < 1.0:
        return None

    bounds = outline.bounds
    raw = Slab3D(
        normal=normal / np.linalg.norm(normal),
        origin=origin.copy(),
        thickness_mm=thickness,
        width_mm=bounds[2] - bounds[0],
        height_mm=bounds[3] - bounds[1],
        basis_u=basis_u.copy(),
        basis_v=basis_v.copy(),
        outline_2d=outline,
        material_key=material_key,
        source=source,
    )
    return _canonicalize_slab_frame(raw)


def _candidates_from_paired_interior(
    patches: List[PlanarPatch],
    config: Slab3DConfig,
    thickness: float,
) -> List[Slab3D]:
    """Generate interior slabs between opposing, near-parallel face patches."""
    if len(patches) < 2:
        return []

    candidates: List[Slab3D] = []
    cos_parallel = np.cos(np.radians(config.paired_parallel_angle_deg))

    for i in range(len(patches)):
        if len(candidates) >= config.max_paired_candidates:
            break
        p_i = patches[i]
        n_i = p_i.plane_normal / np.linalg.norm(p_i.plane_normal)
        c_i = p_i.centroid_3d
        for j in range(i + 1, len(patches)):
            if len(candidates) >= config.max_paired_candidates:
                break
            p_j = patches[j]
            n_j = p_j.plane_normal / np.linalg.norm(p_j.plane_normal)
            if abs(float(n_i @ n_j)) < cos_parallel:
                continue

            c_j = p_j.centroid_3d
            delta = c_j - c_i
            gap = abs(float(delta @ n_i))
            if gap < max(config.paired_min_gap_mm, thickness * 1.5):
                continue

            # Orient pair normal so +n points roughly from patch i to j.
            pair_n = n_i if float(delta @ n_i) >= 0.0 else -n_i
            pair_u, pair_v = _make_2d_basis(pair_n)
            mid_origin = (c_i + c_j) / 2.0

            poly_i = _patch_polygon_in_frame(p_i, mid_origin, pair_u, pair_v)
            poly_j = _patch_polygon_in_frame(p_j, mid_origin, pair_u, pair_v)
            if poly_i is None or poly_j is None:
                continue

            overlap = poly_i.intersection(poly_j)
            if overlap.is_empty:
                continue
            if isinstance(overlap, MultiPolygon):
                overlap = max(overlap.geoms, key=lambda g: g.area)
            if not isinstance(overlap, Polygon):
                continue
            if overlap.area < config.min_slab_area_mm2:
                continue

            denom = max(min(poly_i.area, poly_j.area), 1.0)
            overlap_ratio = float(overlap.area / denom)
            if overlap_ratio < config.paired_min_overlap_ratio:
                continue

            if gap <= config.paired_dual_gap_factor * thickness:
                slab_mid = _build_slab_from_outline(
                    normal=pair_n,
                    origin=mid_origin,
                    basis_u=pair_u,
                    basis_v=pair_v,
                    outline=overlap,
                    thickness=thickness,
                    material_key=config.material_key,
                    source="paired_interior",
                )
                if slab_mid is not None:
                    candidates.append(slab_mid)
            else:
                origin_a = c_i + pair_n * (thickness / 2.0)
                origin_b = c_j - pair_n * (thickness / 2.0)
                slab_a = _build_slab_from_outline(
                    normal=pair_n,
                    origin=origin_a,
                    basis_u=pair_u,
                    basis_v=pair_v,
                    outline=overlap,
                    thickness=thickness,
                    material_key=config.material_key,
                    source="paired_interior",
                )
                slab_b = _build_slab_from_outline(
                    normal=pair_n,
                    origin=origin_b,
                    basis_u=pair_u,
                    basis_v=pair_v,
                    outline=overlap,
                    thickness=thickness,
                    material_key=config.material_key,
                    source="paired_interior",
                )
                if slab_a is not None:
                    candidates.append(slab_a)
                if slab_b is not None and len(candidates) < config.max_paired_candidates:
                    candidates.append(slab_b)

    return candidates


# ─── Source B: Axis-aligned cross-sections ────────────────────────────────────

def _center_outline(
    outline: Polygon,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> Tuple[Polygon, np.ndarray]:
    """Shift an outline polygon so its bounding-box center is at (0,0).

    Adjusts origin in 3D to compensate, so world_to_uv / uv_to_world
    remain consistent.

    Returns:
        (centered_outline, adjusted_origin)
    """
    from shapely.affinity import translate

    bounds = outline.bounds  # (minx, miny, maxx, maxy)
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0

    if abs(cx) < 1e-6 and abs(cy) < 1e-6:
        return outline, origin

    shifted = translate(outline, -cx, -cy)
    new_origin = origin + cx * basis_u + cy * basis_v
    return shifted, new_origin


def _canonicalize_slab_frame(slab: Slab3D) -> Slab3D:
    """Return a copy of *slab* with a canonical UV frame.

    * Outline centred at (0, 0)
    * 3D origin adjusted to match
    * width_mm / height_mm recomputed from centred bounds

    This is the single place where centering + dimension recomputation
    happens, preventing drift between sources.
    """
    outline, origin = _center_outline(
        slab.outline_2d, slab.origin.copy(), slab.basis_u, slab.basis_v,
    )
    bounds = outline.bounds  # (minx, miny, maxx, maxy)
    return Slab3D(
        normal=slab.normal,
        origin=origin,
        thickness_mm=slab.thickness_mm,
        width_mm=bounds[2] - bounds[0],
        height_mm=bounds[3] - bounds[1],
        basis_u=slab.basis_u,
        basis_v=slab.basis_v,
        outline_2d=outline,
        material_key=slab.material_key,
        source=slab.source,
    )


def _candidates_from_axis_slices(
    mesh: trimesh.Trimesh,
    config: Slab3DConfig,
    thickness: float,
) -> List[Slab3D]:
    """Create Slab3D candidates from axis-aligned cross-sections."""
    slabs: List[Slab3D] = []
    bounds = mesh.bounds  # (2, 3): min, max

    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    for axis in axes:
        axis_idx = int(np.argmax(np.abs(axis)))
        lo = bounds[0][axis_idx]
        hi = bounds[1][axis_idx]
        span = hi - lo

        if span < config.axis_slice_spacing_mm:
            positions = [lo + span / 2]
        else:
            n_slices = max(2, int(span / config.axis_slice_spacing_mm))
            positions = np.linspace(lo, hi, n_slices + 2)[1:-1].tolist()

        for pos in positions:
            slab = _slice_mesh_at(mesh, axis, pos, thickness,
                                  config.material_key, config.min_slab_area_mm2)
            if slab is not None:
                slabs.append(slab)

    return slabs


def _slice_mesh_at(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    position: float,
    thickness: float,
    material_key: str,
    min_area: float,
) -> Optional[Slab3D]:
    """Slice mesh at a position along a normal axis and create a Slab3D.

    Uses trimesh's polygon assembly for correct topology, then re-projects
    polygon vertices into the slab's own (basis_u, basis_v) frame so
    outline_2d, width/height, and basis vectors are all consistent.
    """
    n = normal / np.linalg.norm(normal)
    origin_pt = n * position
    basis_u, basis_v = _make_2d_basis(n)

    try:
        section = mesh.section(plane_origin=origin_pt, plane_normal=n)
    except Exception:
        return None

    if section is None:
        return None

    # Use trimesh's polygon assembly for correct topology
    try:
        path_2d, transform = section.to_2D()
    except Exception:
        return None

    if path_2d is None or len(path_2d.entities) == 0:
        return None

    try:
        polys_trimesh = path_2d.polygons_full
        if not polys_trimesh or len(polys_trimesh) == 0:
            return None
    except Exception:
        return None

    # Re-project each polygon from trimesh's arbitrary 2D frame into our
    # (basis_u, basis_v) frame via 3D.
    T_inv = np.linalg.inv(transform)
    our_polys = []
    for poly_2d in polys_trimesh:
        coords_2d = list(poly_2d.exterior.coords[:-1])
        our_coords = []
        for x, y in coords_2d:
            # trimesh 2D → 3D via inverse transform
            pt_3d_h = T_inv @ np.array([x, y, 0.0, 1.0])
            pt_3d = pt_3d_h[:3]
            # 3D → our UV frame
            d = pt_3d - origin_pt
            u = float(d @ basis_u)
            v = float(d @ basis_v)
            our_coords.append((u, v))
        try:
            p = Polygon(our_coords)
            if p.is_valid and not p.is_empty and p.area >= min_area:
                our_polys.append(p)
        except Exception:
            continue

    if not our_polys:
        return None

    outline = max(our_polys, key=lambda p: p.area)

    # Build a raw slab, then canonicalize (center + recompute dims)
    bounds = outline.bounds
    raw = Slab3D(
        normal=n.copy(),
        origin=origin_pt.copy(),
        thickness_mm=thickness,
        width_mm=bounds[2] - bounds[0],
        height_mm=bounds[3] - bounds[1],
        basis_u=basis_u,
        basis_v=basis_v,
        outline_2d=outline,
        material_key=material_key,
        source="axis_slice",
    )
    return _canonicalize_slab_frame(raw)


# ─── Source C: Surface-aligned slices ─────────────────────────────────────────

def _candidates_from_surface_aligned(
    mesh: trimesh.Trimesh,
    config: Slab3DConfig,
    thickness: float,
) -> List[Slab3D]:
    """Create slabs from dominant face normal clusters, sliced at surface depths."""
    decomp_config = DecompositionConfig(
        normal_cluster_threshold_deg=config.normal_cluster_threshold_deg,
    )
    orientations = find_dominant_orientations(mesh, decomp_config)

    face_centres = mesh.triangles_center
    face_normals = mesh.face_normals
    cos_thresh = np.cos(np.radians(config.normal_cluster_threshold_deg))

    slabs: List[Slab3D] = []

    for orient_normal, _area in orientations:
        n = orient_normal / np.linalg.norm(orient_normal)

        # Find faces aligned with this normal
        dots = np.abs(face_normals @ n)
        aligned = dots > cos_thresh
        if not aligned.any():
            continue

        # Cluster depths of aligned face centres
        depths = face_centres[aligned] @ n
        sorted_depths = np.sort(depths)

        # Simple clustering: merge depths within thickness*2
        clusters: List[List[float]] = [[sorted_depths[0]]]
        for d in sorted_depths[1:]:
            if d - clusters[-1][-1] > thickness * 2:
                clusters.append([d])
            else:
                clusters[-1].append(d)

        surface_positions = [float(np.mean(c)) for c in clusters]

        for pos in surface_positions:
            slab = _slice_mesh_at(mesh, n, pos, thickness,
                                  config.material_key, config.min_slab_area_mm2)
            if slab is not None:
                slab.source = "surface_aligned"
                slabs.append(slab)

    return slabs


# ─── Source D: AABB bounding-box fallback ──────────────────────────────────

def _candidates_from_bounding_box(
    mesh: trimesh.Trimesh,
    config: Slab3DConfig,
    thickness: float,
    max_orientations: int = 6,
    max_depths_per_orientation: int = 3,
) -> List[Slab3D]:
    """Generate candidates using vertex projection (no mesh.section).

    Provides diversity candidates in key orientations. Uses dominant
    orientations (top N by face area) and projects all vertices to get
    rectangular outlines.
    """
    decomp_config = DecompositionConfig(
        normal_cluster_threshold_deg=config.normal_cluster_threshold_deg,
    )
    orientations = find_dominant_orientations(mesh, decomp_config)

    # If no dominant orientations, use axis-aligned
    if not orientations:
        orientations = [
            (np.array([1.0, 0.0, 0.0]), 0.0),
            (np.array([0.0, 1.0, 0.0]), 0.0),
            (np.array([0.0, 0.0, 1.0]), 0.0),
        ]

    # Limit to top orientations by area for performance
    orientations = orientations[:max_orientations]

    vertices = mesh.vertices
    face_normals = mesh.face_normals
    face_centres = mesh.triangles_center
    cos_thresh = np.cos(np.radians(config.normal_cluster_threshold_deg))

    slabs: List[Slab3D] = []

    for orient_normal, _area in orientations:
        n = orient_normal / np.linalg.norm(orient_normal)
        basis_u, basis_v = _make_2d_basis(n)

        # Project ALL vertices to get width/height
        proj_u = vertices @ basis_u
        proj_v = vertices @ basis_v

        u_min, u_max = float(proj_u.min()), float(proj_u.max())
        v_min, v_max = float(proj_v.min()), float(proj_v.max())
        width = u_max - u_min
        height = v_max - v_min

        if width * height < config.min_slab_area_mm2:
            continue

        # Cluster face depths for surface positions
        dots = np.abs(face_normals @ n)
        aligned = dots > cos_thresh
        if aligned.any():
            depths = face_centres[aligned] @ n
            sorted_depths = np.sort(depths)
            clusters: List[List[float]] = [[sorted_depths[0]]]
            for d in sorted_depths[1:]:
                if d - clusters[-1][-1] > thickness * 2:
                    clusters.append([d])
                else:
                    clusters[-1].append(d)
            surface_positions = [float(np.mean(c)) for c in clusters]
        else:
            # Fallback: min/max/center depth positions
            all_depths = vertices @ n
            d_min, d_max = float(all_depths.min()), float(all_depths.max())
            surface_positions = [d_min, (d_min + d_max) / 2.0, d_max]

        # Limit depth positions: keep first, middle, last (evenly spaced)
        if len(surface_positions) > max_depths_per_orientation:
            indices = np.linspace(0, len(surface_positions) - 1,
                                  max_depths_per_orientation, dtype=int)
            surface_positions = [surface_positions[i] for i in indices]

        # Create rectangular outline centered at (0,0)
        hw = width / 2.0
        hh = height / 2.0
        outline = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])

        # Center of the UV projection
        u_center = (u_min + u_max) / 2.0
        v_center = (v_min + v_max) / 2.0

        for pos in surface_positions:
            origin = n * pos + basis_u * u_center + basis_v * v_center
            raw = Slab3D(
                normal=n.copy(),
                origin=origin,
                thickness_mm=thickness,
                width_mm=width,
                height_mm=height,
                basis_u=basis_u.copy(),
                basis_v=basis_v.copy(),
                outline_2d=outline,
                material_key=config.material_key,
                source="bbox_fallback",
            )
            slabs.append(_canonicalize_slab_frame(raw))

    return slabs


# ─── Oversized splitting ──────────────────────────────────────────────────

def _split_oversized_candidates(
    candidates: List[Slab3D],
    config: Slab3DConfig,
) -> List[Slab3D]:
    """Split candidates that exceed material sheet limits into tiles.

    Candidates that fit are passed through unchanged.
    """
    material = MATERIALS[config.material_key]
    max_w, max_h = material.max_size_mm

    result: List[Slab3D] = []
    for cand in candidates:
        # Check both orientations (width/height can map to either sheet axis)
        fits_normal = cand.width_mm <= max_w and cand.height_mm <= max_h
        fits_rotated = cand.width_mm <= max_h and cand.height_mm <= max_w
        if fits_normal or fits_rotated:
            result.append(cand)
            continue

        # Tile into grid of sheet-sized rectangles
        bounds = cand.outline_2d.bounds  # (minx, miny, maxx, maxy)
        tile_w = max_w * 0.95  # slight margin for safety
        tile_h = max_h * 0.95

        x = bounds[0]
        while x < bounds[2]:
            y = bounds[1]
            while y < bounds[3]:
                tile_rect = Polygon([
                    (x, y), (x + tile_w, y),
                    (x + tile_w, y + tile_h), (x, y + tile_h),
                ])
                clipped = cand.outline_2d.intersection(tile_rect)
                if clipped.is_empty:
                    y += tile_h
                    continue
                if isinstance(clipped, MultiPolygon):
                    clipped = max(clipped.geoms, key=lambda g: g.area)
                if not isinstance(clipped, Polygon) or clipped.area < config.min_slab_area_mm2:
                    y += tile_h
                    continue

                # Compute new origin for this tile (center of clipped outline)
                cb = clipped.bounds
                tile_cx = (cb[0] + cb[2]) / 2.0
                tile_cy = (cb[1] + cb[3]) / 2.0
                new_origin = cand.origin + tile_cx * cand.basis_u + tile_cy * cand.basis_v

                # Center the clipped outline at (0,0)
                from shapely.affinity import translate
                centered = translate(clipped, -tile_cx, -tile_cy)
                cb2 = centered.bounds

                tile_slab = Slab3D(
                    normal=cand.normal.copy(),
                    origin=new_origin,
                    thickness_mm=cand.thickness_mm,
                    width_mm=cb2[2] - cb2[0],
                    height_mm=cb2[3] - cb2[1],
                    basis_u=cand.basis_u.copy(),
                    basis_v=cand.basis_v.copy(),
                    outline_2d=centered,
                    material_key=cand.material_key,
                    source=cand.source,
                )
                result.append(tile_slab)
                y += tile_h
            x += tile_w

    if len(result) != len(candidates):
        logger.info("Split oversized: %d -> %d candidates", len(candidates), len(result))
    return result


# ─── Conversion ───────────────────────────────────────────────────────────────

def slab3d_to_part_profile(slab: Slab3D) -> PartProfile2D:
    """Convert a Slab3D to a PartProfile2D preserving the 3D basis."""
    return PartProfile2D(
        outline=slab.outline_2d,
        material_key=slab.material_key,
        thickness_mm=slab.thickness_mm,
        basis_u=slab.basis_u.copy(),
        basis_v=slab.basis_v.copy(),
        origin_3d=slab.origin.copy(),
    )
