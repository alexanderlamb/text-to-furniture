"""
Mesh-to-flat-pack decomposition.

Takes 3D meshes (from AI image-to-mesh tools like TripoSR, InstantMesh, etc.)
and decomposes them into flat rectangular components that can be manufactured
by SendCutSend. Outputs a standard FurnitureDesign.

Algorithm:
1. Load & repair mesh (handles non-watertight AI meshes)
2. Cluster face normals to find dominant orientations
3. Generate candidate slabs along each orientation
4. Greedy set-cover selection using voxel coverage
5. Optional local optimization
6. Infer joints from slab overlaps
7. Convert to FurnitureDesign
"""

import math

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

import trimesh
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from furniture import (
    Component, ComponentType, Joint, JointType,
    AssemblyGraph, FurnitureDesign,
)
from materials import MATERIALS, Material, MIN_OVERLAP_MM
from mesh_cleanup import (
    MeshCleanupConfig,
    clean_mesh_geometry,
    simplify_mesh_geometry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DecompositionConfig:
    """Configuration for mesh decomposition."""
    default_material: str = "plywood_baltic_birch"
    target_height_mm: float = 750.0
    auto_scale: bool = True
    max_slabs: int = 15
    min_slab_dimension_mm: float = 50.0
    coverage_target: float = 0.80
    normal_cluster_threshold_deg: float = 15.0
    voxel_resolution_mm: float = 5.0
    optimize_iterations: int = 50
    min_coverage_contribution: float = 0.05
    selection_objective_mode: str = "volume_fill"  # UI pass-through for v2
    target_volume_fill: float = 0.55
    min_volume_contribution: float = 0.002
    plane_penalty_weight: float = 0.0005
    mesh_cleanup: MeshCleanupConfig = field(default_factory=MeshCleanupConfig)


@dataclass
class SlabCandidate:
    """A candidate rectangular slab to approximate part of the mesh."""
    normal: np.ndarray        # Unit normal vector (slab's thin direction)
    position: float           # Position along the normal axis (center of thickness)
    thickness: float          # SCS material thickness in mm
    width: float              # Slab dimension perpendicular to normal (axis 1)
    height: float             # Slab dimension perpendicular to normal (axis 2)
    material_key: str
    rotation: np.ndarray      # (rx, ry, rz) Euler angles
    origin: np.ndarray        # (x, y, z) world position of slab center
    coverage_score: float = 0.0

    def get_corners(self) -> np.ndarray:
        """Get the 8 corners of this slab in world space.

        Returns:
            (8, 3) array of corner positions.
        """
        # Build local half-extents along the slab's local axes.
        # Local frame: u=width axis, v=height axis, n=normal (thickness) axis.
        u_axis, v_axis = _slab_local_axes(self.normal)
        hw = self.width / 2.0
        hh = self.height / 2.0
        ht = self.thickness / 2.0

        corners = []
        for su in (-1, 1):
            for sv in (-1, 1):
                for sn in (-1, 1):
                    pt = (
                        self.origin
                        + u_axis * (su * hw)
                        + v_axis * (sv * hh)
                        + self.normal * (sn * ht)
                    )
                    corners.append(pt)
        return np.array(corners)

    def get_aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box in world space.

        Returns:
            (min_corner, max_corner) each shape (3,).
        """
        corners = self.get_corners()
        return corners.min(axis=0), corners.max(axis=0)


# =============================================================================
# Helper: local axes for a slab given its normal
# =============================================================================

def _slab_local_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal axes perpendicular to *normal*.

    These define the slab's width (u) and height (v) directions.
    """
    n = normal / np.linalg.norm(normal)
    # Pick a reference vector not parallel to n
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


# =============================================================================
# Step 1: Load & Prepare Mesh
# =============================================================================

def load_mesh(filepath: str, config: DecompositionConfig) -> trimesh.Trimesh:
    """Load a mesh file and repair it for decomposition.

    Handles STL, OBJ, GLB, PLY.  For GLB scenes, extracts the largest
    mesh by face count.  Applies AI-mesh repairs (merge vertices, fix
    normals, fill holes).  Scales to *config.target_height_mm* and
    centres at the origin with bottom at z=0.
    """
    scene_or_mesh = trimesh.load(filepath)

    # Extract single mesh from scenes / multi-body files
    # Use scene.to_mesh() to flatten scene-level transforms (rotation,
    # non-uniform scale) so the geometry matches the viewer path.
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_mesh()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            # Fallback: extract largest geometry without scene transforms
            meshes = [
                g for g in scene_or_mesh.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError(f"No triangle meshes found in {filepath}")
            mesh = max(meshes, key=lambda m: len(m.faces))
        else:
            # Detect up-axis using same heuristic as viewer (argmin of extents)
            extents = mesh.bounds[1] - mesh.bounds[0]
            up_axis = int(np.argmin(extents))
            if up_axis == 1:
                # Y-up → rotate +90° around X to make Z-up
                rot = trimesh.transformations.rotation_matrix(
                    math.pi / 2, [1, 0, 0])
                mesh.apply_transform(rot)
            elif up_axis == 0:
                # X-up → rotate -90° around Y to make Z-up
                rot = trimesh.transformations.rotation_matrix(
                    -math.pi / 2, [0, 1, 0])
                mesh.apply_transform(rot)
            # up_axis == 2 → already Z-up, no rotation needed
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        raise ValueError(
            f"Unsupported type from trimesh.load: {type(scene_or_mesh)}"
        )

    # --- Repair for AI-generated meshes ---
    mesh.merge_vertices()
    mesh.fix_normals()
    mesh.fill_holes()

    # --- Scale to target height ---
    if config.auto_scale:
        bounds = mesh.bounds  # (2, 3) min / max
        current_height = bounds[1][2] - bounds[0][2]
        if current_height > 0:
            scale_factor = config.target_height_mm / current_height
            mesh.apply_scale(scale_factor)

    # --- Centre at origin, bottom at z=0 ---
    bounds = mesh.bounds
    centroid_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    translation = np.array([
        -centroid_xy[0],
        -centroid_xy[1],
        -bounds[0][2],
    ])
    mesh.apply_translation(translation)

    preprocess = {
        "cleanup_enabled": bool(config.mesh_cleanup.enabled),
        "cleanup_before_faces": int(len(mesh.faces)),
        "cleanup_after_faces": int(len(mesh.faces)),
    }
    if config.mesh_cleanup.enabled:
        before_faces = len(mesh.faces)
        mesh = clean_mesh_geometry(mesh, config.mesh_cleanup)
        preprocess["cleanup_before_faces"] = int(before_faces)
        preprocess["cleanup_after_faces"] = int(len(mesh.faces))
        logger.info(
            "Mesh cleanup applied: %d -> %d faces",
            before_faces,
            len(mesh.faces),
        )

    mesh, simplify_stats = simplify_mesh_geometry(mesh, config.mesh_cleanup)
    preprocess["simplify"] = simplify_stats
    if simplify_stats.get("mode") not in {"disabled", "skip_small", "target_zero", "no_gain"}:
        logger.info(
            "Mesh simplification (%s): %d -> %d faces "
            "(bbox drift %.4f, normal drift %.2fdeg%s)",
            simplify_stats.get("mode"),
            int(simplify_stats.get("before_faces", 0)),
            int(simplify_stats.get("after_faces", 0)),
            float(simplify_stats.get("bbox_drift_ratio", 0.0)),
            float(simplify_stats.get("normal_change_deg", 0.0)),
            ", reverted" if simplify_stats.get("guard_reverted") else "",
        )

    mesh.metadata = dict(getattr(mesh, "metadata", {}) or {})
    mesh.metadata["preprocess"] = preprocess

    logger.info(
        "Loaded mesh: %d vertices, %d faces, bounds %s",
        len(mesh.vertices), len(mesh.faces), mesh.bounds.tolist(),
    )
    return mesh


# =============================================================================
# Step 2: Find Dominant Orientations
# =============================================================================

def find_dominant_orientations(
    mesh: trimesh.Trimesh,
    config: DecompositionConfig,
) -> List[Tuple[np.ndarray, float]]:
    """Cluster face normals (weighted by area) into dominant orientations.

    Returns list of (unit_normal, total_area) sorted descending by area.
    Opposite normals are merged (a panel seen from front/back is one
    orientation).  The three cardinal axes are always included as fallbacks.
    """
    normals = mesh.face_normals.copy()  # (N, 3)
    areas = mesh.area_faces.copy()      # (N,)

    # Canonicalise: flip normals so they all point into the same hemisphere.
    # Convention: the component with largest absolute value should be positive.
    for i in range(len(normals)):
        dominant_axis = np.argmax(np.abs(normals[i]))
        if normals[i][dominant_axis] < 0:
            normals[i] = -normals[i]

    # Cluster using angular distance
    threshold_rad = np.radians(config.normal_cluster_threshold_deg)

    if len(normals) == 0:
        # Degenerate mesh — return cardinal axes only
        return [
            (np.array([1.0, 0, 0]), 0.0),
            (np.array([0, 1.0, 0]), 0.0),
            (np.array([0, 0, 1.0]), 0.0),
        ]

    # Pairwise angular distances
    # Using cosine distance (1 - cos(angle)) which is monotonic with angle
    # for angles in [0, pi].  threshold_rad -> 1 - cos(threshold_rad).
    cos_threshold = 1.0 - np.cos(threshold_rad)

    # For very large meshes, sub-sample to keep linkage tractable
    max_faces_for_clustering = 50_000
    if len(normals) > max_faces_for_clustering:
        indices = np.random.choice(
            len(normals), max_faces_for_clustering, replace=False,
        )
        sample_normals = normals[indices]
        sample_areas = areas[indices]
    else:
        sample_normals = normals
        sample_areas = areas
        indices = np.arange(len(normals))

    # Compute condensed distance matrix (cosine)
    dists = pdist(sample_normals, metric="cosine")
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=cos_threshold, criterion="distance")

    # Aggregate clusters
    cluster_map: Dict[int, Tuple[np.ndarray, float]] = {}
    for idx, label in enumerate(labels):
        n = sample_normals[idx]
        a = sample_areas[idx]
        if label not in cluster_map:
            cluster_map[label] = (n * a, a)
        else:
            weighted_n, total_a = cluster_map[label]
            cluster_map[label] = (weighted_n + n * a, total_a + a)

    orientations: List[Tuple[np.ndarray, float]] = []
    for weighted_n, total_a in cluster_map.values():
        centroid = weighted_n / np.linalg.norm(weighted_n)
        orientations.append((centroid, total_a))

    # Sort by area descending
    orientations.sort(key=lambda x: x[1], reverse=True)

    # Ensure cardinal axes are present
    cardinals = [
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
    ]
    for c in cardinals:
        already = any(
            np.dot(o[0], c) > np.cos(threshold_rad) for o in orientations
        )
        if not already:
            orientations.append((c, 0.0))

    logger.info("Found %d dominant orientations", len(orientations))
    return orientations


# =============================================================================
# Step 3: Voxelize for Coverage Tracking
# =============================================================================

class VoxelGrid:
    """Thin wrapper around a trimesh VoxelGrid for coverage scoring."""

    def __init__(self, mesh: trimesh.Trimesh, resolution_mm: float):
        self._vox = mesh.voxelized(pitch=resolution_mm)
        self._vox.fill()
        self.pitch = resolution_mm
        filled = self._vox.matrix.copy()                # (I, J, K) bool
        self.origin = self._vox.transform[:3, 3].copy() # (3,) world origin

        # Use only a surface shell (hollow) rather than the solid interior.
        # This ensures thin surface slabs cover a meaningful fraction of
        # the target voxels.
        self.matrix = self._extract_shell(filled)
        self.covered = np.zeros_like(self.matrix, dtype=bool)
        self.total_filled = int(self.matrix.sum())

    @staticmethod
    def _extract_shell(filled: np.ndarray) -> np.ndarray:
        """Extract surface shell from a filled voxel grid.

        A voxel is on the shell if it is filled and has at least one
        empty neighbor along any axis.
        """
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(filled)
        return filled & ~eroded

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.matrix.shape

    def coverage_fraction(self) -> float:
        if self.total_filled == 0:
            return 1.0
        return float((self.matrix & self.covered).sum()) / self.total_filled

    def _world_to_ijk(self, pts: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel grid indices (float)."""
        return (pts - self.origin) / self.pitch

    def count_new_coverage(self, slab: SlabCandidate) -> int:
        """Count how many currently-uncovered filled voxels this slab covers."""
        mask = self._slab_voxel_mask(slab)
        return int((mask & self.matrix & ~self.covered).sum())

    def mark_covered(self, slab: SlabCandidate):
        """Mark voxels covered by *slab* as covered."""
        mask = self._slab_voxel_mask(slab)
        self.covered |= (mask & self.matrix)

    def _slab_voxel_mask(self, slab: SlabCandidate) -> np.ndarray:
        """Boolean mask of voxels inside the slab's OBB."""
        # Build the slab's local frame
        u_axis, v_axis = _slab_local_axes(slab.normal)
        n_axis = slab.normal / np.linalg.norm(slab.normal)
        hw, hh = slab.width / 2.0, slab.height / 2.0
        # Use at least one voxel pitch for thickness so thin slabs still
        # capture voxels at the surface.
        ht = max(slab.thickness / 2.0, self.pitch)

        # Get all voxel centres
        ii, jj, kk = np.where(self.matrix)
        if len(ii) == 0:
            return np.zeros_like(self.matrix, dtype=bool)

        centres = self.origin + np.column_stack([ii, jj, kk]) * self.pitch + self.pitch / 2.0

        # Project onto slab axes relative to slab centre
        diff = centres - slab.origin  # (N, 3)
        proj_u = diff @ u_axis
        proj_v = diff @ v_axis
        proj_n = diff @ n_axis

        inside = (
            (np.abs(proj_u) <= hw)
            & (np.abs(proj_v) <= hh)
            & (np.abs(proj_n) <= ht)
        )

        mask = np.zeros_like(self.matrix, dtype=bool)
        mask[ii[inside], jj[inside], kk[inside]] = True
        return mask


def voxelize(mesh: trimesh.Trimesh, config: DecompositionConfig) -> VoxelGrid:
    """Create a voxel grid from the mesh for coverage scoring."""
    vg = VoxelGrid(mesh, config.voxel_resolution_mm)
    logger.info(
        "Voxelized: shape=%s, filled=%d voxels",
        vg.shape, vg.total_filled,
    )
    return vg


# =============================================================================
# Step 4: Normal → Euler Rotation
# =============================================================================

def normal_to_rotation(normal: np.ndarray) -> np.ndarray:
    """Convert a unit normal vector to (rx, ry, rz) Euler angles.

    The convention matches Component.rotation: the slab's thin (thickness)
    direction aligns with the normal *after* applying the rotation.

    For an axis-aligned normal the rotation is zero on the corresponding
    axes; for arbitrary normals we decompose into Euler XYZ.
    """
    n = normal / np.linalg.norm(normal)

    # We want the rotation R such that R @ [0, 0, 1] = n  (Z is thickness).
    # Construct the rotation matrix from two axes.
    u, v = _slab_local_axes(n)
    # Rotation matrix columns = new x, y, z in world frame
    R = np.column_stack([u, v, n])  # 3x3

    # Extract Euler angles for R = Rz @ Ry @ Rx convention
    # (matches _rotation_matrix_xyz in workers.py).
    # R[2,0] = -sin(ry)
    ry = np.arcsin(np.clip(-R[2, 0], -1, 1))
    if np.abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rz = 0.0
        if R[2, 0] < 0:  # ry ≈ +π/2
            rx = np.arctan2(R[0, 1], R[1, 1])
        else:             # ry ≈ -π/2
            rx = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([rx, ry, rz])


# =============================================================================
# Step 5: Generate Candidate Slabs
# =============================================================================

def generate_candidates(
    mesh: trimesh.Trimesh,
    orientations: List[Tuple[np.ndarray, float]],
    config: DecompositionConfig,
) -> List[SlabCandidate]:
    """Generate candidate slabs for each dominant orientation.

    For each orientation (normal n):
    1. Project *all* mesh vertices onto the plane perpendicular to n to
       determine slab width/height (cross-section extent).
    2. Cluster face depths along n to find distinct surface positions
       (e.g. top & bottom of a box are two distinct positions).
    3. For each surface cluster and each SCS thickness, create a slab
       candidate positioned at the surface (inside the mesh).
    4. Reject slabs smaller than min_slab_dimension_mm.
    """
    material = MATERIALS[config.default_material]
    thicknesses = material.thicknesses_mm
    max_w, max_h = material.max_size_mm
    threshold_rad = np.radians(config.normal_cluster_threshold_deg)

    face_normals = mesh.face_normals
    face_centres = mesh.triangles_center

    # Project ALL vertices for cross-section sizing
    vertices = mesh.vertices  # (V, 3)

    candidates: List[SlabCandidate] = []

    for orientation_normal, _area in orientations:
        n = orientation_normal / np.linalg.norm(orientation_normal)

        # Build local 2D frame
        u_axis, v_axis = _slab_local_axes(n)

        # Cross-section extent from all vertices
        proj_u = vertices @ u_axis
        proj_v = vertices @ v_axis
        slab_width = float(proj_u.max() - proj_u.min())
        slab_height = float(proj_v.max() - proj_v.min())
        u_centre = float((proj_u.max() + proj_u.min()) / 2.0)
        v_centre = float((proj_v.max() + proj_v.min()) / 2.0)

        # Clamp to material sheet size
        slab_width = min(slab_width, max_w)
        slab_height = min(slab_height, max_h)

        # Reject tiny slabs
        if (slab_width < config.min_slab_dimension_mm
                or slab_height < config.min_slab_dimension_mm):
            continue

        rotation = normal_to_rotation(n)

        # Find faces whose normal is close to this orientation (or opposite)
        dots = face_normals @ n
        similar = np.abs(dots) > np.cos(threshold_rad)
        if not similar.any():
            # No matching faces; place candidates at mesh depth extremes
            all_depths = vertices @ n
            surface_positions = [float(all_depths.min()), float(all_depths.max())]
        else:
            # Cluster the depths of matching face centres to find distinct
            # surface positions (e.g. top vs bottom of a box).
            similar_centres = face_centres[similar]
            depths = similar_centres @ n
            surface_positions = _cluster_depths(
                depths, min_gap=max(thicknesses) * 2,
            )

        # Also include the overall mesh depth extremes along n
        all_depths = vertices @ n
        mesh_depth_min = float(all_depths.min())
        mesh_depth_max = float(all_depths.max())
        for d in [mesh_depth_min, mesh_depth_max]:
            if not any(abs(d - sp) < max(thicknesses) for sp in surface_positions):
                surface_positions.append(d)

        # Create candidates at each surface position × each thickness
        for surface_depth in surface_positions:
            for thickness in thicknesses:
                # Position the slab just inside the mesh from the surface.
                # Determine which side "inside" is:
                # if this surface is closer to mesh_depth_max, slab goes inward (-)
                # if closer to mesh_depth_min, slab goes inward (+)
                mid_depth = (mesh_depth_min + mesh_depth_max) / 2.0
                if surface_depth >= mid_depth:
                    pos = surface_depth - thickness / 2.0
                else:
                    pos = surface_depth + thickness / 2.0

                origin = (
                    n * pos
                    + u_axis * u_centre
                    + v_axis * v_centre
                )

                candidates.append(SlabCandidate(
                    normal=n.copy(),
                    position=pos,
                    thickness=thickness,
                    width=slab_width,
                    height=slab_height,
                    material_key=config.default_material,
                    rotation=rotation.copy(),
                    origin=origin.copy(),
                ))

    logger.info("Generated %d candidate slabs", len(candidates))
    return candidates


def _cluster_depths(depths: np.ndarray, min_gap: float) -> List[float]:
    """Cluster 1D depth values into distinct surface positions.

    Returns the mean depth of each cluster, sorted ascending.
    """
    sorted_depths = np.sort(depths)
    clusters: List[List[float]] = [[sorted_depths[0]]]

    for d in sorted_depths[1:]:
        if d - clusters[-1][-1] > min_gap:
            clusters.append([d])
        else:
            clusters[-1].append(d)

    return sorted([float(np.mean(c)) for c in clusters])


# =============================================================================
# Step 6: Greedy Slab Selection
# =============================================================================

def select_slabs(
    candidates: List[SlabCandidate],
    voxel_grid: VoxelGrid,
    config: DecompositionConfig,
) -> List[SlabCandidate]:
    """Greedy set-cover: pick slabs that maximise new voxel coverage.

    Stops when coverage_target is reached, max_slabs hit, or no
    candidate covers enough new volume.
    """
    selected: List[SlabCandidate] = []
    remaining = list(candidates)

    for _ in range(config.max_slabs):
        if voxel_grid.coverage_fraction() >= config.coverage_target:
            break

        # Score every remaining candidate
        best_score = -1
        best_idx = -1
        for i, cand in enumerate(remaining):
            score = voxel_grid.count_new_coverage(cand)
            cand.coverage_score = score
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        # Check minimum coverage contribution
        fraction = best_score / max(voxel_grid.total_filled, 1)
        if fraction < config.min_coverage_contribution:
            break

        chosen = remaining.pop(best_idx)
        voxel_grid.mark_covered(chosen)
        selected.append(chosen)

        logger.info(
            "Selected slab %d: %.1f x %.1f x %.1f mm, "
            "new coverage %d voxels (%.1f%% total)",
            len(selected),
            chosen.width, chosen.height, chosen.thickness,
            best_score,
            voxel_grid.coverage_fraction() * 100,
        )

    logger.info(
        "Selected %d slabs, final coverage %.1f%%",
        len(selected),
        voxel_grid.coverage_fraction() * 100,
    )
    return selected


# =============================================================================
# Step 7: Local Optimization (optional)
# =============================================================================

def optimize_arrangement(
    slabs: List[SlabCandidate],
    mesh: trimesh.Trimesh,
    voxel_grid: VoxelGrid,
    config: DecompositionConfig,
) -> List[SlabCandidate]:
    """Random perturbation optimiser: shift, resize, or remove slabs.

    Accepts a move only if coverage improves or stays the same.
    """
    material = MATERIALS[config.default_material]
    thicknesses = material.thicknesses_mm
    max_w, max_h = material.max_size_mm
    rng = np.random.default_rng()

    def _recompute_coverage(trial_slabs: List[SlabCandidate]) -> float:
        vg_trial = VoxelGrid(mesh, config.voxel_resolution_mm)
        for s in trial_slabs:
            vg_trial.mark_covered(s)
        return vg_trial.coverage_fraction()

    best_coverage = _recompute_coverage(slabs)
    best_slabs = [_copy_slab(s) for s in slabs]

    for it in range(config.optimize_iterations):
        trial = [_copy_slab(s) for s in best_slabs]
        if not trial:
            break

        idx = rng.integers(0, len(trial))
        move = rng.choice(["shift", "resize", "thickness", "remove"])

        if move == "shift":
            delta = rng.uniform(-20.0, 20.0)
            trial[idx].origin = trial[idx].origin + trial[idx].normal * delta
            trial[idx].position += delta
        elif move == "resize":
            factor = rng.uniform(0.9, 1.1)
            if rng.random() < 0.5:
                trial[idx].width = np.clip(
                    trial[idx].width * factor,
                    config.min_slab_dimension_mm, max_w,
                )
            else:
                trial[idx].height = np.clip(
                    trial[idx].height * factor,
                    config.min_slab_dimension_mm, max_h,
                )
        elif move == "thickness":
            trial[idx].thickness = float(rng.choice(thicknesses))
        elif move == "remove" and len(trial) > 1:
            trial.pop(idx)

        cov = _recompute_coverage(trial)
        if cov >= best_coverage:
            best_coverage = cov
            best_slabs = trial

    logger.info(
        "Optimization: %d iterations, coverage %.1f%% -> %.1f%%",
        config.optimize_iterations,
        voxel_grid.coverage_fraction() * 100,
        best_coverage * 100,
    )
    return best_slabs


def _copy_slab(s: SlabCandidate) -> SlabCandidate:
    return SlabCandidate(
        normal=s.normal.copy(),
        position=s.position,
        thickness=s.thickness,
        width=s.width,
        height=s.height,
        material_key=s.material_key,
        rotation=s.rotation.copy(),
        origin=s.origin.copy(),
        coverage_score=s.coverage_score,
    )


# =============================================================================
# Step 8: Infer Joints
# =============================================================================

def infer_joints(slabs: List[SlabCandidate]) -> List[Joint]:
    """Detect joints between pairs of slabs based on bounding-box overlap.

    Joint type heuristic:
    - Perpendicular slabs with edge inside the other -> TAB_SLOT
    - Perpendicular slabs with edge-to-face contact  -> BUTT
    - Parallel slabs stacked with overlap             -> THROUGH_BOLT
    """
    joints: List[Joint] = []

    for i in range(len(slabs)):
        for j in range(i + 1, len(slabs)):
            a, b = slabs[i], slabs[j]
            joint = _check_joint(a, b, i, j)
            if joint is not None:
                joints.append(joint)

    logger.info("Inferred %d joints", len(joints))
    return joints


def _check_joint(
    a: SlabCandidate,
    b: SlabCandidate,
    idx_a: int,
    idx_b: int,
) -> Optional[Joint]:
    """Check whether two slabs are close enough to form a joint.

    Uses proximity-based detection: expands AABBs by a tolerance so that
    adjacent surface panels (which may not volumetrically overlap) still
    produce joints.
    """
    # Tolerance: slabs within this distance are considered adjacent.
    # Use a generous proximity so surface panels at opposite edges of
    # a large mesh still form joints even when clamped to sheet size.
    proximity_mm = max(MIN_OVERLAP_MM, max(a.width, a.height, b.width, b.height) * 0.15)

    a_min, a_max = a.get_aabb()
    b_min, b_max = b.get_aabb()

    # Expand both AABBs by the proximity tolerance
    a_min_exp = a_min - proximity_mm
    a_max_exp = a_max + proximity_mm
    b_min_exp = b_min - proximity_mm
    b_max_exp = b_max + proximity_mm

    # Overlap of expanded AABBs
    overlap = np.minimum(a_max_exp, b_max_exp) - np.maximum(a_min_exp, b_min_exp)
    if np.any(overlap < 0):
        return None

    # Require significant overlap extent in at least 2 dimensions
    # (after expansion, check against original MIN_OVERLAP_MM)
    significant = overlap >= MIN_OVERLAP_MM
    if significant.sum() < 2:
        return None

    # Determine angle between normals -> joint type
    dot = abs(float(np.dot(a.normal, b.normal)))
    if dot < 0.3:
        # Roughly perpendicular
        overlap_along_a = abs(float(np.dot(
            b.origin - a.origin, a.normal
        )))
        if overlap_along_a < a.width / 2.0 and overlap_along_a < a.height / 2.0:
            jtype = JointType.TAB_SLOT
        else:
            jtype = JointType.BUTT
    elif dot > 0.85:
        jtype = JointType.THROUGH_BOLT
    else:
        jtype = JointType.BUTT

    name_a = f"slab_{idx_a}"
    name_b = f"slab_{idx_b}"

    # Joint position: midpoint between slab centres
    mid = (a.origin + b.origin) / 2.0
    pos_a = tuple((mid - a.origin).tolist())
    pos_b = tuple((mid - b.origin).tolist())

    return Joint(
        component_a=name_a,
        component_b=name_b,
        joint_type=jtype,
        position_a=pos_a,
        position_b=pos_b,
    )


# =============================================================================
# Step 9: Build FurnitureDesign
# =============================================================================

def _classify_component(slab: SlabCandidate) -> ComponentType:
    """Classify a slab by its orientation."""
    n = slab.normal / np.linalg.norm(slab.normal)
    # If normal is mostly vertical (Z), the slab is horizontal → PANEL/SHELF
    if abs(n[2]) > 0.7:
        return ComponentType.SHELF if slab.width > 200 and slab.height > 200 else ComponentType.PANEL
    # If normal is mostly horizontal → vertical panel
    return ComponentType.SUPPORT if max(slab.width, slab.height) > 300 else ComponentType.PANEL


def build_design(
    slabs: List[SlabCandidate],
    joints: List[Joint],
    name: str = "decomposed",
) -> FurnitureDesign:
    """Convert selected slabs and joints into a FurnitureDesign."""
    design = FurnitureDesign(name=name)
    assembly = AssemblyGraph(joints=list(joints))

    for i, slab in enumerate(slabs):
        comp_type = _classify_component(slab)
        profile = [
            (0.0, 0.0),
            (slab.width, 0.0),
            (slab.width, slab.height),
            (0.0, slab.height),
        ]
        comp = Component(
            name=f"slab_{i}",
            type=comp_type,
            profile=profile,
            thickness=slab.thickness,
            position=slab.origin.copy(),
            rotation=slab.rotation.copy(),
            material=slab.material_key,
        )
        design.add_component(comp)

    design.assembly = assembly

    # Determine assembly order from joints (simple: order of appearance)
    for joint in joints:
        assembly.assembly_order.append((joint.component_a, joint.component_b))

    return design


# =============================================================================
# Orchestrator
# =============================================================================

def decompose(
    filepath: str,
    config: Optional[DecompositionConfig] = None,
    optimize: bool = True,
) -> FurnitureDesign:
    """Full mesh-to-flat-pack decomposition pipeline.

    Args:
        filepath: Path to mesh file (STL, OBJ, GLB, PLY).
        config: Decomposition parameters.  Uses defaults if None.
        optimize: Run optional local optimization pass.

    Returns:
        FurnitureDesign with rectangular components and inferred joints.
    """
    if config is None:
        config = DecompositionConfig()

    # 1. Load & prepare
    mesh = load_mesh(filepath, config)

    # 2. Find dominant orientations
    orientations = find_dominant_orientations(mesh, config)

    # 3. Voxelize
    vgrid = voxelize(mesh, config)

    # 4. Generate candidate slabs
    candidates = generate_candidates(mesh, orientations, config)
    if not candidates:
        logger.warning("No valid candidates generated — returning empty design")
        return FurnitureDesign(name="decomposed")

    # 5. Greedy selection
    selected = select_slabs(candidates, vgrid, config)

    # 6. Optional optimization
    if optimize and config.optimize_iterations > 0 and selected:
        selected = optimize_arrangement(selected, mesh, vgrid, config)

    # 7. Infer joints
    joints = infer_joints(selected)

    # 8. Build design
    design = build_design(selected, joints)
    design.metadata["coverage"] = vgrid.coverage_fraction()
    preprocess = {}
    if isinstance(mesh.metadata, dict):
        preprocess = mesh.metadata.get("preprocess", {}) or {}
    simplify_stats = preprocess.get("simplify", {}) or {}
    design.metadata["decomposition_debug"] = {
        "mesh_bounds_mm": mesh.bounds.tolist(),
        "mesh_extents_mm": (mesh.bounds[1] - mesh.bounds[0]).tolist(),
        "voxel_resolution_mm": float(config.voxel_resolution_mm),
        "voxel_shape": [int(v) for v in vgrid.shape],
        "voxel_filled_shell_count": int(vgrid.total_filled),
        "dominant_orientation_count": len(orientations),
        "dominant_orientations": [
            {
                "normal": [float(x) for x in normal.tolist()],
                "area_mm2": float(area),
            }
            for normal, area in orientations[:12]
        ],
        "candidate_slab_count": len(candidates),
        "selected_slab_count": len(selected),
        "cleanup_enabled": bool(preprocess.get("cleanup_enabled", False)),
        "cleanup_before_faces": preprocess.get("cleanup_before_faces"),
        "cleanup_after_faces": preprocess.get("cleanup_after_faces"),
        "simplify_enabled": bool(simplify_stats.get("enabled", False)),
        "simplify_mode": simplify_stats.get("mode"),
        "simplify_before_faces": simplify_stats.get("before_faces"),
        "simplify_after_faces": simplify_stats.get("after_faces"),
        "simplify_guard_reverted": bool(simplify_stats.get("guard_reverted", False)),
        "simplify_bbox_drift_ratio": simplify_stats.get("bbox_drift_ratio"),
        "simplify_normal_change_deg": simplify_stats.get("normal_change_deg"),
    }

    is_valid, errors = design.validate()
    if not is_valid:
        logger.warning("Design validation issues: %s", errors)

    logger.info(
        "Decomposition complete: %d components, %d joints",
        len(design.components), len(design.assembly.joints),
    )
    return design
