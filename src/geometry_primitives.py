"""
Core geometry types for manufacturing-aware decomposition.

Built on Shapely for 2D polygon operations. Provides PlanarPatch (from mesh
face extraction), PartProfile2D (CNC-ready 2D part with cutouts), and
conversions between Shapely and Component.profile formats.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity


class FeatureType(Enum):
    """Types of manufacturing features on a 2D part."""
    SLOT = "slot"
    TAB = "tab"
    FINGER_SLOT = "finger_slot"
    DOGBONE = "dogbone"
    BOLT_HOLE = "bolt_hole"
    ENGRAVE = "engrave"


@dataclass
class PartFeature:
    """A manufacturing feature on a 2D part profile."""
    feature_type: FeatureType
    geometry: shapely.Geometry  # Polygon for areas, LineString for engravings
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlanarPatch:
    """A group of coplanar mesh faces with a 2D boundary.

    Extracted by RANSAC + region growing from a 3D mesh.
    """
    plane_normal: np.ndarray        # (3,) unit normal
    plane_offset: float             # signed distance from origin (n . p = d)
    face_indices: List[int]         # indices into mesh.faces
    boundary_2d: Polygon            # 2D outline projected onto the patch's plane
    area_mm2: float
    centroid_3d: np.ndarray         # (3,) world-space centroid
    # 2D basis vectors used for projection (set by plane_extraction)
    basis_u: Optional[np.ndarray] = None  # (3,) first basis vector
    basis_v: Optional[np.ndarray] = None  # (3,) second basis vector


@dataclass
class PartProfile2D:
    """A CNC-ready 2D part with outline, cutouts, and features.

    This is the canonical representation for DFM checking, joint synthesis,
    and DXF/SVG export.
    """
    outline: Polygon                      # outer boundary
    cutouts: List[Polygon] = field(default_factory=list)   # slots, holes
    features: List[PartFeature] = field(default_factory=list)
    material_key: str = "plywood_baltic_birch"
    thickness_mm: float = 6.35
    # 2D basis vectors for converting between 3D and local 2D coordinates.
    # Set by plane_extraction when projecting faces to 2D.
    basis_u: Optional[np.ndarray] = None  # (3,) first basis vector
    basis_v: Optional[np.ndarray] = None  # (3,) second basis vector
    origin_3d: Optional[np.ndarray] = None  # (3,) origin point on the plane

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """Project a 3D point into this part's local 2D coordinate frame."""
        if self.basis_u is None or self.basis_v is None or self.origin_3d is None:
            raise ValueError("Part has no 2D basis vectors set")
        d = point_3d - self.origin_3d
        return (float(d @ self.basis_u), float(d @ self.basis_v))

    def project_2d_to_3d(self, u: float, v: float) -> np.ndarray:
        """Convert a local 2D coordinate back to 3D."""
        if self.basis_u is None or self.basis_v is None or self.origin_3d is None:
            raise ValueError("Part has no 2D basis vectors set")
        return self.origin_3d + u * self.basis_u + v * self.basis_v

    def net_polygon(self) -> Polygon:
        """Outline minus all cutouts."""
        result = self.outline
        for cutout in self.cutouts:
            result = result.difference(cutout)
        return result

    def validate_geometry(self) -> List[str]:
        """Check for geometry issues.

        Returns list of warning/error strings (empty = ok).
        """
        issues = []
        if not self.outline.is_valid:
            issues.append("Outline polygon is invalid")
        if self.outline.is_empty:
            issues.append("Outline polygon is empty")
        if self.outline.area < 1.0:
            issues.append(f"Outline area too small: {self.outline.area:.2f} mm2")
        for i, cutout in enumerate(self.cutouts):
            if not cutout.is_valid:
                issues.append(f"Cutout {i} is invalid")
            if not self.outline.contains(cutout):
                issues.append(f"Cutout {i} extends outside outline")
        return issues


# ─── Conversion functions ────────────────────────────────────────────────────

def project_faces_to_2d(
    mesh,
    face_indices: List[int],
    plane_normal: np.ndarray,
    plane_offset: float,
) -> Polygon:
    """Project mesh faces onto a plane and return 2D boundary polygon.

    Args:
        mesh: trimesh.Trimesh
        face_indices: indices of faces belonging to this patch
        plane_normal: unit normal of the plane
        plane_offset: signed distance from origin
    """
    polygon, _, _, _ = project_faces_to_2d_with_basis(
        mesh, face_indices, plane_normal, plane_offset,
    )
    return polygon


def project_faces_to_2d_with_basis(
    mesh,
    face_indices: List[int],
    plane_normal: np.ndarray,
    plane_offset: float,
) -> Tuple[Polygon, np.ndarray, np.ndarray, np.ndarray]:
    """Project mesh faces onto a plane and return 2D boundary + basis vectors.

    Args:
        mesh: trimesh.Trimesh
        face_indices: indices of faces belonging to this patch
        plane_normal: unit normal of the plane
        plane_offset: signed distance from origin

    Returns:
        (polygon, u_axis, v_axis, origin_3d)
    """
    n = plane_normal / np.linalg.norm(plane_normal)
    u_axis, v_axis = _make_2d_basis(n)

    # Compute a 3D origin on the plane
    origin_3d = n * plane_offset

    # Collect all triangle vertices projected to 2D
    triangles_3d = mesh.vertices[mesh.faces[face_indices]]  # (N, 3, 3)

    # Project each triangle vertex onto the u-v plane
    polygons_2d = []
    for tri in triangles_3d:
        pts_2d = []
        for pt in tri:
            pts_2d.append((float(pt @ u_axis), float(pt @ v_axis)))
        try:
            p = Polygon(pts_2d)
            if p.is_valid and p.area > 0:
                polygons_2d.append(p)
        except Exception:
            continue

    if not polygons_2d:
        return Polygon(), u_axis, v_axis, origin_3d

    merged = unary_union(polygons_2d)
    if isinstance(merged, MultiPolygon):
        # Return the largest polygon
        merged = max(merged.geoms, key=lambda g: g.area)

    # Simplify slightly to reduce vertex count
    return merged.simplify(0.5, preserve_topology=True), u_axis, v_axis, origin_3d


def polygon_to_profile(polygon) -> List[Tuple[float, float]]:
    """Convert a Shapely Polygon to Component.profile format (list of (x,y) tuples).

    If a MultiPolygon is passed, uses the largest polygon by area.
    Translates so min corner is at origin.
    """
    if polygon.is_empty:
        return []
    # Handle MultiPolygon by picking the largest sub-polygon
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    coords = list(polygon.exterior.coords[:-1])  # drop closing duplicate
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, min_y = min(xs), min(ys)
    return [(c[0] - min_x, c[1] - min_y) for c in coords]


def profile_to_polygon(profile: List[Tuple[float, float]]) -> Polygon:
    """Convert a Component.profile to a Shapely Polygon."""
    if len(profile) < 3:
        return Polygon()
    return Polygon(profile)


def compute_obb_2d(polygon: Polygon) -> Tuple[float, float, float]:
    """Compute 2D oriented bounding box.

    Returns:
        (width, height, rotation_rad) of the minimum-area bounding rectangle.
    """
    if polygon.is_empty:
        return (0.0, 0.0, 0.0)

    # Use Shapely's minimum_rotated_rectangle
    obb = polygon.minimum_rotated_rectangle
    coords = list(obb.exterior.coords)

    # Compute edge lengths
    edge1 = np.array(coords[1]) - np.array(coords[0])
    edge2 = np.array(coords[2]) - np.array(coords[1])

    len1 = float(np.linalg.norm(edge1))
    len2 = float(np.linalg.norm(edge2))

    # Width = longer edge, height = shorter edge
    if len1 >= len2:
        width, height = len1, len2
        angle = float(np.arctan2(edge1[1], edge1[0]))
    else:
        width, height = len2, len1
        angle = float(np.arctan2(edge2[1], edge2[0]))

    return (width, height, angle)


# ─── Internal helpers ────────────────────────────────────────────────────────

def _make_2d_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal (u, v) basis perpendicular to normal."""
    n = normal / np.linalg.norm(normal)
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v
