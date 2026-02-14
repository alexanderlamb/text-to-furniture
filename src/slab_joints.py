"""
3D joint detection and 2D projection for v2 manufacturing pipeline.

Detects joints between selected slabs entirely in 3D, then projects
to each slab's 2D frame only at the end. This avoids the v1 bug of
comparing values from incompatible 2D frames.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from slab_candidates import Slab3D
from furniture import JointType
from joint_synthesizer import JointSpec

logger = logging.getLogger(__name__)


@dataclass
class SlabIntersection3D:
    """A detected intersection between two slabs, computed in 3D."""
    slab_a_idx: int
    slab_b_idx: int
    line_origin_3d: np.ndarray    # point on intersection line
    line_dir_3d: np.ndarray       # direction of intersection line
    edge_start_3d: np.ndarray     # clipped start point
    edge_end_3d: np.ndarray       # clipped end point
    joint_type: JointType
    edge_on_a: Tuple[Tuple[float, float], Tuple[float, float]]  # in slab_a's 2D
    edge_on_b: Tuple[Tuple[float, float], Tuple[float, float]]  # in slab_b's 2D


def detect_joints(
    slabs: List[Slab3D],
    proximity_mm: float = 25.0,
    min_segment_mm: float = 15.0,
) -> List[SlabIntersection3D]:
    """Detect joints between all pairs of slabs using 3D geometry.

    For each pair:
    1. Check 3D AABB proximity
    2. Compute plane-plane intersection line
    3. Clip to both slabs' extents
    4. Reject segments shorter than min_segment_mm
    5. Classify joint type from normal dot product
    6. Project endpoints into each slab's 2D frame

    Args:
        slabs: Selected slabs.
        proximity_mm: AABB expansion for proximity check.
        min_segment_mm: Minimum intersection segment length to form a joint.

    Returns:
        List of SlabIntersection3D.
    """
    intersections: List[SlabIntersection3D] = []

    for i in range(len(slabs)):
        for j in range(i + 1, len(slabs)):
            result = _check_pair(slabs[i], slabs[j], i, j, proximity_mm,
                                 min_segment_mm)
            if result is not None:
                intersections.append(result)

    logger.info("Detected %d slab intersections", len(intersections))
    return intersections


def intersections_to_joint_specs(
    intersections: List[SlabIntersection3D],
    slabs: List[Slab3D],
) -> List[JointSpec]:
    """Convert SlabIntersection3D objects to JointSpec for joint_synthesizer.

    The edge_start/edge_end are in slab_a's 2D frame, and
    mating_thickness is slab_b's thickness.

    Args:
        intersections: Detected 3D intersections.
        slabs: The slab list (for thickness lookup).

    Returns:
        List of JointSpec ready for synthesize_joints().
    """
    specs: List[JointSpec] = []

    for ix in intersections:
        part_a_key = f"part_{ix.slab_a_idx}"
        part_b_key = f"part_{ix.slab_b_idx}"

        specs.append(JointSpec(
            part_a_key=part_a_key,
            part_b_key=part_b_key,
            joint_type=ix.joint_type,
            edge_start=ix.edge_on_a[0],
            edge_end=ix.edge_on_a[1],
            mating_thickness_mm=slabs[ix.slab_b_idx].thickness_mm,
        ))

    return specs


# ─── Internal ─────────────────────────────────────────────────────────────────

def _get_slab_aabb(slab: Slab3D) -> Tuple[np.ndarray, np.ndarray]:
    """Compute axis-aligned bounding box of a slab in world space."""
    n = slab.normal / np.linalg.norm(slab.normal)
    hw = slab.width_mm / 2.0
    hh = slab.height_mm / 2.0
    ht = slab.thickness_mm / 2.0

    corners = []
    for su in (-1, 1):
        for sv in (-1, 1):
            for sn in (-1, 1):
                pt = (
                    slab.origin
                    + slab.basis_u * (su * hw)
                    + slab.basis_v * (sv * hh)
                    + n * (sn * ht)
                )
                corners.append(pt)

    corners = np.array(corners)
    return corners.min(axis=0), corners.max(axis=0)


def _check_pair(
    slab_a: Slab3D,
    slab_b: Slab3D,
    idx_a: int,
    idx_b: int,
    proximity_mm: float,
    min_segment_mm: float = 15.0,
) -> Optional[SlabIntersection3D]:
    """Check whether two slabs form a joint."""
    # 1. AABB proximity check
    a_min, a_max = _get_slab_aabb(slab_a)
    b_min, b_max = _get_slab_aabb(slab_b)

    # Expand AABBs
    a_min_exp = a_min - proximity_mm
    a_max_exp = a_max + proximity_mm
    b_min_exp = b_min - proximity_mm
    b_max_exp = b_max + proximity_mm

    overlap = np.minimum(a_max_exp, b_max_exp) - np.maximum(a_min_exp, b_min_exp)
    if np.any(overlap < 0):
        return None

    na = slab_a.normal / np.linalg.norm(slab_a.normal)
    nb = slab_b.normal / np.linalg.norm(slab_b.normal)

    # 2. Classify joint type from normal dot product
    dot = abs(float(np.dot(na, nb)))

    if dot < 0.3:
        joint_type = JointType.TAB_SLOT
    elif dot > 0.85:
        joint_type = JointType.THROUGH_BOLT
    else:
        joint_type = JointType.FINGER

    # 3. Compute plane-plane intersection line
    line_dir = np.cross(na, nb)
    line_dir_norm = np.linalg.norm(line_dir)

    if line_dir_norm < 1e-8:
        # Near-parallel planes — use THROUGH_BOLT with fallback edge
        joint_type = JointType.THROUGH_BOLT
        return _parallel_joint(slab_a, slab_b, idx_a, idx_b, joint_type)

    line_dir = line_dir / line_dir_norm

    # Find a point on the intersection line via lstsq
    d_a = float(np.dot(na, slab_a.origin))
    d_b = float(np.dot(nb, slab_b.origin))
    A_mat = np.vstack([na, nb])
    b_vec = np.array([d_a, d_b])
    line_point, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # 4. Clip intersection line to both slabs' extents
    t_min_a, t_max_a = _clip_to_slab(slab_a, line_point, line_dir)
    t_min_b, t_max_b = _clip_to_slab(slab_b, line_point, line_dir)

    t_min = max(t_min_a, t_min_b)
    t_max = min(t_max_a, t_max_b)

    if t_max <= t_min:
        return None

    edge_start_3d = line_point + line_dir * t_min
    edge_end_3d = line_point + line_dir * t_max

    # Reject very short intersection segments
    segment_length = float(np.linalg.norm(edge_end_3d - edge_start_3d))
    if segment_length < min_segment_mm:
        return None

    # 5. Project into each slab's 2D frame
    edge_on_a = (
        slab_a.world_to_uv(edge_start_3d),
        slab_a.world_to_uv(edge_end_3d),
    )
    edge_on_b = (
        slab_b.world_to_uv(edge_start_3d),
        slab_b.world_to_uv(edge_end_3d),
    )

    return SlabIntersection3D(
        slab_a_idx=idx_a,
        slab_b_idx=idx_b,
        line_origin_3d=line_point,
        line_dir_3d=line_dir,
        edge_start_3d=edge_start_3d,
        edge_end_3d=edge_end_3d,
        joint_type=joint_type,
        edge_on_a=edge_on_a,
        edge_on_b=edge_on_b,
    )


def _clip_to_slab(
    slab: Slab3D,
    line_point: np.ndarray,
    line_dir: np.ndarray,
) -> Tuple[float, float]:
    """Clip an infinite line to a slab's OBB extent. Returns (t_min, t_max)."""
    n = slab.normal / np.linalg.norm(slab.normal)
    hw = slab.width_mm / 2.0
    hh = slab.height_mm / 2.0
    ht = slab.thickness_mm / 2.0

    # Project slab corners onto the line
    corners = []
    for su in (-1, 1):
        for sv in (-1, 1):
            for sn in (-1, 1):
                pt = (
                    slab.origin
                    + slab.basis_u * (su * hw)
                    + slab.basis_v * (sv * hh)
                    + n * (sn * ht)
                )
                corners.append(pt)

    t_values = [float(np.dot(c - line_point, line_dir)) for c in corners]
    return min(t_values), max(t_values)


def _project_to_slab_2d(
    slab: Slab3D,
    point_3d: np.ndarray,
) -> Tuple[float, float]:
    """Project a 3D point into a slab's local 2D (u,v) frame.

    Delegates to Slab3D.world_to_uv(). Kept for backward compatibility.
    """
    return slab.world_to_uv(point_3d)


def _parallel_joint(
    slab_a: Slab3D,
    slab_b: Slab3D,
    idx_a: int,
    idx_b: int,
    joint_type: JointType,
) -> SlabIntersection3D:
    """Create a joint between near-parallel slabs."""
    # Use the midpoint between origins as the joint location
    mid_3d = (slab_a.origin + slab_b.origin) / 2.0

    # Create an edge along slab_a's u-axis through the midpoint
    proj_a = slab_a.world_to_uv(mid_3d)
    proj_b = slab_b.world_to_uv(mid_3d)

    # Edge along u direction in slab_a's frame
    bounds_a = slab_a.outline_2d.bounds
    edge_len = (bounds_a[2] - bounds_a[0]) / 2 if not slab_a.outline_2d.is_empty else 50.0
    edge_on_a = (
        (proj_a[0] - edge_len / 2, proj_a[1]),
        (proj_a[0] + edge_len / 2, proj_a[1]),
    )

    bounds_b = slab_b.outline_2d.bounds
    edge_len_b = (bounds_b[2] - bounds_b[0]) / 2 if not slab_b.outline_2d.is_empty else 50.0
    edge_on_b = (
        (proj_b[0] - edge_len_b / 2, proj_b[1]),
        (proj_b[0] + edge_len_b / 2, proj_b[1]),
    )

    edge_start_3d = slab_a.origin + slab_a.basis_u * (-edge_len / 2)
    edge_end_3d = slab_a.origin + slab_a.basis_u * (edge_len / 2)

    return SlabIntersection3D(
        slab_a_idx=idx_a,
        slab_b_idx=idx_b,
        line_origin_3d=mid_3d,
        line_dir_3d=slab_a.basis_u.copy(),
        edge_start_3d=edge_start_3d,
        edge_end_3d=edge_end_3d,
        joint_type=joint_type,
        edge_on_a=edge_on_a,
        edge_on_b=edge_on_b,
    )
