"""Emit LLM-friendly spatial capsule payloads from current ManufacturingPart objects."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import Polygon

from .contracts import SpatialCapsule, SpatialPart, SpatialRelation


class _SnapshotProfile:
    def __init__(
        self,
        outline: Polygon,
        cutouts: List[Polygon],
        basis_u: Optional[np.ndarray],
        basis_v: Optional[np.ndarray],
        origin_3d: Optional[np.ndarray],
    ) -> None:
        self.outline = outline
        self.cutouts = cutouts
        self.basis_u = basis_u
        self.basis_v = basis_v
        self.origin_3d = origin_3d


class _SnapshotPart:
    def __init__(
        self,
        part_id: str,
        thickness_mm: float,
        position_3d: np.ndarray,
        rotation_3d: np.ndarray,
        profile: _SnapshotProfile,
    ) -> None:
        self.part_id = part_id
        self.thickness_mm = thickness_mm
        self.position_3d = position_3d
        self.rotation_3d = rotation_3d
        self.profile = profile


def emit_spatial_capsule_from_manufacturing_parts(
    parts: Sequence[Any],
    joint_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    contact_tolerance_mm: float = 0.5,
    overlap_tolerance_mm: float = 0.01,
) -> SpatialCapsule:
    """Build a spatial capsule from current `ManufacturingPart`-like objects.

    Required part fields:
    - `part_id`, `thickness_mm`, `position_3d`, `rotation_3d`, `profile`
    Required profile fields:
    - `outline`
    Optional profile fields:
    - `basis_u`, `basis_v`, `origin_3d`
    """
    normalized_joint_pairs: Set[Tuple[str, str]] = set()
    if joint_pairs is not None:
        for a, b in joint_pairs:
            aa = str(a).strip()
            bb = str(b).strip()
            if not aa or not bb or aa == bb:
                continue
            normalized_joint_pairs.add(tuple(sorted((aa, bb))))

    spatial_parts: List[SpatialPart] = []
    for part in sorted(parts, key=lambda p: str(getattr(p, "part_id", ""))):
        spatial_parts.append(_spatial_part_from_part(part))

    relations: List[SpatialRelation] = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pa = parts[i]
            pb = parts[j]
            pair_key = tuple(sorted((str(pa.part_id), str(pb.part_id))))
            relation = _relation_from_pair(
                pa,
                pb,
                pair_is_jointed=pair_key in normalized_joint_pairs,
                contact_tolerance_mm=contact_tolerance_mm,
                overlap_tolerance_mm=overlap_tolerance_mm,
            )
            relations.append(relation)

    capsule = SpatialCapsule(parts=spatial_parts, relations=relations)
    capsule.validate()
    return capsule


def emit_spatial_capsule_from_snapshot_payload(
    snapshot_payload: Dict[str, Any],
    contact_tolerance_mm: float = 0.5,
    overlap_tolerance_mm: float = 0.01,
) -> SpatialCapsule:
    """Build a spatial capsule from a phase snapshot payload.

    Expected shape:
    - `parts[*]`: `part_id`, `thickness_mm`, `outline_2d`, `cutouts_2d`,
      `position_3d`, `rotation_3d`, optional `origin_3d`
    - `joints[*]`: optional `part_a`, `part_b`
    """
    raw_parts = snapshot_payload.get("parts", [])
    if not isinstance(raw_parts, list):
        raw_parts = []
    adapters: List[_SnapshotPart] = []
    for part in raw_parts:
        if not isinstance(part, dict):
            continue
        adapter = _snapshot_part_adapter(part)
        if adapter is None:
            continue
        adapters.append(adapter)

    raw_joints = snapshot_payload.get("joints", [])
    if not isinstance(raw_joints, list):
        raw_joints = []
    joint_pairs: List[Tuple[str, str]] = []
    for joint in raw_joints:
        if not isinstance(joint, dict):
            continue
        part_a = str(joint.get("part_a", "")).strip()
        part_b = str(joint.get("part_b", "")).strip()
        if not part_a or not part_b or part_a == part_b:
            continue
        joint_pairs.append((part_a, part_b))

    return emit_spatial_capsule_from_manufacturing_parts(
        adapters,
        joint_pairs=joint_pairs,
        contact_tolerance_mm=contact_tolerance_mm,
        overlap_tolerance_mm=overlap_tolerance_mm,
    )


def write_spatial_capsule_json(path: str | Path, capsule: SpatialCapsule) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = capsule.to_dict()
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _snapshot_part_adapter(payload: Dict[str, Any]) -> Optional[_SnapshotPart]:
    part_id = str(payload.get("part_id", "")).strip()
    if not part_id:
        return None
    outline_2d = payload.get("outline_2d", [])
    if not isinstance(outline_2d, list) or len(outline_2d) < 3:
        return None
    try:
        shell = [(float(pt[0]), float(pt[1])) for pt in outline_2d]
    except Exception:
        return None
    if len(shell) > 1 and np.allclose(shell[0], shell[-1]):
        shell = shell[:-1]
    if len(shell) < 3:
        return None
    try:
        outline = Polygon(shell)
    except Exception:
        return None
    if outline.is_empty:
        return None
    outline = outline if outline.is_valid else outline.buffer(0)
    if outline.is_empty or not isinstance(outline, Polygon):
        return None

    cutouts_raw = payload.get("cutouts_2d", [])
    cutouts: List[Polygon] = []
    if isinstance(cutouts_raw, list):
        for ring in cutouts_raw:
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            try:
                coords = [(float(pt[0]), float(pt[1])) for pt in ring]
            except Exception:
                continue
            if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            if len(coords) < 3:
                continue
            try:
                poly = Polygon(coords)
            except Exception:
                continue
            if poly.is_empty:
                continue
            poly = poly if poly.is_valid else poly.buffer(0)
            if isinstance(poly, Polygon) and not poly.is_empty and poly.area > 1e-9:
                cutouts.append(poly)

    origin = payload.get("origin_3d")
    position = payload.get("position_3d")
    if origin is None:
        origin = position
    try:
        origin_3d = np.asarray(origin, dtype=float)
        if origin_3d.shape != (3,):
            origin_3d = None
    except Exception:
        origin_3d = None

    rotation = payload.get("rotation_3d", [0.0, 0.0, 0.0])
    try:
        rotation_3d = np.asarray(rotation, dtype=float)
        if rotation_3d.shape != (3,):
            rotation_3d = np.zeros(3, dtype=float)
    except Exception:
        rotation_3d = np.zeros(3, dtype=float)

    try:
        position_3d = np.asarray(position, dtype=float)
        if position_3d.shape != (3,):
            position_3d = None
    except Exception:
        position_3d = None

    if origin_3d is None:
        origin_3d = np.zeros(3, dtype=float)
    if position_3d is None:
        position_3d = origin_3d.copy()

    rot = _rotation_matrix_xyz(rotation_3d)
    profile = _SnapshotProfile(
        outline=outline,
        cutouts=cutouts,
        basis_u=rot[:, 0],
        basis_v=rot[:, 1],
        origin_3d=origin_3d,
    )
    return _SnapshotPart(
        part_id=part_id,
        thickness_mm=float(payload.get("thickness_mm", 0.0) or 0.0),
        position_3d=position_3d,
        rotation_3d=rotation_3d,
        profile=profile,
    )


def _spatial_part_from_part(part: Any) -> SpatialPart:
    profile = part.profile
    rot = _rotation_matrix_xyz(np.asarray(part.rotation_3d, dtype=float))
    n_axis = rot[:, 2] / max(np.linalg.norm(rot[:, 2]), 1e-8)

    basis_u = (
        np.asarray(profile.basis_u, dtype=float)
        if getattr(profile, "basis_u", None) is not None
        else rot[:, 0]
    )
    basis_v = (
        np.asarray(profile.basis_v, dtype=float)
        if getattr(profile, "basis_v", None) is not None
        else rot[:, 1]
    )
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-8)
    basis_v = basis_v / max(np.linalg.norm(basis_v), 1e-8)

    origin_3d = (
        np.asarray(profile.origin_3d, dtype=float)
        if getattr(profile, "origin_3d", None) is not None
        else np.asarray(part.position_3d, dtype=float)
    )

    outline_coords = np.asarray(profile.outline.exterior.coords, dtype=float)
    if len(outline_coords) > 1 and np.allclose(outline_coords[0], outline_coords[-1]):
        outline_coords = outline_coords[:-1]
    outline_2d = [[float(p[0]), float(p[1])] for p in outline_coords]

    holes_2d: List[List[List[float]]] = []
    for ring in profile.outline.interiors:
        ring_coords = np.asarray(ring.coords, dtype=float)
        if len(ring_coords) > 1 and np.allclose(ring_coords[0], ring_coords[-1]):
            ring_coords = ring_coords[:-1]
        holes_2d.append([[float(p[0]), float(p[1])] for p in ring_coords])
    for cutout in getattr(profile, "cutouts", []):
        ring_coords = np.asarray(cutout.exterior.coords, dtype=float)
        if len(ring_coords) > 1 and np.allclose(ring_coords[0], ring_coords[-1]):
            ring_coords = ring_coords[:-1]
        holes_2d.append([[float(p[0]), float(p[1])] for p in ring_coords])

    bounds = profile.outline.bounds
    min_x, min_y, max_x, max_y = [float(v) for v in bounds]
    half_u = max(0.0, 0.5 * (max_x - min_x))
    half_v = max(0.0, 0.5 * (max_y - min_y))
    half_n = max(0.0, 0.5 * float(part.thickness_mm))
    local_center = np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.0])
    center_world = origin_3d + local_center[0] * basis_u + local_center[1] * basis_v
    center_world = center_world + half_n * n_axis

    obb = {
        "center": [float(v) for v in center_world],
        "axes": [
            [float(v) for v in basis_u],
            [float(v) for v in basis_v],
            [float(v) for v in n_axis],
        ],
        "half_extents": [float(half_u), float(half_v), float(half_n)],
    }

    return SpatialPart(
        part_id=str(part.part_id),
        thickness_mm=float(part.thickness_mm),
        origin_3d=[float(v) for v in origin_3d],
        basis_u=[float(v) for v in basis_u],
        basis_v=[float(v) for v in basis_v],
        outline_2d=outline_2d,
        holes_2d=holes_2d,
        obb=obb,
    )


def _relation_from_pair(
    part_a: Any,
    part_b: Any,
    pair_is_jointed: bool,
    contact_tolerance_mm: float,
    overlap_tolerance_mm: float,
) -> SpatialRelation:
    ca, aa, ha = _part_obb(part_a)
    cb, ab, hb = _part_obb(part_b)
    signed_gap = _obb_max_separation(ca, aa, ha, cb, ab, hb)
    penetration_mm = _obb_penetration_depth(ca, aa, ha, cb, ab, hb)

    if penetration_mm > max(0.0, overlap_tolerance_mm):
        relation_class = "overlapping"
    elif pair_is_jointed and signed_gap <= max(0.0, contact_tolerance_mm):
        relation_class = "jointed"
    elif signed_gap <= max(0.0, contact_tolerance_mm):
        relation_class = "touching"
    else:
        relation_class = "disjoint"

    return SpatialRelation(
        part_a=str(part_a.part_id),
        part_b=str(part_b.part_id),
        relation_class=relation_class,
        penetration_mm=float(penetration_mm),
        contact_area_mm2=0.0,
    )


def _part_obb(part: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _rotation_matrix_xyz(np.asarray(part.rotation_3d, dtype=float))
    bounds = part.profile.outline.bounds
    min_x, min_y, max_x, max_y = [float(v) for v in bounds]
    half = np.array(
        [
            max(0.1, 0.5 * (max_x - min_x)),
            max(0.1, 0.5 * (max_y - min_y)),
            max(0.1, 0.5 * float(part.thickness_mm)),
        ],
        dtype=float,
    )
    center_local = np.array([0.5 * (min_x + max_x), 0.5 * (min_y + max_y), 0.0])
    center_world = rot @ center_local + np.asarray(part.position_3d, dtype=float)
    axes = np.column_stack(
        [
            rot[:, 0] / max(np.linalg.norm(rot[:, 0]), 1e-8),
            rot[:, 1] / max(np.linalg.norm(rot[:, 1]), 1e-8),
            rot[:, 2] / max(np.linalg.norm(rot[:, 2]), 1e-8),
        ]
    )
    return center_world, axes, half


def _rotation_matrix_xyz(rotation_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(v) for v in rotation_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _obb_max_separation(
    ca: np.ndarray,
    aa: np.ndarray,
    ha: np.ndarray,
    cb: np.ndarray,
    ab: np.ndarray,
    hb: np.ndarray,
) -> float:
    eps = 1e-8
    r = aa.T @ ab
    abs_r = np.abs(r) + eps
    t_world = cb - ca
    t = aa.T @ t_world
    max_sep = -float("inf")

    for i in range(3):
        ra = ha[i]
        rb = hb[0] * abs_r[i, 0] + hb[1] * abs_r[i, 1] + hb[2] * abs_r[i, 2]
        sep = abs(t[i]) - (ra + rb)
        if sep > max_sep:
            max_sep = sep

    for j in range(3):
        ra = ha[0] * abs_r[0, j] + ha[1] * abs_r[1, j] + ha[2] * abs_r[2, j]
        rb = hb[j]
        proj = abs(t[0] * r[0, j] + t[1] * r[1, j] + t[2] * r[2, j])
        sep = proj - (ra + rb)
        if sep > max_sep:
            max_sep = sep

    for i in range(3):
        for j in range(3):
            axis_norm = np.linalg.norm(np.cross(aa[:, i], ab[:, j]))
            if axis_norm < 1e-6:
                continue
            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            ra = ha[i1] * abs_r[i2, j] + ha[i2] * abs_r[i1, j]
            rb = hb[j1] * abs_r[i, j2] + hb[j2] * abs_r[i, j1]
            proj = abs(t[i2] * r[i1, j] - t[i1] * r[i2, j])
            sep = proj - (ra + rb)
            if sep > max_sep:
                max_sep = sep

    return float(max_sep)


def _obb_penetration_depth(
    ca: np.ndarray,
    aa: np.ndarray,
    ha: np.ndarray,
    cb: np.ndarray,
    ab: np.ndarray,
    hb: np.ndarray,
) -> float:
    """Approximate SAT penetration depth in mm (0 when disjoint)."""
    eps = 1e-8
    r = aa.T @ ab
    abs_r = np.abs(r) + eps
    t_world = cb - ca
    t = aa.T @ t_world
    min_overlap = float("inf")

    for i in range(3):
        ra = ha[i]
        rb = hb[0] * abs_r[i, 0] + hb[1] * abs_r[i, 1] + hb[2] * abs_r[i, 2]
        overlap = (ra + rb) - abs(t[i])
        if overlap < 0.0:
            return 0.0
        min_overlap = min(min_overlap, overlap)

    for j in range(3):
        ra = ha[0] * abs_r[0, j] + ha[1] * abs_r[1, j] + ha[2] * abs_r[2, j]
        rb = hb[j]
        proj = abs(t[0] * r[0, j] + t[1] * r[1, j] + t[2] * r[2, j])
        overlap = (ra + rb) - proj
        if overlap < 0.0:
            return 0.0
        min_overlap = min(min_overlap, overlap)

    for i in range(3):
        for j in range(3):
            axis_norm = np.linalg.norm(np.cross(aa[:, i], ab[:, j]))
            if axis_norm < 1e-6:
                continue
            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            ra = ha[i1] * abs_r[i2, j] + ha[i2] * abs_r[i1, j]
            rb = hb[j1] * abs_r[i, j2] + hb[j2] * abs_r[i, j1]
            proj = abs(t[i2] * r[i1, j] - t[i1] * r[i2, j])
            overlap = (ra + rb) - proj
            if overlap < 0.0:
                return 0.0
            min_overlap = min(min_overlap, overlap)

    if min_overlap == float("inf"):
        return 0.0
    return float(max(0.0, min_overlap))
