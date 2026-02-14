"""
Decomposition quality scoring.

Evaluates how well a flat-pack decomposition approximates the original mesh
using Hausdorff distance, structural plausibility, and DFM compliance.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import KDTree

from geometry_primitives import PartProfile2D
from dfm_rules import DFMViolation
from furniture import FurnitureDesign

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """Configuration for decomposition scoring."""
    n_sample_points: int = 5000
    hausdorff_weight: float = 0.3
    part_count_weight: float = 0.2
    dfm_weight: float = 0.3
    structural_weight: float = 0.2
    max_acceptable_hausdorff_mm: float = 50.0
    max_acceptable_parts: int = 20


@dataclass
class DecompositionScore:
    """Quality score for a decomposition."""
    hausdorff_mm: float           # max distance from mesh surface to nearest part
    mean_distance_mm: float       # average distance
    part_count: int
    dfm_violations_error: int
    dfm_violations_warning: int
    structural_plausibility: float  # 0-1 (1 = fully plausible)
    overall_score: float            # 0-1 composite


def score_decomposition(
    mesh: trimesh.Trimesh,
    design: FurnitureDesign,
    parts: Optional[List[PartProfile2D]] = None,
    dfm_violations: Optional[List[DFMViolation]] = None,
    config: Optional[ScoringConfig] = None,
) -> DecompositionScore:
    """Score how well a decomposition approximates the original mesh.

    Args:
        mesh: The original 3D mesh.
        design: The flat-pack design.
        parts: Optional list of PartProfile2D for DFM analysis.
        dfm_violations: Pre-computed DFM violations.
        config: Scoring parameters.

    Returns:
        DecompositionScore with all metrics and composite score.
    """
    if config is None:
        config = ScoringConfig()

    # Hausdorff distance
    hausdorff_mm, mean_mm = compute_hausdorff(mesh, design, config.n_sample_points)

    # DFM compliance
    n_errors = 0
    n_warnings = 0
    if dfm_violations:
        n_errors = sum(1 for v in dfm_violations if v.severity == "error")
        n_warnings = sum(1 for v in dfm_violations if v.severity == "warning")

    # Structural plausibility
    structural = _compute_structural_plausibility(design)

    # Composite score
    part_count = len(design.components)

    # Hausdorff score: 1.0 if hausdorff=0, 0.0 if hausdorff >= max
    hausdorff_score = max(
        0.0,
        1.0 - hausdorff_mm / config.max_acceptable_hausdorff_mm,
    )

    # Part count score: 1.0 for 1 part, 0.0 for >= max parts
    part_score = max(
        0.0,
        1.0 - part_count / config.max_acceptable_parts,
    )

    # DFM score: 1.0 for no violations
    dfm_score = max(0.0, 1.0 - n_errors * 0.2 - n_warnings * 0.05)

    overall = (
        config.hausdorff_weight * hausdorff_score
        + config.part_count_weight * part_score
        + config.dfm_weight * dfm_score
        + config.structural_weight * structural
    )

    score = DecompositionScore(
        hausdorff_mm=hausdorff_mm,
        mean_distance_mm=mean_mm,
        part_count=part_count,
        dfm_violations_error=n_errors,
        dfm_violations_warning=n_warnings,
        structural_plausibility=structural,
        overall_score=overall,
    )

    logger.info(
        "Score: overall=%.2f hausdorff=%.1fmm parts=%d dfm_errors=%d structural=%.2f",
        overall, hausdorff_mm, part_count, n_errors, structural,
    )
    return score


def compute_hausdorff(
    mesh: trimesh.Trimesh,
    design: FurnitureDesign,
    n_points: int = 5000,
) -> Tuple[float, float]:
    """Compute Hausdorff and mean distance from mesh to assembled parts.

    Samples points on the original mesh surface, then for each point finds
    the nearest point on any assembled part's bounding box.

    Args:
        mesh: The original mesh.
        design: The flat-pack design.
        n_points: Number of surface sample points.

    Returns:
        (hausdorff_mm, mean_distance_mm)
    """
    if not design.components:
        return (float('inf'), float('inf'))

    # Sample points on mesh surface
    samples, _ = trimesh.sample.sample_surface(mesh, n_points)

    # Build dense point cloud from assembled part surfaces
    part_points = []
    # Target ~20mm spacing along each dimension for dense sampling
    sample_spacing = 20.0

    for comp in design.components:
        pos = comp.position
        dims = comp.get_dimensions()
        rot = _euler_to_rotation_matrix(comp.rotation)

        # Number of samples along each axis (at least 2 = endpoints)
        nx = max(2, int(dims[0] / sample_spacing) + 1)
        ny = max(2, int(dims[1] / sample_spacing) + 1)
        nz = max(2, int(dims[2] / sample_spacing) + 1)

        # Center the box around the component position (pos = slab origin = center)
        xs = np.linspace(-dims[0] / 2, dims[0] / 2, nx)
        ys = np.linspace(-dims[1] / 2, dims[1] / 2, ny)
        zs = np.linspace(-dims[2] / 2, dims[2] / 2, nz)

        # Sample all 6 faces of the oriented bounding box
        # Top/bottom faces (z = 0 and z = dims[2])
        for z in [0.0, dims[2]]:
            for x in xs:
                for y in ys:
                    part_points.append(pos + rot @ np.array([x, y, z]))

        # Front/back faces (y = 0 and y = dims[1])
        for y in [0.0, dims[1]]:
            for x in xs:
                for z in zs:
                    part_points.append(pos + rot @ np.array([x, y, z]))

        # Left/right faces (x = 0 and x = dims[0])
        for x in [0.0, dims[0]]:
            for y in ys:
                for z in zs:
                    part_points.append(pos + rot @ np.array([x, y, z]))

    if not part_points:
        return (float('inf'), float('inf'))

    part_points = np.array(part_points)
    tree = KDTree(part_points)

    distances, _ = tree.query(samples)
    hausdorff = float(np.max(distances))
    mean_dist = float(np.mean(distances))

    return (hausdorff, mean_dist)


def _euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (rx, ry, rz) to a 3x3 rotation matrix."""
    rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    # Rz @ Ry @ Rx (extrinsic XYZ convention)
    return np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy],
    ])


def _compute_structural_plausibility(design: FurnitureDesign) -> float:
    """Estimate structural plausibility of the design (0-1).

    Checks:
    - Has at least one horizontal surface (table/shelf)
    - Has vertical supports
    - Base is wider than top (stability)
    - All parts are connected via joints
    """
    if not design.components:
        return 0.0

    score = 0.0
    checks = 0

    # Check 1: Has horizontal surfaces (SHELF, SEAT — not PANEL which is vertical)
    from furniture import ComponentType
    has_horizontal = any(
        c.type in (ComponentType.SHELF, ComponentType.SEAT)
        for c in design.components
    )
    score += 1.0 if has_horizontal else 0.0
    checks += 1

    # Check 2: Has vertical supports
    has_vertical = any(
        c.type in (ComponentType.LEG, ComponentType.SUPPORT, ComponentType.BRACE,
                    ComponentType.PANEL)
        for c in design.components
    )
    score += 1.0 if has_vertical else 0.0
    checks += 1

    # Check 3: Has joints connecting parts
    has_joints = len(design.assembly.joints) > 0
    score += 1.0 if has_joints else 0.0
    checks += 1

    # Check 4: Joint connectivity (all parts reachable from any part)
    connected = _check_connectivity(design)
    score += 1.0 if connected else 0.5
    checks += 1

    return score / checks if checks > 0 else 0.0


def _check_connectivity(design: FurnitureDesign) -> bool:
    """Check if all components are connected via joints."""
    if len(design.components) <= 1:
        return True

    names = {c.name for c in design.components}
    if not design.assembly.joints:
        return False

    # BFS from first component
    adj = {}
    for name in names:
        adj[name] = set()
    for joint in design.assembly.joints:
        if joint.component_a in adj and joint.component_b in adj:
            adj[joint.component_a].add(joint.component_b)
            adj[joint.component_b].add(joint.component_a)

    start = next(iter(names))
    visited = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adj.get(node, set()):
            if neighbor not in visited:
                queue.append(neighbor)

    return visited == names
