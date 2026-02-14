"""
Joint geometry synthesizer.

Generates tab/slot, finger joint, through-bolt, and half-lap joint geometry
as Shapely polygon operations on PartProfile2D outlines. Adds dogbone relief
to all internal right-angle corners.
"""
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box, Point
from shapely.ops import unary_union

from geometry_primitives import PartProfile2D, PartFeature, FeatureType
from dfm_rules import DFMConfig, add_dogbone_relief
from furniture import Joint, JointType

logger = logging.getLogger(__name__)


@dataclass
class JointSynthesisConfig:
    """Parameters for joint geometry generation."""
    tab_width_mm: float = 30.0
    tab_depth_factor: float = 0.8        # fraction of mating thickness
    min_tabs_per_edge: int = 2
    max_tabs_per_edge: int = 8
    finger_joint_pitch_mm: float = 15.0
    slip_fit_clearance_mm: float = 0.254  # 0.010"
    add_dogbones: bool = True
    dogbone_radius_mm: float = 1.6        # 1/16"
    bolt_hole_diameter_mm: float = 6.5    # M6 clearance hole
    edge_inset_mm: float = 5.0            # inset joint edge from part boundary


@dataclass
class JointSpec:
    """Specification for a joint between two parts."""
    part_a_key: str
    part_b_key: str
    joint_type: JointType
    edge_start: Tuple[float, float]   # 2D start of intersection line on part A
    edge_end: Tuple[float, float]     # 2D end of intersection line on part A
    mating_thickness_mm: float        # thickness of the mating part


def synthesize_joints(
    parts: Dict[str, PartProfile2D],
    joint_specs: List[JointSpec],
    config: Optional[JointSynthesisConfig] = None,
    dfm_config: Optional[DFMConfig] = None,
) -> Tuple[Dict[str, PartProfile2D], List[Joint]]:
    """Apply joint geometry to parts.

    Modifies part outlines with tabs, slots, and other joint features.

    Args:
        parts: Dictionary of part key -> PartProfile2D.
        joint_specs: Joint specifications between part pairs.
        config: Joint geometry parameters.
        dfm_config: DFM constraints for dogbone sizing.

    Returns:
        (modified_parts, joints) - updated parts dict and Joint objects.
    """
    if config is None:
        config = JointSynthesisConfig()
    if dfm_config is None:
        dfm_config = DFMConfig()

    modified_parts = {k: _clone_part(v) for k, v in parts.items()}
    joints: List[Joint] = []

    for spec in joint_specs:
        if spec.part_a_key not in modified_parts or spec.part_b_key not in modified_parts:
            logger.warning(
                "Joint spec references missing parts: %s, %s",
                spec.part_a_key, spec.part_b_key,
            )
            continue

        part_a = modified_parts[spec.part_a_key]
        part_b = modified_parts[spec.part_b_key]

        if spec.joint_type == JointType.TAB_SLOT:
            _apply_tab_slot(part_a, part_b, spec, config)
        elif spec.joint_type == JointType.FINGER:
            _apply_finger_joint(part_a, part_b, spec, config)
        elif spec.joint_type == JointType.THROUGH_BOLT:
            _apply_through_bolt(part_a, part_b, spec, config)
        elif spec.joint_type == JointType.HALF_LAP:
            _apply_half_lap(part_a, part_b, spec, config)

        # Create Joint object with positions in each part's local frame
        edge_mid = (
            (spec.edge_start[0] + spec.edge_end[0]) / 2,
            (spec.edge_start[1] + spec.edge_end[1]) / 2,
        )
        position_a = (edge_mid[0], edge_mid[1], 0.0)

        # Compute position_b by converting edge midpoint from part_a's 3D
        # frame to part_b's 2D frame
        position_b = _compute_position_b(
            part_a, part_b, edge_mid,
        )

        joints.append(Joint(
            component_a=spec.part_a_key,
            component_b=spec.part_b_key,
            joint_type=spec.joint_type,
            position_a=position_a,
            position_b=position_b,
            parameters={
                "tab_width_mm": config.tab_width_mm,
                "clearance_mm": config.slip_fit_clearance_mm,
            },
        ))

    logger.info("Synthesized %d joints", len(joints))
    return modified_parts, joints


# ─── Tab-slot joints ─────────────────────────────────────────────────────────

def _apply_tab_slot(
    part_a: PartProfile2D,
    part_b: PartProfile2D,
    spec: JointSpec,
    config: JointSynthesisConfig,
) -> None:
    """Apply tab-slot joint: tabs on part_a, slots on part_b.

    Part A gets tab extensions along the edge.
    Part B gets slot cutouts matching the tabs with clearance.
    """
    edge_start = np.array(spec.edge_start)
    edge_end = np.array(spec.edge_end)
    edge_vec = edge_end - edge_start
    edge_len = float(np.linalg.norm(edge_vec))

    if edge_len < config.tab_width_mm:
        return

    edge_dir = edge_vec / edge_len

    # Inset the effective edge from part boundary to maintain bridge width
    if config.edge_inset_mm > 0 and edge_len > config.edge_inset_mm * 2 + config.tab_width_mm:
        edge_start = edge_start + edge_dir * config.edge_inset_mm
        edge_end = edge_end - edge_dir * config.edge_inset_mm
        edge_vec = edge_end - edge_start
        edge_len = float(np.linalg.norm(edge_vec))

    # Normal perpendicular to edge (into the mating part)
    edge_normal = np.array([-edge_dir[1], edge_dir[0]])

    # Determine number of tabs
    n_tabs = max(
        config.min_tabs_per_edge,
        min(config.max_tabs_per_edge, int(edge_len / (config.tab_width_mm * 2))),
    )

    tab_depth = spec.mating_thickness_mm * config.tab_depth_factor
    clearance = config.slip_fit_clearance_mm

    # Evenly space tabs along the edge
    spacing = edge_len / (n_tabs * 2 + 1)

    for i in range(n_tabs):
        # Tab center position along edge
        t = spacing * (2 * i + 1.5)
        tab_center = edge_start + edge_dir * t

        # Tab rectangle (on part A = extension outward)
        tab_hw = config.tab_width_mm / 2
        tab_corners = [
            tab_center - edge_dir * tab_hw,
            tab_center + edge_dir * tab_hw,
            tab_center + edge_dir * tab_hw + edge_normal * tab_depth,
            tab_center - edge_dir * tab_hw + edge_normal * tab_depth,
        ]
        tab_poly = Polygon([(c[0], c[1]) for c in tab_corners])

        # Add tab to part A (union with outline)
        part_a.outline = part_a.outline.union(tab_poly)
        part_a.features.append(PartFeature(
            feature_type=FeatureType.TAB,
            geometry=tab_poly,
            parameters={"width_mm": config.tab_width_mm, "depth_mm": tab_depth},
        ))

        # Slot rectangle (on part B = cutout with clearance)
        slot_hw = (config.tab_width_mm + 2 * clearance) / 2
        slot_depth = spec.mating_thickness_mm + 2 * clearance
        slot_corners = [
            tab_center - edge_dir * slot_hw,
            tab_center + edge_dir * slot_hw,
            tab_center + edge_dir * slot_hw + edge_normal * slot_depth,
            tab_center - edge_dir * slot_hw + edge_normal * slot_depth,
        ]
        slot_poly = Polygon([(c[0], c[1]) for c in slot_corners])

        # Add dogbone relief to slot corners
        if config.add_dogbones:
            slot_poly = add_dogbone_relief(slot_poly, config.dogbone_radius_mm)

        # Skip slot if it's too close to the part outline boundary
        outline = part_b.outline
        if isinstance(outline, MultiPolygon):
            outline = max(outline.geoms, key=lambda g: g.area)
        dist_to_edge = outline.exterior.distance(slot_poly)
        if dist_to_edge < config.edge_inset_mm * 0.5:
            # Check if the slot would leave insufficient bridge material
            # Only skip if the slot actually touches or nearly touches the boundary
            if dist_to_edge < 1.0:
                continue

        part_b.cutouts.append(slot_poly)
        part_b.features.append(PartFeature(
            feature_type=FeatureType.SLOT,
            geometry=slot_poly,
            parameters={"width_mm": float(slot_hw * 2), "depth_mm": slot_depth},
        ))


# ─── Finger joints ──────────────────────────────────────────────────────────

def _apply_finger_joint(
    part_a: PartProfile2D,
    part_b: PartProfile2D,
    spec: JointSpec,
    config: JointSynthesisConfig,
) -> None:
    """Apply finger joint: alternating fingers on both parts.

    Part A gets odd fingers, Part B gets even fingers.
    Both get matching slot cutouts.
    """
    edge_start = np.array(spec.edge_start)
    edge_end = np.array(spec.edge_end)
    edge_vec = edge_end - edge_start
    edge_len = float(np.linalg.norm(edge_vec))

    if edge_len < config.finger_joint_pitch_mm * 2:
        return

    edge_dir = edge_vec / edge_len

    # Inset the effective edge from part boundary to maintain bridge width
    if config.edge_inset_mm > 0 and edge_len > config.edge_inset_mm * 2 + config.finger_joint_pitch_mm * 2:
        edge_start = edge_start + edge_dir * config.edge_inset_mm
        edge_end = edge_end - edge_dir * config.edge_inset_mm
        edge_vec = edge_end - edge_start
        edge_len = float(np.linalg.norm(edge_vec))

    edge_normal = np.array([-edge_dir[1], edge_dir[0]])

    pitch = config.finger_joint_pitch_mm
    clearance = config.slip_fit_clearance_mm
    n_fingers = int(edge_len / pitch)
    finger_depth = spec.mating_thickness_mm * config.tab_depth_factor

    for i in range(n_fingers):
        t_start = i * pitch
        t_end = (i + 1) * pitch
        center = edge_start + edge_dir * (t_start + pitch / 2)

        finger_corners = [
            edge_start + edge_dir * t_start,
            edge_start + edge_dir * t_end,
            edge_start + edge_dir * t_end + edge_normal * finger_depth,
            edge_start + edge_dir * t_start + edge_normal * finger_depth,
        ]
        finger_poly = Polygon([(c[0], c[1]) for c in finger_corners])

        # Slot with clearance
        slot_corners = [
            edge_start + edge_dir * (t_start - clearance),
            edge_start + edge_dir * (t_end + clearance),
            edge_start + edge_dir * (t_end + clearance) + edge_normal * (finger_depth + clearance),
            edge_start + edge_dir * (t_start - clearance) + edge_normal * (finger_depth + clearance),
        ]
        slot_poly = Polygon([(c[0], c[1]) for c in slot_corners])

        if config.add_dogbones:
            slot_poly = add_dogbone_relief(slot_poly, config.dogbone_radius_mm)

        if i % 2 == 0:
            # Part A gets the finger, Part B gets the slot
            part_a.outline = part_a.outline.union(finger_poly)
            part_a.features.append(PartFeature(
                feature_type=FeatureType.TAB,
                geometry=finger_poly,
                parameters={"pitch_mm": pitch},
            ))
            part_b.cutouts.append(slot_poly)
            part_b.features.append(PartFeature(
                feature_type=FeatureType.FINGER_SLOT,
                geometry=slot_poly,
                parameters={"pitch_mm": pitch},
            ))
        else:
            # Part B gets the finger, Part A gets the slot
            part_b.outline = part_b.outline.union(finger_poly)
            part_b.features.append(PartFeature(
                feature_type=FeatureType.TAB,
                geometry=finger_poly,
                parameters={"pitch_mm": pitch},
            ))
            part_a.cutouts.append(slot_poly)
            part_a.features.append(PartFeature(
                feature_type=FeatureType.FINGER_SLOT,
                geometry=slot_poly,
                parameters={"pitch_mm": pitch},
            ))


# ─── Through-bolt ────────────────────────────────────────────────────────────

def _apply_through_bolt(
    part_a: PartProfile2D,
    part_b: PartProfile2D,
    spec: JointSpec,
    config: JointSynthesisConfig,
) -> None:
    """Apply through-bolt holes on both parts along the intersection edge."""
    edge_start = np.array(spec.edge_start)
    edge_end = np.array(spec.edge_end)
    edge_vec = edge_end - edge_start
    edge_len = float(np.linalg.norm(edge_vec))

    if edge_len < config.bolt_hole_diameter_mm * 3:
        return

    edge_dir = edge_vec / edge_len
    radius = config.bolt_hole_diameter_mm / 2

    # Place bolts near each end of the edge
    margin = config.bolt_hole_diameter_mm * 2
    bolt_positions = [margin, edge_len - margin]

    if edge_len > config.bolt_hole_diameter_mm * 8:
        bolt_positions.append(edge_len / 2)

    for t in bolt_positions:
        center = edge_start + edge_dir * t
        hole = Point(float(center[0]), float(center[1])).buffer(
            radius, quad_segs=8,
        )

        part_a.cutouts.append(hole)
        part_a.features.append(PartFeature(
            feature_type=FeatureType.BOLT_HOLE,
            geometry=hole,
            parameters={"diameter_mm": config.bolt_hole_diameter_mm},
        ))

        part_b.cutouts.append(hole)
        part_b.features.append(PartFeature(
            feature_type=FeatureType.BOLT_HOLE,
            geometry=hole,
            parameters={"diameter_mm": config.bolt_hole_diameter_mm},
        ))


# ─── Half-lap (for ribcage) ─────────────────────────────────────────────────

def _apply_half_lap(
    part_a: PartProfile2D,
    part_b: PartProfile2D,
    spec: JointSpec,
    config: JointSynthesisConfig,
) -> None:
    """Apply half-lap notches at crossing points for interlocking ribs.

    Part A gets a notch cut from the top, part B gets a notch from the bottom.
    Each notch is half the part height deep.
    """
    edge_start = np.array(spec.edge_start)
    edge_end = np.array(spec.edge_end)
    edge_mid = (edge_start + edge_end) / 2

    clearance = config.slip_fit_clearance_mm

    # For half-lap, the slot width = mating part thickness + clearance
    slot_width = spec.mating_thickness_mm + 2 * clearance

    # Get part heights from their outlines
    a_bounds = part_a.outline.bounds
    b_bounds = part_b.outline.bounds
    a_height = a_bounds[3] - a_bounds[1]
    b_height = b_bounds[3] - b_bounds[1]

    # Part A: notch from top (y = max down to mid)
    notch_a = box(
        edge_mid[0] - slot_width / 2,
        a_bounds[1] + a_height / 2,  # mid height
        edge_mid[0] + slot_width / 2,
        a_bounds[3] + clearance,  # top edge + clearance
    )

    # Part B: notch from bottom (y = min up to mid)
    notch_b = box(
        edge_mid[0] - slot_width / 2,
        b_bounds[1] - clearance,  # bottom edge - clearance
        edge_mid[0] + slot_width / 2,
        b_bounds[1] + b_height / 2,  # mid height
    )

    if config.add_dogbones:
        notch_a = add_dogbone_relief(notch_a, config.dogbone_radius_mm)
        notch_b = add_dogbone_relief(notch_b, config.dogbone_radius_mm)

    part_a.cutouts.append(notch_a)
    part_a.features.append(PartFeature(
        feature_type=FeatureType.SLOT,
        geometry=notch_a,
        parameters={"type": "half_lap", "depth_factor": 0.5},
    ))

    part_b.cutouts.append(notch_b)
    part_b.features.append(PartFeature(
        feature_type=FeatureType.SLOT,
        geometry=notch_b,
        parameters={"type": "half_lap", "depth_factor": 0.5},
    ))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_position_b(
    part_a: PartProfile2D,
    part_b: PartProfile2D,
    edge_mid_a: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Convert joint midpoint from part_a's 2D frame to part_b's 2D frame."""
    has_basis_a = (
        part_a.basis_u is not None
        and part_a.basis_v is not None
        and part_a.origin_3d is not None
    )
    has_basis_b = (
        part_b.basis_u is not None
        and part_b.basis_v is not None
        and part_b.origin_3d is not None
    )

    if has_basis_a and has_basis_b:
        # Convert part_a 2D → 3D → part_b 2D
        pt_3d = part_a.project_2d_to_3d(edge_mid_a[0], edge_mid_a[1])
        u_b, v_b = part_b.project_3d_to_2d(pt_3d)
        return (u_b, v_b, 0.0)

    # Fallback: use part_b's centroid
    centroid = part_b.outline.centroid
    return (centroid.x, centroid.y, 0.0)


def _clone_part(part: PartProfile2D) -> PartProfile2D:
    """Deep-copy a PartProfile2D."""
    return PartProfile2D(
        outline=Polygon(part.outline),
        cutouts=[Polygon(c) for c in part.cutouts],
        features=list(part.features),
        material_key=part.material_key,
        thickness_mm=part.thickness_mm,
        basis_u=part.basis_u.copy() if part.basis_u is not None else None,
        basis_v=part.basis_v.copy() if part.basis_v is not None else None,
        origin_3d=part.origin_3d.copy() if part.origin_3d is not None else None,
    )
