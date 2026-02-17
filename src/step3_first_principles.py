"""First-principles Step 3: single-strategy decomposition for rapid iteration."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from dfm_rules import DFMConfig, DFMViolation, add_dogbone_relief, check_part_dfm
from furniture import (
    Component,
    ComponentType,
    FurnitureDesign,
    Joint,
    JointType,
)
from geometry_primitives import PartProfile2D, polygon_to_profile
from materials import MATERIALS

logger = logging.getLogger(__name__)


class RegionType(Enum):
    PLANAR_CUT = "planar_cut"
    BENDABLE_SHEET = "bendable_sheet"


@dataclass
class BendCapability:
    enabled: bool = True
    allowed_materials: List[str] = field(
        default_factory=lambda: ["mild_steel", "aluminum_5052", "stainless_304"]
    )
    min_radius_multiplier: float = 1.5
    max_bend_angle_deg: float = 135.0
    min_bend_length_mm: float = 20.0


@dataclass
class CapabilityProfile:
    profile_name: str
    supported_materials: List[str]
    min_feature_mm: float
    min_internal_radius_mm: float
    min_bridge_width_mm: float
    max_sheet_width_mm: float
    max_sheet_height_mm: float
    default_kerf_mm: float
    controlled_bending: BendCapability

    def supports_material(self, material_key: str) -> bool:
        return material_key in self.supported_materials

    def supports_bending(self, material_key: str) -> bool:
        return self.controlled_bending.enabled and (
            material_key in self.controlled_bending.allowed_materials
        )


@dataclass
class Step3Input:
    mesh_path: str
    design_name: str = "design"
    fidelity_weight: float = 0.75
    part_budget_max: int = 10
    material_preferences: List[str] = field(
        default_factory=lambda: ["plywood_baltic_birch"]
    )
    scs_capabilities: CapabilityProfile = field(
        default_factory=lambda: build_default_capability_profile()
    )
    target_height_mm: float = 750.0
    auto_scale: bool = True
    min_planar_region_area_mm2: float = 500.0
    min_bend_region_area_mm2: float = 1600.0
    max_bend_regions: int = 4
    joint_distance_mm: float = 240.0
    joint_contact_tolerance_mm: float = 2.0
    joint_parallel_dot_threshold: float = 0.95
    enable_planar_stacking: bool = True
    max_stack_layers_per_region: int = 4
    stack_roundup_bias: float = 0.35
    stack_extra_layer_gain: float = 0.65
    enable_thin_side_suppression: bool = True
    thin_side_dim_multiplier: float = 1.10
    thin_side_aspect_limit: float = 0.25
    thin_side_coverage_penalty_start: float = 0.40
    thin_side_coverage_drop_threshold: float = 0.62
    enable_intersection_filter: bool = True
    allow_joint_intent_intersections: bool = True
    intersection_clearance_mm: float = 0.5
    mesh_clip_min_area_ratio: float = 0.25
    joint_enable_geometry: bool = True
    joint_tab_spacing_mm: float = 80.0
    joint_tab_length_mm: float = 20.0
    joint_min_contact_mm: float = 15.0
    joint_fit_validation: bool = True  # enable joint fit + assembly checks


@dataclass
class BendOp:
    line: Tuple[Tuple[float, float], Tuple[float, float]]
    angle_deg: float
    radius_mm: float
    direction: str
    sequence_index: int


@dataclass
class JointSpec:
    joint_type: str
    part_a: str
    part_b: str
    geometry: Dict[str, float]
    clearance_mm: float
    fastener_spec: Optional[str] = None


@dataclass
class ManufacturingPart:
    part_id: str
    material_key: str
    thickness_mm: float
    profile: PartProfile2D
    region_type: RegionType
    position_3d: np.ndarray
    rotation_3d: np.ndarray
    source_area_mm2: float
    source_faces: List[int] = field(default_factory=list)
    bend_ops: List[BendOp] = field(default_factory=list)
    metadata: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class Step3Violation:
    code: str
    severity: str
    message: str
    part_id: Optional[str] = None


@dataclass
class Step3QualityMetrics:
    hausdorff_mm: float
    mean_distance_mm: float
    normal_error_deg: float
    connectivity_score: float
    overlap_score: float
    violation_score: float
    part_count: int
    overall_score: float


@dataclass
class Step3Output:
    design: FurnitureDesign
    parts: Dict[str, ManufacturingPart]
    joints: List[JointSpec]
    quality_metrics: Step3QualityMetrics
    violations: List[Step3Violation]
    status: str
    debug: Dict[str, object] = field(default_factory=dict)

    def to_manufacturing_json(self) -> dict:
        parts_payload = []
        for part in self.parts.values():
            parts_payload.append(
                {
                    "part_id": part.part_id,
                    "region_type": part.region_type.value,
                    "material_key": part.material_key,
                    "thickness_mm": part.thickness_mm,
                    "outline_2d": list(part.profile.outline.exterior.coords),
                    "cutouts_2d": [
                        list(c.exterior.coords) for c in part.profile.cutouts
                    ]
                    + [list(h.coords) for h in part.profile.outline.interiors],
                    "bend_ops": [
                        {
                            "line": [list(op.line[0]), list(op.line[1])],
                            "angle_deg": op.angle_deg,
                            "radius_mm": op.radius_mm,
                            "direction": op.direction,
                            "sequence_index": op.sequence_index,
                        }
                        for op in part.bend_ops
                    ],
                    "position_3d": [float(v) for v in part.position_3d],
                    "rotation_3d": [float(v) for v in part.rotation_3d],
                    "source_area_mm2": float(part.source_area_mm2),
                    "metadata": part.metadata,
                }
            )

        joints_payload = [
            {
                "joint_type": j.joint_type,
                "part_a": j.part_a,
                "part_b": j.part_b,
                "geometry": j.geometry,
                "clearance_mm": j.clearance_mm,
                "fastener_spec": j.fastener_spec,
            }
            for j in self.joints
        ]

        return {
            "units": "mm",
            "status": self.status,
            "parts": parts_payload,
            "joints": joints_payload,
            "quality_metrics": {
                "hausdorff_mm": self.quality_metrics.hausdorff_mm,
                "mean_distance_mm": self.quality_metrics.mean_distance_mm,
                "normal_error_deg": self.quality_metrics.normal_error_deg,
                "connectivity_score": self.quality_metrics.connectivity_score,
                "overlap_score": self.quality_metrics.overlap_score,
                "violation_score": self.quality_metrics.violation_score,
                "part_count": self.quality_metrics.part_count,
                "overall_score": self.quality_metrics.overall_score,
            },
            "violations": [
                {
                    "code": v.code,
                    "severity": v.severity,
                    "message": v.message,
                    "part_id": v.part_id,
                }
                for v in self.violations
            ],
            "debug": self.debug,
        }


def _serialize_parts_snapshot(
    label: str,
    index: int,
    parts: List[ManufacturingPart],
    joints: Optional[List[JointSpec]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> dict:
    """Serialize current parts state for phase step-through debugging."""
    parts_payload = []
    for part in parts:
        parts_payload.append(
            {
                "part_id": part.part_id,
                "region_type": part.region_type.value,
                "material_key": part.material_key,
                "thickness_mm": part.thickness_mm,
                "outline_2d": list(part.profile.outline.exterior.coords),
                "cutouts_2d": [list(c.exterior.coords) for c in part.profile.cutouts]
                + [list(h.coords) for h in part.profile.outline.interiors],
                "bend_ops": [
                    {
                        "line": [list(op.line[0]), list(op.line[1])],
                        "angle_deg": op.angle_deg,
                        "radius_mm": op.radius_mm,
                        "direction": op.direction,
                        "sequence_index": op.sequence_index,
                    }
                    for op in part.bend_ops
                ],
                "position_3d": [float(v) for v in part.position_3d],
                "origin_3d": (
                    [
                        float(v)
                        for v in (
                            part.profile.origin_3d
                            + part.metadata.get("stack_layer_offset_mm", 0.0)
                            * _rotation_to_normal(part.rotation_3d)
                        )
                    ]
                    if part.profile.origin_3d is not None
                    else [float(v) for v in part.position_3d]
                ),
                "rotation_3d": [float(v) for v in part.rotation_3d],
                "source_area_mm2": float(part.source_area_mm2),
                "metadata": part.metadata,
            }
        )

    joints_payload = []
    if joints:
        joints_payload = [
            {
                "joint_type": j.joint_type,
                "part_a": j.part_a,
                "part_b": j.part_b,
                "geometry": j.geometry,
                "clearance_mm": j.clearance_mm,
                "fastener_spec": j.fastener_spec,
            }
            for j in joints
        ]

    payload = {
        "phase_label": label,
        "phase_index": index,
        "part_count": len(parts_payload),
        "units": "mm",
        "parts": parts_payload,
        "joints": joints_payload,
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def _extract_phase_overlap_metrics(
    phase_snapshots: List[dict],
    phase_index: int,
    prefix: str,
) -> Dict[str, object]:
    """Extract overlap metrics from a snapshot diagnostics payload."""
    defaults: Dict[str, object] = {
        f"{prefix}pairs": 0,
        f"{prefix}max_mm": 0.0,
        f"{prefix}total_mm": 0.0,
        f"{prefix}count": 0,
        f"{prefix}details": [],
        f"{prefix}regions": [],
    }
    for snap in phase_snapshots:
        if int(snap.get("phase_index", -1)) != int(phase_index):
            continue
        diagnostics = snap.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return defaults
        overlap = diagnostics.get("plane_overlap")
        if not isinstance(overlap, dict):
            return defaults
        return {
            f"{prefix}pairs": int(overlap.get("plane_overlap_pairs", 0) or 0),
            f"{prefix}max_mm": float(overlap.get("plane_overlap_max_mm", 0.0) or 0.0),
            f"{prefix}total_mm": float(
                overlap.get("plane_overlap_total_mm", 0.0) or 0.0
            ),
            f"{prefix}count": int(overlap.get("plane_overlap_count", 0) or 0),
            f"{prefix}details": overlap.get("plane_overlap_details", []),
            f"{prefix}regions": overlap.get("plane_overlap_regions", []),
        }
    return defaults


def _extract_phase_trim_debug(
    phase_snapshots: List[dict], phase_index: int
) -> Dict[str, Any]:
    """Extract trim-decision diagnostics from a phase snapshot."""
    defaults: Dict[str, Any] = {
        "step2_trim_debug": {},
    }
    for snap in phase_snapshots:
        if int(snap.get("phase_index", -1)) != int(phase_index):
            continue
        diagnostics = snap.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return defaults
        trim_debug = diagnostics.get("trim_decisions")
        if isinstance(trim_debug, dict):
            return {"step2_trim_debug": trim_debug}
        return defaults
    return defaults


@dataclass
class _RegionCandidate:
    region_type: RegionType
    profile: PartProfile2D
    normal: np.ndarray
    position_3d: np.ndarray
    area_mm2: float
    source_faces: List[int]
    bend_ops: List[BendOp]
    score: float
    metadata: Dict[str, float | int | str] = field(default_factory=dict)
    basis_u: Optional[np.ndarray] = None
    basis_v: Optional[np.ndarray] = None
    origin_3d: Optional[np.ndarray] = None


def build_default_capability_profile(
    material_key: str = "plywood_baltic_birch",
    allow_controlled_bending: bool = True,
) -> CapabilityProfile:
    dfm = DFMConfig.from_material(material_key)
    bend = BendCapability(enabled=allow_controlled_bending)
    if not allow_controlled_bending:
        bend.allowed_materials = []
    return CapabilityProfile(
        profile_name="sendcutsend_flat_cut_plus_controlled_bending",
        supported_materials=sorted(MATERIALS.keys()),
        min_feature_mm=dfm.min_slot_width_mm,
        min_internal_radius_mm=dfm.min_internal_radius_mm,
        min_bridge_width_mm=dfm.min_bridge_width_mm,
        max_sheet_width_mm=dfm.max_sheet_width_mm,
        max_sheet_height_mm=dfm.max_sheet_height_mm,
        default_kerf_mm=0.20,
        controlled_bending=bend,
    )


def decompose_first_principles(input_spec: Step3Input) -> Step3Output:
    timers: Dict[str, float] = {}
    t0 = time.perf_counter()

    material_key = _pick_material(
        input_spec.material_preferences,
        input_spec.scs_capabilities,
    )
    if not input_spec.scs_capabilities.supports_material(material_key):
        raise ValueError(f"Unsupported material: {material_key}")

    mesh = _load_and_normalize_mesh(
        input_spec.mesh_path,
        target_height_mm=input_spec.target_height_mm,
        auto_scale=input_spec.auto_scale,
    )
    timers["mesh_load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    thickness_mm = _pick_thickness_mm(material_key)
    raw_candidates, unsupported_faces = _extract_candidates(
        mesh=mesh,
        material_key=material_key,
        thickness_mm=thickness_mm,
        input_spec=input_spec,
    )
    coplanar_grouped = _merge_coplanar_candidates(raw_candidates, mesh)
    candidates, merged_planar_pairs = _collapse_planar_face_pairs(
        coplanar_grouped, mesh=mesh
    )
    timers["candidate_generation_s"] = time.perf_counter() - t1

    if not candidates:
        return _empty_output(
            design_name=input_spec.design_name,
            reason="no_candidates_generated",
            debug={"timings": timers},
        )

    t2 = time.perf_counter()
    selected_parts, selection_debug = _select_candidates(candidates, input_spec)
    phase_snapshots = [
        _serialize_parts_snapshot(
            "Selected parts",
            0,
            selected_parts,
            diagnostics=_phase_diagnostics(selected_parts),
        ),
    ]
    # Identify joint contact pairs BEFORE trimming so the trimmer can
    # make informed decisions about which direction to cut.
    joint_contact_pairs = _identify_joint_contact_pairs(
        selected_parts,
        contact_tolerance_mm=input_spec.joint_contact_tolerance_mm,
        parallel_dot_threshold=input_spec.joint_parallel_dot_threshold,
        joint_distance_mm=input_spec.joint_distance_mm,
    )
    selected_parts, constrain_snapshots = _constrain_outlines(
        selected_parts, mesh, input_spec, joint_pairs=joint_contact_pairs
    )
    phase_snapshots.extend(constrain_snapshots)
    # Do NOT pass contact_pairs here — part indices are invalidated by
    # _constrain_outlines (which deep-copies, drops, and reorders parts).
    # Let _synthesize_joint_specs do its own full scan on the final parts.
    joints = _synthesize_joint_specs(
        selected_parts,
        joint_distance_mm=input_spec.joint_distance_mm,
        contact_tolerance_mm=input_spec.joint_contact_tolerance_mm,
        parallel_dot_threshold=input_spec.joint_parallel_dot_threshold,
    )
    selected_parts, joints, joint_geom_debug = _apply_joint_geometry(
        selected_parts, joints, input_spec
    )
    # Safety-net: bounded post-joint cleanup for residual overlaps after
    # joint geometry application. Should rarely trigger with correct trim.
    selected_parts = _post_joint_overlap_cleanup(
        selected_parts, joints
    )
    phase_snapshots.append(
        _serialize_parts_snapshot(
            "With joint geometry",
            5,
            selected_parts,
            joints,
            diagnostics=_phase_diagnostics(
                selected_parts,
                joint_count=int(len(joints)),
            ),
        )
    )
    phase_snapshots.append(
        _serialize_parts_snapshot(
            "With joints",
            6,
            selected_parts,
            joints,
            diagnostics=_phase_diagnostics(
                selected_parts,
                joint_count=int(len(joints)),
            ),
        )
    )
    design = _build_design(input_spec.design_name, selected_parts, joints, material_key)
    timers["selection_and_assembly_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    violations = _validate_parts_and_bends(
        selected_parts,
        capabilities=input_spec.scs_capabilities,
        material_key=material_key,
    )
    if unsupported_faces > 0:
        unsupported_fraction = unsupported_faces / max(1, len(mesh.faces))
        violations.append(
            Step3Violation(
                code="unsupported_geometry",
                severity="warning" if unsupported_fraction < 0.2 else "error",
                message=(
                    f"{unsupported_faces} faces were not representable under the "
                    "flat-cut + controlled-bending capability profile"
                ),
            )
        )
    else:
        unsupported_fraction = 0.0

    if input_spec.joint_fit_validation:
        joint_fit_violations, joint_fit_debug = _validate_joint_fit_2d(
            selected_parts, joints
        )
        violations.extend(joint_fit_violations)

        assembly_violations, assembly_debug = _validate_joint_assembly_3d(
            selected_parts, joints
        )
        violations.extend(assembly_violations)
    else:
        joint_fit_debug = {}
        assembly_debug = {}

    overlap_debug = _compute_plane_overlap(selected_parts, joints=joints)
    step2_overlap_debug = _extract_phase_overlap_metrics(
        phase_snapshots, phase_index=2, prefix="step2_plane_overlap_"
    )
    step2_trim_debug = _extract_phase_trim_debug(phase_snapshots, phase_index=2)

    error_count = sum(1 for v in violations if v.severity == "error")
    quality = _compute_quality_metrics(
        mesh=mesh,
        selected_parts=selected_parts,
        joints=joints,
        part_budget_max=input_spec.part_budget_max,
        fidelity_weight=input_spec.fidelity_weight,
        overlap_pairs=int(overlap_debug.get("plane_overlap_pairs", 0)),
        error_count=error_count,
    )
    timers["validation_metrics_s"] = time.perf_counter() - t3

    status = _compute_status(selected_parts, violations)
    mesh_area_mm2 = max(float(mesh.area), 1e-6)
    unique_coverage_ratio = _unique_face_coverage_ratio(mesh, selected_parts)
    summed_source_area_ratio = min(
        1.0,
        sum(max(0.0, p.source_area_mm2) for p in selected_parts) / mesh_area_mm2,
    )
    debug = {
        "candidate_count": len(candidates),
        "candidate_count_raw": len(raw_candidates),
        "merged_planar_pairs": int(merged_planar_pairs),
        "selected_count": len(selected_parts),
        "unsupported_face_count": int(unsupported_faces),
        "unsupported_fraction": float(unsupported_fraction),
        "material_key": material_key,
        "mesh_faces": int(len(mesh.faces)),
        "mesh_area_mm2": float(mesh_area_mm2),
        "coverage_ratio_unique_faces": float(unique_coverage_ratio),
        "coverage_ratio_summed_source_area": float(summed_source_area_ratio),
        "joint_contact_tolerance_mm": float(input_spec.joint_contact_tolerance_mm),
        "joint_parallel_dot_threshold": float(input_spec.joint_parallel_dot_threshold),
        "timings": timers,
        "phase_snapshots": phase_snapshots,
        **selection_debug,
        **joint_geom_debug,
        **joint_fit_debug,
        **assembly_debug,
        **overlap_debug,
        **step2_overlap_debug,
        **step2_trim_debug,
    }
    design.metadata["step3_first_principles"] = debug

    return Step3Output(
        design=design,
        parts={part.part_id: part for part in selected_parts},
        joints=joints,
        quality_metrics=quality,
        violations=violations,
        status=status,
        debug=debug,
    )


def _pick_material(
    preferences: Sequence[str],
    capabilities: CapabilityProfile,
) -> str:
    for key in preferences:
        if key in MATERIALS and capabilities.supports_material(key):
            return key
    return "plywood_baltic_birch"


def _pick_thickness_mm(material_key: str) -> float:
    options = MATERIALS[material_key].thicknesses_mm
    return min(options, key=lambda t: abs(t - 6.35))


def _load_and_normalize_mesh(
    mesh_path: str,
    target_height_mm: float,
    auto_scale: bool,
) -> trimesh.Trimesh:
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    scene_or_mesh = trimesh.load(mesh_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_mesh()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            meshes = [
                g
                for g in scene_or_mesh.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError(f"No triangle mesh found in scene: {mesh_path}")
            mesh = max(meshes, key=lambda m: len(m.faces))
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        raise ValueError(f"Unsupported mesh object: {type(scene_or_mesh)}")

    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.fix_normals()
    try:
        mesh.fill_holes()
    except Exception as exc:
        logger.warning("fill_holes failed: %s", exc)

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    mesh_height = float(extents[2])

    if auto_scale and mesh_height > 1e-6:
        mesh.apply_scale(target_height_mm / mesh_height)

    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    translate = np.array([-center_xy[0], -center_xy[1], -bounds[0][2]])
    mesh.apply_translation(translate)

    return mesh


def _extract_candidates(
    mesh: trimesh.Trimesh,
    material_key: str,
    thickness_mm: float,
    input_spec: Step3Input,
) -> Tuple[List[_RegionCandidate], int]:
    candidates: List[_RegionCandidate] = []
    covered_faces: set[int] = set()

    # Planar regions from coplanar facets.
    for idx, facet in enumerate(mesh.facets):
        face_indices = [int(f) for f in facet]
        area = float(np.sum(mesh.area_faces[face_indices]))
        if area < input_spec.min_planar_region_area_mm2:
            continue

        normal = np.array(mesh.facets_normal[idx], dtype=float)
        polygon, basis_u, basis_v, proj_origin = _project_faces_to_outline(
            mesh, face_indices, normal
        )
        if polygon is None:
            continue

        centroid = np.mean(mesh.triangles_center[face_indices], axis=0)
        profile = PartProfile2D(
            outline=polygon,
            material_key=material_key,
            thickness_mm=thickness_mm,
            basis_u=basis_u,
            basis_v=basis_v,
            origin_3d=proj_origin,
        )
        candidates.append(
            _RegionCandidate(
                region_type=RegionType.PLANAR_CUT,
                profile=profile,
                normal=normal,
                position_3d=centroid,
                area_mm2=area,
                source_faces=face_indices,
                bend_ops=[],
                score=area,
                metadata={"kind": "facet"},
                basis_u=basis_u,
                basis_v=basis_v,
                origin_3d=proj_origin,
            )
        )
        covered_faces.update(face_indices)

    remaining_faces = [i for i in range(len(mesh.faces)) if i not in covered_faces]
    if not remaining_faces:
        return candidates, 0

    bend_candidates, supported_faces = _extract_bend_candidates(
        mesh=mesh,
        face_indices=remaining_faces,
        material_key=material_key,
        thickness_mm=thickness_mm,
        input_spec=input_spec,
    )
    candidates.extend(bend_candidates)
    unsupported = max(0, len(remaining_faces) - supported_faces)
    return candidates, unsupported


def _extract_bend_candidates(
    mesh: trimesh.Trimesh,
    face_indices: List[int],
    material_key: str,
    thickness_mm: float,
    input_spec: Step3Input,
) -> Tuple[List[_RegionCandidate], int]:
    if not input_spec.scs_capabilities.supports_bending(material_key):
        return [], 0

    components = _connected_face_components(mesh, set(face_indices))
    supported_faces = 0
    candidates: List[_RegionCandidate] = []

    for comp_idx, comp_faces in enumerate(components):
        if comp_idx >= input_spec.max_bend_regions:
            break
        area = float(np.sum(mesh.area_faces[comp_faces]))
        if area < input_spec.min_bend_region_area_mm2:
            continue

        candidate = _build_bend_candidate(
            mesh,
            comp_faces,
            material_key,
            thickness_mm,
            area,
            input_spec,
        )
        if candidate is None:
            continue

        candidates.append(candidate)
        supported_faces += len(comp_faces)

    return candidates, supported_faces


def _build_bend_candidate(
    mesh: trimesh.Trimesh,
    face_indices: List[int],
    material_key: str,
    thickness_mm: float,
    area_mm2: float,
    input_spec: Step3Input,
) -> Optional[_RegionCandidate]:
    pts = mesh.vertices[mesh.faces[face_indices]].reshape(-1, 3)
    if len(pts) < 4:
        return None

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    u_axis = vh[0]
    v_axis = vh[1]
    normal = np.cross(u_axis, v_axis)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-8:
        return None
    normal /= normal_norm

    u = centered @ u_axis
    v = centered @ v_axis
    min_u, max_u = float(np.min(u)), float(np.max(u))
    min_v, max_v = float(np.min(v)), float(np.max(v))
    width = max_u - min_u
    height = max_v - min_v
    if width < 1.0 or height < 1.0:
        return None

    bend_length = max(width, height)
    if bend_length < input_spec.scs_capabilities.controlled_bending.min_bend_length_mm:
        return None

    outline = Polygon(
        [
            (min_u, min_v),
            (max_u, min_v),
            (max_u, max_v),
            (min_u, max_v),
        ]
    )
    outline = _clean_polygon(outline)
    if outline is None:
        return None

    face_normals = mesh.face_normals[face_indices]
    angle_deg = _estimate_bend_angle(face_normals)
    radius_mm = max(
        input_spec.scs_capabilities.min_internal_radius_mm,
        thickness_mm
        * input_spec.scs_capabilities.controlled_bending.min_radius_multiplier,
    )

    if width >= height:
        line = ((min_u, (min_v + max_v) / 2.0), (max_u, (min_v + max_v) / 2.0))
    else:
        line = (((min_u + max_u) / 2.0, min_v), ((min_u + max_u) / 2.0, max_v))

    return _RegionCandidate(
        region_type=RegionType.BENDABLE_SHEET,
        profile=PartProfile2D(
            outline=outline,
            material_key=material_key,
            thickness_mm=thickness_mm,
        ),
        normal=normal,
        position_3d=centroid,
        area_mm2=area_mm2,
        source_faces=face_indices,
        bend_ops=[
            BendOp(
                line=line,
                angle_deg=angle_deg,
                radius_mm=radius_mm,
                direction="up",
                sequence_index=0,
            )
        ],
        score=area_mm2 * 1.1,
        metadata={"kind": "bend_component", "width_mm": width, "height_mm": height},
    )


def _project_faces_to_outline(
    mesh: trimesh.Trimesh,
    face_indices: List[int],
    normal: np.ndarray,
) -> Tuple[Optional[Polygon], np.ndarray, np.ndarray, np.ndarray]:
    """Project mesh faces to 2D using triangle union (concave-accurate).

    Returns (polygon, basis_u, basis_v, origin_3d).
    Falls back to convex hull only if the union is degenerate.
    """
    vertices = mesh.vertices[mesh.faces[face_indices]].reshape(-1, 3)
    if len(vertices) < 3:
        return None, np.zeros(3), np.zeros(3), np.zeros(3)

    centroid = np.mean(vertices, axis=0)
    n = normal / max(np.linalg.norm(normal), 1e-8)
    u_axis, v_axis = _plane_basis(n)

    # Project each triangle individually and union them
    triangles_3d = mesh.vertices[mesh.faces[face_indices]]  # (N, 3, 3)
    tri_polys: List[Polygon] = []
    for tri in triangles_3d:
        local = tri - centroid
        pts_2d = [(float(p @ u_axis), float(p @ v_axis)) for p in local]
        try:
            p = Polygon(pts_2d)
            if p.is_valid and p.area > 0:
                tri_polys.append(p)
        except Exception:
            continue

    polygon: Optional[Polygon] = None
    if tri_polys:
        merged = unary_union(tri_polys)
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)
        if merged.geom_type == "Polygon" and not merged.is_empty:
            polygon = merged.simplify(0.5, preserve_topology=True)
            polygon = _clean_polygon(polygon)

    # Fallback to convex hull if triangle union failed
    if polygon is None:
        local = vertices - centroid
        points_2d = np.column_stack([local @ u_axis, local @ v_axis])
        hull = MultiPoint(points_2d).convex_hull
        if hull.geom_type != "Polygon":
            return None, u_axis, v_axis, centroid
        polygon = _clean_polygon(hull)

    if polygon is None or polygon.area < 1.0:
        return None, u_axis, v_axis, centroid
    return polygon, u_axis, v_axis, centroid


def _clean_polygon(polygon: Polygon) -> Optional[Polygon]:
    if polygon.is_empty:
        return None
    clean = polygon if polygon.is_valid else polygon.buffer(0)
    if clean.is_empty:
        return None
    if clean.geom_type == "Polygon":
        return clean
    if clean.geom_type == "MultiPolygon":
        return max(clean.geoms, key=lambda g: g.area)
    return None


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if abs(float(normal[2])) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, ref)
    u /= max(np.linalg.norm(u), 1e-8)
    v = np.cross(normal, u)
    v /= max(np.linalg.norm(v), 1e-8)
    return u, v


def _connected_face_components(
    mesh: trimesh.Trimesh, faces: set[int]
) -> List[List[int]]:
    adjacency: Dict[int, List[int]] = {}
    for a, b in mesh.face_adjacency:
        a_idx = int(a)
        b_idx = int(b)
        if a_idx not in faces or b_idx not in faces:
            continue
        adjacency.setdefault(a_idx, []).append(b_idx)
        adjacency.setdefault(b_idx, []).append(a_idx)

    unseen = set(faces)
    comps: List[List[int]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = [start]
        while stack:
            node = stack.pop()
            for neigh in adjacency.get(node, []):
                if neigh in unseen:
                    unseen.remove(neigh)
                    stack.append(neigh)
                    comp.append(neigh)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _estimate_bend_angle(face_normals: np.ndarray) -> float:
    mean_normal = np.mean(face_normals, axis=0)
    mean_norm = float(np.linalg.norm(mean_normal))
    if mean_norm < 1e-8:
        return 15.0
    mean_normal /= mean_norm
    dots = np.clip(face_normals @ mean_normal, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    spread = float(np.percentile(angles, 90))
    return float(np.clip(max(15.0, spread * 2.0), 15.0, 120.0))


def _candidate_outline_dims(cand: _RegionCandidate) -> Tuple[float, float]:
    bounds = cand.profile.outline.bounds
    w = float(max(0.0, bounds[2] - bounds[0]))
    h = float(max(0.0, bounds[3] - bounds[1]))
    lo, hi = sorted((w, h))
    return lo, hi


def _candidate_mesh_constraint_pressure(cand: _RegionCandidate) -> float:
    """Estimate how tightly candidate geometry is constrained by mesh support.

    Uses area mismatch between projected outline and source-face area as a proxy:
    0.0 => tightly constrained, 1.0 => highly unconstrained.
    """
    outline_area = float(max(0.0, cand.profile.outline.area))
    source_area = float(max(0.0, cand.area_mm2))
    denom = max(outline_area, source_area, 1e-6)
    return float(np.clip(abs(outline_area - source_area) / denom, 0.0, 1.0))


def _canonicalize_normal(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float)
    n /= max(np.linalg.norm(n), 1e-8)
    eps = 1e-8
    if abs(float(n[0])) > eps:
        return n if n[0] >= 0.0 else -n
    if abs(float(n[1])) > eps:
        return n if n[1] >= 0.0 else -n
    return n if n[2] >= 0.0 else -n


def _is_opposite_planar_pair(a: _RegionCandidate, b: _RegionCandidate) -> bool:
    if a.region_type != RegionType.PLANAR_CUT or b.region_type != RegionType.PLANAR_CUT:
        return False

    n_a = a.normal / max(np.linalg.norm(a.normal), 1e-8)
    n_b = b.normal / max(np.linalg.norm(b.normal), 1e-8)
    if float(np.dot(n_a, n_b)) > -0.96:
        return False

    area_rel = abs(a.area_mm2 - b.area_mm2) / max(a.area_mm2, b.area_mm2, 1e-6)
    if area_rel > 0.30:
        return False

    dims_a = _candidate_outline_dims(a)
    dims_b = _candidate_outline_dims(b)
    dim_rel = max(
        abs(dims_a[0] - dims_b[0]) / max(dims_a[0], dims_b[0], 1e-6),
        abs(dims_a[1] - dims_b[1]) / max(dims_a[1], dims_b[1], 1e-6),
    )
    if dim_rel > 0.25:
        return False

    delta = b.position_3d - a.position_3d
    delta_n_signed = float(np.dot(delta, n_a))
    delta_n = abs(delta_n_signed)
    delta_t = float(np.linalg.norm(delta - delta_n_signed * n_a))
    max_dim = max(dims_a[1], dims_b[1], 1.0)

    if delta_n < 0.5:
        return False
    if delta_n > max(40.0, 0.45 * max_dim):
        return False
    if delta_t > max(8.0, 0.15 * max_dim):
        return False
    return True


def _classify_face_exterior(
    mesh: trimesh.Trimesh,
    face_centroid: np.ndarray,
    face_normal: np.ndarray,
    epsilon: float = 0.5,
) -> bool:
    """Return True if *face_normal* points into open space (exterior face).

    Casts a ray from ``centroid + epsilon * normal`` along ``normal``.
    If the ray hits nothing, the face looks outward (exterior).
    Relies on ``mesh.fix_normals()`` having been called so that face normals
    point outward.
    """
    n = face_normal / max(float(np.linalg.norm(face_normal)), 1e-12)
    origin = np.asarray(face_centroid, dtype=float) + epsilon * n
    try:
        locations, _ray_idx, _tri_idx = mesh.ray.intersects_location(
            ray_origins=origin.reshape(1, 3),
            ray_directions=n.reshape(1, 3),
        )
        return len(locations) == 0
    except Exception:
        return False


def _merge_planar_pair(
    a: _RegionCandidate,
    b: _RegionCandidate,
    member_thickness_mm: float,
    stack_alignment: str = "centered",
    exterior_face_origin_3d: Optional[np.ndarray] = None,
    exterior_face_normal: Optional[np.ndarray] = None,
) -> _RegionCandidate:
    chosen = a if a.area_mm2 >= b.area_mm2 else b
    merged_faces = sorted(set(a.source_faces) | set(b.source_faces))
    merged_metadata = dict(chosen.metadata)
    merged_metadata["kind"] = "merged_facet_pair"
    merged_metadata["merged_source_face_count"] = int(len(merged_faces))
    merged_metadata["member_thickness_mm"] = float(max(0.0, member_thickness_mm))
    merged_metadata["stack_alignment"] = stack_alignment
    if exterior_face_origin_3d is not None:
        merged_metadata["exterior_face_origin_3d"] = [
            float(v) for v in exterior_face_origin_3d
        ]
    if exterior_face_normal is not None:
        merged_metadata["exterior_face_normal"] = [
            float(v) for v in exterior_face_normal
        ]

    return _RegionCandidate(
        region_type=RegionType.PLANAR_CUT,
        profile=chosen.profile,
        # Keep the chosen face normal so local profile coordinates and
        # downstream rotation_3d stay in the same frame.
        normal=chosen.normal,
        position_3d=0.5 * (a.position_3d + b.position_3d),
        area_mm2=max(a.area_mm2, b.area_mm2),
        source_faces=merged_faces,
        bend_ops=[],
        score=max(a.score, b.score),
        metadata=merged_metadata,
        basis_u=chosen.basis_u,
        basis_v=chosen.basis_v,
        origin_3d=chosen.origin_3d,
    )


def _merge_coplanar_candidates(
    candidates: List[_RegionCandidate],
    mesh: trimesh.Trimesh,
) -> List[_RegionCandidate]:
    """Group same-plane planar candidates into compound candidates.

    Candidates with near-identical normals (dot > 0.99) and plane offsets
    (within 1 mm) are merged.  Their source faces are combined and
    re-projected into a single outline.
    """
    planar = [
        (idx, cand)
        for idx, cand in enumerate(candidates)
        if cand.region_type == RegionType.PLANAR_CUT
    ]
    non_planar = [
        cand for cand in candidates if cand.region_type != RegionType.PLANAR_CUT
    ]
    if not planar:
        return list(candidates)

    # Bucket by canonical normal direction + plane offset
    groups: Dict[int, List[int]] = {}  # group_id -> list of indices into planar
    group_of: Dict[int, int] = {}  # planar list index -> group_id
    next_group = 0

    for i, (_, cand_i) in enumerate(planar):
        if i in group_of:
            continue
        group_of[i] = next_group
        groups[next_group] = [i]
        n_i = cand_i.normal / max(np.linalg.norm(cand_i.normal), 1e-8)
        offset_i = float(np.dot(cand_i.position_3d, n_i))

        for j in range(i + 1, len(planar)):
            if j in group_of:
                continue
            _, cand_j = planar[j]
            n_j = cand_j.normal / max(np.linalg.norm(cand_j.normal), 1e-8)
            # Same direction (not opposite — opposite pairs are handled later)
            if abs(float(np.dot(n_i, n_j))) < 0.99:
                continue
            # Same plane offset
            sign = 1.0 if float(np.dot(n_i, n_j)) > 0 else -1.0
            offset_j = float(np.dot(cand_j.position_3d, sign * n_j))
            if abs(offset_i - offset_j) > 1.0:
                continue
            group_of[j] = next_group
            groups[next_group].append(j)

        next_group += 1

    merged_planar: List[_RegionCandidate] = []
    for gid, members in groups.items():
        if len(members) == 1:
            merged_planar.append(planar[members[0]][1])
            continue

        # Merge: combine source_faces, re-project outline
        all_faces: List[int] = []
        total_area = 0.0
        for m in members:
            cand_m = planar[m][1]
            all_faces.extend(cand_m.source_faces)
            total_area += cand_m.area_mm2
        all_faces = sorted(set(all_faces))

        # Use the largest member as the reference for normal / basis
        ref_idx = max(members, key=lambda m: planar[m][1].area_mm2)
        ref_cand = planar[ref_idx][1]
        normal = ref_cand.normal

        polygon, basis_u, basis_v, proj_origin = _project_faces_to_outline(
            mesh, all_faces, normal
        )
        if polygon is None:
            # Fallback: keep members unmerged
            for m in members:
                merged_planar.append(planar[m][1])
            continue

        profile = PartProfile2D(
            outline=polygon,
            material_key=ref_cand.profile.material_key,
            thickness_mm=ref_cand.profile.thickness_mm,
            basis_u=basis_u,
            basis_v=basis_v,
            origin_3d=proj_origin,
        )
        merged_planar.append(
            _RegionCandidate(
                region_type=RegionType.PLANAR_CUT,
                profile=profile,
                normal=normal,
                position_3d=proj_origin,
                area_mm2=total_area,
                source_faces=all_faces,
                bend_ops=[],
                score=total_area,
                metadata={"kind": "facet", "coplanar_merged_count": len(members)},
                basis_u=basis_u,
                basis_v=basis_v,
                origin_3d=proj_origin,
            )
        )

    return merged_planar + non_planar


def _collapse_planar_face_pairs(
    candidates: List[_RegionCandidate],
    mesh: Optional[trimesh.Trimesh] = None,
) -> Tuple[List[_RegionCandidate], int]:
    planar_indices = [
        idx
        for idx, cand in enumerate(candidates)
        if cand.region_type == RegionType.PLANAR_CUT
    ]
    used: set[int] = set()
    planar_out: List[_RegionCandidate] = []
    merged_pairs = 0

    for i in planar_indices:
        if i in used:
            continue
        a = candidates[i]
        best_j: Optional[int] = None
        best_cost = float("inf")
        best_delta_n = 0.0

        for j in planar_indices:
            if j <= i or j in used:
                continue
            b = candidates[j]
            if not _is_opposite_planar_pair(a, b):
                continue

            n_a = a.normal / max(np.linalg.norm(a.normal), 1e-8)
            delta = b.position_3d - a.position_3d
            delta_n_signed = float(np.dot(delta, n_a))
            delta_n = abs(delta_n_signed)
            delta_t = float(np.linalg.norm(delta - delta_n_signed * n_a))
            area_rel = abs(a.area_mm2 - b.area_mm2) / max(a.area_mm2, b.area_mm2, 1e-6)
            cost = delta_t + 0.25 * delta_n + 30.0 * area_rel
            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_delta_n = delta_n

        if best_j is not None:
            used.add(i)
            used.add(best_j)
            b = candidates[best_j]

            # Classify faces as exterior/interior for stack alignment
            stack_alignment = "centered"
            exterior_face_origin_3d = None
            exterior_face_normal = None
            if mesh is not None:
                a_ext = _classify_face_exterior(mesh, a.position_3d, a.normal)
                b_ext = _classify_face_exterior(mesh, b.position_3d, b.normal)
                if a_ext and not b_ext:
                    stack_alignment = "exterior_flush"
                    exterior_face_origin_3d = np.asarray(
                        a.origin_3d if a.origin_3d is not None else a.position_3d,
                        dtype=float,
                    )
                    exterior_face_normal = a.normal / max(
                        float(np.linalg.norm(a.normal)), 1e-12
                    )
                elif b_ext and not a_ext:
                    stack_alignment = "exterior_flush"
                    exterior_face_origin_3d = np.asarray(
                        b.origin_3d if b.origin_3d is not None else b.position_3d,
                        dtype=float,
                    )
                    exterior_face_normal = b.normal / max(
                        float(np.linalg.norm(b.normal)), 1e-12
                    )

            planar_out.append(
                _merge_planar_pair(
                    a,
                    b,
                    member_thickness_mm=best_delta_n,
                    stack_alignment=stack_alignment,
                    exterior_face_origin_3d=exterior_face_origin_3d,
                    exterior_face_normal=exterior_face_normal,
                )
            )
            merged_pairs += 1
        else:
            used.add(i)
            planar_out.append(a)

    non_planar = [c for c in candidates if c.region_type != RegionType.PLANAR_CUT]
    return planar_out + non_planar, merged_pairs


def _select_candidates(
    candidates: List[_RegionCandidate],
    input_spec: Step3Input,
) -> Tuple[List[ManufacturingPart], Dict[str, object]]:
    part_budget_max = max(1, int(input_spec.part_budget_max))
    ranked_indices = _greedy_coverage_rank(candidates)
    if not ranked_indices:
        return [], {"stack_enabled": bool(input_spec.enable_planar_stacking)}

    extra_layer_gain = float(np.clip(input_spec.stack_extra_layer_gain, 0.0, 1.5))
    layer_options: List[Tuple[float, float, int, int]] = (
        []
    )  # (sort_gain, gain, rank, cand_idx)
    target_layers_by_candidate: Dict[int, int] = {}
    member_thickness_by_candidate: Dict[int, float] = {}
    score_factor_by_candidate: Dict[int, float] = {}
    thin_side_state_by_candidate: Dict[int, str] = {}

    for rank, cand_idx in enumerate(ranked_indices):
        cand = candidates[cand_idx]
        member_thickness = _candidate_member_thickness_mm(cand)
        if member_thickness is not None:
            member_thickness_by_candidate[cand_idx] = float(member_thickness)

        target_layers = _target_stack_layers(
            cand,
            input_spec=input_spec,
            member_thickness_mm=member_thickness,
        )
        target_layers_by_candidate[cand_idx] = target_layers

    coverage_refs = _stack_coverage_refs(
        candidates,
        target_layers_by_candidate=target_layers_by_candidate,
        member_thickness_by_candidate=member_thickness_by_candidate,
    )
    thin_side_matched_count = 0
    thin_side_penalized_count = 0
    thin_side_dropped_count = 0

    for cand_idx in ranked_indices:
        cand = candidates[cand_idx]
        score_factor, thin_side_state = _thin_side_candidate_score_factor(
            cand,
            member_thickness_mm=member_thickness_by_candidate.get(cand_idx),
            coverage_refs=coverage_refs,
            input_spec=input_spec,
        )
        score_factor_by_candidate[cand_idx] = score_factor
        thin_side_state_by_candidate[cand_idx] = thin_side_state
        if thin_side_state != "none":
            thin_side_matched_count += 1
        if thin_side_state == "penalized":
            thin_side_penalized_count += 1
        elif thin_side_state == "dropped":
            thin_side_dropped_count += 1

    # Collect all source faces across candidates for novelty computation
    all_source_faces: set[int] = set()
    for cand_idx in ranked_indices:
        all_source_faces.update(candidates[cand_idx].source_faces)

    for rank, cand_idx in enumerate(ranked_indices):
        cand = candidates[cand_idx]
        score_factor = score_factor_by_candidate.get(cand_idx, 1.0)
        if score_factor <= 0.0:
            continue
        target_layers = target_layers_by_candidate[cand_idx]
        for layer_idx in range(target_layers):
            if layer_idx == 0:
                gain = float(cand.score)
            else:
                gain = float(cand.score) * extra_layer_gain * (0.85 ** (layer_idx - 1))
            gain *= score_factor
            sort_gain = gain
            # Coverage breadth boost: first-layer candidates get a novelty bonus
            # proportional to how many faces they uniquely cover. Stack layers
            # (layer_idx > 0) add no new face coverage, so they get no boost.
            if layer_idx == 0 and cand.source_faces:
                novelty = len(cand.source_faces) / max(len(all_source_faces), 1)
                sort_gain = gain * (1.0 + 0.6 * novelty)
            layer_options.append((sort_gain, gain, rank, cand_idx))

    layer_options.sort(key=lambda item: (-item[0], item[2]))
    selected = layer_options[:part_budget_max]

    selected_layer_count: Dict[int, int] = {}
    for _, _, _, cand_idx in selected:
        selected_layer_count[cand_idx] = selected_layer_count.get(cand_idx, 0) + 1

    parts: List[ManufacturingPart] = []
    part_counter = 0
    stacked_regions = 0
    selected_regions = 0
    for cand_idx in ranked_indices:
        selected_layers = int(selected_layer_count.get(cand_idx, 0))
        if selected_layers <= 0:
            continue
        selected_regions += 1
        if selected_layers > 1:
            stacked_regions += 1

        cand = candidates[cand_idx]
        normal = cand.normal / max(np.linalg.norm(cand.normal), 1e-8)
        member_thickness = member_thickness_by_candidate.get(cand_idx)
        sheet_thick = float(cand.profile.thickness_mm)

        # Determine stack alignment mode from merged-pair metadata
        cand_alignment = cand.metadata.get("stack_alignment", "centered")
        ext_origin_raw = cand.metadata.get("exterior_face_origin_3d")
        ext_normal_raw = cand.metadata.get("exterior_face_normal")
        use_exterior_flush = (
            cand_alignment == "exterior_flush"
            and ext_origin_raw is not None
            and ext_normal_raw is not None
        )

        if use_exterior_flush:
            # Exterior-flush: place outermost sheet flush with exterior face,
            # stack inward along the *opposite* of the exterior normal.
            ext_origin = np.asarray(ext_origin_raw, dtype=float)
            ext_normal = np.asarray(ext_normal_raw, dtype=float)
            ext_normal = ext_normal / max(float(np.linalg.norm(ext_normal)), 1e-12)
            # Sheet k center = ext_origin - (k + 0.5) * sheet_thick * ext_normal
            # Clamp so sheet center stays within member thickness from exterior
            max_inward = (
                float(member_thickness) - 0.5 * sheet_thick
                if member_thickness is not None and member_thickness > 1e-6
                else float("inf")
            )
            sheet_centers = []
            for k in range(selected_layers):
                inward_dist = (k + 0.5) * sheet_thick
                inward_dist = min(inward_dist, max_inward)
                center = ext_origin - inward_dist * ext_normal
                sheet_centers.append(center)
        else:
            # Centered mode: use symmetric offsets.
            # For merged pairs where member_thickness is comparable to the total
            # stack height, center at position_3d (midpoint between faces).
            # When member_thickness is much larger (faces span the whole mesh),
            # keep origin_3d as anchor since the midpoint is in empty space.
            stack_height = selected_layers * sheet_thick
            use_midpoint = (
                cand.metadata.get("kind") == "merged_facet_pair"
                and member_thickness is not None
                and member_thickness > 1e-6
                and stack_height >= 0.5 * member_thickness
            )
            if use_midpoint:
                center_anchor = np.asarray(cand.position_3d, dtype=float)
            else:
                center_anchor = (
                    np.asarray(cand.origin_3d, dtype=float)
                    if cand.origin_3d is not None
                    else np.asarray(cand.position_3d, dtype=float)
                )
            offsets = _stack_layer_offsets_mm(
                selected_layers,
                sheet_thickness_mm=sheet_thick,
                member_thickness_mm=member_thickness,
            )
            sheet_centers = [center_anchor + off * normal for off in offsets]

        stack_group_id = f"cand_{cand_idx:04d}"

        # profile_origin for computing stack_layer_offset_mm
        profile_origin = (
            np.asarray(cand.origin_3d, dtype=float)
            if cand.origin_3d is not None
            else np.asarray(cand.position_3d, dtype=float)
        )

        for layer_idx, sheet_center in enumerate(sheet_centers):
            part_id = f"part_{part_counter:02d}"
            part_counter += 1
            metadata = dict(cand.metadata)
            outline_area_mm2 = float(max(0.0, cand.profile.outline.area))
            source_area_mm2 = float(max(0.0, cand.area_mm2))
            metadata["stack_group_id"] = stack_group_id
            metadata["stack_layer_index"] = int(layer_idx + 1)
            metadata["stack_layer_count"] = int(selected_layers)
            metadata["stack_target_layers"] = int(target_layers_by_candidate[cand_idx])
            # Compute offset as dot(sheet_center - profile_origin, normal) so the
            # serialization formula origin_3d + offset * normal still works.
            layer_offset = float(np.dot(sheet_center - profile_origin, normal))
            metadata["stack_layer_offset_mm"] = layer_offset
            metadata["candidate_outline_area_mm2"] = outline_area_mm2
            metadata["candidate_source_area_mm2"] = source_area_mm2
            metadata["mesh_constraint_pressure"] = _candidate_mesh_constraint_pressure(
                cand
            )
            if member_thickness is not None:
                metadata["member_thickness_mm"] = float(member_thickness)
            score_factor = score_factor_by_candidate.get(cand_idx, 1.0)
            if score_factor < 0.999:
                metadata["thin_side_score_factor"] = float(score_factor)
            thin_state = thin_side_state_by_candidate.get(cand_idx, "none")
            if thin_state != "none":
                metadata["thin_side_state"] = thin_state
            base_gain = float(cand.score)
            if layer_idx > 0:
                base_gain *= extra_layer_gain * (0.85 ** (layer_idx - 1))
            metadata["selection_gain"] = float(base_gain * score_factor)

            parts.append(
                ManufacturingPart(
                    part_id=part_id,
                    material_key=cand.profile.material_key,
                    thickness_mm=cand.profile.thickness_mm,
                    profile=cand.profile,
                    region_type=cand.region_type,
                    position_3d=sheet_center,
                    rotation_3d=_normal_to_rotation(cand.normal),
                    source_area_mm2=cand.area_mm2,
                    source_faces=list(cand.source_faces),
                    bend_ops=list(cand.bend_ops),
                    metadata=metadata,
                )
            )

    parts, intersection_debug = _filter_invalid_part_intersections(
        parts=parts,
        joint_distance_mm=input_spec.joint_distance_mm,
        enabled=input_spec.enable_intersection_filter,
        allow_joint_intent=input_spec.allow_joint_intent_intersections,
        clearance_mm=input_spec.intersection_clearance_mm,
    )
    final_stack_groups: Dict[str, int] = {}
    for part in parts:
        group_id = str(part.metadata.get("stack_group_id", "")).strip()
        if not group_id:
            group_id = part.part_id
        final_stack_groups[group_id] = final_stack_groups.get(group_id, 0) + 1
    final_selected_regions = len(final_stack_groups)
    final_stacked_regions = sum(1 for cnt in final_stack_groups.values() if cnt > 1)
    final_extra_layers = sum(max(0, cnt - 1) for cnt in final_stack_groups.values())
    constraint_debug = _compute_constraint_difficulty_summary(parts)

    selection_debug: Dict[str, object] = {
        "stack_enabled": bool(input_spec.enable_planar_stacking),
        "selected_region_count": int(final_selected_regions),
        "stacked_region_count": int(final_stacked_regions),
        "stacked_extra_layers": int(final_extra_layers),
        "prefilter_selected_region_count": int(selected_regions),
        "prefilter_stacked_region_count": int(stacked_regions),
        "max_layers_on_selected_region": int(
            max((int(v) for v in selected_layer_count.values()), default=0)
        ),
        "thin_side_suppression_enabled": bool(input_spec.enable_thin_side_suppression),
        "thin_side_matched_count": int(thin_side_matched_count),
        "thin_side_penalized_count": int(thin_side_penalized_count),
        "thin_side_dropped_count": int(thin_side_dropped_count),
        **intersection_debug,
        **constraint_debug,
    }
    return parts, selection_debug


def _candidate_member_thickness_mm(cand: _RegionCandidate) -> Optional[float]:
    value = cand.metadata.get("member_thickness_mm")
    if value is None:
        return None
    try:
        thickness = float(value)
    except (TypeError, ValueError):
        return None
    return thickness if thickness > 0.0 else None


def _stack_coverage_refs(
    candidates: List[_RegionCandidate],
    target_layers_by_candidate: Dict[int, int],
    member_thickness_by_candidate: Dict[int, float],
) -> List[Tuple[float, float, float]]:
    refs: List[Tuple[float, float, float]] = []
    for cand_idx, member_thickness in member_thickness_by_candidate.items():
        cand = candidates[cand_idx]
        if cand.region_type != RegionType.PLANAR_CUT:
            continue
        layers = int(target_layers_by_candidate.get(cand_idx, 1))
        if layers <= 1:
            continue
        sheet_thickness = float(max(1e-6, cand.profile.thickness_mm))
        coverage = float(layers * sheet_thickness / max(member_thickness, 1e-6))
        refs.append((float(member_thickness), coverage, float(cand.area_mm2)))
    refs.sort(key=lambda item: item[2], reverse=True)
    return refs


def _thin_side_candidate_score_factor(
    cand: _RegionCandidate,
    member_thickness_mm: Optional[float],
    coverage_refs: List[Tuple[float, float, float]],
    input_spec: Step3Input,
) -> Tuple[float, str]:
    if not input_spec.enable_thin_side_suppression:
        return 1.0, "none"
    if not input_spec.enable_planar_stacking:
        return 1.0, "none"
    if cand.region_type != RegionType.PLANAR_CUT:
        return 1.0, "none"
    if member_thickness_mm is not None:
        return 1.0, "none"
    if not coverage_refs:
        return 1.0, "none"

    lo, hi = _candidate_outline_dims(cand)
    if hi <= 1e-6:
        return 1.0, "none"
    aspect = lo / hi
    if aspect > float(np.clip(input_spec.thin_side_aspect_limit, 0.05, 0.95)):
        return 1.0, "none"

    dim_multiplier = max(1.0, float(input_spec.thin_side_dim_multiplier))
    best_rel = float("inf")
    matched_coverage: Optional[float] = None
    for ref_thickness, coverage, _ in coverage_refs:
        if lo < 0.35 * ref_thickness:
            continue
        if lo > dim_multiplier * ref_thickness:
            continue
        rel = abs(lo - ref_thickness) / max(ref_thickness, 1e-6)
        if rel < best_rel:
            best_rel = rel
            matched_coverage = float(coverage)

    if matched_coverage is None:
        return 1.0, "none"

    start = max(0.0, float(input_spec.thin_side_coverage_penalty_start))
    drop = max(start + 1e-3, float(input_spec.thin_side_coverage_drop_threshold))
    if matched_coverage >= drop:
        return 0.0, "dropped"
    if matched_coverage <= start:
        return 1.0, "kept"

    t = (matched_coverage - start) / max(drop - start, 1e-6)
    factor = max(0.1, 1.0 - 0.9 * t)
    return float(factor), "penalized"


def _target_stack_layers(
    cand: _RegionCandidate,
    input_spec: Step3Input,
    member_thickness_mm: Optional[float],
) -> int:
    if not input_spec.enable_planar_stacking:
        return 1
    if cand.region_type != RegionType.PLANAR_CUT:
        return 1
    if member_thickness_mm is None:
        return 1

    sheet_thickness = float(max(1e-6, cand.profile.thickness_mm))
    ratio = member_thickness_mm / sheet_thickness
    if ratio <= 1.0:
        return 1

    bias = float(np.clip(input_spec.stack_roundup_bias, 0.0, 0.95))
    target = int(math.ceil(max(1.0, ratio - bias)))
    max_layers = max(1, int(input_spec.max_stack_layers_per_region))
    return int(np.clip(target, 1, max_layers))


def _stack_layer_offsets_mm(
    selected_layers: int,
    sheet_thickness_mm: float,
    member_thickness_mm: Optional[float],
) -> List[float]:
    if selected_layers <= 1:
        return [0.0]

    step = float(max(0.1, sheet_thickness_mm))
    offsets = (
        np.arange(selected_layers, dtype=float) - 0.5 * (selected_layers - 1)
    ) * step

    if member_thickness_mm is not None and member_thickness_mm > 1e-6:
        max_abs = float(np.max(np.abs(offsets)))
        max_allowed = 0.5 * float(member_thickness_mm)
        if max_abs > 1e-9 and max_allowed < max_abs:
            offsets *= max_allowed / max_abs
    return [float(v) for v in offsets]


def _compute_constraint_difficulty_summary(
    parts: List[ManufacturingPart],
) -> Dict[str, float | List[Dict[str, float | str | int]]]:
    if not parts:
        return {
            "plane_constraint_difficulty_mean": 0.0,
            "plane_constraint_difficulty_weighted": 0.0,
            "plane_constraint_difficulty_max": 0.0,
            "mesh_constraint_pressure_mean": 0.0,
            "mesh_constraint_pressure_weighted": 0.0,
            "intersection_constraint_pressure_mean": 0.0,
            "intersection_constraint_pressure_weighted": 0.0,
            "constraint_difficulty_parts": [],
        }

    rows: List[Dict[str, float | str | int]] = []
    mesh_vals: List[float] = []
    inter_vals: List[float] = []
    diff_vals: List[float] = []
    weights: List[float] = []

    for part in parts:
        metadata = part.metadata
        mesh_pressure = float(
            np.clip(float(metadata.get("mesh_constraint_pressure", 0.0)), 0.0, 1.0)
        )
        conflicts = max(0, int(metadata.get("intersection_conflict_count", 0)))
        intersection_pressure = float(1.0 - (1.0 / (1.0 + float(conflicts))))
        difficulty = float(
            np.clip(0.75 * mesh_pressure + 0.25 * intersection_pressure, 0.0, 1.0)
        )

        weight_mm2 = float(
            max(
                1e-6,
                metadata.get("candidate_outline_area_mm2", part.profile.outline.area),
            )
        )
        mesh_vals.append(mesh_pressure)
        inter_vals.append(intersection_pressure)
        diff_vals.append(difficulty)
        weights.append(weight_mm2)

        metadata["intersection_constraint_pressure"] = intersection_pressure
        metadata["plane_constraint_difficulty"] = difficulty

        rows.append(
            {
                "part_id": part.part_id,
                "mesh_constraint_pressure": mesh_pressure,
                "intersection_constraint_pressure": intersection_pressure,
                "plane_constraint_difficulty": difficulty,
                "intersection_conflict_count": conflicts,
                "area_weight_mm2": weight_mm2,
            }
        )

    w = np.asarray(weights, dtype=float)
    w /= max(1e-6, float(np.sum(w)))
    mesh_arr = np.asarray(mesh_vals, dtype=float)
    inter_arr = np.asarray(inter_vals, dtype=float)
    diff_arr = np.asarray(diff_vals, dtype=float)

    return {
        "plane_constraint_difficulty_mean": float(np.mean(diff_arr)),
        "plane_constraint_difficulty_weighted": float(np.sum(diff_arr * w)),
        "plane_constraint_difficulty_max": float(np.max(diff_arr)),
        "mesh_constraint_pressure_mean": float(np.mean(mesh_arr)),
        "mesh_constraint_pressure_weighted": float(np.sum(mesh_arr * w)),
        "intersection_constraint_pressure_mean": float(np.mean(inter_arr)),
        "intersection_constraint_pressure_weighted": float(np.sum(inter_arr * w)),
        "constraint_difficulty_parts": rows,
    }


def _filter_invalid_part_intersections(
    parts: List[ManufacturingPart],
    joint_distance_mm: float,
    enabled: bool,
    allow_joint_intent: bool,
    clearance_mm: float,
) -> Tuple[List[ManufacturingPart], Dict[str, Any]]:
    if not enabled or len(parts) < 2:
        return (
            parts,
            {
                "intersection_filter_enabled": bool(enabled),
                "intersection_clearance_mm": float(clearance_mm),
                "intersection_dropped_count": 0,
                "intersection_allowed_stack_count": 0,
                "intersection_allowed_joint_intent_count": 0,
                "intersection_events": [],
                "intersection_part_decisions": [],
                "intersection_dropped_part_ids": [],
            },
        )

    ordered = sorted(
        parts,
        key=lambda p: float(p.metadata.get("selection_gain", p.source_area_mm2)),
        reverse=True,
    )
    kept: List[ManufacturingPart] = []
    dropped_count = 0
    allowed_stack = 0
    allowed_joint_intent_count = 0
    intersection_events: List[Dict[str, Any]] = []
    part_decisions: List[Dict[str, Any]] = []
    dropped_part_ids: List[str] = []

    for part in ordered:
        candidate_id = str(part.part_id)
        rejected = False
        local_conflicts = 0
        local_allowed_stack = 0
        local_allowed_joint_intent = 0
        decision_reason = "kept_no_conflict"
        for prev in kept:
            if not _parts_intersect(part, prev, clearance_mm=clearance_mm):
                continue

            local_conflicts += 1
            group_a = str(part.metadata.get("stack_group_id", "")).strip()
            group_b = str(prev.metadata.get("stack_group_id", "")).strip()
            if group_a and group_b and group_a == group_b:
                allowed_stack += 1
                local_allowed_stack += 1
                intersection_events.append(
                    {
                        "candidate_part_id": candidate_id,
                        "against_part_id": str(prev.part_id),
                        "action": "allow_stack_group",
                        "stack_group_id": group_a,
                        "clearance_mm": float(clearance_mm),
                    }
                )
                continue

            # Use plane geometry to decide: only parallel parts on the same
            # plane can genuinely occupy the same space.  Non-parallel OBB
            # overlap is expected at joint edges.
            if not _is_genuine_plane_conflict(part, prev, clearance_mm):
                allowed_joint_intent_count += 1
                local_allowed_joint_intent += 1
                intersection_events.append(
                    {
                        "candidate_part_id": candidate_id,
                        "against_part_id": str(prev.part_id),
                        "action": "allow_nonparallel_joint_intent",
                        "clearance_mm": float(clearance_mm),
                    }
                )
                continue

            rejected = True
            decision_reason = "reject_genuine_plane_conflict"
            intersection_events.append(
                {
                    "candidate_part_id": candidate_id,
                    "against_part_id": str(prev.part_id),
                    "action": "reject_genuine_plane_conflict",
                    "clearance_mm": float(clearance_mm),
                }
            )
            break

        if rejected:
            dropped_count += 1
            dropped_part_ids.append(candidate_id)
            part_decisions.append(
                {
                    "part_id_before_reindex": candidate_id,
                    "decision": "dropped",
                    "reason": decision_reason,
                    "conflict_count": int(local_conflicts),
                    "allowed_stack_contacts": int(local_allowed_stack),
                    "allowed_joint_intent_contacts": int(local_allowed_joint_intent),
                }
            )
            continue
        part.metadata["intersection_conflict_count"] = int(local_conflicts)
        part.metadata["intersection_allowed_stack_contacts"] = int(local_allowed_stack)
        part.metadata["intersection_allowed_joint_intent_contacts"] = int(
            local_allowed_joint_intent
        )
        part_decisions.append(
            {
                "part_id_before_reindex": candidate_id,
                "decision": "kept",
                "reason": decision_reason,
                "conflict_count": int(local_conflicts),
                "allowed_stack_contacts": int(local_allowed_stack),
                "allowed_joint_intent_contacts": int(local_allowed_joint_intent),
            }
        )
        kept.append(part)

    kept.sort(key=lambda p: p.part_id)
    reindex_map: Dict[str, str] = {}
    for idx, part in enumerate(kept):
        old_id = str(part.part_id)
        new_id = f"part_{idx:02d}"
        part.part_id = new_id
        reindex_map[old_id] = new_id
    for row in part_decisions:
        old = str(row.get("part_id_before_reindex", ""))
        if old in reindex_map:
            row["part_id_after_reindex"] = reindex_map[old]
    for event in intersection_events:
        old_cand = str(event.get("candidate_part_id", ""))
        old_against = str(event.get("against_part_id", ""))
        if old_cand in reindex_map:
            event["candidate_part_id_after_reindex"] = reindex_map[old_cand]
        if old_against in reindex_map:
            event["against_part_id_after_reindex"] = reindex_map[old_against]

    debug = {
        "intersection_filter_enabled": bool(enabled),
        "intersection_clearance_mm": float(clearance_mm),
        "intersection_dropped_count": int(dropped_count),
        "intersection_allowed_stack_count": int(allowed_stack),
        "intersection_allowed_joint_intent_count": int(allowed_joint_intent_count),
        "intersection_events": intersection_events,
        "intersection_part_decisions": part_decisions,
        "intersection_dropped_part_ids": dropped_part_ids,
        "intersection_reindex_map": reindex_map,
    }
    return kept, debug


def _is_genuine_plane_conflict(
    a: ManufacturingPart,
    b: ManufacturingPart,
    clearance_mm: float,
) -> bool:
    """Check if two parts with overlapping OBBs genuinely occupy the same space.

    Only near-parallel parts on the same plane are real conflicts.
    Non-parallel parts (perpendicular walls meeting a shelf, etc.) overlap
    at joint edges — that's expected geometry, not a conflict.
    """
    n_a = _rotation_to_normal(a.rotation_3d)
    n_b = _rotation_to_normal(b.rotation_3d)

    # Non-parallel parts can't conflict: OBB overlap is at joint edges
    if abs(float(np.dot(n_a, n_b))) < 0.85:
        return False

    # Parallel parts: only conflict if on the same plane
    delta = b.position_3d - a.position_3d
    normal_dist = abs(float(np.dot(delta, n_a)))
    max_thickness = max(a.thickness_mm, b.thickness_mm)

    # Separated by more than one sheet thickness → different parallel planes
    if normal_dist > max_thickness + clearance_mm:
        return False

    return True


def _part_obb(part: ManufacturingPart) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = _rotation_matrix_xyz(part.rotation_3d)
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
    center_local = np.array(
        [
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
            0.0,
        ],
        dtype=float,
    )
    center_world = rot @ center_local + np.asarray(part.position_3d, dtype=float)
    axes = np.column_stack(
        [
            rot[:, 0] / max(np.linalg.norm(rot[:, 0]), 1e-8),
            rot[:, 1] / max(np.linalg.norm(rot[:, 1]), 1e-8),
            rot[:, 2] / max(np.linalg.norm(rot[:, 2]), 1e-8),
        ]
    )
    return center_world, axes, half


def _parts_intersect(
    a: ManufacturingPart,
    b: ManufacturingPart,
    clearance_mm: float,
) -> bool:
    return _parts_signed_gap_mm(a, b, clearance_mm=clearance_mm) <= 0.0


def _parts_signed_gap_mm(
    a: ManufacturingPart,
    b: ManufacturingPart,
    clearance_mm: float = 0.0,
) -> float:
    ca, aa, ha = _part_obb(a)
    cb, ab, hb = _part_obb(b)
    shrink = max(0.0, float(clearance_mm)) * 0.5
    ha = np.maximum(0.05, ha - shrink)
    hb = np.maximum(0.05, hb - shrink)
    return _obb_max_separation(ca, aa, ha, cb, ab, hb)


def _parts_in_contact_band(
    a: ManufacturingPart,
    b: ManufacturingPart,
    contact_tolerance_mm: float,
) -> Tuple[bool, float]:
    signed_gap = _parts_signed_gap_mm(a, b, clearance_mm=0.0)
    tolerance = max(0.0, float(contact_tolerance_mm))
    return signed_gap <= tolerance, signed_gap


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


def _identify_joint_contact_pairs(
    parts: List[ManufacturingPart],
    contact_tolerance_mm: float = 2.0,
    parallel_dot_threshold: float = 0.95,
    joint_distance_mm: float = 240.0,
) -> Set[Tuple[int, int]]:
    """Identify part index pairs that will need joints (pre-trim).

    Uses OBB contact detection + parallelism filtering only.
    Does NOT need trimmed outlines.
    Returns set of (i, j) tuples where i < j.
    """
    pairs: Set[Tuple[int, int]] = set()
    parallel_threshold = float(np.clip(parallel_dot_threshold, 0.0, 0.999999))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pa = parts[i]
            pb = parts[j]
            group_a = str(pa.metadata.get("stack_group_id", "")).strip()
            group_b = str(pb.metadata.get("stack_group_id", "")).strip()
            if group_a and group_b and group_a == group_b:
                continue

            n_a = _rotation_to_normal(pa.rotation_3d)
            n_b = _rotation_to_normal(pb.rotation_3d)
            if abs(float(np.dot(n_a, n_b))) >= parallel_threshold:
                continue

            in_contact, signed_gap = _parts_in_contact_band(
                pa, pb, contact_tolerance_mm=contact_tolerance_mm,
            )
            if not in_contact:
                continue

            distance = float(np.linalg.norm(pa.position_3d - pb.position_3d))
            if distance > float(max(1.0, joint_distance_mm * 1.8)):
                continue

            pairs.add((i, j))
    return pairs


def _synthesize_joint_specs(
    parts: List[ManufacturingPart],
    joint_distance_mm: float,
    contact_tolerance_mm: float,
    parallel_dot_threshold: float = 0.95,
    contact_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> List[JointSpec]:
    joints: List[JointSpec] = []
    parallel_threshold = float(np.clip(parallel_dot_threshold, 0.0, 0.999999))

    # If contact_pairs were pre-computed, iterate only those pairs.
    if contact_pairs is not None:
        candidate_pairs = sorted(contact_pairs)
    else:
        candidate_pairs = [
            (i, j) for i in range(len(parts)) for j in range(i + 1, len(parts))
        ]

    for i, j in candidate_pairs:
        pa = parts[i]
        pb = parts[j]
        group_a = str(pa.metadata.get("stack_group_id", "")).strip()
        group_b = str(pb.metadata.get("stack_group_id", "")).strip()
        if group_a and group_b and group_a == group_b:
            continue

        n_a = _rotation_to_normal(pa.rotation_3d)
        n_b = _rotation_to_normal(pb.rotation_3d)
        if abs(float(np.dot(n_a, n_b))) >= parallel_threshold:
            continue

        # Re-check contact (outlines may have changed after trimming)
        in_contact, signed_gap = _parts_in_contact_band(
            pa, pb, contact_tolerance_mm=contact_tolerance_mm,
        )
        if not in_contact:
            continue

        distance = float(np.linalg.norm(pa.position_3d - pb.position_3d))
        if distance > float(max(1.0, joint_distance_mm * 1.8)):
            continue

        if pa.bend_ops or pb.bend_ops:
            joint_type = "through_bolt"
            fastener = "M6_bolt"
        else:
            joint_type = "tab_slot"
            fastener = None

        joints.append(
            JointSpec(
                joint_type=joint_type,
                part_a=pa.part_id,
                part_b=pb.part_id,
                geometry={
                    "distance_mm": distance,
                    "contact_gap_mm": float(max(0.0, signed_gap)),
                    "overlap_mm": float(max(0.0, -signed_gap)),
                },
                clearance_mm=0.254,
                fastener_spec=fastener,
            )
        )
    return joints


# ─── Joint geometry: tabs, slots, cross-laps ─────────────────────────────────


def _compute_contact_line(
    part_a: ManufacturingPart,
    part_b: ManufacturingPart,
) -> Optional[Tuple[np.ndarray, LineString, LineString]]:
    """Compute 3D plane-plane intersection and clip to both outlines.

    Returns (line_dir_3d, line_2d_a, line_2d_b) or None.
    """
    prof_a = part_a.profile
    prof_b = part_b.profile
    if (
        prof_a.basis_u is None
        or prof_a.basis_v is None
        or prof_a.origin_3d is None
        or prof_b.basis_u is None
        or prof_b.basis_v is None
        or prof_b.origin_3d is None
    ):
        return None

    n_a = _rotation_to_normal(part_a.rotation_3d)
    n_b = _rotation_to_normal(part_b.rotation_3d)
    line_dir = np.cross(n_a, n_b)
    if np.linalg.norm(line_dir) < 0.1:
        return None
    line_dir = line_dir / np.linalg.norm(line_dir)

    d_a = float(np.dot(n_a, prof_a.origin_3d))
    d_b = float(np.dot(n_b, prof_b.origin_3d))

    A = np.array([n_a, n_b, line_dir])
    b = np.array([d_a, d_b, 0.0])
    try:
        p0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # Build long 3D segment along the intersection line
    far = 5000.0
    p_start = p0 - far * line_dir
    p_end = p0 + far * line_dir

    # Project into part_a's 2D frame and clip
    a0 = prof_a.project_3d_to_2d(p_start)
    a1 = prof_a.project_3d_to_2d(p_end)
    line_a = LineString([a0, a1])
    clipped_a = prof_a.outline.intersection(line_a)
    if clipped_a.is_empty or clipped_a.length < 1e-3:
        return None

    # Project into part_b's 2D frame and clip
    b0 = prof_b.project_3d_to_2d(p_start)
    b1 = prof_b.project_3d_to_2d(p_end)
    line_b = LineString([b0, b1])
    clipped_b = prof_b.outline.intersection(line_b)
    if clipped_b.is_empty or clipped_b.length < 1e-3:
        return None

    # Ensure we have LineStrings (not MultiLineString)
    if clipped_a.geom_type == "MultiLineString":
        clipped_a = max(clipped_a.geoms, key=lambda g: g.length)
    if clipped_b.geom_type == "MultiLineString":
        clipped_b = max(clipped_b.geoms, key=lambda g: g.length)

    if clipped_a.geom_type != "LineString" or clipped_b.geom_type != "LineString":
        return None

    return (line_dir, clipped_a, clipped_b)


def _classify_joint_type(
    part_a: ManufacturingPart,
    part_b: ManufacturingPart,
    line_2d_a: LineString,
    line_2d_b: LineString,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Classify a joint as tab_slot, cross_lap, or butt.

    Returns (type, tab_part_id_or_None, slot_part_id_or_None).
    """
    mid_a = line_2d_a.interpolate(0.5, normalized=True)
    mid_b = line_2d_b.interpolate(0.5, normalized=True)

    edge_dist_a = float(part_a.profile.outline.boundary.distance(mid_a))
    edge_dist_b = float(part_b.profile.outline.boundary.distance(mid_b))

    thresh_a = max(2.0 * part_a.thickness_mm, 10.0)
    thresh_b = max(2.0 * part_b.thickness_mm, 10.0)

    near_a = edge_dist_a < thresh_a
    near_b = edge_dist_b < thresh_b

    if near_a and not near_b:
        return ("tab_slot", part_a.part_id, part_b.part_id)
    if near_b and not near_a:
        return ("tab_slot", part_b.part_id, part_a.part_id)
    if not near_a and not near_b:
        return ("cross_lap", None, None)
    return ("butt", None, None)


def _outward_perpendicular(
    outline: Polygon,
    point_2d: np.ndarray,
    tangent_2d: np.ndarray,
) -> np.ndarray:
    """Return the perpendicular to tangent that points away from outline interior."""
    perp1 = np.array([-tangent_2d[1], tangent_2d[0]])
    perp2 = np.array([tangent_2d[1], -tangent_2d[0]])
    test_dist = 1.0
    p1 = Point(point_2d[0] + perp1[0] * test_dist, point_2d[1] + perp1[1] * test_dist)
    if outline.contains(p1):
        return perp2
    return perp1


def _apply_tab_slot(
    tab_part: ManufacturingPart,
    slot_part: ManufacturingPart,
    line_2d_tab: LineString,
    line_2d_slot: LineString,
    line_dir_3d: np.ndarray,
    clearance: float,
    input_spec: Step3Input,
) -> Optional[dict]:
    """Add tab protrusions to tab_part and slot cutouts to slot_part.

    Returns geometry debug dict, or None to fall back to butt.
    """
    min_slot_width = DFMConfig.from_material(slot_part.material_key).min_slot_width_mm
    tab_width = slot_part.thickness_mm - 2.0 * clearance
    if tab_width < min_slot_width:
        return None

    tab_depth = tab_part.thickness_mm
    tab_length = input_spec.joint_tab_length_mm
    contact_length = line_2d_tab.length
    tab_spacing = input_spec.joint_tab_spacing_mm

    # Tab placement: anchor tabs near both ends first, then fill the middle.
    # Edge inset must keep the full slot footprint (half slot length + dogbone
    # relief) inside the slot part's outline, plus the DFM minimum bridge width
    # between the cutout edge and the part outline edge.
    dfm = DFMConfig.from_material(slot_part.material_key)
    dfm_relief = dfm.min_internal_radius_mm
    slot_half_footprint = (tab_length + clearance) / 2.0 + dfm_relief + 0.5
    edge_inset = slot_half_footprint + dfm.min_bridge_width_mm
    min_two_tab_contact = 2 * edge_inset + tab_length

    if contact_length >= min_two_tab_contact:
        # Two end tabs, then fill the middle gap if spacing allows.
        frac_first = edge_inset / contact_length
        frac_last = 1.0 - frac_first
        middle_span = contact_length - 2 * edge_inset
        n_middle = max(0, int(middle_span / tab_spacing) - 1)
        n_middle = min(n_middle, 3)  # cap total at 5
        tab_fracs = [frac_first]
        for mi in range(n_middle):
            tab_fracs.append(
                frac_first + (mi + 1) / (n_middle + 1) * (frac_last - frac_first)
            )
        tab_fracs.append(frac_last)
    else:
        # Too short for two — single centered tab.
        tab_fracs = [0.5]

    tabs_info = []
    coords_tab = np.array(line_2d_tab.coords)
    tangent_tab = coords_tab[-1] - coords_tab[0]
    tang_len = np.linalg.norm(tangent_tab)
    if tang_len < 1e-6:
        return None
    tangent_tab = tangent_tab / tang_len

    # Precompute tab part's normal for thickness-center offset
    n_tab = _rotation_to_normal(tab_part.rotation_3d)
    outline_boundary = tab_part.profile.outline.boundary

    # Compute tab protrusion direction: outward from the part interior toward
    # the contact edge.  We use the direction from the outline's centroid to
    # the contact midpoint, projected onto the perpendicular to the contact
    # tangent.  This is robust regardless of outline expansion (stacking) or
    # degenerate slot-origin projections.
    contact_mid = np.array(line_2d_tab.interpolate(0.5, normalized=True).coords[0])
    centroid = np.array(tab_part.profile.outline.centroid.coords[0])
    outward_from_centroid = contact_mid - centroid
    perp_candidate = np.array([-tangent_tab[1], tangent_tab[0]])
    if np.dot(outward_from_centroid, perp_candidate) >= 0:
        outward_tab = perp_candidate
    else:
        outward_tab = -perp_candidate

    pre_tab_bbox = list(tab_part.profile.outline.bounds)  # [minx, miny, maxx, maxy]

    for frac in tab_fracs:
        pt_on_tab = np.array(line_2d_tab.interpolate(frac, normalized=True).coords[0])

        # Snap tab base to the nearest point on the outline boundary so the
        # tab protrudes BEYOND the current outline edge, not inside it.
        nearest = outline_boundary.interpolate(
            outline_boundary.project(Point(pt_on_tab[0], pt_on_tab[1]))
        )
        pt_on_edge = np.array([nearest.x, nearest.y])

        # Build tab rectangle from the outline edge, extending outward
        half_len = tab_length / 2.0
        corners = [
            pt_on_edge - half_len * tangent_tab,
            pt_on_edge + half_len * tangent_tab,
            pt_on_edge + half_len * tangent_tab + tab_depth * outward_tab,
            pt_on_edge - half_len * tangent_tab + tab_depth * outward_tab,
        ]
        tab_rect = Polygon(corners)
        tab_rect = _clean_polygon(tab_rect)
        if tab_rect is None:
            continue

        # Union tab rectangle with tab part's outline
        new_outline = tab_part.profile.outline.union(tab_rect)
        new_outline = _clean_polygon(new_outline)
        if new_outline is not None:
            tab_part.profile.outline = new_outline
            # Update boundary reference after outline change
            outline_boundary = tab_part.profile.outline.boundary

        # Project tab center to 3D, offset by half thickness toward slot part,
        # then project to slot part's 2D so the slot is centered on the tab
        # part's thickness, not at one surface.
        center_3d = tab_part.profile.project_2d_to_3d(pt_on_tab[0], pt_on_tab[1])
        toward_slot = slot_part.profile.origin_3d - center_3d
        sign = 1.0 if np.dot(toward_slot, n_tab) > 0 else -1.0
        center_3d_centered = center_3d + sign * 0.5 * tab_part.thickness_mm * n_tab
        center_slot_2d = slot_part.profile.project_3d_to_2d(center_3d_centered)
        center_slot = np.array(center_slot_2d)

        # Build slot rectangle in slot part's 2D frame.
        # Use the contact line direction in the slot part's own 2D frame
        # (not the tab tangent projected through 3D, which can rotate).
        coords_slot = np.array(line_2d_slot.coords)
        slot_tangent = coords_slot[-1] - coords_slot[0]
        st_len = np.linalg.norm(slot_tangent)
        if st_len < 1e-8:
            continue
        slot_tangent = slot_tangent / st_len
        slot_perp = np.array([-slot_tangent[1], slot_tangent[0]])

        slot_half_len = (tab_length + clearance) / 2.0
        slot_half_wid = (tab_part.thickness_mm + 2.0 * clearance) / 2.0
        slot_corners = [
            center_slot - slot_half_len * slot_tangent - slot_half_wid * slot_perp,
            center_slot + slot_half_len * slot_tangent - slot_half_wid * slot_perp,
            center_slot + slot_half_len * slot_tangent + slot_half_wid * slot_perp,
            center_slot - slot_half_len * slot_tangent + slot_half_wid * slot_perp,
        ]
        slot_rect = Polygon(slot_corners)
        slot_rect = _clean_polygon(slot_rect)
        if slot_rect is None:
            continue

        # Add dogbone relief for CNC internal corners
        slot_rect = add_dogbone_relief(slot_rect, dfm.min_internal_radius_mm)
        slot_part.profile.cutouts.append(slot_rect)

        tabs_info.append(
            {
                "center_2d_tab": [float(pt_on_tab[0]), float(pt_on_tab[1])],
                "center_2d_slot": [float(center_slot[0]), float(center_slot[1])],
            }
        )

    if not tabs_info:
        return None

    slot_width = tab_part.thickness_mm + 2.0 * clearance
    return {
        "tab_part": tab_part.part_id,
        "slot_part": slot_part.part_id,
        "tab_count": len(tabs_info),
        "tab_width_mm": float(tab_width),
        "tab_depth_mm": float(tab_depth),
        "slot_width_mm": float(slot_width),
        "tabs": tabs_info,
        "pre_tab_bbox": pre_tab_bbox,
    }


def _apply_cross_lap(
    part_a: ManufacturingPart,
    part_b: ManufacturingPart,
    line_2d_a: LineString,
    line_2d_b: LineString,
    line_dir_3d: np.ndarray,
    clearance: float,
) -> Optional[dict]:
    """Add cross-lap slots to both parts. Each gets a slot from one end to the midpoint."""
    dfm_a = DFMConfig.from_material(part_a.material_key)
    dfm_b = DFMConfig.from_material(part_b.material_key)

    slot_a_width = part_b.thickness_mm + 2.0 * clearance
    slot_b_width = part_a.thickness_mm + 2.0 * clearance

    for part, line_2d, slot_width, dfm, from_start in [
        (part_a, line_2d_a, slot_a_width, dfm_a, True),
        (part_b, line_2d_b, slot_b_width, dfm_b, False),
    ]:
        coords = np.array(line_2d.coords)
        midpoint = np.array(line_2d.interpolate(0.5, normalized=True).coords[0])
        if from_start:
            slot_start = coords[0]
            slot_end = midpoint
        else:
            slot_start = midpoint
            slot_end = coords[-1]

        slot_vec = slot_end - slot_start
        slot_len = float(np.linalg.norm(slot_vec))
        if slot_len < 1e-3:
            continue
        slot_dir = slot_vec / slot_len
        slot_perp = np.array([-slot_dir[1], slot_dir[0]])

        half_w = slot_width / 2.0
        corners = [
            slot_start - half_w * slot_perp,
            slot_end - half_w * slot_perp,
            slot_end + half_w * slot_perp,
            slot_start + half_w * slot_perp,
        ]
        slot_rect = Polygon(corners)
        slot_rect = _clean_polygon(slot_rect)
        if slot_rect is None:
            continue

        slot_rect = add_dogbone_relief(slot_rect, dfm.min_internal_radius_mm)
        part.profile.cutouts.append(slot_rect)

    return {
        "slot_a_width_mm": float(slot_a_width),
        "slot_b_width_mm": float(slot_b_width),
    }


def _apply_joint_geometry(
    parts: List[ManufacturingPart],
    joints: List[JointSpec],
    input_spec: Step3Input,
) -> Tuple[List[ManufacturingPart], List[JointSpec], dict]:
    """Apply tab/slot/cross-lap geometry to parts based on joint specs."""
    debug = {
        "joint_geometry_enabled": input_spec.joint_enable_geometry,
        "joints_processed": 0,
        "joints_skipped_no_basis": 0,
        "joints_skipped_parallel": 0,
        "joints_skipped_no_contact": 0,
        "tab_slot_count": 0,
        "cross_lap_count": 0,
        "butt_count": 0,
        "tabs_created": 0,
        "slots_created": 0,
        "cutout_overlap_warnings": 0,
    }

    if not input_spec.joint_enable_geometry:
        return parts, joints, debug

    part_map: Dict[str, ManufacturingPart] = {p.part_id: p for p in parts}

    # Build a lookup from stack_group_id → list of layer indices
    stack_groups: Dict[str, List[ManufacturingPart]] = {}
    for p in parts:
        gid = str(p.metadata.get("stack_group_id", "")).strip()
        if gid:
            stack_groups.setdefault(gid, []).append(p)

    for spec in joints:
        if spec.joint_type == "through_bolt":
            continue

        pa = part_map.get(spec.part_a)
        pb = part_map.get(spec.part_b)
        if pa is None or pb is None:
            continue

        # Skip if either part lacks basis vectors
        if (
            pa.profile.basis_u is None
            or pa.profile.origin_3d is None
            or pb.profile.basis_u is None
            or pb.profile.origin_3d is None
        ):
            debug["joints_skipped_no_basis"] += 1
            continue

        # For stacked parts, only process the layer closest to the partner
        for part, other in [(pa, pb), (pb, pa)]:
            gid = str(part.metadata.get("stack_group_id", "")).strip()
            if gid and gid in stack_groups and len(stack_groups[gid]) > 1:
                layers = stack_groups[gid]
                closest = min(
                    layers,
                    key=lambda lp: float(
                        np.linalg.norm(lp.position_3d - other.position_3d)
                    ),
                )
                if part.part_id != closest.part_id:
                    # This is not the closest layer — skip this joint pair
                    break
        else:
            # Both parts are the closest layers (or not stacked) — proceed
            pass

        # This is a bit awkward — we need to detect the break above
        # Rewrite: check both parts
        skip = False
        for part, other in [(pa, pb), (pb, pa)]:
            gid = str(part.metadata.get("stack_group_id", "")).strip()
            if gid and gid in stack_groups and len(stack_groups[gid]) > 1:
                layers = stack_groups[gid]
                closest = min(
                    layers,
                    key=lambda lp: float(
                        np.linalg.norm(lp.position_3d - other.position_3d)
                    ),
                )
                if part.part_id != closest.part_id:
                    skip = True
                    break
        if skip:
            continue

        contact = _compute_contact_line(pa, pb)
        if contact is None:
            debug["joints_skipped_parallel"] += 1
            continue

        line_dir_3d, line_2d_a, line_2d_b = contact

        # Check minimum contact length
        contact_length = min(line_2d_a.length, line_2d_b.length)
        if contact_length < input_spec.joint_min_contact_mm:
            debug["joints_skipped_no_contact"] += 1
            spec.joint_type = "butt"
            debug["butt_count"] += 1
            continue

        debug["joints_processed"] += 1

        jtype, tab_id, slot_id = _classify_joint_type(pa, pb, line_2d_a, line_2d_b)

        # Store classification debug info
        mid_a = line_2d_a.interpolate(0.5, normalized=True)
        mid_b = line_2d_b.interpolate(0.5, normalized=True)
        edge_dist_a = float(pa.profile.outline.boundary.distance(mid_a))
        edge_dist_b = float(pb.profile.outline.boundary.distance(mid_b))
        spec.geometry.update(
            {
                "classification": jtype,
                "edge_dist_a_mm": edge_dist_a,
                "edge_dist_b_mm": edge_dist_b,
                "edge_threshold_a_mm": max(2.0 * pa.thickness_mm, 10.0),
                "edge_threshold_b_mm": max(2.0 * pb.thickness_mm, 10.0),
                "contact_length_mm": float(contact_length),
                "contact_line_2d_a": [list(c) for c in line_2d_a.coords],
                "contact_line_2d_b": [list(c) for c in line_2d_b.coords],
            }
        )

        if jtype == "tab_slot":
            assert tab_id is not None and slot_id is not None
            tab_p = part_map[tab_id]
            slot_p = part_map[slot_id]
            line_tab = line_2d_a if tab_id == pa.part_id else line_2d_b
            line_slot = line_2d_b if tab_id == pa.part_id else line_2d_a

            geom = _apply_tab_slot(
                tab_p,
                slot_p,
                line_tab,
                line_slot,
                line_dir_3d,
                spec.clearance_mm,
                input_spec,
            )
            if geom is not None:
                spec.joint_type = "tab_slot"
                spec.geometry.update(geom)
                debug["tab_slot_count"] += 1
                debug["tabs_created"] += geom["tab_count"]
                debug["slots_created"] += geom["tab_count"]
            else:
                spec.joint_type = "butt"
                debug["butt_count"] += 1

        elif jtype == "cross_lap":
            geom = _apply_cross_lap(
                pa,
                pb,
                line_2d_a,
                line_2d_b,
                line_dir_3d,
                spec.clearance_mm,
            )
            if geom is not None:
                spec.joint_type = "cross_lap"
                spec.geometry.update(geom)
                debug["cross_lap_count"] += 1
            else:
                spec.joint_type = "butt"
                debug["butt_count"] += 1

        else:
            spec.joint_type = "butt"
            debug["butt_count"] += 1

    # Cutout overlap check
    overlap_warnings = 0
    for part in parts:
        cutouts = part.profile.cutouts
        for i in range(len(cutouts)):
            for j in range(i + 1, len(cutouts)):
                if cutouts[i].intersects(cutouts[j]):
                    logger.warning(
                        "Part %s: cutouts %d and %d overlap",
                        part.part_id,
                        i,
                        j,
                    )
                    overlap_warnings += 1
    debug["cutout_overlap_warnings"] = overlap_warnings

    return parts, joints, debug


def _validate_joint_fit_2d(
    parts: List[ManufacturingPart],
    joints: List[JointSpec],
) -> Tuple[List[Step3Violation], dict]:
    """Verify 2D fit properties for each joint with geometry.

    Checks:
    - Tab cross-section fits inside slot (tab_slot joints)
    - Slot cutouts are inside part outlines
    - No cutout-cutout collisions within a part
    """
    violations: List[Step3Violation] = []
    debug = {
        "joint_fit_checks_run": 0,
        "joint_fit_tab_slot_ok": 0,
        "joint_fit_cross_lap_ok": 0,
        "joint_fit_errors": 0,
        "joint_fit_warnings": 0,
    }

    part_map: Dict[str, ManufacturingPart] = {p.part_id: p for p in parts}

    for jspec in joints:
        geom = jspec.geometry
        classification = geom.get("classification")
        if classification not in ("tab_slot", "cross_lap"):
            continue

        debug["joint_fit_checks_run"] += 1
        joint_ok = True

        if classification == "tab_slot":
            tab_id = geom.get("tab_part")
            slot_id = geom.get("slot_part")
            if not tab_id or not slot_id:
                continue
            tab_part = part_map.get(tab_id)
            slot_part = part_map.get(slot_id)
            if tab_part is None or slot_part is None:
                continue

            # Check 1: Tab cross-section fits inside slot
            tab_width = geom.get("tab_width_mm", 0)
            tab_depth = geom.get("tab_depth_mm", 0)
            slot_width = geom.get("slot_width_mm", 0)
            clearance = jspec.clearance_mm
            tabs = geom.get("tabs", [])

            # Get slot tangent direction from contact line
            contact_coords_slot = geom.get(
                "contact_line_2d_b" if tab_id == jspec.part_a else "contact_line_2d_a"
            )
            if contact_coords_slot and len(contact_coords_slot) >= 2:
                slot_tangent = np.array(contact_coords_slot[-1]) - np.array(
                    contact_coords_slot[0]
                )
                st_len = np.linalg.norm(slot_tangent)
                if st_len > 1e-8:
                    slot_tangent = slot_tangent / st_len
                else:
                    slot_tangent = np.array([1.0, 0.0])
            else:
                slot_tangent = np.array([1.0, 0.0])
            slot_perp = np.array([-slot_tangent[1], slot_tangent[0]])

            for tab_info in tabs:
                center_slot = np.array(tab_info.get("center_2d_slot", [0, 0]))

                # Build tab cross-section rectangle at center_2d_slot
                tab_half_w = tab_width / 2.0
                tab_half_d = tab_depth / 2.0
                tab_corners = [
                    center_slot - tab_half_w * slot_perp - tab_half_d * slot_tangent,
                    center_slot + tab_half_w * slot_perp - tab_half_d * slot_tangent,
                    center_slot + tab_half_w * slot_perp + tab_half_d * slot_tangent,
                    center_slot - tab_half_w * slot_perp + tab_half_d * slot_tangent,
                ]
                tab_cross = Polygon(tab_corners)

                # Build slot rectangle at same center
                slot_half_w = slot_width / 2.0
                slot_half_d = (tab_depth + 2.0 * clearance) / 2.0
                slot_corners = [
                    center_slot - slot_half_w * slot_perp - slot_half_d * slot_tangent,
                    center_slot + slot_half_w * slot_perp - slot_half_d * slot_tangent,
                    center_slot + slot_half_w * slot_perp + slot_half_d * slot_tangent,
                    center_slot - slot_half_w * slot_perp + slot_half_d * slot_tangent,
                ]
                slot_poly = Polygon(slot_corners)

                if tab_cross.is_valid and slot_poly.is_valid:
                    # Check with 0.1mm tolerance buffer
                    buffered_slot = slot_poly.buffer(0.1)
                    if not buffered_slot.contains(tab_cross):
                        violations.append(
                            Step3Violation(
                                code="joint_tab_exceeds_slot",
                                severity="error",
                                message=(
                                    f"Tab cross-section doesn't fit inside slot "
                                    f"(tab {tab_width:.1f}x{tab_depth:.1f}mm, "
                                    f"slot width {slot_width:.1f}mm) "
                                    f"between {tab_id} and {slot_id}"
                                ),
                                part_id=slot_id,
                            )
                        )
                        debug["joint_fit_errors"] += 1
                        joint_ok = False

            # Check 2: Tab protrusion overextension
            pre_bbox = geom.get("pre_tab_bbox")
            if pre_bbox and tab_depth > 0:
                post_bbox = tab_part.profile.outline.bounds
                tolerance = tab_depth + 1.0  # allow tab_depth + 1mm margin
                for axis, label in [
                    (0, "min-x"),
                    (1, "min-y"),
                    (2, "max-x"),
                    (3, "max-y"),
                ]:
                    if axis < 2:
                        growth = pre_bbox[axis] - post_bbox[axis]
                    else:
                        growth = post_bbox[axis] - pre_bbox[axis]
                    if growth > tolerance:
                        violations.append(
                            Step3Violation(
                                code="joint_tab_overextension",
                                severity="warning",
                                message=(
                                    f"Tab protrusions extend {growth:.1f}mm "
                                    f"beyond pre-tab outline ({label}) on "
                                    f"{tab_id} (expected max {tab_depth:.1f}mm)"
                                ),
                                part_id=tab_id,
                            )
                        )
                        debug["joint_fit_warnings"] += 1
                        joint_ok = False

            # Check 3: Slot cutouts inside slot part outline
            outline_buffered = slot_part.profile.outline.buffer(-0.1)
            for cutout in slot_part.profile.cutouts:
                if cutout.is_valid and not cutout.is_empty:
                    if not outline_buffered.contains(cutout):
                        violations.append(
                            Step3Violation(
                                code="joint_slot_outside_outline",
                                severity="error",
                                message=(
                                    f"Slot cutout extends beyond part outline "
                                    f"on {slot_id}"
                                ),
                                part_id=slot_id,
                            )
                        )
                        debug["joint_fit_errors"] += 1
                        joint_ok = False

            if joint_ok:
                debug["joint_fit_tab_slot_ok"] += 1

        elif classification == "cross_lap":
            pa = part_map.get(jspec.part_a)
            pb = part_map.get(jspec.part_b)
            if pa is None or pb is None:
                continue

            joint_ok = True

            # Check 1: Slot cutouts inside outlines for both parts
            for part in [pa, pb]:
                outline_buffered = part.profile.outline.buffer(-0.1)
                for cutout in part.profile.cutouts:
                    if cutout.is_valid and not cutout.is_empty:
                        if not outline_buffered.contains(cutout):
                            violations.append(
                                Step3Violation(
                                    code="joint_slot_outside_outline",
                                    severity="error",
                                    message=(
                                        f"Cross-lap slot cutout extends beyond "
                                        f"part outline on {part.part_id}"
                                    ),
                                    part_id=part.part_id,
                                )
                            )
                            debug["joint_fit_errors"] += 1
                            joint_ok = False

            if joint_ok:
                debug["joint_fit_cross_lap_ok"] += 1

    # Check 3: Cutout-cutout collisions across all parts
    for part in parts:
        cutouts = part.profile.cutouts
        for i in range(len(cutouts)):
            for j in range(i + 1, len(cutouts)):
                if cutouts[i].is_valid and cutouts[j].is_valid:
                    inter = cutouts[i].intersection(cutouts[j])
                    if inter.area > 0.5:
                        violations.append(
                            Step3Violation(
                                code="joint_cutout_collision",
                                severity="warning",
                                message=(
                                    f"Cutouts {i} and {j} overlap "
                                    f"({inter.area:.1f} mm²) on {part.part_id}"
                                ),
                                part_id=part.part_id,
                            )
                        )
                        debug["joint_fit_warnings"] += 1

    return violations, debug


def _validate_joint_assembly_3d(
    parts: List[ManufacturingPart],
    joints: List[JointSpec],
) -> Tuple[List[Step3Violation], dict]:
    """Project joint features into 3D and verify mating pairs align.

    Checks:
    - Tab-slot 3D alignment (centers meet within tolerance)
    - Part normal compatibility (tab-slot should be roughly perpendicular)
    - Gap detection (parts not too far apart at joint contact)
    """
    violations: List[Step3Violation] = []
    debug = {
        "assembly_checks_run": 0,
        "assembly_alignment_ok": 0,
        "assembly_alignment_errors": 0,
        "assembly_alignment_warnings": 0,
        "assembly_max_misalignment_mm": 0.0,
        "assembly_mean_misalignment_mm": 0.0,
    }

    part_map: Dict[str, ManufacturingPart] = {p.part_id: p for p in parts}
    misalignments: List[float] = []

    for jspec in joints:
        geom = jspec.geometry
        classification = geom.get("classification")

        # --- Normal compatibility check (tab_slot only) ---
        if classification == "tab_slot":
            tab_id = geom.get("tab_part")
            slot_id = geom.get("slot_part")
            if not tab_id or not slot_id:
                continue
            tab_part = part_map.get(tab_id)
            slot_part = part_map.get(slot_id)
            if tab_part is None or slot_part is None:
                continue

            debug["assembly_checks_run"] += 1

            n_tab = _rotation_to_normal(tab_part.rotation_3d)
            n_slot = _rotation_to_normal(slot_part.rotation_3d)
            dot_val = abs(float(np.dot(n_tab, n_slot)))
            if dot_val > 0.95:
                violations.append(
                    Step3Violation(
                        code="joint_3d_parallel_normals",
                        severity="warning",
                        message=(
                            f"Tab-slot joint between {tab_id} and {slot_id} "
                            f"has near-parallel normals (dot={dot_val:.3f})"
                        ),
                        part_id=tab_id,
                    )
                )
                debug["assembly_alignment_warnings"] += 1

            # --- Tab-slot 3D alignment ---
            tabs = geom.get("tabs", [])
            clearance = jspec.clearance_mm
            tolerance = max(2.0, clearance * 3.0)
            tab_depth = geom.get("tab_depth_mm", tab_part.thickness_mm)
            joint_aligned = True

            for tab_info in tabs:
                c2d_tab = tab_info.get("center_2d_tab")
                c2d_slot = tab_info.get("center_2d_slot")
                if c2d_tab is None or c2d_slot is None:
                    continue

                if (
                    tab_part.profile.basis_u is None
                    or slot_part.profile.basis_u is None
                ):
                    continue

                pt3d_tab = tab_part.profile.project_2d_to_3d(c2d_tab[0], c2d_tab[1])
                pt3d_slot = slot_part.profile.project_2d_to_3d(c2d_slot[0], c2d_slot[1])
                dist = float(np.linalg.norm(pt3d_tab - pt3d_slot))
                misalignments.append(dist)

                if dist > tab_depth + tolerance:
                    violations.append(
                        Step3Violation(
                            code="joint_3d_misalignment",
                            severity="warning",
                            message=(
                                f"Tab-slot 3D misalignment: {dist:.1f}mm "
                                f"between {tab_id} and {slot_id} "
                                f"(limit {tab_depth + tolerance:.1f}mm)"
                            ),
                            part_id=tab_id,
                        )
                    )
                    debug["assembly_alignment_warnings"] += 1
                    joint_aligned = False

            if joint_aligned:
                debug["assembly_alignment_ok"] += 1

        elif classification == "cross_lap":
            pa = part_map.get(jspec.part_a)
            pb = part_map.get(jspec.part_b)
            if pa is None or pb is None:
                continue

            debug["assembly_checks_run"] += 1

            # Project contact line midpoints to 3D and check alignment
            cl_a = geom.get("contact_line_2d_a")
            cl_b = geom.get("contact_line_2d_b")
            if cl_a and cl_b and len(cl_a) >= 2 and len(cl_b) >= 2:
                if pa.profile.basis_u is not None and pb.profile.basis_u is not None:
                    mid_a_2d = (
                        (cl_a[0][0] + cl_a[-1][0]) / 2.0,
                        (cl_a[0][1] + cl_a[-1][1]) / 2.0,
                    )
                    mid_b_2d = (
                        (cl_b[0][0] + cl_b[-1][0]) / 2.0,
                        (cl_b[0][1] + cl_b[-1][1]) / 2.0,
                    )
                    pt3d_a = pa.profile.project_2d_to_3d(mid_a_2d[0], mid_a_2d[1])
                    pt3d_b = pb.profile.project_2d_to_3d(mid_b_2d[0], mid_b_2d[1])
                    dist = float(np.linalg.norm(pt3d_a - pt3d_b))
                    misalignments.append(dist)

                    tolerance = max(2.0, jspec.clearance_mm * 3.0)
                    max_gap = pa.thickness_mm + pb.thickness_mm + tolerance
                    if dist > max_gap:
                        violations.append(
                            Step3Violation(
                                code="joint_3d_misalignment",
                                severity="warning",
                                message=(
                                    f"Cross-lap 3D misalignment: {dist:.1f}mm "
                                    f"between {jspec.part_a} and {jspec.part_b} "
                                    f"(limit {max_gap:.1f}mm)"
                                ),
                                part_id=jspec.part_a,
                            )
                        )
                        debug["assembly_alignment_warnings"] += 1
                    else:
                        debug["assembly_alignment_ok"] += 1

        # --- Gap detection for all classified joints ---
        if classification in ("tab_slot", "cross_lap", "butt"):
            pa = part_map.get(jspec.part_a)
            pb = part_map.get(jspec.part_b)
            if pa is None or pb is None:
                continue

            cl_a = geom.get("contact_line_2d_a")
            cl_b = geom.get("contact_line_2d_b")
            if cl_a and cl_b and len(cl_a) >= 2 and len(cl_b) >= 2:
                if pa.profile.basis_u is not None and pb.profile.basis_u is not None:
                    mid_a_2d = (
                        (cl_a[0][0] + cl_a[-1][0]) / 2.0,
                        (cl_a[0][1] + cl_a[-1][1]) / 2.0,
                    )
                    mid_b_2d = (
                        (cl_b[0][0] + cl_b[-1][0]) / 2.0,
                        (cl_b[0][1] + cl_b[-1][1]) / 2.0,
                    )
                    pt3d_a = pa.profile.project_2d_to_3d(mid_a_2d[0], mid_a_2d[1])
                    pt3d_b = pb.profile.project_2d_to_3d(mid_b_2d[0], mid_b_2d[1])
                    gap = float(np.linalg.norm(pt3d_a - pt3d_b))
                    max_acceptable = pa.thickness_mm + pb.thickness_mm + 5.0
                    if gap > max_acceptable:
                        violations.append(
                            Step3Violation(
                                code="joint_3d_excessive_gap",
                                severity="error",
                                message=(
                                    f"Excessive gap {gap:.1f}mm between "
                                    f"{jspec.part_a} and {jspec.part_b} "
                                    f"(limit {max_acceptable:.1f}mm)"
                                ),
                                part_id=jspec.part_a,
                            )
                        )
                        debug["assembly_alignment_errors"] += 1

    if misalignments:
        debug["assembly_max_misalignment_mm"] = float(max(misalignments))
        debug["assembly_mean_misalignment_mm"] = float(
            sum(misalignments) / len(misalignments)
        )

    return violations, debug


def _build_design(
    design_name: str,
    parts: List[ManufacturingPart],
    joints: List[JointSpec],
    material_key: str,
) -> FurnitureDesign:
    design = FurnitureDesign(name=design_name)

    for part in parts:
        component = Component(
            name=part.part_id,
            type=_classify_component(part),
            profile=polygon_to_profile(part.profile.outline),
            thickness=part.thickness_mm,
            position=part.position_3d.copy(),
            rotation=part.rotation_3d.copy(),
            material=material_key,
            features=[],
        )
        for op in part.bend_ops:
            component.features.append(
                {
                    "type": "engrave",
                    "coords": [list(op.line[0]), list(op.line[1])],
                    "label": "bend_line",
                }
            )
        for cutout in part.profile.cutouts:
            if not cutout.is_empty:
                component.features.append(
                    {
                        "type": "slot",
                        "outline": list(cutout.exterior.coords),
                    }
                )
        design.add_component(component)

    _JOINT_TYPE_MAP = {
        "through_bolt": JointType.THROUGH_BOLT,
        "tab_slot": JointType.TAB_SLOT,
        "cross_lap": JointType.HALF_LAP,
        "butt": JointType.BUTT,
    }

    for spec in joints:
        jt = _JOINT_TYPE_MAP.get(spec.joint_type, JointType.TAB_SLOT)
        design.assembly.add_joint(
            Joint(
                component_a=spec.part_a,
                component_b=spec.part_b,
                joint_type=jt,
                position_a=(0.0, 0.0, 0.0),
                position_b=(0.0, 0.0, 0.0),
                parameters={"clearance_mm": spec.clearance_mm, **spec.geometry},
            )
        )

    ordered = sorted(design.components, key=lambda c: float(c.position[2]))
    design.assembly.assembly_order = [(comp.name, "") for comp in ordered]
    return design


def _classify_component(part: ManufacturingPart) -> ComponentType:
    normal = _rotation_to_normal(part.rotation_3d)
    nz = abs(float(normal[2]))
    if part.region_type == RegionType.BENDABLE_SHEET:
        return ComponentType.PANEL
    if nz < 0.35:
        return ComponentType.LEG
    if nz < 0.75:
        return ComponentType.SUPPORT
    return ComponentType.SHELF


def _validate_parts_and_bends(
    parts: Iterable[ManufacturingPart],
    capabilities: CapabilityProfile,
    material_key: str,
) -> List[Step3Violation]:
    issues: List[Step3Violation] = []

    dfm = DFMConfig.from_material(material_key)
    dfm.max_sheet_width_mm = min(
        dfm.max_sheet_width_mm, capabilities.max_sheet_width_mm
    )
    dfm.max_sheet_height_mm = min(
        dfm.max_sheet_height_mm, capabilities.max_sheet_height_mm
    )

    for part in parts:
        violations = check_part_dfm(part.profile, dfm)
        issues.extend(_convert_dfm_violations(violations, part.part_id))

        bounds = part.profile.outline.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if min(width, height) < capabilities.min_feature_mm:
            issues.append(
                Step3Violation(
                    code="min_feature",
                    severity="error",
                    message=(
                        f"Part dimension below min feature size "
                        f"({min(width, height):.2f}mm < {capabilities.min_feature_mm:.2f}mm)"
                    ),
                    part_id=part.part_id,
                )
            )

        for bend in part.bend_ops:
            if not capabilities.supports_bending(part.material_key):
                issues.append(
                    Step3Violation(
                        code="bend_material_unsupported",
                        severity="error",
                        message=f"Controlled bending unsupported for {part.material_key}",
                        part_id=part.part_id,
                    )
                )
                continue

            if bend.angle_deg > capabilities.controlled_bending.max_bend_angle_deg:
                issues.append(
                    Step3Violation(
                        code="bend_angle_limit",
                        severity="error",
                        message=(
                            f"Bend angle {bend.angle_deg:.1f} exceeds "
                            f"limit {capabilities.controlled_bending.max_bend_angle_deg:.1f}"
                        ),
                        part_id=part.part_id,
                    )
                )

            min_radius = (
                part.thickness_mm
                * capabilities.controlled_bending.min_radius_multiplier
            )
            if bend.radius_mm < min_radius:
                issues.append(
                    Step3Violation(
                        code="bend_radius_limit",
                        severity="error",
                        message=(
                            f"Bend radius {bend.radius_mm:.2f}mm below required "
                            f"{min_radius:.2f}mm"
                        ),
                        part_id=part.part_id,
                    )
                )

    return issues


def _convert_dfm_violations(
    violations: List[DFMViolation],
    part_id: str,
) -> List[Step3Violation]:
    return [
        Step3Violation(
            code=f"dfm_{v.rule_name}",
            severity=v.severity,
            message=v.message,
            part_id=part_id,
        )
        for v in violations
    ]


_SLAB_INTERIOR_TOL = 0.5  # mm — ignore vertices within this distance of slab faces


def _part_geometry_summary(parts: List[ManufacturingPart]) -> Dict[str, Any]:
    """Summarize part geometry for phase-level debugging."""
    rows: List[Dict[str, Any]] = []
    total_area = 0.0
    total_perimeter = 0.0
    for part in parts:
        outline = part.profile.outline
        area = float(max(0.0, outline.area))
        perimeter = float(max(0.0, outline.length))
        bounds = outline.bounds
        rows.append(
            {
                "part_id": part.part_id,
                "area_mm2": round(area, 4),
                "perimeter_mm": round(perimeter, 4),
                "bbox_mm": [
                    round(float(bounds[0]), 4),
                    round(float(bounds[1]), 4),
                    round(float(bounds[2]), 4),
                    round(float(bounds[3]), 4),
                ],
                "region_type": part.region_type.value,
                "source_area_mm2": round(float(part.source_area_mm2), 4),
                "selection_gain": round(
                    float(part.metadata.get("selection_gain", part.source_area_mm2)), 4
                ),
                "stack_group_id": str(part.metadata.get("stack_group_id", "")),
                "stack_layer_offset_mm": round(
                    float(part.metadata.get("stack_layer_offset_mm", 0.0)), 4
                ),
            }
        )
        total_area += area
        total_perimeter += perimeter
    rows.sort(key=lambda r: str(r.get("part_id", "")))
    return {
        "part_count": int(len(parts)),
        "total_area_mm2": round(float(total_area), 4),
        "total_perimeter_mm": round(float(total_perimeter), 4),
        "parts": rows,
    }


def _phase_diagnostics(parts: List[ManufacturingPart], **extra: Any) -> Dict[str, Any]:
    """Compose rich diagnostics payload for a phase snapshot."""
    diagnostics: Dict[str, Any] = {
        "part_geometry": _part_geometry_summary(parts),
        "plane_overlap": _compute_plane_overlap(parts),
    }
    for key, value in extra.items():
        diagnostics[key] = value
    return diagnostics


def _extract_polygons_from_geometry(geom: Any) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    clean = geom if geom.is_valid else geom.buffer(0)
    if clean.is_empty:
        return []
    if clean.geom_type == "Polygon":
        return [clean] if clean.area > 1e-6 else []
    if clean.geom_type == "MultiPolygon":
        return [g for g in clean.geoms if g.area > 1e-6]
    if hasattr(clean, "geoms"):
        return [g for g in clean.geoms if g.geom_type == "Polygon" and g.area > 1e-6]
    return []


def _clip_polygon_half_plane(
    poly: Polygon, a: float, b: float, c_rhs: float
) -> List[Polygon]:
    """Clip polygon to the half-plane a*x + b*y >= c_rhs."""
    ab_norm = float(math.hypot(a, b))
    if ab_norm < 1e-9:
        return []

    line_normal = np.array([a, b], dtype=float) / ab_norm
    line_dir = np.array([-line_normal[1], line_normal[0]], dtype=float)
    if abs(a) > abs(b):
        line_pt = np.array([c_rhs / a, 0.0], dtype=float)
    else:
        line_pt = np.array([0.0, c_rhs / b], dtype=float)

    bounds = poly.bounds
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 100.0) * 4.0
    p1 = line_pt - extent * line_dir
    p2 = line_pt + extent * line_dir
    p3 = p2 + extent * line_normal
    p4 = p1 + extent * line_normal
    half_plane = Polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])
    try:
        clipped = poly.intersection(half_plane)
    except Exception:
        return []
    return _extract_polygons_from_geometry(clipped)


def _clip_polygon_strip(
    poly: Polygon,
    a: float,
    b: float,
    c0: float,
    lo: float,
    hi: float,
) -> List[Polygon]:
    """Clip polygon to strip lo <= a*x + b*y + c0 <= hi."""
    if hi <= lo:
        return []
    first = _clip_polygon_half_plane(poly, a, b, lo - c0)
    if not first:
        return []
    result: List[Polygon] = []
    for p in first:
        result.extend(_clip_polygon_half_plane(p, -a, -b, c0 - hi))
    return result


def _directional_overlap_regions(
    src_part: ManufacturingPart,
    dst_part: ManufacturingPart,
    overlap_mm: float,
) -> List[Dict[str, Any]]:
    """Compute overlap-region polygons on src_part where it intrudes dst_part slab."""
    if overlap_mm <= 0.01:
        return []
    prof_src = src_part.profile
    prof_dst = dst_part.profile
    if (
        prof_src.basis_u is None
        or prof_src.basis_v is None
        or prof_src.origin_3d is None
        or prof_dst.origin_3d is None
    ):
        return []

    # Use net_polygon (outline minus cutouts) so slot voids are excluded
    src_outline = prof_src.net_polygon()
    if src_outline.is_empty or src_outline.area <= 1e-6:
        return []

    n_dst = _rotation_to_normal(dst_part.rotation_3d)
    a = float(np.dot(n_dst, prof_src.basis_u))
    b = float(np.dot(n_dst, prof_src.basis_v))
    if math.hypot(a, b) < 1e-9:
        return []

    # Slab origin = outward-facing face (position_3d + normal * thickness).
    # Material extends in anti-normal direction per _slab_penetration convention.
    # For stacked parts, position_3d differs per layer (unlike origin_3d).
    dst_slab_origin = np.asarray(dst_part.position_3d, dtype=float) + n_dst * dst_part.thickness_mm
    d_surface = float(np.dot(n_dst, dst_slab_origin))
    c0 = float(np.dot(n_dst, prof_src.origin_3d) - d_surface)
    tol = _SLAB_INTERIOR_TOL
    lo = -float(dst_part.thickness_mm) + tol
    hi = -tol
    clipped_polys = _clip_polygon_strip(src_outline, a, b, c0, lo, hi)
    if not clipped_polys:
        return []

    out: List[Dict[str, Any]] = []
    for poly in clipped_polys:
        if poly.is_empty or poly.area <= 1e-6:
            continue
        ext2 = [
            [round(float(u), 4), round(float(v), 4)] for u, v in poly.exterior.coords
        ]
        ext3 = []
        for u, v in poly.exterior.coords:
            p3 = prof_src.project_2d_to_3d(float(u), float(v))
            ext3.append(
                [round(float(p3[0]), 4), round(float(p3[1]), 4), round(float(p3[2]), 4)]
            )
        holes2: List[List[List[float]]] = []
        for ring in poly.interiors:
            holes2.append(
                [[round(float(u), 4), round(float(v), 4)] for u, v in ring.coords]
            )
        out.append(
            {
                "part_id": src_part.part_id,
                "against_part_id": dst_part.part_id,
                "overlap_mm": round(float(overlap_mm), 4),
                "area_mm2": round(float(poly.area), 4),
                "outline_2d": ext2,
                "outline_3d": ext3,
                "holes_2d": holes2,
            }
        )
    return out


def _slab_penetration(
    prof: "PartProfile2D",
    slab_normal: np.ndarray,
    slab_origin: np.ndarray,
    slab_thickness: float,
) -> float:
    """Return how far *prof*'s outline extends into a material slab.

    The slab surface is at ``dot(slab_normal, slab_origin)`` and material
    extends in the anti-normal direction by *slab_thickness*.

    Only vertices clearly in the **interior** of the slab are counted
    (at least ``_SLAB_INTERIOR_TOL`` from both the surface and far face).
    This avoids false positives from perpendicular parts whose edges
    naturally sit at the slab boundary.
    """
    d_surface = float(np.dot(slab_normal, slab_origin))
    coords = list(prof.outline.exterior.coords)
    if not coords:
        return 0.0

    # Evaluate signed distances over the outline. A linear field over a connected
    # polygon produces a continuous distance interval, so range overlap catches
    # edge crossings even when no vertex lies inside the slab interior.
    d_vals: List[float] = []
    for u, v in coords:
        pt3d = prof.project_2d_to_3d(u, v)
        d_vals.append(float(np.dot(slab_normal, pt3d)) - d_surface)
    d_min = min(d_vals)
    d_max = max(d_vals)

    tol = _SLAB_INTERIOR_TOL
    interior_lo = -slab_thickness + tol
    interior_hi = -tol

    overlap_lo = max(d_min, interior_lo)
    overlap_hi = min(d_max, interior_hi)
    if overlap_hi <= overlap_lo:
        return 0.0

    # Penetration is maximal at slab mid-plane when reachable.
    center = -0.5 * slab_thickness
    d_star = min(max(center, overlap_lo), overlap_hi)
    return max(0.0, min(-d_star, slab_thickness + d_star))


def _compute_plane_overlap(
    parts: List[ManufacturingPart],
    joints: Optional[List[JointSpec]] = None,
) -> Dict[str, Any]:
    """Measure how much parts physically penetrate each other's material slabs.

    For every pair of non-parallel parts, project outline vertices of each into
    the other's plane and check if they lie inside the material slab.

    Cutout-aware: after slab penetration is detected, overlap regions are
    computed using the source outline (which includes slot cutouts), so
    tab-through-slot junctions produce near-zero overlap area and are
    filtered out by the area gate.

    Returns a dict of overlap metrics suitable for merging into the debug dict.
    """
    overlap_pairs = 0
    overlap_max_mm = 0.0
    overlap_total_mm = 0.0
    pairs_checked = 0
    overlap_details: List[Dict[str, Any]] = []
    overlap_regions: List[Dict[str, Any]] = []

    for idx_a in range(len(parts)):
        pa = parts[idx_a]
        prof_a = pa.profile
        if prof_a.basis_u is None or prof_a.basis_v is None or prof_a.origin_3d is None:
            continue
        n_a = _rotation_to_normal(pa.rotation_3d)
        # Slab origin = outward-facing face (position_3d + normal * thickness).
        # Material extends in anti-normal direction per _slab_penetration convention.
        slab_origin_a = np.asarray(pa.position_3d, dtype=float) + n_a * pa.thickness_mm

        for idx_b in range(idx_a + 1, len(parts)):
            pb = parts[idx_b]
            prof_b = pb.profile
            if (
                prof_b.basis_u is None
                or prof_b.basis_v is None
                or prof_b.origin_3d is None
            ):
                continue
            n_b = _rotation_to_normal(pb.rotation_3d)
            slab_origin_b = np.asarray(pb.position_3d, dtype=float) + n_b * pb.thickness_mm

            cross_norm = float(np.linalg.norm(np.cross(n_a, n_b)))
            if cross_norm < 0.1:
                continue  # Nearly parallel — skip

            pairs_checked += 1
            # Require mutual slab intrusion. One-way intrusion often happens at
            # legitimate face contacts (e.g., butt joints) and should not count
            # as volumetric overlap.
            pen_a_into_b = _slab_penetration(
                prof_a,
                n_b,
                slab_origin_b,
                pb.thickness_mm,
            )
            pen_b_into_a = _slab_penetration(
                prof_b,
                n_a,
                slab_origin_a,
                pa.thickness_mm,
            )
            pair_max = min(pen_a_into_b, pen_b_into_a)

            if pair_max > 0.01:  # 0.01mm tolerance
                # Compute cutout-aware overlap regions and check total area.
                # Tab-through-slot leaves near-zero area after cutout subtraction.
                regions_a = _directional_overlap_regions(pa, pb, pair_max)
                regions_b = _directional_overlap_regions(pb, pa, pair_max)
                total_area = sum(
                    r.get("area_mm2", 0.0) for r in regions_a + regions_b
                )
                if total_area < 1.0:
                    # Tab-through-slot or negligible overlap after cutout subtraction
                    continue
                overlap_pairs += 1
                overlap_total_mm += pair_max
                overlap_details.append(
                    {
                        "part_a": pa.part_id,
                        "part_b": pb.part_id,
                        "penetration_a_into_b_mm": round(float(pen_a_into_b), 4),
                        "penetration_b_into_a_mm": round(float(pen_b_into_a), 4),
                        "overlap_mm": round(float(pair_max), 4),
                    }
                )
                overlap_regions.extend(regions_a)
                overlap_regions.extend(regions_b)
            if pair_max > overlap_max_mm:
                overlap_max_mm = pair_max

    overlap_details.sort(key=lambda item: float(item["overlap_mm"]), reverse=True)
    overlap_regions.sort(
        key=lambda item: float(item.get("area_mm2", 0.0)),
        reverse=True,
    )

    return {
        "plane_overlap_pairs": overlap_pairs,
        "plane_overlap_max_mm": round(float(overlap_max_mm), 4),
        "plane_overlap_total_mm": round(float(overlap_total_mm), 4),
        "plane_overlap_count": pairs_checked,
        "plane_overlap_details": overlap_details,
        "plane_overlap_regions": overlap_regions,
    }


_POST_JOINT_MAX_TRIM_LOSS = 0.15  # Don't trim more than 15% in safety-net pass


def _post_joint_overlap_cleanup(
    parts: List[ManufacturingPart],
    joints: List[JointSpec],
) -> List[ManufacturingPart]:
    """Bounded post-joint cleanup: trim residual real overlaps after joint geometry.

    Single pass, no iteration. Skips pairs that share a joint. Only applies
    the lower-loss trim direction if loss < _POST_JOINT_MAX_TRIM_LOSS.
    """
    # Build set of joint-connected part_id pairs
    joint_pairs_ids = set()
    for j in joints:
        pair = tuple(sorted([j.part_a, j.part_b]))
        joint_pairs_ids.add(pair)

    id_to_idx = {p.part_id: idx for idx, p in enumerate(parts)}

    for idx_a in range(len(parts)):
        pa = parts[idx_a]
        if pa.profile.basis_u is None:
            continue
        n_a = _rotation_to_normal(pa.rotation_3d)
        for idx_b in range(idx_a + 1, len(parts)):
            pb = parts[idx_b]
            if pb.profile.basis_u is None:
                continue

            # Skip pairs that share a joint (tab-through-slot is expected)
            pair_ids = tuple(sorted([pa.part_id, pb.part_id]))
            if pair_ids in joint_pairs_ids:
                continue

            n_b = _rotation_to_normal(pb.rotation_3d)
            cross_norm = float(np.linalg.norm(np.cross(n_a, n_b)))
            if cross_norm < 0.1:
                continue

            slab_origin_a = np.asarray(pa.position_3d, dtype=float) + n_a * pa.thickness_mm
            slab_origin_b = np.asarray(pb.position_3d, dtype=float) + n_b * pb.thickness_mm
            pen_a = _slab_penetration(
                pa.profile, n_b, slab_origin_b, pb.thickness_mm,
            )
            pen_b = _slab_penetration(
                pb.profile, n_a, slab_origin_a, pa.thickness_mm,
            )
            pair_max = min(pen_a, pen_b)
            if pair_max <= 0.01:
                continue

            # Check cutout-aware area
            regions_a = _directional_overlap_regions(pa, pb, pair_max)
            regions_b = _directional_overlap_regions(pb, pa, pair_max)
            total_area = sum(
                r.get("area_mm2", 0.0) for r in regions_a + regions_b
            )
            if total_area < 1.0:
                continue

            # Try both trim directions, pick the lower-loss one
            loss_a = _trim_loss_fraction(pa, n_b, slab_origin_b, pb.thickness_mm)
            loss_b = _trim_loss_fraction(pb, n_a, slab_origin_a, pa.thickness_mm)
            if loss_a <= loss_b and loss_a < _POST_JOINT_MAX_TRIM_LOSS:
                _trim_part_against_plane(pa, n_b, slab_origin_b, pb.thickness_mm)
            elif loss_b < _POST_JOINT_MAX_TRIM_LOSS:
                _trim_part_against_plane(pb, n_a, slab_origin_a, pa.thickness_mm)
            # else: both losses too high, skip

    return parts


def _compute_quality_metrics(
    mesh: trimesh.Trimesh,
    selected_parts: List[ManufacturingPart],
    joints: List[JointSpec],
    part_budget_max: int,
    fidelity_weight: float,
    overlap_pairs: int = 0,
    error_count: int = 0,
) -> Step3QualityMetrics:
    if not selected_parts:
        return Step3QualityMetrics(
            hausdorff_mm=1e6,
            mean_distance_mm=1e6,
            normal_error_deg=180.0,
            connectivity_score=0.0,
            overlap_score=0.0,
            violation_score=0.0,
            part_count=0,
            overall_score=0.0,
        )

    coverage_ratio = _unique_face_coverage_ratio(mesh, selected_parts)

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))
    hausdorff_mm = max_extent * (1.0 - coverage_ratio)
    mean_distance_mm = 0.5 * hausdorff_mm

    normal_error_deg = _compute_normal_error_deg(mesh, selected_parts)

    # --- Sub-scores, all in [0, 1], higher = better ---

    hausdorff_score = 1.0 / (1.0 + hausdorff_mm / max(1.0, max_extent))
    normal_score = 1.0 / (1.0 + normal_error_deg / 45.0)
    fidelity_score = 0.8 * hausdorff_score + 0.2 * normal_score

    # Connectivity: a spanning tree of N members needs N-1 joints.
    effective_members = _effective_member_count(selected_parts)
    expected_joints = max(1, effective_members - 1)
    connectivity_score = min(1.0, len(joints) / expected_joints)

    # Overlap: each overlap pair indicates physical impossibility.
    overlap_score = 1.0 / (1.0 + overlap_pairs)

    # Violations: each error-severity violation is a manufacturing problem.
    violation_score = 1.0 / (1.0 + error_count)

    # --- Weighted combination ---
    # Remainder after fidelity is split: 40% connectivity, 40% overlap, 20% violations
    remainder = max(1e-6, 1.0 - fidelity_weight)
    connectivity_w = remainder * 0.40
    overlap_w = remainder * 0.40
    violation_w = remainder * 0.20

    overall = (
        fidelity_weight * fidelity_score
        + connectivity_w * connectivity_score
        + overlap_w * overlap_score
        + violation_w * violation_score
    )

    return Step3QualityMetrics(
        hausdorff_mm=float(hausdorff_mm),
        mean_distance_mm=float(mean_distance_mm),
        normal_error_deg=float(normal_error_deg),
        connectivity_score=float(connectivity_score),
        overlap_score=float(overlap_score),
        violation_score=float(violation_score),
        part_count=len(selected_parts),
        overall_score=float(np.clip(overall, 0.0, 1.0)),
    )


def _unique_face_coverage_ratio(
    mesh: trimesh.Trimesh,
    selected_parts: List[ManufacturingPart],
) -> float:
    mesh_area = max(float(mesh.area), 1e-6)
    face_count = len(mesh.faces)
    unique_faces: set[int] = set()
    for part in selected_parts:
        for face_idx in part.source_faces:
            idx = int(face_idx)
            if 0 <= idx < face_count:
                unique_faces.add(idx)

    if unique_faces:
        covered_area = float(np.sum(mesh.area_faces[list(unique_faces)]))
    else:
        covered_area = float(sum(max(0.0, p.source_area_mm2) for p in selected_parts))

    return float(np.clip(covered_area / mesh_area, 0.0, 1.0))


def _effective_member_count(parts: List[ManufacturingPart]) -> int:
    groups: set[str] = set()
    singletons = 0
    for part in parts:
        group = str(part.metadata.get("stack_group_id", "")).strip()
        if group:
            groups.add(group)
        else:
            singletons += 1
    return len(groups) + singletons


def _compute_normal_error_deg(
    mesh: trimesh.Trimesh,
    selected_parts: List[ManufacturingPart],
) -> float:
    errors: List[float] = []
    for part in selected_parts:
        if not part.source_faces:
            continue
        src_normals = mesh.face_normals[part.source_faces]
        part_normal = _rotation_to_normal(part.rotation_3d)
        dots = np.clip(src_normals @ part_normal, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(np.abs(dots)))
        errors.append(float(np.mean(angle_deg)))
    return float(np.mean(errors)) if errors else 0.0


def _normal_to_rotation(normal: np.ndarray) -> np.ndarray:
    n = normal / max(np.linalg.norm(normal), 1e-8)
    u_axis, v_axis = _plane_basis(n)
    rot = np.column_stack([u_axis, v_axis, n])

    ry = math.asin(float(np.clip(-rot[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        rz = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        rz = 0.0
        rx = math.atan2(float(rot[0, 1]), float(rot[1, 1]))

    return np.array([rx, ry, rz], dtype=float)


def _rotation_matrix_xyz(rotation_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(v) for v in rotation_xyz]

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _rotation_to_normal(rotation_xyz: np.ndarray) -> np.ndarray:
    rot = _rotation_matrix_xyz(rotation_xyz)
    n = rot @ np.array([0.0, 0.0, 1.0])
    return n / max(np.linalg.norm(n), 1e-8)


def _compute_status(
    selected_parts: List[ManufacturingPart],
    violations: List[Step3Violation],
) -> str:
    if not selected_parts:
        return "fail"
    if any(v.severity == "error" for v in violations):
        return "partial"
    return "success"


def _empty_output(
    design_name: str, reason: str, debug: Optional[dict] = None
) -> Step3Output:
    if debug is None:
        debug = {}

    return Step3Output(
        design=FurnitureDesign(name=design_name),
        parts={},
        joints=[],
        quality_metrics=Step3QualityMetrics(
            hausdorff_mm=1e6,
            mean_distance_mm=1e6,
            normal_error_deg=180.0,
            connectivity_score=0.0,
            overlap_score=0.0,
            violation_score=0.0,
            part_count=0,
            overall_score=0.0,
        ),
        violations=[
            Step3Violation(
                code=reason,
                severity="error",
                message=f"First-principles Step 3 failed: {reason}",
            )
        ],
        status="fail",
        debug=debug,
    )


# ─── Greedy coverage ranking ────────────────────────────────────────────────


def _greedy_coverage_rank(candidates: List[_RegionCandidate]) -> List[int]:
    """Rank candidates using greedy set-cover to maximize face coverage.

    Instead of pure area/score ranking, each iteration picks the candidate
    whose source faces provide the most marginal gain (uncovered faces).
    """
    covered_faces: set[int] = set()
    ranked: List[int] = []
    remaining = set(range(len(candidates)))

    while remaining:
        best_idx = -1
        best_gain = -float("inf")

        for idx in remaining:
            cand = candidates[idx]
            if not cand.source_faces:
                gain = cand.score
            else:
                uncovered = sum(1 for f in cand.source_faces if f not in covered_faces)
                total = max(1, len(cand.source_faces))
                gain = cand.score * (uncovered / total)

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx < 0:
            break

        ranked.append(best_idx)
        remaining.remove(best_idx)
        covered_faces.update(candidates[best_idx].source_faces)

    return ranked


# ─── Grow-then-constrain phase ───────────────────────────────────────────────


def _constrain_outlines(
    parts: List[ManufacturingPart],
    mesh: trimesh.Trimesh,
    input_spec: Step3Input,
    joint_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[List[ManufacturingPart], List[dict]]:
    """Grow-then-constrain: expand outlines freely, clip by planes, clip by mesh."""
    snapshots: List[dict] = []
    if not parts:
        return parts, snapshots

    # Phase A: grow each outline to cover coplanar mesh faces
    grown: List[ManufacturingPart] = []
    for part in parts:
        if part.region_type != RegionType.PLANAR_CUT:
            grown.append(part)
            continue

        profile = part.profile
        if (
            profile.basis_u is None
            or profile.basis_v is None
            or profile.origin_3d is None
        ):
            grown.append(part)
            continue

        grown_outline = _compute_coplanar_extent(mesh, profile, part.rotation_3d)
        if grown_outline.is_empty or grown_outline.area <= profile.outline.area:
            grown.append(part)
            continue

        # Grow outward but preserve holes: expand the exterior using the
        # grown extent, then re-punch the original interior rings.
        new_exterior = unary_union([Polygon(profile.outline.exterior), grown_outline])
        if not isinstance(new_exterior, Polygon):
            new_exterior = _largest_polygon(new_exterior)
        if new_exterior is None or new_exterior.is_empty:
            grown.append(part)
            continue
        # Re-apply original holes (only those that remain inside the new exterior)
        kept_holes = []
        for hole in profile.outline.interiors:
            hole_poly = Polygon(hole)
            if new_exterior.contains(hole_poly) or new_exterior.intersects(hole_poly):
                kept_holes.append(hole)
        if kept_holes:
            new_outline = Polygon(new_exterior.exterior, kept_holes)
            new_outline = _clean_polygon(new_outline)
        else:
            new_outline = new_exterior
        if new_outline is None or new_outline.is_empty:
            grown.append(part)
            continue

        new_profile = PartProfile2D(
            outline=new_outline,
            cutouts=profile.cutouts,
            features=profile.features,
            material_key=profile.material_key,
            thickness_mm=profile.thickness_mm,
            basis_u=profile.basis_u,
            basis_v=profile.basis_v,
            origin_3d=profile.origin_3d,
        )
        grown.append(
            ManufacturingPart(
                part_id=part.part_id,
                material_key=part.material_key,
                thickness_mm=part.thickness_mm,
                profile=new_profile,
                region_type=part.region_type,
                position_3d=part.position_3d.copy(),
                rotation_3d=part.rotation_3d.copy(),
                source_area_mm2=part.source_area_mm2,
                source_faces=list(part.source_faces),
                bend_ops=list(part.bend_ops),
                metadata=dict(part.metadata),
            )
        )

    snapshots.append(
        _serialize_parts_snapshot(
            "Grown (coplanar)",
            1,
            grown,
            diagnostics=_phase_diagnostics(grown),
        )
    )

    # Phase B: trim at plane-plane intersections
    trimmed, trim_debug = _trim_at_plane_intersections(
        grown, input_spec, joint_pairs=joint_pairs
    )

    snapshots.append(
        _serialize_parts_snapshot(
            "Trimmed (planes)",
            2,
            trimmed,
            diagnostics=_phase_diagnostics(trimmed, trim_decisions=trim_debug),
        )
    )

    # Phase C skipped: Phase A now grows outlines to the mesh cross-section
    # directly, so a redundant mesh clip is no longer needed.
    clipped = trimmed

    snapshots.append(
        _serialize_parts_snapshot(
            "Clipped (mesh)",
            3,
            clipped,
            diagnostics=_phase_diagnostics(clipped),
        )
    )

    # Phase D: split oversized parts
    max_w = input_spec.scs_capabilities.max_sheet_width_mm
    max_h = input_spec.scs_capabilities.max_sheet_height_mm
    final: List[ManufacturingPart] = []
    for part in clipped:
        bounds = part.profile.outline.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]
        if w > max_w or h > max_h:
            final.extend(_split_oversized_part(part, max_w, max_h, input_spec))
        else:
            final.append(part)

    # Phase E: drop parts below minimum area
    result: List[ManufacturingPart] = []
    for part in final:
        if part.profile.outline.is_empty:
            continue
        if part.profile.outline.area < input_spec.min_planar_region_area_mm2:
            continue
        result.append(part)

    snapshots.append(
        _serialize_parts_snapshot(
            "Final",
            4,
            result,
            diagnostics=_phase_diagnostics(result),
        )
    )

    return result, snapshots


def _compute_mesh_cross_section(
    mesh: trimesh.Trimesh,
    profile: PartProfile2D,
    rotation_3d: np.ndarray,
) -> Optional[Polygon]:
    """Compute the mesh cross-section at this part's plane, in the part's 2D frame.

    Uses trimesh.section() to slice the mesh volume and projects the resulting
    contour onto the part's local coordinate system.  Falls back to a slightly
    inward-offset slice when the plane sits exactly on a mesh face.
    """
    normal = _rotation_to_normal(rotation_3d)
    origin = profile.origin_3d
    basis_u = profile.basis_u
    basis_v = profile.basis_v

    # Collect cross-sections from all offsets and union them.
    # On step boundaries the solid volume differs on each side of the face,
    # so a single offset captures only one side.  Unioning all offsets gives
    # the complete profile.
    polys: List[Polygon] = []
    for offset in [0.0, -0.1, 0.1]:
        try:
            plane_pt = origin + offset * normal
            section = mesh.section(plane_origin=plane_pt, plane_normal=normal)
        except Exception:
            continue
        if section is None:
            continue

        try:
            discrete = section.discrete
        except Exception:
            continue

        for path_verts in discrete:
            local = path_verts - origin
            pts_2d = [(float(p @ basis_u), float(p @ basis_v)) for p in local]
            if len(pts_2d) < 3:
                continue
            try:
                p = Polygon(pts_2d)
                if not p.is_valid:
                    p = p.buffer(0)
                if isinstance(p, MultiPolygon):
                    p = max(p.geoms, key=lambda g: g.area)
                if p.is_valid and p.area > 1.0:
                    polys.append(p)
            except Exception:
                continue

    if not polys:
        return None

    merged = unary_union(polys)
    result = _largest_polygon(merged)
    if result is not None:
        result = result.simplify(0.5, preserve_topology=True)
    return result


def _compute_coplanar_extent(
    mesh: trimesh.Trimesh,
    profile: PartProfile2D,
    rotation_3d: np.ndarray,
    inset_mm: float = 0.5,
) -> Polygon:
    """Grow a part's outline by cross-sectioning the mesh just inside the surface.

    Takes a cross-section of the mesh at ``origin - inset_mm * normal``,
    projects the resulting contour into the part's 2-D frame, and keeps
    only the connected component that overlaps the original outline.
    Returns the bounding box of that region.

    Falls back to the original outline when the section is empty or
    no connected component is found.
    """
    normal = _rotation_to_normal(rotation_3d)
    origin = profile.origin_3d
    basis_u = profile.basis_u
    basis_v = profile.basis_v

    # Try both inset directions — the canonical normal may point away from
    # the mesh interior.  Pick whichever section yields more geometry.
    best_section = None
    for sign in (-1.0, 1.0):
        s = mesh.section(
            plane_origin=origin + sign * inset_mm * normal,
            plane_normal=normal,
        )
        if s is None:
            continue
        if best_section is None or len(s.vertices) > len(best_section.vertices):
            best_section = s
    section = best_section

    if section is None:
        return profile.outline

    # Build 2-D polygons from section entities in the part's local frame
    polys: List[Polygon] = []
    for ent in section.entities:
        pts_3d = section.vertices[ent.points]
        local = pts_3d - origin
        pts_2d = np.column_stack([local @ basis_u, local @ basis_v])
        try:
            poly = Polygon(pts_2d)
            if poly.is_valid and poly.area > 0.01:
                polys.append(poly)
        except Exception:
            continue

    if not polys:
        return profile.outline

    merged = unary_union(polys)
    connected = _keep_connected(merged, profile.outline)

    if connected.is_empty or connected.area < 1.0:
        return profile.outline

    return connected


def _keep_connected(geometry, reference: Polygon) -> Polygon:
    """Keep only parts of a geometry that are connected to the reference polygon."""
    if geometry.geom_type == "Polygon":
        return geometry
    if geometry.geom_type == "MultiPolygon":
        connected = [g for g in geometry.geoms if g.intersects(reference)]
        if not connected:
            return reference
        result = unary_union(connected)
        if isinstance(result, MultiPolygon):
            return max(result.geoms, key=lambda g: g.area)
        if result.geom_type == "Polygon":
            return result
        return reference
    return reference


def _largest_polygon(geom) -> Optional[Polygon]:
    """Extract the largest Polygon from a geometry."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return None


def _deep_copy_part(part: ManufacturingPart) -> ManufacturingPart:
    """Deep-copy a ManufacturingPart so mutations don't affect the original."""
    p = part.profile
    new_profile = PartProfile2D(
        outline=Polygon(
            p.outline.exterior.coords, [list(h.coords) for h in p.outline.interiors]
        ),
        cutouts=list(p.cutouts),
        features=list(p.features),
        material_key=p.material_key,
        thickness_mm=p.thickness_mm,
        basis_u=p.basis_u.copy() if p.basis_u is not None else None,
        basis_v=p.basis_v.copy() if p.basis_v is not None else None,
        origin_3d=p.origin_3d.copy() if p.origin_3d is not None else None,
    )
    return ManufacturingPart(
        part_id=part.part_id,
        material_key=part.material_key,
        thickness_mm=part.thickness_mm,
        profile=new_profile,
        region_type=part.region_type,
        position_3d=part.position_3d.copy(),
        rotation_3d=part.rotation_3d.copy(),
        source_area_mm2=part.source_area_mm2,
        source_faces=list(part.source_faces),
        bend_ops=list(part.bend_ops),
        metadata=dict(part.metadata),
    )


def _apply_trim_combination(
    parts: List[ManufacturingPart],
    groups: List[List[Tuple[int, int]]],
    mask: int,
) -> List[ManufacturingPart]:
    """Apply one combination of trim directions and return the modified parts.

    *groups* is a list of pair-groups.  Each group shares one decision bit:
      bit=0 → for every (i, j) in the group, trim j against i's plane
      bit=1 → for every (i, j) in the group, trim i against j's plane
    """
    copied = [_deep_copy_part(p) for p in parts]
    for k, group in enumerate(groups):
        for i, j in group:
            pi = copied[i]
            pj = copied[j]
            n_i = _rotation_to_normal(pi.rotation_3d)
            n_j = _rotation_to_normal(pj.rotation_3d)
            slab_origin_i = np.asarray(pi.position_3d, dtype=float) + n_i * pi.thickness_mm
            slab_origin_j = np.asarray(pj.position_3d, dtype=float) + n_j * pj.thickness_mm
            if mask & (1 << k):
                _trim_part_against_plane(pi, n_j, slab_origin_j, pj.thickness_mm)
            else:
                _trim_part_against_plane(pj, n_i, slab_origin_i, pi.thickness_mm)
    return copied


def _trim_loss_fraction(
    part: ManufacturingPart,
    other_normal: np.ndarray,
    other_origin: np.ndarray,
    other_thickness: float,
) -> float:
    """Return the fraction of part outline area removed by a trial trim.

    Does NOT mutate the part — works on a copy.
    """
    area_before = part.profile.outline.area
    if area_before < 1.0:
        return 0.0
    trial = _deep_copy_part(part)
    _trim_part_against_plane(trial, other_normal, other_origin, other_thickness)
    area_after = trial.profile.outline.area
    return 1.0 - area_after / area_before


_SIGNIFICANT_TRIM_THRESHOLD = 0.20  # 20 % area loss
_MIN_SHARED_BOUNDARY_MM = 1.0  # Skip pairs with < 1mm shared intersection


def _line_outline_segment(
    part: ManufacturingPart,
    line_point: np.ndarray,
    line_dir: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Return (t_min, t_max) where 3D line passes through part's outline.

    The line is parameterised as P(t) = line_point + t * line_dir.  We project
    it into the part's 2D frame and intersect with the outline polygon.
    The outline is buffered by the part's material thickness so that parts
    meeting at their edges (within one thickness distance) still register a
    shared boundary.
    Returns None if the line doesn't cross the outline.
    """
    profile = part.profile
    if profile.basis_u is None or profile.basis_v is None or profile.origin_3d is None:
        return None

    delta = line_point - profile.origin_3d
    u0 = float(np.dot(profile.basis_u, delta))
    v0 = float(np.dot(profile.basis_v, delta))
    du = float(np.dot(profile.basis_u, line_dir))
    dv = float(np.dot(profile.basis_v, line_dir))

    if abs(du) < 1e-12 and abs(dv) < 1e-12:
        return None  # Line perpendicular to this part's plane

    T = 10000.0
    line_2d = LineString([
        (u0 - T * du, v0 - T * dv),
        (u0 + T * du, v0 + T * dv),
    ])

    # Buffer by material thickness so edges meeting within one thickness
    # distance still count as sharing a boundary.
    check_outline = profile.outline.buffer(part.thickness_mm, quad_segs=2)

    try:
        inter = line_2d.intersection(check_outline)
    except Exception:
        return None

    if inter.is_empty:
        return None

    # Collect all coordinates from the intersection geometry
    coords: list = []
    if inter.geom_type == "LineString":
        coords = list(inter.coords)
    elif inter.geom_type == "MultiLineString":
        for ls in inter.geoms:
            coords.extend(list(ls.coords))
    elif inter.geom_type == "GeometryCollection":
        for g in inter.geoms:
            if g.geom_type == "LineString":
                coords.extend(list(g.coords))
    if len(coords) < 2:
        return None

    # Convert 2D coords back to t-parameter along the 3D line
    use_u = abs(du) > abs(dv)
    t_values = []
    for u, v in coords:
        t = (u - u0) / du if use_u else (v - v0) / dv
        t_values.append(t)

    return (min(t_values), max(t_values))


def _shared_boundary_length(
    pi: ManufacturingPart,
    pj: ManufacturingPart,
    n_i: np.ndarray,
    n_j: np.ndarray,
) -> float:
    """Length (mm) of the plane-plane intersection line through BOTH outlines.

    Two parts only need trim resolution if their planes' intersection line
    passes through both finite outlines simultaneously — i.e. they share a
    physical boundary.  If the intersection line misses one outline, the
    infinite-plane trim is a phantom with no real material conflict.
    """
    L = np.cross(n_i, n_j)
    L_norm = float(np.linalg.norm(L))
    if L_norm < 1e-8:
        return 0.0
    L = L / L_norm

    # A point on the intersection line (least-squares solution)
    d_i = float(np.dot(n_i, pi.profile.origin_3d))
    d_j = float(np.dot(n_j, pj.profile.origin_3d))
    A = np.vstack([n_i, n_j])
    P0 = np.linalg.lstsq(A, np.array([d_i, d_j]), rcond=None)[0]

    seg_i = _line_outline_segment(pi, P0, L)
    seg_j = _line_outline_segment(pj, P0, L)

    if seg_i is None or seg_j is None:
        return 0.0

    overlap_start = max(seg_i[0], seg_j[0])
    overlap_end = min(seg_i[1], seg_j[1])
    return max(0.0, overlap_end - overlap_start)


def _group_pairs_by_stack(
    pairs: List[Tuple[int, int]],
    parts: List[ManufacturingPart],
) -> List[List[Tuple[int, int]]]:
    """Group trim pairs that share the same stack-group combination.

    Stacked siblings have the same plane orientation, so trimming any member
    of stack A against any member of stack B produces nearly the same cut.
    Grouping them into one decision bit collapses the search space.
    """
    key_to_pairs: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for i, j in pairs:
        sg_i = parts[i].metadata.get("stack_group_id", f"_solo_{i}")
        sg_j = parts[j].metadata.get("stack_group_id", f"_solo_{j}")
        key = (sg_i, sg_j)
        key_to_pairs.setdefault(key, []).append((i, j))
    return list(key_to_pairs.values())


def _trim_at_plane_intersections(
    parts: List[ManufacturingPart],
    input_spec: Step3Input,
    joint_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[List[ManufacturingPart], Dict[str, Any]]:
    """Trim part outlines at plane-plane intersection boundaries.

    A shared-boundary gate checks whether the plane-plane intersection line
    passes through both finite outlines; pairs with no shared boundary are
    skipped (the infinite-plane trim would be a phantom with no material
    conflict).

    Surviving pairs are classified as *significant* (either direction removes
    >20 % of a part's area) or *minor* (small corner clips / butt joints).
    Minor pairs use lowest-loss-wins.  Significant pairs are grouped by stack
    membership (stacked siblings share one decision bit) and searched
    exhaustively over the group mask to maximise total outline area.
    """
    # -- Collect intersecting pairs and classify --
    trim_debug: Dict[str, Any] = {
        "significant_trim_threshold": float(_SIGNIFICANT_TRIM_THRESHOLD),
        "pairs_considered_total": 0,
        "pairs_skipped_non_planar": 0,
        "pairs_skipped_missing_basis": 0,
        "pairs_skipped_parallel": 0,
        "pair_trials": [],
        "minor_pairs_count": 0,
        "significant_pairs_count": 0,
        "minor_decisions": [],
        "minor_mask": 0,
        "significant_group_count": 0,
        "significant_groups": [],
        "search_mode": "none",
        "fallback_used": False,
        "best_mask": 0,
        "best_score": 0.0,
        "combos_evaluated": 0,
        "joint_aware_pairs": len(joint_pairs) if joint_pairs else 0,
    }
    if joint_pairs is None:
        joint_pairs = set()
    minor_pairs: List[Tuple[int, int, float, float]] = []  # (i, j, loss_i, loss_j)
    significant_pairs: List[Tuple[int, int]] = []

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            trim_debug["pairs_considered_total"] = (
                int(trim_debug["pairs_considered_total"]) + 1
            )
            pi = parts[i]
            pj = parts[j]

            if (
                pi.region_type != RegionType.PLANAR_CUT
                or pj.region_type != RegionType.PLANAR_CUT
            ):
                trim_debug["pairs_skipped_non_planar"] = (
                    int(trim_debug["pairs_skipped_non_planar"]) + 1
                )
                continue
            if pi.profile.basis_u is None or pj.profile.basis_u is None:
                trim_debug["pairs_skipped_missing_basis"] = (
                    int(trim_debug["pairs_skipped_missing_basis"]) + 1
                )
                continue

            n_i = _rotation_to_normal(pi.rotation_3d)
            n_j = _rotation_to_normal(pj.rotation_3d)

            cross_norm = float(np.linalg.norm(np.cross(n_i, n_j)))
            if cross_norm < 0.1:
                trim_debug["pairs_skipped_parallel"] = (
                    int(trim_debug["pairs_skipped_parallel"]) + 1
                )
                continue

            # Volume gate: only trim if the plane-plane intersection line
            # passes through both finite outlines (shared physical boundary).
            shared = _shared_boundary_length(pi, pj, n_i, n_j)
            if shared < _MIN_SHARED_BOUNDARY_MM:
                continue

            # Trial-trim both directions to measure area loss
            slab_origin_i = np.asarray(pi.position_3d, dtype=float) + n_i * pi.thickness_mm
            slab_origin_j = np.asarray(pj.position_3d, dtype=float) + n_j * pj.thickness_mm
            loss_j = _trim_loss_fraction(pj, n_i, slab_origin_i, pi.thickness_mm)
            loss_i = _trim_loss_fraction(pi, n_j, slab_origin_j, pj.thickness_mm)
            pair_kind = (
                "significant"
                if (
                    loss_i > _SIGNIFICANT_TRIM_THRESHOLD
                    or loss_j > _SIGNIFICANT_TRIM_THRESHOLD
                )
                else "minor"
            )
            trim_debug["pair_trials"].append(
                {
                    "part_i": pi.part_id,
                    "part_j": pj.part_id,
                    "loss_i": round(float(loss_i), 6),
                    "loss_j": round(float(loss_j), 6),
                    "cross_norm": round(float(cross_norm), 6),
                    "classification": pair_kind,
                }
            )

            if (
                loss_i > _SIGNIFICANT_TRIM_THRESHOLD
                or loss_j > _SIGNIFICANT_TRIM_THRESHOLD
            ):
                significant_pairs.append((i, j))
            else:
                minor_pairs.append((i, j, loss_i, loss_j))
    trim_debug["minor_pairs_count"] = int(len(minor_pairs))
    trim_debug["significant_pairs_count"] = int(len(significant_pairs))

    if not minor_pairs and not significant_pairs:
        return parts, trim_debug

    # -- Apply minor pairs: trim the part that extends past the other's boundary --
    # Lowest-loss-wins: trim the part that loses less area (it's more "at its
    # end").  Tie-break on area (larger area wins).
    minor_groups = [[(i, j)] for i, j, _, _ in minor_pairs]
    minor_mask = 0
    for k, (i, j, loss_i, loss_j) in enumerate(minor_pairs):
        decision = "trim_j"
        if loss_i < loss_j:
            # Part i loses less → i is more at its end → trim i
            minor_mask |= 1 << k
            decision = "trim_i"
        elif loss_i == loss_j:
            # Tie: trim the smaller-area part
            if parts[i].profile.outline.area < parts[j].profile.outline.area:
                minor_mask |= 1 << k
                decision = "trim_i"
            else:
                decision = "trim_j_tie"
        else:
            decision = "trim_j"
        trim_debug["minor_decisions"].append(
            {
                "group_index": int(k),
                "part_i": parts[i].part_id,
                "part_j": parts[j].part_id,
                "loss_i": round(float(loss_i), 6),
                "loss_j": round(float(loss_j), 6),
                "decision": decision,
                "mask_bit": int((minor_mask >> k) & 1),
            }
        )
    trim_debug["minor_mask"] = int(minor_mask)
    if minor_groups:
        parts = _apply_trim_combination(parts, minor_groups, minor_mask)

    if not significant_pairs:
        trim_debug["search_mode"] = "minor_only"
        trim_debug["best_mask"] = int(minor_mask)
        trim_debug["best_score"] = 0.0
        return parts, trim_debug

    # -- Group significant pairs by stack membership --
    sig_groups = _group_pairs_by_stack(significant_pairs, parts)
    n_groups = len(sig_groups)
    trim_debug["significant_group_count"] = int(n_groups)
    sig_group_rows: List[Dict[str, Any]] = []
    for k, group in enumerate(sig_groups):
        i0, j0 = group[0]
        sg_i = str(parts[i0].metadata.get("stack_group_id", f"_solo_{i0}"))
        sg_j = str(parts[j0].metadata.get("stack_group_id", f"_solo_{j0}"))
        sig_group_rows.append(
            {
                "group_index": int(k),
                "stack_key": [sg_i, sg_j],
                "pairs": [
                    {
                        "part_i": parts[i].part_id,
                        "part_j": parts[j].part_id,
                    }
                    for i, j in group
                ],
            }
        )
    trim_debug["significant_groups"] = sig_group_rows

    # -- Safety valve on GROUP count (not raw pair count) --
    if n_groups > 16:
        fallback_mask = 0
        for k, group in enumerate(sig_groups):
            i, j = group[0]
            area_i = parts[i].source_area_mm2
            area_j = parts[j].source_area_mm2
            if area_i < area_j:
                fallback_mask |= 1 << k
        trim_debug["search_mode"] = "significant_fallback"
        trim_debug["fallback_used"] = True
        trim_debug["best_mask"] = int(fallback_mask)
        trim_debug["best_score"] = 0.0
        trim_debug["combos_evaluated"] = 0
        return _apply_trim_combination(parts, sig_groups, fallback_mask), trim_debug

    # -- Exhaustive search over group bits --
    # Score rewards useful area (capped at source geometry) and penalizes
    # excess bbox padding that Phase C will clip away.  This prevents
    # inflated bounding boxes from biasing trim direction decisions.
    best_mask = 0
    best_score = -float("inf")
    n_combos = 1 << n_groups

    for mask in range(n_combos):
        candidate = _apply_trim_combination(parts, sig_groups, mask)
        score = sum(
            2.0 * min(p.profile.outline.area, p.source_area_mm2)
            - p.profile.outline.area
            for p in candidate
        )
        if score > best_score:
            best_score = score
            best_mask = mask
    trim_debug["search_mode"] = "significant_exhaustive"
    trim_debug["best_mask"] = int(best_mask)
    trim_debug["best_score"] = float(best_score)
    trim_debug["combos_evaluated"] = int(n_combos)
    return _apply_trim_combination(parts, sig_groups, best_mask), trim_debug


def _trim_part_against_plane(
    part: ManufacturingPart,
    other_normal: np.ndarray,
    other_origin: np.ndarray,
    other_thickness_mm: float = 0.0,
) -> None:
    """Trim a part's outline to stay on its side of another part's plane.

    Uses the half-plane inequality: n_other . (origin + u*x + v*y) >= d_other
    which becomes a*x + b*y >= c in the part's 2D frame.

    The cutting plane is offset to the *nearest* face of the winning part's
    material slab.  The slab occupies [d_surface - thickness, d_surface] along
    the other normal.  When the trimmed part sits on the normal side
    (d_trimmed >= d_surface) the nearest face is the surface itself; when it
    sits on the anti-normal side the nearest face is the far face at
    d_surface - thickness.
    """
    profile = part.profile
    if profile.basis_u is None or profile.basis_v is None or profile.origin_3d is None:
        return

    a = float(np.dot(other_normal, profile.basis_u))
    b = float(np.dot(other_normal, profile.basis_v))
    # Determine which face of the winning slab is closest to the trimmed part.
    d_surface = float(np.dot(other_normal, other_origin))
    trimmed_d = float(np.dot(other_normal, profile.origin_3d))
    if trimmed_d >= d_surface:
        # Trimmed part on normal side → nearest face is the surface
        d_other = d_surface
    else:
        # Trimmed part on anti-normal side → nearest face is the far face
        d_other = d_surface - other_thickness_mm
    c = d_other - float(np.dot(other_normal, profile.origin_3d))

    ab_norm = math.sqrt(a * a + b * b)
    if ab_norm < 1e-8:
        return  # Other plane is parallel in this 2D projection

    # Determine which side of the half-plane the outline centroid is on
    cx = float(profile.outline.centroid.x)
    cy = float(profile.outline.centroid.y)
    centroid_side = a * cx + b * cy - c

    if centroid_side >= 0:
        # Centroid is on the >= side, keep that side (a*x + b*y >= c)
        line_normal_2d = np.array([a, b]) / ab_norm
    else:
        # Centroid is on the < side, keep the opposite half-plane (a*x + b*y <= c)
        # Equivalently: (-a)*x + (-b)*y >= -c
        line_normal_2d = np.array([-a, -b]) / ab_norm

    line_dir_2d = np.array([-line_normal_2d[1], line_normal_2d[0]])

    # A point on the boundary line a*x + b*y = c
    if abs(a) > abs(b):
        line_pt = np.array([c / a, 0.0])
    else:
        line_pt = np.array([0.0, c / b])

    # Build a large half-plane polygon
    bounds = profile.outline.bounds
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 100.0) * 3.0

    p1 = line_pt - extent * line_dir_2d
    p2 = line_pt + extent * line_dir_2d
    p3 = p2 + extent * line_normal_2d
    p4 = p1 + extent * line_normal_2d

    half_plane = Polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])

    try:
        result = profile.outline.intersection(half_plane)
    except Exception:
        return

    result = _clean_polygon(result)
    if result is not None and not result.is_empty and result.area > 1.0:
        part.profile = PartProfile2D(
            outline=result,
            cutouts=profile.cutouts,
            features=profile.features,
            material_key=profile.material_key,
            thickness_mm=profile.thickness_mm,
            basis_u=profile.basis_u,
            basis_v=profile.basis_v,
            origin_3d=profile.origin_3d,
        )


def _split_oversized_part(
    part: ManufacturingPart,
    max_width: float,
    max_height: float,
    input_spec: Step3Input,
) -> List[ManufacturingPart]:
    """Split a part that exceeds sheet dimensions into tiles."""
    outline = part.profile.outline
    bounds = outline.bounds
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]

    cols = max(1, int(math.ceil(w / max_width)))
    rows = max(1, int(math.ceil(h / max_height)))

    if cols == 1 and rows == 1:
        return [part]

    tile_w = w / cols
    tile_h = h / rows

    result: List[ManufacturingPart] = []
    for r in range(rows):
        for c_idx in range(cols):
            tile_minx = bounds[0] + c_idx * tile_w
            tile_miny = bounds[1] + r * tile_h
            tile_box = Polygon(
                [
                    (tile_minx, tile_miny),
                    (tile_minx + tile_w, tile_miny),
                    (tile_minx + tile_w, tile_miny + tile_h),
                    (tile_minx, tile_miny + tile_h),
                ]
            )

            tile_outline = outline.intersection(tile_box)
            tile_polygon = _largest_polygon(tile_outline)

            if tile_polygon is None or tile_polygon.is_empty:
                continue
            if tile_polygon.area < input_spec.min_planar_region_area_mm2:
                continue

            tile_profile = PartProfile2D(
                outline=tile_polygon,
                material_key=part.profile.material_key,
                thickness_mm=part.profile.thickness_mm,
                basis_u=part.profile.basis_u,
                basis_v=part.profile.basis_v,
                origin_3d=part.profile.origin_3d,
            )
            result.append(
                ManufacturingPart(
                    part_id=f"{part.part_id}_tile_{r}_{c_idx}",
                    material_key=part.material_key,
                    thickness_mm=part.thickness_mm,
                    profile=tile_profile,
                    region_type=part.region_type,
                    position_3d=part.position_3d.copy(),
                    rotation_3d=part.rotation_3d.copy(),
                    source_area_mm2=float(tile_polygon.area),
                    source_faces=list(part.source_faces),
                    bend_ops=[],
                    metadata=dict(part.metadata),
                )
            )

    return result if result else [part]
