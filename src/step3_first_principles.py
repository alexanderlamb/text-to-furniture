"""First-principles Step 3: single-strategy decomposition for rapid iteration."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from dfm_rules import DFMConfig, DFMViolation, check_part_dfm
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
    seam_weight: float = 0.15
    assembly_weight: float = 0.10
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
    seam_penalty: float
    assembly_penalty: float
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
                    ] + [
                        list(h.coords) for h in part.profile.outline.interiors
                    ],
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
                "seam_penalty": self.quality_metrics.seam_penalty,
                "assembly_penalty": self.quality_metrics.assembly_penalty,
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
                "cutouts_2d": [
                    list(c.exterior.coords) for c in part.profile.cutouts
                ] + [
                    list(h.coords) for h in part.profile.outline.interiors
                ],
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
                "origin_3d": [
                    float(v)
                    for v in (
                        part.profile.origin_3d
                        + part.metadata.get("stack_layer_offset_mm", 0.0)
                        * _rotation_to_normal(part.rotation_3d)
                    )
                ]
                if part.profile.origin_3d is not None
                else [float(v) for v in part.position_3d],
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

    return {
        "phase_label": label,
        "phase_index": index,
        "part_count": len(parts_payload),
        "units": "mm",
        "parts": parts_payload,
        "joints": joints_payload,
    }


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
    candidates, merged_planar_pairs = _collapse_planar_face_pairs(coplanar_grouped)
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
        _serialize_parts_snapshot("Selected parts", 0, selected_parts),
    ]
    selected_parts, constrain_snapshots = _constrain_outlines(
        selected_parts, mesh, input_spec
    )
    phase_snapshots.extend(constrain_snapshots)
    joints = _synthesize_joint_specs(
        selected_parts,
        joint_distance_mm=input_spec.joint_distance_mm,
        contact_tolerance_mm=input_spec.joint_contact_tolerance_mm,
        parallel_dot_threshold=input_spec.joint_parallel_dot_threshold,
    )
    phase_snapshots.append(
        _serialize_parts_snapshot("With joints", 5, selected_parts, joints)
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

    quality = _compute_quality_metrics(
        mesh=mesh,
        selected_parts=selected_parts,
        joints=joints,
        part_budget_max=input_spec.part_budget_max,
        fidelity_weight=input_spec.fidelity_weight,
        seam_weight=input_spec.seam_weight,
        assembly_weight=input_spec.assembly_weight,
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


def _merge_planar_pair(
    a: _RegionCandidate,
    b: _RegionCandidate,
    member_thickness_mm: float,
) -> _RegionCandidate:
    chosen = a if a.area_mm2 >= b.area_mm2 else b
    merged_faces = sorted(set(a.source_faces) | set(b.source_faces))
    merged_metadata = dict(chosen.metadata)
    merged_metadata["kind"] = "merged_facet_pair"
    merged_metadata["merged_source_face_count"] = int(len(merged_faces))
    merged_metadata["member_thickness_mm"] = float(max(0.0, member_thickness_mm))

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
            planar_out.append(
                _merge_planar_pair(a, candidates[best_j], member_thickness_mm=best_delta_n)
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
    layer_options: List[Tuple[float, float, int, int]] = []  # (sort_gain, gain, rank, cand_idx)
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
        offsets = _stack_layer_offsets_mm(
            selected_layers,
            sheet_thickness_mm=float(cand.profile.thickness_mm),
            member_thickness_mm=member_thickness,
        )
        stack_group_id = f"cand_{cand_idx:04d}"

        for layer_idx, offset in enumerate(offsets):
            part_id = f"part_{part_counter:02d}"
            part_counter += 1
            metadata = dict(cand.metadata)
            outline_area_mm2 = float(max(0.0, cand.profile.outline.area))
            source_area_mm2 = float(max(0.0, cand.area_mm2))
            metadata["stack_group_id"] = stack_group_id
            metadata["stack_layer_index"] = int(layer_idx + 1)
            metadata["stack_layer_count"] = int(selected_layers)
            metadata["stack_target_layers"] = int(target_layers_by_candidate[cand_idx])
            metadata["stack_layer_offset_mm"] = float(offset)
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

            # outline_2d is defined in the candidate's local plane frame.
            # Use the same frame anchor (origin_3d) for position_3d when available.
            base_anchor = (
                np.asarray(cand.origin_3d, dtype=float)
                if cand.origin_3d is not None
                else np.asarray(cand.position_3d, dtype=float)
            )

            parts.append(
                ManufacturingPart(
                    part_id=part_id,
                    material_key=cand.profile.material_key,
                    thickness_mm=cand.profile.thickness_mm,
                    profile=cand.profile,
                    region_type=cand.region_type,
                    position_3d=base_anchor + offset * normal,
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
    offsets = (np.arange(selected_layers, dtype=float) - 0.5 * (selected_layers - 1)) * step

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
) -> Tuple[List[ManufacturingPart], Dict[str, int | bool | float]]:
    if not enabled or len(parts) < 2:
        return (
            parts,
            {
                "intersection_filter_enabled": bool(enabled),
                "intersection_clearance_mm": float(clearance_mm),
                "intersection_dropped_count": 0,
                "intersection_allowed_stack_count": 0,
                "intersection_allowed_joint_intent_count": 0,
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

    for part in ordered:
        rejected = False
        local_conflicts = 0
        local_allowed_stack = 0
        local_allowed_joint_intent = 0
        for prev in kept:
            if not _parts_intersect(part, prev, clearance_mm=clearance_mm):
                continue

            local_conflicts += 1
            group_a = str(part.metadata.get("stack_group_id", "")).strip()
            group_b = str(prev.metadata.get("stack_group_id", "")).strip()
            if group_a and group_b and group_a == group_b:
                allowed_stack += 1
                local_allowed_stack += 1
                continue

            # Use plane geometry to decide: only parallel parts on the same
            # plane can genuinely occupy the same space.  Non-parallel OBB
            # overlap is expected at joint edges.
            if not _is_genuine_plane_conflict(part, prev, clearance_mm):
                allowed_joint_intent_count += 1
                local_allowed_joint_intent += 1
                continue

            rejected = True
            break

        if rejected:
            dropped_count += 1
            continue
        part.metadata["intersection_conflict_count"] = int(local_conflicts)
        part.metadata["intersection_allowed_stack_contacts"] = int(local_allowed_stack)
        part.metadata["intersection_allowed_joint_intent_contacts"] = int(
            local_allowed_joint_intent
        )
        kept.append(part)

    kept.sort(key=lambda p: p.part_id)
    for idx, part in enumerate(kept):
        part.part_id = f"part_{idx:02d}"

    debug = {
        "intersection_filter_enabled": bool(enabled),
        "intersection_clearance_mm": float(clearance_mm),
        "intersection_dropped_count": int(dropped_count),
        "intersection_allowed_stack_count": int(allowed_stack),
        "intersection_allowed_joint_intent_count": int(allowed_joint_intent_count),
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


def _synthesize_joint_specs(
    parts: List[ManufacturingPart],
    joint_distance_mm: float,
    contact_tolerance_mm: float,
    parallel_dot_threshold: float = 0.95,
) -> List[JointSpec]:
    joints: List[JointSpec] = []
    parallel_threshold = float(np.clip(parallel_dot_threshold, 0.0, 0.999999))
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pa = parts[i]
            pb = parts[j]
            group_a = str(pa.metadata.get("stack_group_id", "")).strip()
            group_b = str(pb.metadata.get("stack_group_id", "")).strip()
            if group_a and group_b and group_a == group_b:
                # Labeled sheet laminations are a single physical member.
                continue

            n_a = _rotation_to_normal(pa.rotation_3d)
            n_b = _rotation_to_normal(pb.rotation_3d)
            if abs(float(np.dot(n_a, n_b))) >= parallel_threshold:
                # Parallel/near-parallel faces are handled by glue/screws, not explicit joints.
                continue

            in_contact, signed_gap = _parts_in_contact_band(
                pa,
                pb,
                contact_tolerance_mm=contact_tolerance_mm,
            )
            if not in_contact:
                continue

            distance = float(np.linalg.norm(pa.position_3d - pb.position_3d))
            # Guard against pathological pairings when loose contact tolerance is used.
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
        design.add_component(component)

    for spec in joints:
        jt = (
            JointType.THROUGH_BOLT
            if spec.joint_type == "through_bolt"
            else JointType.TAB_SLOT
        )
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


def _compute_quality_metrics(
    mesh: trimesh.Trimesh,
    selected_parts: List[ManufacturingPart],
    joints: List[JointSpec],
    part_budget_max: int,
    fidelity_weight: float,
    seam_weight: float,
    assembly_weight: float,
) -> Step3QualityMetrics:
    if not selected_parts:
        return Step3QualityMetrics(
            hausdorff_mm=1e6,
            mean_distance_mm=1e6,
            normal_error_deg=180.0,
            seam_penalty=1.0,
            assembly_penalty=1.0,
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

    effective_members = _effective_member_count(selected_parts)
    seam_penalty = max(0.0, (effective_members - 1) / max(1, part_budget_max))
    assembly_penalty = max(0.0, len(joints) / max(1, len(selected_parts) * 3))

    hausdorff_score = 1.0 / (1.0 + hausdorff_mm / max(1.0, max_extent))
    normal_score = 1.0 / (1.0 + normal_error_deg / 45.0)
    seam_score = max(0.0, 1.0 - seam_penalty)
    assembly_score = max(0.0, 1.0 - assembly_penalty)

    remainder = max(1e-6, 1.0 - fidelity_weight)
    seam_component = remainder * (
        seam_weight / max(1e-6, seam_weight + assembly_weight)
    )
    asm_component = remainder * (
        assembly_weight / max(1e-6, seam_weight + assembly_weight)
    )

    overall = (
        fidelity_weight * (0.8 * hausdorff_score + 0.2 * normal_score)
        + seam_component * seam_score
        + asm_component * assembly_score
    )

    return Step3QualityMetrics(
        hausdorff_mm=float(hausdorff_mm),
        mean_distance_mm=float(mean_distance_mm),
        normal_error_deg=float(normal_error_deg),
        seam_penalty=float(seam_penalty),
        assembly_penalty=float(assembly_penalty),
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
            seam_penalty=1.0,
            assembly_penalty=1.0,
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
        if profile.basis_u is None or profile.basis_v is None or profile.origin_3d is None:
            grown.append(part)
            continue

        grown_outline = _compute_coplanar_extent(mesh, profile, part.rotation_3d)
        if grown_outline.is_empty or grown_outline.area <= profile.outline.area:
            grown.append(part)
            continue

        # Grow outward but preserve holes: expand the exterior using the
        # grown extent, then re-punch the original interior rings.
        new_exterior = unary_union(
            [Polygon(profile.outline.exterior), grown_outline]
        )
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

    snapshots.append(_serialize_parts_snapshot("Grown (coplanar)", 1, grown))

    # Phase B: trim at plane-plane intersections
    trimmed = _trim_at_plane_intersections(grown, input_spec)

    snapshots.append(_serialize_parts_snapshot("Trimmed (planes)", 2, trimmed))

    # Phase C skipped: Phase A now grows outlines to the mesh cross-section
    # directly, so a redundant mesh clip is no longer needed.
    clipped = trimmed

    snapshots.append(_serialize_parts_snapshot("Clipped (mesh)", 3, clipped))

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

    snapshots.append(_serialize_parts_snapshot("Final", 4, result))

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
        outline=Polygon(p.outline.exterior.coords,
                        [list(h.coords) for h in p.outline.interiors]),
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
            if mask & (1 << k):
                _trim_part_against_plane(pi, n_j, pj.profile.origin_3d, pj.thickness_mm)
            else:
                _trim_part_against_plane(pj, n_i, pi.profile.origin_3d, pi.thickness_mm)
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
) -> List[ManufacturingPart]:
    """Trim part outlines at plane-plane intersection boundaries.

    Pairs are classified as *significant* (either direction removes >20% of a
    part's area) or *minor* (small corner clips / butt joints).  Minor pairs
    use largest-area-wins directly.  Significant pairs are grouped by stack
    membership (stacked siblings share one decision bit) and searched
    exhaustively over the group mask to maximize total outline area.
    """
    # -- Collect intersecting pairs and classify --
    minor_pairs: List[Tuple[int, int, float, float]] = []  # (i, j, loss_i, loss_j)
    significant_pairs: List[Tuple[int, int]] = []

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            pi = parts[i]
            pj = parts[j]

            if pi.region_type != RegionType.PLANAR_CUT or pj.region_type != RegionType.PLANAR_CUT:
                continue
            if pi.profile.basis_u is None or pj.profile.basis_u is None:
                continue

            n_i = _rotation_to_normal(pi.rotation_3d)
            n_j = _rotation_to_normal(pj.rotation_3d)

            cross_norm = float(np.linalg.norm(np.cross(n_i, n_j)))
            if cross_norm < 0.1:
                continue

            # Trial-trim both directions to measure area loss
            loss_j = _trim_loss_fraction(pj, n_i, pi.profile.origin_3d, pi.thickness_mm)
            loss_i = _trim_loss_fraction(pi, n_j, pj.profile.origin_3d, pj.thickness_mm)

            if loss_i > _SIGNIFICANT_TRIM_THRESHOLD or loss_j > _SIGNIFICANT_TRIM_THRESHOLD:
                significant_pairs.append((i, j))
            else:
                minor_pairs.append((i, j, loss_i, loss_j))

    if not minor_pairs and not significant_pairs:
        return parts

    # -- Apply minor pairs: trim the part that loses less (it's more "at its end") --
    # The part with lower trim loss has the intersection line closer to its edge,
    # meaning it barely extends past the other part. Trim it so the continuing
    # part keeps its full extent.  Tie-break on area (larger area wins).
    minor_groups = [[(i, j)] for i, j, _, _ in minor_pairs]
    minor_mask = 0
    for k, (i, j, loss_i, loss_j) in enumerate(minor_pairs):
        if loss_i < loss_j:
            # Part i loses less → i is more at its end → trim i
            minor_mask |= 1 << k
        elif loss_i == loss_j:
            # Tie: trim the smaller-area part
            if parts[i].profile.outline.area < parts[j].profile.outline.area:
                minor_mask |= 1 << k
        # else: loss_j < loss_i → trim j → bit stays 0
    if minor_groups:
        parts = _apply_trim_combination(parts, minor_groups, minor_mask)

    if not significant_pairs:
        return parts

    # -- Group significant pairs by stack membership --
    sig_groups = _group_pairs_by_stack(significant_pairs, parts)
    n_groups = len(sig_groups)

    # -- Safety valve on GROUP count (not raw pair count) --
    if n_groups > 16:
        fallback_mask = 0
        for k, group in enumerate(sig_groups):
            i, j = group[0]
            area_i = parts[i].source_area_mm2
            area_j = parts[j].source_area_mm2
            if area_i < area_j:
                fallback_mask |= 1 << k
        return _apply_trim_combination(parts, sig_groups, fallback_mask)

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

    return _apply_trim_combination(parts, sig_groups, best_mask)


def _trim_part_against_plane(
    part: ManufacturingPart,
    other_normal: np.ndarray,
    other_origin: np.ndarray,
    other_thickness_mm: float = 0.0,
) -> None:
    """Trim a part's outline to stay on its side of another part's plane.

    Uses the half-plane inequality: n_other . (origin + u*x + v*y) >= d_other
    which becomes a*x + b*y >= c in the part's 2D frame.

    The cutting plane is offset inward by other_thickness_mm so that the
    trimmed part butts against the inner face of the winning part, preventing
    physical overlap of material volumes.
    """
    profile = part.profile
    if profile.basis_u is None or profile.basis_v is None or profile.origin_3d is None:
        return

    a = float(np.dot(other_normal, profile.basis_u))
    b = float(np.dot(other_normal, profile.basis_v))
    # Offset the cutting plane inward by the winning part's thickness.
    # The part surface is at d_other; material extends inward (against normal)
    # by thickness_mm, so the inner face is at d_other - thickness.
    d_other = float(np.dot(other_normal, other_origin)) - other_thickness_mm
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
