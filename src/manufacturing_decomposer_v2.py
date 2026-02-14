"""
3D-first manufacturing decomposer (v2).

Keeps all geometric reasoning in 3D until DXF/SVG export. Pipeline:
  1. Load mesh
  2. Generate candidate slabs (RANSAC + axis + surface)
  3. Greedy set-cover selection
  4. 3D joint detection → 2D projection
  5. Joint synthesis (existing)
  6. Build FurnitureDesign
  7. Score and assembly sequence
"""
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon

from furniture import (
    Component, ComponentType, Joint, JointType,
    AssemblyGraph, FurnitureDesign,
)
from geometry_primitives import PartProfile2D, polygon_to_profile
from slab_candidates import Slab3D, Slab3DConfig, generate_candidates, slab3d_to_part_profile
from slab_selector import SlabSelectionConfig, select_slabs
from slab_joints import detect_joints, intersections_to_joint_specs
from joint_synthesizer import JointSynthesisConfig, synthesize_joints
from dfm_rules import DFMConfig, DFMViolation, check_part_dfm
from scoring import ScoringConfig, DecompositionScore, score_decomposition
from assembly_sequence import AssemblyStep, optimize_assembly_sequence
from mesh_decomposer import load_mesh, DecompositionConfig, normal_to_rotation
from mesh_cleanup import MeshCleanupConfig

logger = logging.getLogger(__name__)


@dataclass
class ManufacturingDecompositionConfigV2:
    """Configuration for v2 manufacturing-aware decomposition."""
    slab_candidates: Slab3DConfig = field(default_factory=Slab3DConfig)
    slab_selection: SlabSelectionConfig = field(
        default_factory=SlabSelectionConfig
    )
    joint_synthesis: JointSynthesisConfig = field(default_factory=JointSynthesisConfig)
    dfm: DFMConfig = field(default_factory=DFMConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    default_material: str = "plywood_baltic_birch"
    target_height_mm: float = 750.0
    auto_scale: bool = True
    mesh_cleanup: MeshCleanupConfig = field(
        default_factory=lambda: MeshCleanupConfig(enabled=True)
    )
    joint_proximity_mm: float = 25.0
    pose_alignment_fix: bool = True


@dataclass
class ManufacturingDecompositionResult:
    """Output of the manufacturing-aware decomposition pipeline."""
    design: FurnitureDesign
    parts: Dict[str, PartProfile2D]
    dfm_violations: List[DFMViolation]
    score: DecompositionScore
    assembly_steps: List[AssemblyStep]
    candidates: list = field(default_factory=list)
    segmentation: Optional[Any] = None

    def to_manufacturing_json(self) -> dict:
        """Export to manufacturing JSON schema."""
        parts_json = []
        for name, profile in self.parts.items():
            comp = self.design.get_component(name)
            bounds = profile.outline.bounds

            outline = profile.outline
            if isinstance(outline, MultiPolygon):
                outline = max(outline.geoms, key=lambda g: g.area)
            part_data = {
                "name": name,
                "material": profile.material_key,
                "thickness_mm": profile.thickness_mm,
                "outline_2d": list(outline.exterior.coords),
                "cutouts_2d": [
                    list(c.exterior.coords) for c in profile.cutouts
                ],
                "features": [
                    {
                        "type": f.feature_type.value,
                        "parameters": f.parameters,
                    }
                    for f in profile.features
                ],
                "width_mm": bounds[2] - bounds[0],
                "height_mm": bounds[3] - bounds[1],
            }
            if comp:
                part_data["position_3d"] = comp.position.tolist()
                part_data["rotation_3d"] = comp.rotation.tolist()
                part_data["component_type"] = comp.type.value

            parts_json.append(part_data)

        joints_json = [
            {
                "component_a": j.component_a,
                "component_b": j.component_b,
                "joint_type": j.joint_type.value,
                "parameters": j.parameters,
            }
            for j in self.design.assembly.joints
        ]

        assembly_json = [
            {
                "step": s.step_number,
                "component": s.component_name,
                "action": s.action,
                "attach_to": s.attach_to,
                "notes": s.notes,
            }
            for s in self.assembly_steps
        ]

        return {
            "units": "mm",
            "material": self.design.components[0].material if self.design.components else "",
            "parts": parts_json,
            "assembly_graph": {
                "joints": joints_json,
                "assembly_order": assembly_json,
            },
            "metrics": {
                "hausdorff_mm": self.score.hausdorff_mm,
                "mean_distance_mm": self.score.mean_distance_mm,
                "part_count": self.score.part_count,
                "dfm_errors": self.score.dfm_violations_error,
                "dfm_warnings": self.score.dfm_violations_warning,
                "structural_plausibility": self.score.structural_plausibility,
                "overall_score": self.score.overall_score,
            },
        }


def decompose_manufacturing_v2(
    filepath: str,
    config: Optional[ManufacturingDecompositionConfigV2] = None,
) -> ManufacturingDecompositionResult:
    """Full v2 manufacturing-aware decomposition pipeline.

    Args:
        filepath: Path to mesh file (STL, OBJ, GLB, PLY).
        config: V2 decomposition parameters.

    Returns:
        ManufacturingDecompositionResult (same type as v1 for compatibility).
    """
    if config is None:
        config = ManufacturingDecompositionConfigV2()

    # Step 1: Load mesh
    decomp_config = DecompositionConfig(
        default_material=config.default_material,
        target_height_mm=config.target_height_mm,
        auto_scale=config.auto_scale,
        mesh_cleanup=config.mesh_cleanup,
    )
    mesh = load_mesh(filepath, decomp_config)
    logger.info("Loaded mesh: %d faces, bounds %s",
                len(mesh.faces), mesh.bounds.tolist())

    # Step 2: Generate candidate slabs
    config.slab_candidates.material_key = config.default_material
    config.slab_candidates.target_height_mm = config.target_height_mm
    candidates = generate_candidates(mesh, config.slab_candidates)
    logger.info("Generated %d candidate slabs", len(candidates))

    if not candidates:
        logger.warning("No candidates generated — returning empty result")
        return _empty_result(
            mesh=mesh, candidates=[], config=config,
            failure_reason="no_candidates_generated",
        )

    # Resolve DFM config with material-specific sheet size unless explicitly overridden.
    effective_dfm = _resolve_dfm_config(config.default_material, config.dfm)

    # Step 3: Select slabs via greedy set cover / volume fill
    selection = select_slabs(
        mesh, candidates, config.slab_selection, effective_dfm,
    )
    logger.info(
        "Selected %d slabs, mode=%s, objective %.1f%%",
        len(selection.selected_slabs),
        selection.objective_mode,
        selection.coverage_fraction * 100,
    )

    if not selection.selected_slabs:
        logger.warning("No slabs selected — returning empty result")
        return _empty_result(
            mesh=mesh, candidates=candidates, config=config,
            selection=selection, failure_reason="no_slabs_selected",
        )

    selected_slabs = selection.selected_slabs

    # Step 4: Detect joints in 3D
    intersections = detect_joints(selected_slabs, config.joint_proximity_mm)
    logger.info("Detected %d intersections", len(intersections))

    # Step 5: Convert to parts and joint specs
    parts: Dict[str, PartProfile2D] = {}
    for i, slab in enumerate(selected_slabs):
        parts[f"part_{i}"] = slab3d_to_part_profile(slab)

    joint_specs = intersections_to_joint_specs(intersections, selected_slabs)

    # Step 6: Synthesize joints (existing module)
    if joint_specs:
        parts, joints = synthesize_joints(
            parts, joint_specs, config.joint_synthesis, effective_dfm,
        )
    else:
        joints = []

    # Step 7: DFM validation on final parts
    all_violations: List[DFMViolation] = []
    for name, profile in parts.items():
        violations = check_part_dfm(profile, effective_dfm)
        all_violations.extend(violations)

    # Step 8: Build FurnitureDesign
    design = _build_furniture_design(
        selected_slabs, parts, joints,
        pose_alignment_fix=config.pose_alignment_fix,
    )

    # Step 8b: Fit-ratio logging (components AABB vs mesh extents)
    if design.components:
        positions = np.array([c.position for c in design.components])
        comp_min = positions.min(axis=0)
        comp_max = positions.max(axis=0)
        mesh_extents = mesh.bounds[1] - mesh.bounds[0]
        comp_extents = comp_max - comp_min
        safe_mesh = np.where(mesh_extents > 1e-6, mesh_extents, 1.0)
        fit_ratios = comp_extents / safe_mesh
        logger.info(
            "Fit ratios: X=%.3f, Y=%.3f, Z=%.3f",
            fit_ratios[0], fit_ratios[1], fit_ratios[2],
        )

    # Step 9: Score
    score = score_decomposition(
        mesh, design, list(parts.values()),
        all_violations, config.scoring,
    )

    # Step 10: Assembly sequence
    assembly_steps = optimize_assembly_sequence(design)

    # Step 11: Populate debug telemetry
    source_counts = Counter(s.source for s in selected_slabs)
    debug_dict = {
        "mesh_bounds_mm": mesh.bounds.tolist(),
        "mesh_extents_mm": (mesh.bounds[1] - mesh.bounds[0]).tolist(),
        "candidate_count": len(candidates),
        "candidate_count_after_dfm": int(selection.candidate_count_after_dfm),
        "selection_dfm_rejected_count": int(selection.dfm_rejected_count),
        "selected_slab_count": len(selected_slabs),
        "coverage_fraction": float(selection.coverage_fraction),
        "objective_mode": selection.objective_mode,
        "objective_fraction": float(selection.coverage_fraction),
        "surface_coverage_fraction": float(selection.surface_coverage_fraction),
        "volume_fill_fraction": float(selection.volume_fill_fraction),
        "target_volume_fill": float(config.slab_selection.target_volume_fill),
        "plane_penalty_weight": float(config.slab_selection.plane_penalty_weight),
        "voxel_resolution_mm": float(config.slab_selection.voxel_resolution_mm),
        "intersection_count": len(intersections),
        "joint_count": len(joints),
        "dfm_error_count": sum(1 for v in all_violations if v.severity == "error"),
        "dfm_warning_count": sum(1 for v in all_violations if v.severity == "warning"),
        "pose_alignment_fix": config.pose_alignment_fix,
        "selected_source_histogram": dict(source_counts),
        "selection_trace": selection.selection_trace,
        "selection_stop_reason": selection.stop_reason,
    }
    if design.components:
        debug_dict["fit_ratios"] = fit_ratios.tolist()
    design.metadata["decomposition_debug"] = debug_dict

    logger.info(
        "V2 decomposition complete: %d parts, %d joints, score=%.2f",
        len(design.components), len(joints), score.overall_score,
    )

    return ManufacturingDecompositionResult(
        design=design,
        parts=parts,
        dfm_violations=all_violations,
        score=score,
        assembly_steps=assembly_steps,
        candidates=[],  # v2 doesn't use ManufacturingCandidate
        segmentation=None,
    )


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _classify_slab_component(
    slab: Slab3D,
    slab_idx: int,
    all_slabs: List[Slab3D],
    joints: List[Joint],
    mesh_bounds: np.ndarray,
) -> ComponentType:
    """Classify a slab into a ComponentType based on geometry and connectivity.

    Rules:
    - |normal.z| > 0.7 AND has >= 2 vertical neighbors → SHELF/SEAT
    - |normal.z| > 0.7 AND at top of bounding box → SHELF
    - |normal.z| < 0.3 AND narrow + tall → LEG
    - |normal.z| < 0.3 AND wide → PANEL
    - Angled → BRACE
    """
    n = slab.normal / np.linalg.norm(slab.normal)
    name = f"part_{slab_idx}"
    mesh_height = mesh_bounds[1][2] - mesh_bounds[0][2]

    # Count vertical neighbors (slabs with |n.z| < 0.3 connected via joints)
    neighbors = []
    for j in joints:
        if j.component_a == name:
            neighbors.append(j.component_b)
        elif j.component_b == name:
            neighbors.append(j.component_a)

    vertical_neighbors = 0
    for nb_name in neighbors:
        # Extract index from "part_X"
        try:
            nb_idx = int(nb_name.split("_")[1])
            if nb_idx < len(all_slabs):
                nb_n = all_slabs[nb_idx].normal / np.linalg.norm(all_slabs[nb_idx].normal)
                if abs(nb_n[2]) < 0.3:
                    vertical_neighbors += 1
        except (ValueError, IndexError):
            pass

    if abs(n[2]) > 0.7:
        if vertical_neighbors >= 2:
            # Check if near top of mesh
            if slab.origin[2] > mesh_bounds[0][2] + mesh_height * 0.6:
                return ComponentType.SHELF
            return ComponentType.SHELF
        if slab.origin[2] > mesh_bounds[0][2] + mesh_height * 0.7:
            return ComponentType.SHELF
        return ComponentType.SHELF

    if abs(n[2]) < 0.3:
        aspect = max(slab.width_mm, slab.height_mm) / max(min(slab.width_mm, slab.height_mm), 1.0)
        if aspect > 4.0 and max(slab.width_mm, slab.height_mm) > mesh_height * 0.5:
            return ComponentType.LEG
        return ComponentType.PANEL

    return ComponentType.BRACE


def _basis_to_rotation(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Compute Euler XYZ rotation from an actual slab basis.

    The rotation R maps local (x, y, z) → world so that:
      R @ [1,0,0] = basis_u
      R @ [0,1,0] = basis_v
      R @ [0,0,1] = normal

    Extraction uses the R = Rz @ Ry @ Rx convention matching
    ``_rotation_matrix_xyz`` in workers.py.

    For R = Rz @ Ry @ Rx the matrix elements are::

        R[2,0] = -sin(ry)
        R[2,1] =  cos(ry)*sin(rx)
        R[2,2] =  cos(ry)*cos(rx)
        R[1,0] =  sin(rz)*cos(ry)
        R[0,0] =  cos(rz)*cos(ry)
    """
    n = normal / np.linalg.norm(normal)
    u = basis_u / np.linalg.norm(basis_u)
    v = basis_v / np.linalg.norm(basis_v)
    R = np.column_stack([u, v, n])

    # R[2,0] = -sin(ry)
    ry = np.arcsin(np.clip(-R[2, 0], -1, 1))
    if np.abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock — set rz = 0 and recover rx
        rz = 0.0
        if R[2, 0] < 0:  # sy > 0, ry ≈ +π/2
            rx = np.arctan2(R[0, 1], R[1, 1])
        else:             # sy < 0, ry ≈ -π/2
            rx = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([rx, ry, rz])


def _build_furniture_design(
    slabs: List[Slab3D],
    parts: Dict[str, PartProfile2D],
    joints: List[Joint],
    pose_alignment_fix: bool = True,
) -> FurnitureDesign:
    """Convert slabs, parts, and joints into a FurnitureDesign."""
    design = FurnitureDesign(name="manufacturing_v2")
    assembly = AssemblyGraph(joints=list(joints))

    # Get mesh bounds approximation from slab origins
    if slabs:
        origins = np.array([s.origin for s in slabs])
        approx_bounds = np.array([origins.min(axis=0), origins.max(axis=0)])
    else:
        approx_bounds = np.array([[0, 0, 0], [750, 750, 750]], dtype=float)

    for i, slab in enumerate(slabs):
        name = f"part_{i}"
        if name not in parts:
            continue

        profile = parts[name]
        comp_profile = polygon_to_profile(profile.outline)
        comp_type = _classify_slab_component(slab, i, slabs, joints, approx_bounds)
        n = slab.normal / np.linalg.norm(slab.normal)

        features_list = [
            {
                "type": f.feature_type.value,
                "parameters": f.parameters,
            }
            for f in profile.features
        ]

        if pose_alignment_fix:
            # Use the slab's actual basis for rotation (not just normal)
            rotation = _basis_to_rotation(slab.basis_u, slab.basis_v, n)
            position = slab.origin.copy()
        else:
            rotation = normal_to_rotation(n)
            position = slab.origin.copy()

        # Diagnostics logging
        ob = profile.outline.bounds
        center_u = (ob[0] + ob[2]) / 2.0
        center_v = (ob[1] + ob[3]) / 2.0
        logger.debug(
            "Part %s: origin=[%.1f,%.1f,%.1f], outline_center_uv=(%.1f,%.1f), "
            "bounds_uv=(%.1f,%.1f,%.1f,%.1f), pos=[%.1f,%.1f,%.1f], rot=[%.3f,%.3f,%.3f]",
            name,
            slab.origin[0], slab.origin[1], slab.origin[2],
            center_u, center_v,
            ob[0], ob[1], ob[2], ob[3],
            position[0], position[1], position[2],
            rotation[0], rotation[1], rotation[2],
        )

        comp = Component(
            name=name,
            type=comp_type,
            profile=comp_profile,
            thickness=slab.thickness_mm,
            position=position,
            rotation=rotation,
            material=slab.material_key,
            features=features_list,
        )
        design.add_component(comp)

    design.assembly = assembly
    for joint in joints:
        assembly.assembly_order.append((joint.component_a, joint.component_b))

    return design


def _resolve_dfm_config(material_key: str, configured_dfm: DFMConfig) -> DFMConfig:
    """Apply material sheet size unless the caller explicitly overrides it."""
    defaults = DFMConfig()
    material_dfm = DFMConfig.from_material(material_key)

    # Keep user-provided sheet limits only if they differ from global defaults.
    custom_sheet_width = abs(configured_dfm.max_sheet_width_mm - defaults.max_sheet_width_mm) > 1e-6
    custom_sheet_height = abs(configured_dfm.max_sheet_height_mm - defaults.max_sheet_height_mm) > 1e-6

    return DFMConfig(
        min_slot_width_inch=configured_dfm.min_slot_width_inch,
        min_internal_radius_inch=configured_dfm.min_internal_radius_inch,
        slip_fit_clearance_inch=configured_dfm.slip_fit_clearance_inch,
        min_bridge_width_inch=configured_dfm.min_bridge_width_inch,
        max_aspect_ratio=configured_dfm.max_aspect_ratio,
        max_sheet_width_mm=(
            configured_dfm.max_sheet_width_mm
            if custom_sheet_width else material_dfm.max_sheet_width_mm
        ),
        max_sheet_height_mm=(
            configured_dfm.max_sheet_height_mm
            if custom_sheet_height else material_dfm.max_sheet_height_mm
        ),
    )


def _empty_result(
    mesh: Optional[trimesh.Trimesh] = None,
    candidates: Optional[list] = None,
    config: Optional[ManufacturingDecompositionConfigV2] = None,
    selection=None,
    failure_reason: str = "",
) -> ManufacturingDecompositionResult:
    """Return an empty result for edge cases, with debug telemetry."""
    design = FurnitureDesign(name="manufacturing_v2")

    debug_dict: Dict[str, Any] = {}
    if failure_reason:
        debug_dict["failure_reason"] = failure_reason
    if mesh is not None:
        debug_dict["mesh_bounds_mm"] = mesh.bounds.tolist()
        debug_dict["mesh_extents_mm"] = (mesh.bounds[1] - mesh.bounds[0]).tolist()
        debug_dict["mesh_faces"] = int(len(mesh.faces))
        debug_dict["mesh_is_watertight"] = bool(mesh.is_watertight)
    if candidates is not None:
        debug_dict["candidate_count"] = len(candidates)
    if config is not None:
        debug_dict["objective_mode"] = config.slab_selection.objective_mode
        debug_dict["default_material"] = config.default_material
        debug_dict["dfm_check"] = config.slab_selection.dfm_check
        debug_dict["min_slab_area_mm2"] = config.slab_candidates.min_slab_area_mm2
        debug_dict["voxel_resolution_mm"] = config.slab_selection.voxel_resolution_mm
    if selection is not None:
        debug_dict["selection_stop_reason"] = selection.stop_reason
        debug_dict["selection_dfm_rejected_count"] = int(selection.dfm_rejected_count)
        debug_dict["candidate_count_after_dfm"] = int(selection.candidate_count_after_dfm)

    if debug_dict:
        design.metadata["decomposition_debug"] = debug_dict

    return ManufacturingDecompositionResult(
        design=design,
        parts={},
        dfm_violations=[],
        score=DecompositionScore(
            hausdorff_mm=float('inf'),
            mean_distance_mm=float('inf'),
            part_count=0,
            dfm_violations_error=0,
            dfm_violations_warning=0,
            structural_plausibility=0.0,
            overall_score=0.0,
        ),
        assembly_steps=[],
        candidates=[],
        segmentation=None,
    )
