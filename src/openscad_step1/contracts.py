"""Contracts for the clean-slate OpenSCAD Step 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Step1Config:
    """Configuration for mesh -> parametric OpenSCAD panelization."""

    mesh_path: str
    design_name: str = "design"
    material_key: str = "plywood_baltic_birch"
    preferred_thickness_mm: Optional[float] = None
    auto_scale: bool = True
    target_height_mm: float = 750.0
    part_budget_max: int = 18
    min_region_area_mm2: float = 450.0
    coplanar_angle_tol_deg: float = 2.5
    coplanar_offset_tol_mm: float = 1.0
    pair_angle_tol_deg: float = 12.0
    pair_min_area_ratio: float = 0.55
    pair_lateral_fraction: float = 0.20
    max_stack_layers: int = 6
    stack_roundup_bias: float = 0.35
    min_feature_mm: float = 3.175  # 1/8" baseline CNC feature
    selection_mode: str = "cavity_primary_nonoverlap_v1"
    cavity_axis_policy: str = "single_primary"
    shell_policy: str = "thin_one_panel_else_two_shells"
    overlap_enforcement: str = "hard_prevent"
    thin_gap_clearance_mm: float = 0.5
    base_selection_weight_coverage: float = 0.75
    base_selection_weight_area: float = 0.25
    stack_decay_schedule: Tuple[float, ...] = (1.0, 0.65, 0.45, 0.30, 0.20)
    stack_min_pass1_coverage_ratio: Optional[float] = None

    # Trim resolution
    trim_enabled: bool = True
    trim_parallel_dot_threshold: float = 0.95
    trim_max_loss_fraction: float = 0.40
    trim_min_intrusion_area_mm2: float = 1.0
    trim_structural_tiebreak: bool = True
    trim_vertical_bias: float = 0.02


@dataclass
class CandidateRegion:
    """Planar region candidate extracted from mesh facets."""

    candidate_id: str
    normal: Vec3
    center_3d: Vec3
    basis_u: Vec3
    basis_v: Vec3
    plane_offset: float
    outline_2d: List[Vec2]
    holes_2d: List[List[Vec2]]
    area_mm2: float
    source_faces: List[int] = field(default_factory=list)


@dataclass
class PanelFamily:
    """A selectable family of one or more panel layers."""

    family_id: str
    candidate_ids: List[str]
    representative_candidate_id: str
    face_ids: List[int]
    total_area_mm2: float
    layer_count: int
    selected_layer_count: int = 0
    shell_layers_required: int = 1
    interior_layers_capacity: int = 0
    selected_shell_layers: int = 0
    selected_interior_layers: int = 0
    estimated_gap_mm: float = 0.0
    is_opposite_pair: bool = False
    cavity_id: Optional[str] = None
    axis_role: str = "unpaired_shell"
    notes: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class PanelInstance:
    """Physical panel instance emitted to OpenSCAD."""

    panel_id: str
    family_id: str
    source_candidate_ids: List[str]
    thickness_mm: float
    origin_3d: Vec3
    basis_u: Vec3
    basis_v: Vec3
    basis_n: Vec3
    outline_2d: List[Vec2]
    holes_2d: List[List[Vec2]]
    area_mm2: float
    source_face_count: int
    metadata: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class TrimDecision:
    """Record of a single trim-direction choice between two panels."""

    trimmed_panel_id: str
    receiving_panel_id: str
    loss_trimmed_mm2: float
    loss_receiving_mm2: float
    dihedral_angle_deg: float
    direction_reason: str  # "effective_loss" | "area_tiebreak" | "vertical_bias" | "id_tiebreak"


@dataclass
class Step1Violation:
    """DFM/manufacturing issue observed during Step 1."""

    code: str
    severity: str
    message: str
    panel_id: Optional[str] = None
    value: Optional[float] = None
    limit: Optional[float] = None


@dataclass
class Step1RunResult:
    """In-memory result from Step 1 run."""

    run_id: str
    status: str
    mesh_hash_sha256: str
    material_key: str
    material_thickness_mm: float
    scale_factor: float
    mesh_bounds_mm: Vec3
    panel_families: List[PanelFamily]
    selected_families: List[PanelFamily]
    panels: List[PanelInstance]
    violations: List[Step1Violation]
    checkpoints: List[Path]
    openscad_code: str
    design_payload: Dict[str, object]
    spatial_capsule: Dict[str, object]
    decision_log_path: Path
    decision_hash_chain_path: Path
    trim_decisions: List[TrimDecision] = field(default_factory=list)
    debug: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseSummary:
    """Compact phase summary used in run-level manifests."""

    phase_index: int
    phase_name: str
    checkpoint_path: str


def to_vec3(values: Sequence[float]) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def to_vec2(values: Sequence[float]) -> Vec2:
    return (float(values[0]), float(values[1]))
