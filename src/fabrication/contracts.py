"""Shared contracts for competing fabrication strategies.

The tournament prototype intentionally keeps these contracts strategy-neutral:
flat panels, contour layers, voxel blocks, ribs, and future hybrids can all
describe their output as parts, joints, operations, and score components.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class FabricationConfig:
    """Configuration shared by all strategy candidates."""

    mesh_path: str
    design_name: str = "fabrication_tournament"
    material_key: str = "plywood_baltic_birch"
    preferred_thickness_mm: Optional[float] = None
    auto_scale: bool = True
    target_height_mm: float = 750.0
    strategies: Tuple[str, ...] = (
        "planar_skin",
        "contour_stack",
        "waffle_ribs",
        "voxel_blocks",
    )
    part_budget_max: int = 48
    min_feature_mm: float = 3.175
    voxel_pitch_multiplier: float = 4.0
    max_voxels_per_axis: int = 36
    max_hybrid_regions: int = 6


@dataclass
class Part:
    """A manufacturable or proto-manufacturable part emitted by a strategy."""

    part_id: str
    strategy_id: str
    kind: str
    quantity: int = 1
    material_thickness_mm: float = 0.0
    area_mm2: float = 0.0
    volume_mm3: float = 0.0
    aabb_min: Vec3 = (0.0, 0.0, 0.0)
    aabb_max: Vec3 = (0.0, 0.0, 0.0)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Joint:
    """A proposed relationship between parts."""

    joint_id: str
    strategy_id: str
    part_ids: List[str]
    kind: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Operation:
    """A manufacturing or assembly operation."""

    operation_id: str
    strategy_id: str
    kind: str
    part_ids: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class FabricationPlan:
    """Normalized output from any fabrication strategy."""

    strategy_id: str
    status: str
    parts: List[Part] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    operations: List[Operation] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    debug: Dict[str, object] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        return float(self.scores.get("overall", 0.0))

    def to_payload(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["part_count"] = len(self.parts)
        payload["joint_count"] = len(self.joints)
        payload["operation_count"] = len(self.operations)
        payload["overall_score"] = self.overall_score
        return payload


@dataclass
class HybridRegion:
    """A spatial region of the input mesh that can receive a strategy assignment."""

    region_id: str
    kind: str
    aabb_min: Vec3
    aabb_max: Vec3
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RegionStrategyAssignment:
    """The strategy outputs selected for one region of a hybrid plan."""

    assignment_id: str
    region_id: str
    strategy_id: str
    part_ids: List[str]
    fit_score: float
    reason_codes: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class HybridFabricationPlan:
    """Composed plan assembled from multiple strategy outputs."""

    status: str
    regions: List[HybridRegion] = field(default_factory=list)
    assignments: List[RegionStrategyAssignment] = field(default_factory=list)
    parts: List[Part] = field(default_factory=list)
    joints: List[Joint] = field(default_factory=list)
    operations: List[Operation] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source_strategy_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    debug: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["region_count"] = len(self.regions)
        payload["assignment_count"] = len(self.assignments)
        payload["part_count"] = len(self.parts)
        payload["joint_count"] = len(self.joints)
        payload["operation_count"] = len(self.operations)
        payload["overall_score"] = float(self.scores.get("overall", 0.0))
        return payload


class FabricationStrategy(Protocol):
    """Strategy plugin interface consumed by the tournament runner."""

    strategy_id: str

    def generate(self, context, artifacts_dir=None) -> FabricationPlan:
        """Generate a normalized fabrication plan for a mesh context."""
