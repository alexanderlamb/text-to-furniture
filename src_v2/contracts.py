"""Core V2 contracts for checkpoints, decisions, and spatial capsules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional

SCHEMA_CHECKPOINT_V1 = "v2.checkpoint.1"
SCHEMA_DECISION_V1 = "v2.decision.1"
SCHEMA_SPATIAL_CAPSULE_V1 = "v2.spatial_capsule.1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return sha256(raw).hexdigest()


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = _SLUG_RE.sub("-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "phase"


@dataclass
class CheckpointRecord:
    run_id: str
    phase_index: int
    phase_name: str
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_CHECKPOINT_V1
    input_hashes: Dict[str, str] = field(default_factory=dict)
    invariants: Dict[str, Any] = field(default_factory=lambda: {"units": "mm"})
    counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.run_id.strip():
            raise ValueError("CheckpointRecord.run_id is required")
        if self.phase_index < 0:
            raise ValueError("CheckpointRecord.phase_index must be >= 0")
        if not self.phase_name.strip():
            raise ValueError("CheckpointRecord.phase_name is required")
        if self.schema_version != SCHEMA_CHECKPOINT_V1:
            raise ValueError(
                f"Unexpected checkpoint schema version: {self.schema_version}"
            )
        units = str(self.invariants.get("units", "mm")).strip().lower()
        if units != "mm":
            raise ValueError(f"Checkpoint units must be 'mm', got: {units}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "phase_index": int(self.phase_index),
            "phase_name": self.phase_name,
            "timestamp_utc": self.timestamp_utc,
            "input_hashes": dict(self.input_hashes),
            "invariants": dict(self.invariants),
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
            "outputs": dict(self.outputs),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CheckpointRecord":
        rec = cls(
            run_id=str(payload.get("run_id", "")),
            phase_index=int(payload.get("phase_index", 0)),
            phase_name=str(payload.get("phase_name", "")),
            timestamp_utc=str(payload.get("timestamp_utc", utc_now_iso())),
            schema_version=str(payload.get("schema_version", SCHEMA_CHECKPOINT_V1)),
            input_hashes=dict(payload.get("input_hashes", {}) or {}),
            invariants=dict(payload.get("invariants", {}) or {"units": "mm"}),
            counts=dict(payload.get("counts", {}) or {}),
            metrics=dict(payload.get("metrics", {}) or {}),
            outputs=dict(payload.get("outputs", {}) or {}),
        )
        rec.validate()
        return rec


@dataclass
class DecisionAlternative:
    name: str
    cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "cost": float(self.cost)}


@dataclass
class DecisionRecord:
    phase_index: int
    decision_type: str
    entity_ids: List[str]
    selected: str
    seq: int = 0
    run_id: Optional[str] = None
    timestamp_utc: str = field(default_factory=utc_now_iso)
    schema_version: str = SCHEMA_DECISION_V1
    alternatives: List[DecisionAlternative] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    numeric_evidence: Dict[str, Any] = field(default_factory=dict)
    parent_checkpoint_sha256: Optional[str] = None

    def validate(self) -> None:
        if self.phase_index < 0:
            raise ValueError("DecisionRecord.phase_index must be >= 0")
        if self.seq < 0:
            raise ValueError("DecisionRecord.seq must be >= 0")
        if not self.decision_type.strip():
            raise ValueError("DecisionRecord.decision_type is required")
        if not self.selected.strip():
            raise ValueError("DecisionRecord.selected is required")
        if self.schema_version != SCHEMA_DECISION_V1:
            raise ValueError(
                f"Unexpected decision schema version: {self.schema_version}"
            )
        for entity_id in self.entity_ids:
            if not str(entity_id).strip():
                raise ValueError("DecisionRecord.entity_ids contains blank id")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "seq": int(self.seq),
            "phase_index": int(self.phase_index),
            "decision_type": self.decision_type,
            "entity_ids": list(self.entity_ids),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "selected": self.selected,
            "reason_codes": list(self.reason_codes),
            "numeric_evidence": dict(self.numeric_evidence),
            "timestamp_utc": self.timestamp_utc,
        }
        if self.run_id:
            payload["run_id"] = self.run_id
        if self.parent_checkpoint_sha256:
            payload["parent_checkpoint_sha256"] = self.parent_checkpoint_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecisionRecord":
        alternatives_raw = payload.get("alternatives", []) or []
        alternatives: List[DecisionAlternative] = []
        for item in alternatives_raw:
            if not isinstance(item, dict):
                continue
            alternatives.append(
                DecisionAlternative(
                    name=str(item.get("name", "")),
                    cost=float(item.get("cost", 0.0)),
                )
            )
        rec = cls(
            seq=int(payload.get("seq", 0)),
            run_id=(
                str(payload.get("run_id"))
                if payload.get("run_id") is not None
                else None
            ),
            phase_index=int(payload.get("phase_index", 0)),
            decision_type=str(payload.get("decision_type", "")),
            entity_ids=[str(v) for v in (payload.get("entity_ids", []) or [])],
            alternatives=alternatives,
            selected=str(payload.get("selected", "")),
            reason_codes=[str(v) for v in (payload.get("reason_codes", []) or [])],
            numeric_evidence=dict(payload.get("numeric_evidence", {}) or {}),
            parent_checkpoint_sha256=(
                str(payload.get("parent_checkpoint_sha256"))
                if payload.get("parent_checkpoint_sha256") is not None
                else None
            ),
            timestamp_utc=str(payload.get("timestamp_utc", utc_now_iso())),
            schema_version=str(payload.get("schema_version", SCHEMA_DECISION_V1)),
        )
        rec.validate()
        return rec


@dataclass
class SpatialPart:
    part_id: str
    thickness_mm: float
    origin_3d: List[float]
    basis_u: List[float]
    basis_v: List[float]
    outline_2d: List[List[float]]
    holes_2d: List[List[List[float]]] = field(default_factory=list)
    obb: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id": self.part_id,
            "thickness_mm": float(self.thickness_mm),
            "origin_3d": list(self.origin_3d),
            "basis_u": list(self.basis_u),
            "basis_v": list(self.basis_v),
            "outline_2d": [list(p) for p in self.outline_2d],
            "holes_2d": [[list(point) for point in ring] for ring in self.holes_2d],
            "obb": dict(self.obb),
        }


@dataclass
class SpatialRelation:
    part_a: str
    part_b: str
    relation_class: str
    penetration_mm: float = 0.0
    contact_area_mm2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_a": self.part_a,
            "part_b": self.part_b,
            "class": self.relation_class,
            "penetration_mm": float(self.penetration_mm),
            "contact_area_mm2": float(self.contact_area_mm2),
        }


@dataclass
class SpatialCapsule:
    parts: List[SpatialPart]
    relations: List[SpatialRelation]
    units: str = "mm"
    frame: Dict[str, str] = field(
        default_factory=lambda: {"handedness": "right", "up_axis": "Z"}
    )
    schema_version: str = SCHEMA_SPATIAL_CAPSULE_V1

    def validate(self) -> None:
        if self.schema_version != SCHEMA_SPATIAL_CAPSULE_V1:
            raise ValueError(
                f"Unexpected spatial capsule schema version: {self.schema_version}"
            )
        if self.units.lower() != "mm":
            raise ValueError("Spatial capsule units must be mm")
        part_ids = [p.part_id for p in self.parts]
        if len(part_ids) != len(set(part_ids)):
            raise ValueError("Spatial capsule part ids must be unique")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "units": self.units,
            "frame": dict(self.frame),
            "parts": [p.to_dict() for p in self.parts],
            "relations": [r.to_dict() for r in self.relations],
        }
