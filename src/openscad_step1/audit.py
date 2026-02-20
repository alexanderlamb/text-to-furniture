"""Audit/checkpoint utilities for OpenSCAD Step 1 runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(payload: Dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CheckpointHandle:
    phase_index: int
    phase_name: str
    path: Path
    payload_sha256: str


class AuditTrail:
    """Append-only audit writer with per-decision hash chaining."""

    def __init__(self, run_id: str, artifacts_dir: Path):
        self.run_id = run_id
        self.artifacts_dir = artifacts_dir
        self.checkpoints_dir = artifacts_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.decision_log_path = artifacts_dir / "decision_log.jsonl"
        self.hash_chain_path = artifacts_dir / "decision_hash_chain.json"
        self._sequence = 0
        self._prev_hash = "0" * 64
        self._chain: List[Dict[str, object]] = []
        self._checkpoint_handles: List[CheckpointHandle] = []

    @property
    def checkpoints(self) -> List[CheckpointHandle]:
        return list(self._checkpoint_handles)

    def append_decision(
        self,
        *,
        phase_index: int,
        decision_type: str,
        entity_ids: Iterable[str],
        alternatives: List[Dict[str, object]],
        selected: str,
        reason_codes: Iterable[str],
        numeric_evidence: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        self._sequence += 1
        payload: Dict[str, object] = {
            "schema_version": "openscad_step1.decision.v1",
            "run_id": self.run_id,
            "seq": self._sequence,
            "timestamp_utc": _utc_now_iso(),
            "phase_index": int(phase_index),
            "decision_type": decision_type,
            "entity_ids": list(entity_ids),
            "alternatives": alternatives,
            "selected": selected,
            "reason_codes": list(reason_codes),
            "numeric_evidence": numeric_evidence or {},
            "metadata": metadata or {},
            "previous_hash": self._prev_hash,
        }
        digest = sha256_text(_canonical_json(payload))
        payload["hash"] = digest

        with self.decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

        self._chain.append(
            {
                "seq": self._sequence,
                "hash": digest,
                "previous_hash": self._prev_hash,
            }
        )
        self._prev_hash = digest
        return payload

    def write_checkpoint(
        self,
        *,
        phase_index: int,
        phase_name: str,
        counts: Dict[str, int],
        metrics: Dict[str, float],
        invariants: Dict[str, object],
        outputs: Dict[str, object],
        input_hashes: Optional[Dict[str, str]] = None,
        notes: Optional[List[str]] = None,
    ) -> CheckpointHandle:
        phase_slug = phase_name.lower().replace(" ", "_")
        path = self.checkpoints_dir / f"phase_{phase_index:02d}_{phase_slug}.json"
        payload: Dict[str, object] = {
            "schema_version": "openscad_step1.checkpoint.v1",
            "run_id": self.run_id,
            "phase_index": int(phase_index),
            "phase_name": phase_name,
            "timestamp_utc": _utc_now_iso(),
            "input_hashes": input_hashes or {},
            "invariants": invariants,
            "counts": counts,
            "metrics": metrics,
            "outputs": outputs,
            "notes": notes or [],
        }
        canonical = _canonical_json(payload)
        payload_sha = sha256_text(canonical)
        payload["payload_sha256"] = payload_sha
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        handle = CheckpointHandle(
            phase_index=phase_index,
            phase_name=phase_name,
            path=path,
            payload_sha256=payload_sha,
        )
        self._checkpoint_handles.append(handle)
        return handle

    def finalize(self) -> None:
        payload = {
            "schema_version": "openscad_step1.hash_chain.v1",
            "run_id": self.run_id,
            "final_hash": self._prev_hash,
            "decision_count": self._sequence,
            "entries": self._chain,
            "checkpoint_hashes": [
                {
                    "phase_index": c.phase_index,
                    "phase_name": c.phase_name,
                    "path": str(c.path),
                    "payload_sha256": c.payload_sha256,
                }
                for c in self._checkpoint_handles
            ],
        }
        self.hash_chain_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
