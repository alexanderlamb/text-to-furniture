"""Writers and verifiers for V2 checkpoint and decision audit artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import (
    CheckpointRecord,
    DecisionRecord,
    canonical_json_dumps,
    sha256_hex,
    slugify,
    utc_now_iso,
)

SCHEMA_DECISION_HASH_CHAIN_V1 = "v2.decision_hash_chain.1"
SCHEMA_CHECKPOINT_HASHES_V1 = "v2.checkpoint_hashes.1"
_GENESIS_HASH = "0" * 64


@dataclass
class AuditPaths:
    run_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    decision_log_path: Path
    decision_hash_chain_path: Path
    checkpoint_hashes_path: Path


def build_audit_paths(run_dir: str | Path) -> AuditPaths:
    base = Path(run_dir)
    artifacts = base / "artifacts"
    checkpoints = artifacts / "checkpoints"
    return AuditPaths(
        run_dir=base,
        artifacts_dir=artifacts,
        checkpoints_dir=checkpoints,
        decision_log_path=artifacts / "decision_log.jsonl",
        decision_hash_chain_path=artifacts / "decision_hash_chain.json",
        checkpoint_hashes_path=artifacts / "checkpoint_hashes.json",
    )


class AuditWriter:
    def __init__(self, run_dir: str | Path, run_id: str) -> None:
        self.run_id = run_id
        self.paths = build_audit_paths(run_dir)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._last_checkpoint_sha256: Optional[str] = None
        self._last_chain_hash = _GENESIS_HASH
        self._next_seq = 1
        self._chain_entries: List[Dict[str, Any]] = []
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        if self.paths.checkpoint_hashes_path.exists():
            payload = json.loads(self.paths.checkpoint_hashes_path.read_text("utf-8"))
            latest = payload.get("latest", {}) if isinstance(payload, dict) else {}
            latest_sha = latest.get("sha256")
            if isinstance(latest_sha, str) and len(latest_sha) == 64:
                self._last_checkpoint_sha256 = latest_sha

        if self.paths.decision_hash_chain_path.exists():
            payload = json.loads(self.paths.decision_hash_chain_path.read_text("utf-8"))
            if not isinstance(payload, dict):
                return
            entries = payload.get("entries", [])
            if not isinstance(entries, list):
                return
            cleaned_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                cleaned_entries.append(dict(entry))
            self._chain_entries = cleaned_entries
            if cleaned_entries:
                head = cleaned_entries[-1]
                self._last_chain_hash = str(head.get("chain_hash", _GENESIS_HASH))
                self._next_seq = int(head.get("seq", 0)) + 1

    def write_checkpoint(self, record: CheckpointRecord) -> Tuple[Path, str]:
        if not record.run_id:
            record.run_id = self.run_id
        record.validate()

        if (
            "prev_checkpoint_sha256" not in record.input_hashes
            and self._last_checkpoint_sha256
        ):
            record.input_hashes["prev_checkpoint_sha256"] = self._last_checkpoint_sha256

        payload = record.to_dict()
        canonical = canonical_json_dumps(payload)
        checkpoint_sha = sha256_hex(canonical)

        phase_slug = slugify(record.phase_name)
        checkpoint_path = (
            self.paths.checkpoints_dir
            / f"phase_{record.phase_index:02d}_{phase_slug}.json"
        )
        _write_json(checkpoint_path, payload)

        self._record_checkpoint_hash(
            checkpoint_path, record.phase_index, checkpoint_sha
        )
        self._last_checkpoint_sha256 = checkpoint_sha
        return checkpoint_path, checkpoint_sha

    def append_decision(self, record: DecisionRecord) -> Tuple[int, str]:
        if not record.run_id:
            record.run_id = self.run_id
        if record.seq <= 0:
            record.seq = self._next_seq
        if record.seq != self._next_seq:
            raise ValueError(
                f"Decision sequence mismatch: expected {self._next_seq}, got {record.seq}"
            )
        if (
            record.parent_checkpoint_sha256 is None
            and self._last_checkpoint_sha256 is not None
        ):
            record.parent_checkpoint_sha256 = self._last_checkpoint_sha256
        record.validate()

        payload = record.to_dict()
        canonical = canonical_json_dumps(payload)
        decision_sha = sha256_hex(canonical)
        chain_hash = sha256_hex(f"{self._last_chain_hash}{decision_sha}")

        self.paths.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.decision_log_path.open("a", encoding="utf-8") as f:
            f.write(canonical + "\n")

        entry = {
            "seq": int(record.seq),
            "timestamp_utc": record.timestamp_utc or utc_now_iso(),
            "decision_sha256": decision_sha,
            "prev_chain_hash": self._last_chain_hash,
            "chain_hash": chain_hash,
        }
        self._chain_entries.append(entry)
        self._last_chain_hash = chain_hash
        self._next_seq += 1
        self._write_chain_file()
        return record.seq, chain_hash

    def verify_chain(self) -> Tuple[bool, str]:
        return verify_decision_hash_chain(
            self.paths.decision_log_path,
            self.paths.decision_hash_chain_path,
        )

    def _record_checkpoint_hash(
        self,
        checkpoint_path: Path,
        phase_index: int,
        checkpoint_sha: str,
    ) -> None:
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_CHECKPOINT_HASHES_V1,
            "run_id": self.run_id,
            "latest": {},
            "entries": [],
        }
        if self.paths.checkpoint_hashes_path.exists():
            existing = json.loads(self.paths.checkpoint_hashes_path.read_text("utf-8"))
            if isinstance(existing, dict):
                payload = existing
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        entries.append(
            {
                "phase_index": int(phase_index),
                "checkpoint_path": str(checkpoint_path.relative_to(self.paths.run_dir)),
                "sha256": checkpoint_sha,
                "timestamp_utc": utc_now_iso(),
            }
        )
        payload["entries"] = entries
        payload["latest"] = entries[-1]
        payload["schema_version"] = SCHEMA_CHECKPOINT_HASHES_V1
        payload["run_id"] = self.run_id
        _write_json(self.paths.checkpoint_hashes_path, payload)

    def _write_chain_file(self) -> None:
        payload = {
            "schema_version": SCHEMA_DECISION_HASH_CHAIN_V1,
            "run_id": self.run_id,
            "genesis_hash": _GENESIS_HASH,
            "head": self._chain_entries[-1] if self._chain_entries else None,
            "entries": self._chain_entries,
        }
        _write_json(self.paths.decision_hash_chain_path, payload)


def verify_decision_hash_chain(
    decision_log_path: str | Path,
    decision_hash_chain_path: str | Path,
) -> Tuple[bool, str]:
    log_path = Path(decision_log_path)
    chain_path = Path(decision_hash_chain_path)
    if not log_path.exists():
        return False, f"Missing decision log: {log_path}"
    if not chain_path.exists():
        return False, f"Missing decision hash chain: {chain_path}"

    lines = [line for line in log_path.read_text("utf-8").splitlines() if line.strip()]
    chain_payload = json.loads(chain_path.read_text("utf-8"))
    entries = (
        chain_payload.get("entries", []) if isinstance(chain_payload, dict) else []
    )
    if not isinstance(entries, list):
        return False, "Invalid chain payload: entries is not a list"
    if len(lines) != len(entries):
        return (
            False,
            f"Mismatch: {len(lines)} decision lines vs {len(entries)} chain entries",
        )

    expected_prev = _GENESIS_HASH
    for idx, (line, entry) in enumerate(zip(lines, entries), start=1):
        if not isinstance(entry, dict):
            return False, f"Chain entry {idx} is not an object"
        line_hash = sha256_hex(line)
        expected_chain_hash = sha256_hex(f"{expected_prev}{line_hash}")
        if str(entry.get("decision_sha256")) != line_hash:
            return False, f"Decision hash mismatch at seq {idx}"
        if str(entry.get("prev_chain_hash")) != expected_prev:
            return False, f"Prev chain hash mismatch at seq {idx}"
        if str(entry.get("chain_hash")) != expected_chain_hash:
            return False, f"Chain hash mismatch at seq {idx}"
        expected_prev = expected_chain_hash

    return True, "ok"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
