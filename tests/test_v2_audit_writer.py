from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_v2.audit.writer import AuditWriter, verify_decision_hash_chain
from src_v2.contracts import CheckpointRecord, DecisionAlternative, DecisionRecord


def test_audit_writer_writes_checkpoint_and_hash_index(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    writer = AuditWriter(run_dir=run_dir, run_id="run_001")

    checkpoint = CheckpointRecord(
        run_id="run_001",
        phase_index=0,
        phase_name="preflight",
        counts={"parts": 0},
        metrics={"coverage_ratio_unique_faces": 0.0},
    )
    checkpoint_path, checkpoint_sha = writer.write_checkpoint(checkpoint)

    assert checkpoint_path.exists()
    hashes_path = run_dir / "artifacts" / "checkpoint_hashes.json"
    assert hashes_path.exists()
    payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    assert payload["latest"]["sha256"] == checkpoint_sha
    assert payload["latest"]["phase_index"] == 0


def test_audit_writer_appends_decisions_and_verifies_chain(tmp_path: Path):
    run_dir = tmp_path / "run_002"
    writer = AuditWriter(run_dir=run_dir, run_id="run_002")

    checkpoint = CheckpointRecord(
        run_id="run_002",
        phase_index=1,
        phase_name="candidate_generation",
        counts={"parts": 3},
    )
    writer.write_checkpoint(checkpoint)

    d1 = DecisionRecord(
        phase_index=1,
        decision_type="candidate_filter",
        entity_ids=["cand_01"],
        alternatives=[
            DecisionAlternative(name="keep", cost=0.1),
            DecisionAlternative(name="drop", cost=0.8),
        ],
        selected="keep",
        reason_codes=["min_cost"],
        numeric_evidence={"area_mm2": 12000.0},
    )
    seq1, _ = writer.append_decision(d1)

    d2 = DecisionRecord(
        phase_index=1,
        decision_type="candidate_filter",
        entity_ids=["cand_02"],
        alternatives=[
            DecisionAlternative(name="keep", cost=0.6),
            DecisionAlternative(name="drop", cost=0.2),
        ],
        selected="drop",
        reason_codes=["budget_pressure"],
        numeric_evidence={"area_mm2": 450.0},
    )
    seq2, _ = writer.append_decision(d2)

    assert seq1 == 1
    assert seq2 == 2

    ok, msg = writer.verify_chain()
    assert ok, msg

    log_path = run_dir / "artifacts" / "decision_log.jsonl"
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2

    chain_path = run_dir / "artifacts" / "decision_hash_chain.json"
    ok2, msg2 = verify_decision_hash_chain(log_path, chain_path)
    assert ok2, msg2
