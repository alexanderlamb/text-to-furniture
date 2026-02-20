from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openscad_step1 import Step1Config, run_step1_pipeline
from openscad_step1.audit import AuditTrail


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_step1_audit_log_and_hash_chain(box_mesh_file: str, tmp_path: Path):
    run_id = "audit_case"
    artifacts = tmp_path / run_id / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    config = Step1Config(
        mesh_path=box_mesh_file, design_name="audit_case", auto_scale=False
    )
    audit = AuditTrail(run_id=run_id, artifacts_dir=artifacts)
    result = run_step1_pipeline(
        config=config,
        run_id=run_id,
        artifacts_dir=artifacts,
        audit=audit,
    )

    decision_log = result.decision_log_path
    chain_path = result.decision_hash_chain_path
    assert decision_log.exists()
    assert chain_path.exists()

    records = [
        json.loads(line)
        for line in decision_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records

    expected_seq = list(range(1, len(records) + 1))
    assert [record["seq"] for record in records] == expected_seq

    prev_hash = "0" * 64
    for record in records:
        assert record["previous_hash"] == prev_hash
        payload = {k: v for k, v in record.items() if k != "hash"}
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        assert digest == record["hash"]
        prev_hash = digest

    chain = json.loads(chain_path.read_text(encoding="utf-8"))
    assert chain["decision_count"] == len(records)
    assert chain["final_hash"] == prev_hash


def test_step1_checkpoints_are_ordered(box_mesh_file: str, tmp_path: Path):
    run_id = "checkpoint_case"
    artifacts = tmp_path / run_id / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    config = Step1Config(
        mesh_path=box_mesh_file, design_name="checkpoint_case", auto_scale=False
    )
    audit = AuditTrail(run_id=run_id, artifacts_dir=artifacts)
    result = run_step1_pipeline(
        config=config,
        run_id=run_id,
        artifacts_dir=artifacts,
        audit=audit,
    )

    checkpoint_files = sorted(result.checkpoints)
    assert len(checkpoint_files) >= 6

    indices = []
    for checkpoint_path in checkpoint_files:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        indices.append(int(payload["phase_index"]))
        assert payload["schema_version"] == "openscad_step1.checkpoint.v1"

    assert indices == sorted(indices)
    assert indices[0] == 0
