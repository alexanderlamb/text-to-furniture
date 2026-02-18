# V2 Rewrite Spec: Deterministic, Auditable, No-Overlap Decomposition

Status: draft  
Scope: full backend rewrite in `src_v2/` with side-by-side A/B against current pipeline

## 1. Goals

1. Produce manufacturable flat-pack parts from input mesh with zero unintended slab overlap.
2. Make all geometry decisions deterministic and reproducible.
3. Make every phase auditable via machine-parseable checkpoints and decision logs.
4. Improve worst-case benchmark quality while preserving/raising mean quality.

## 2. Non-Goals (V2 initial)

1. No UI redesign. Reuse existing Streamlit app and run protocol.
2. No immediate replacement of all V1 internals; V2 ships in parallel first.
3. No dependency on interactive CAD tools during core pipeline execution.

## 3. Architecture (from scratch)

```text
mesh_input
  -> phase_00_preflight
  -> phase_01_plane_inference
  -> phase_02_candidate_generation
  -> phase_03_global_selection_solver
  -> phase_04_global_overlap_solver
  -> phase_05_joint_planning_solver
  -> phase_06_joint_geometry_apply
  -> phase_07_dfm_validation_autofix
  -> phase_08_export_nesting
  -> phase_09_scoring_and_reports
```

Modules:

1. `src_v2/geom/`
   - robust transforms, polygon ops, intersection helpers, tolerance policy
2. `src_v2/candidates/`
   - plane primitives, coplanar grouping, opposite-face pairing, stack hypotheses
3. `src_v2/selection/`
   - global part selection under budget using CP-SAT
4. `src_v2/conflicts/`
   - global overlap resolution (single solve, no pair-order dependence)
5. `src_v2/joints/`
   - contact graph, joint typing, fit/assembly checks
6. `src_v2/dfm/`
   - vendor/material rules + auto-fix passes
7. `src_v2/export/`
   - SVG/DXF and optional nesting payloads
8. `src_v2/audit/`
   - checkpoints, decision ledger, hash chain, diff reports

## 4. Checkpoints and Auditability Contract

Each run writes immutable phase checkpoints and append-only decision logs.

Required artifacts per run:

1. `runs/<run_id>/manifest.json`
2. `runs/<run_id>/metrics.json`
3. `runs/<run_id>/artifacts/checkpoints/phase_XX_<name>.json`
4. `runs/<run_id>/artifacts/decision_log.jsonl`
5. `runs/<run_id>/artifacts/decision_hash_chain.json`
6. `runs/<run_id>/artifacts/spatial_capsule_<phase>.json`

### 4.1 Checkpoint schema (machine-first)

```json
{
  "schema_version": "v2.checkpoint.1",
  "run_id": "2026..._name",
  "phase_index": 4,
  "phase_name": "global_overlap_solver",
  "timestamp_utc": "2026-02-18T00:00:00Z",
  "input_hashes": {
    "prev_checkpoint_sha256": "..."
  },
  "invariants": {
    "units": "mm",
    "connected_parts_only": true,
    "self_intersection_count": 0
  },
  "counts": {
    "parts": 9,
    "joints": 12,
    "overlap_pairs": 0
  },
  "metrics": {
    "coverage_ratio_unique_faces": 0.93,
    "trim_loss_total_ratio": 0.11
  },
  "outputs": {
    "spatial_capsule_path": "artifacts/spatial_capsule_phase_04.json"
  }
}
```

### 4.2 Decision log schema (append-only JSONL)

One decision per line:

```json
{
  "schema_version": "v2.decision.1",
  "seq": 1842,
  "phase_index": 4,
  "decision_type": "overlap_resolution",
  "entity_ids": ["part_03", "part_08"],
  "alternatives": [
    {"name": "trim_part_03", "cost": 0.081},
    {"name": "trim_part_08", "cost": 0.226},
    {"name": "drop_part_08", "cost": 0.517}
  ],
  "selected": "trim_part_03",
  "reason_codes": ["min_cost", "connectivity_preserved"],
  "numeric_evidence": {
    "overlap_mm": 4.21,
    "loss_ratio_selected": 0.081
  },
  "parent_checkpoint_sha256": "..."
}
```

### 4.3 Hash chain

Every decision record hash includes previous hash:

`H_n = SHA256(H_(n-1) || canonical_json(decision_n))`

This gives tamper-evident auditability.

## 5. LLM-Parseable Spatial Format (recommended)

There is no single universal “3D-to-text for LLMs” standard that captures both runtime meshes and manufacturing semantics well.

V2 should emit a canonical sidecar: `spatial_capsule.v1`.

### 5.1 Spatial capsule requirements

1. Units fixed to mm, right-handed axis convention explicit.
2. Every part includes:
   - stable id
   - transform (origin + basis)
   - local 2D outline and holes
   - thickness
   - world OBB
   - adjacency/contact list
3. Every pairwise relation includes:
   - min distance
   - penetration depth
   - relation class (`disjoint`, `touching`, `overlapping`, `jointed`)
4. Optional compact voxel occupancy summary for quick spatial reasoning.

### 5.2 Example

```json
{
  "schema_version": "v2.spatial_capsule.1",
  "units": "mm",
  "frame": {"handedness": "right", "up_axis": "Z"},
  "parts": [
    {
      "part_id": "part_03",
      "thickness_mm": 6.35,
      "origin_3d": [0.0, 120.0, 300.0],
      "basis_u": [1.0, 0.0, 0.0],
      "basis_v": [0.0, 1.0, 0.0],
      "outline_2d": [[0,0],[400,0],[400,120],[0,120]],
      "holes_2d": [],
      "obb": {"center":[200,180,303.175],"axes":[[1,0,0],[0,1,0],[0,0,1]],"half_extents":[200,60,3.175]}
    }
  ],
  "relations": [
    {
      "part_a": "part_03",
      "part_b": "part_08",
      "class": "overlapping",
      "penetration_mm": 2.1,
      "contact_area_mm2": 1870.0
    }
  ]
}
```

## 6. Standards Landscape for 3D-to-Text / 3D exchange

## 6.1 Practical answer

1. No single standard is sufficient for this product problem.
2. Use a hybrid:
   - runtime/preview: glTF/GLB
   - CAD/manufacturing interchange: STEP AP242 (where CAD/B-rep needed)
   - optional DCC/pipeline composition: OpenUSD
   - LLM/audit reasoning: custom JSON `spatial_capsule.v1`

## 6.2 Standards matrix (what to use)

1. `glTF 2.0`
   - Good: compact, JSON-based scene graph, broad runtime support.
   - Weak: not ideal as authoritative manufacturing semantic model.
2. `OpenUSD (.usda/.usdc/.usd)`
   - Good: composition, layering, textual `.usda` option, DCC ecosystem.
   - Weak: heavier than needed for simple audit payloads.
3. `STEP AP242 (ISO 10303)`
   - Good: authoritative CAD/manufacturing exchange semantics.
   - Weak: verbose, harder for direct LLM consumption.
4. `IFC`
   - Good for AEC/BIM semantics.
   - Not primary for furniture flat-pack CAM pipeline.
5. `3MF`
   - Good for additive manufacturing workflows.
   - Not primary for sheet-cut/joinery-first pipeline.

## 7. Phase-level invariants (hard gates)

Every phase must pass explicit checks before continuing.

1. `phase_00_preflight`
   - mesh readable, non-empty, units known/scaled
2. `phase_01_plane_inference`
   - candidate planes have valid normals and support area
3. `phase_02_candidate_generation`
   - polygons valid, connected, area > min
4. `phase_03_global_selection_solver`
   - budget respected, selected ids stable
5. `phase_04_global_overlap_solver`
   - no unintended overlap pairs
6. `phase_05_joint_planning_solver`
   - joint graph connectedness target met
7. `phase_06_joint_geometry_apply`
   - generated cutouts valid and non-self-intersecting
8. `phase_07_dfm_validation_autofix`
   - no error-severity DFM violations for release
9. `phase_08_export_nesting`
   - all parts exportable with stable IDs and dimensions

## 8. Rewrite delivery plan with checkpoints

1. Milestone M0: contracts and audit framework
   - deliver schemas, checkpoint writer, hash-chain logger
2. Milestone M1: `geom/` + invariant test harness
   - property tests for clipping/transform consistency
3. Milestone M2: candidate engine
   - parity with V1 candidate count on baseline meshes
4. Milestone M3: global selection solver
   - deterministic selection under budget
5. Milestone M4: global overlap solver
   - zero overlap on easy/medium suite subset
6. Milestone M5: joint planner/apply
   - fit/assembly checks pass on orthogonal cases
7. Milestone M6: DFM + export
   - no new DFM regressions, exporter parity
8. Milestone M7: full suite A/B gate
   - mean score >= strict V1, worst-case improved, deterministic replay pass

## 9. Dashboard observability additions (required)

1. Show per-case:
   - strict resolved flag
   - parts dropped count
   - phase checkpoint pass/fail badges
2. Show suite-level:
   - overlap pairs total trend
   - worst-case score trend
   - DFM error count trend
3. Add direct links to:
   - `decision_log.jsonl`
   - selected phase spatial capsules

## 10. Immediate next implementation tasks

1. Add `src_v2/contracts.py` for checkpoint/decision schemas.
2. Add `src_v2/audit/writer.py` (JSONL + hash chain).
3. Add `src_v2/spatial_capsule.py` emitter and validator.
4. Add `scripts/run_mesh_suite_v2.py` with strict A/B diff output.

## References (standards)

1. glTF registry/spec (Khronos): https://registry.khronos.org/glTF/
2. glTF overview: https://www.khronos.org/gltf
3. OpenUSD layer formats (`.usda/.usdc/.usd`): https://openusd.org/docs/Converting-Between-Layer-Formats.html
4. IFC schema specs (buildingSMART): https://technical.buildingsmart.org/standards/ifc/ifc-schema-specifications/
5. IFC 4.3 documentation: https://standards.buildingsmart.org/IFC/RELEASE/IFC4_3/
6. STEP at NIST (ISO 10303/AP242 context): https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/step-nist
7. STEP analyzer supporting AP242 (NIST): https://www.nist.gov/services-resources/software/step-file-analyzer-and-viewer
8. 3MF spec: https://3mf.io/spec/
