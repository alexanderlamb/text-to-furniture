# Mesh-to-OpenSCAD Step 1

Single-strategy repository for **Step 1** of the rebuild:

`mesh -> planar panel families -> stacked panel instances -> parametric OpenSCAD + audit artifacts`

This repository no longer runs the legacy Step3 decomposition/suite/export pipeline.

## Scope

In scope:
1. Load and normalize mesh geometry.
2. Extract and merge planar regions.
3. Collapse opposite faces into stackable panel families.
4. Select families under a panel budget.
5. Emit parametric OpenSCAD assembly.
6. Emit machine-parseable audit/checkpoint artifacts.

Out of scope (for now):
1. Cut-file generation (DXF/SVG nesting).
2. Joint synthesis.
3. Step 2 CNC toolpath/cut package.

## Quick Start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Run Step 1 on a mesh:

```bash
venv/bin/python3 scripts/generate_openscad_step1.py \
  --mesh benchmarks/meshes/01_box.stl \
  --name step1_box
```

Launch dashboard:

```bash
venv/bin/streamlit run app/streamlit_app.py
```

## Run Artifacts

Each run writes:

```text
runs/<run_id>/
  input/<mesh>
  artifacts/
    design_step1_openscad.json
    model_step1.scad
    spatial_capsule_step1.json
    checkpoints/phase_*.json
    decision_log.jsonl
    decision_hash_chain.json
  manifest.json
  metrics.json
  summary.md
```

## Contracts

Primary contract files:
1. `src/openscad_step1/contracts.py`
2. `src/openscad_step1/pipeline.py`
3. `src/openscad_step1/audit.py`
4. `src/openscad_step1/scad_writer.py`

Strategy identifier in manifests/metrics:
- `openscad_step1_clean_slate`

## Testing

```bash
venv/bin/python3 -m pytest tests/
```

Core tests:
1. `tests/test_openscad_step1_pipeline.py` — core pipeline unit tests
2. `tests/test_openscad_step1_trim.py` — trim resolution logic (24 tests)
3. `tests/test_openscad_step1_audit.py` — audit trail and checkpoints
4. `tests/test_openscad_step1_cli.py` — CLI integration test
5. `tests/test_streamlit_step1_smoke.py` — dashboard smoke test
