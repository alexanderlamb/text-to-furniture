# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This project decomposes 3D meshes into flat CNC-cuttable panels for assembly from sheet stock (e.g. SendCutSend Baltic Birch plywood). The pipeline extracts planar regions from a mesh, merges coplanar faces, selects panels within a budget, resolves perpendicular overlaps via slab-based trimming, and emits a parametric OpenSCAD assembly model.

**Current scope:** Mesh → panel decomposition + OpenSCAD output. Joint synthesis, DXF/SVG cut-file generation, and CNC toolpath packaging are not yet implemented.

## Environment & Commands

```bash
# Python virtual environment (Python 3.12)
venv/bin/python3          # Always use this, not system python
venv/bin/pip install -r requirements.txt

# Run pipeline on a single mesh
venv/bin/python3 scripts/generate_openscad_step1.py --mesh benchmarks/meshes/01_box.stl --name my-run

# Run all 13 benchmark meshes
for f in benchmarks/meshes/*.stl; do
  venv/bin/python3 scripts/generate_openscad_step1.py --mesh "$f" --name "batch-$(basename "$f" .stl)"
done

# Run tests
venv/bin/python3 -m pytest tests/ -v

# Format code
venv/bin/python3 -m black src/ tests/ scripts/ app/

# Launch review UI
venv/bin/streamlit run app/streamlit_app.py
```

## Architecture

### Data Flow

```
Mesh file (.stl/.obj/.glb/.ply)
  → Phase 0: Load, auto-scale to target height, center at origin
  → Phase 1: Extract planar candidate regions from mesh facets
  → Phase 2: Merge coplanar candidates into larger panels
  → Phase 3: Pair opposite-normal candidates into families, select under budget
  → Phase 4: Instantiate panel layers (shell + interior) with 3D positions
  → Phase 5: Trim perpendicular overlaps via slab-based half-plane cuts
  → Phase 6: Validate DFM rules, emit OpenSCAD + spatial capsule + audit trail
```

### Source Layout

```
src/
  openscad_step1/           # Core pipeline package
    __init__.py             # Exports Step1Config, Step1RunResult, run_step1_pipeline
    contracts.py            # Dataclass contracts: CandidateRegion, PanelFamily, PanelInstance, etc.
    pipeline.py             # Main algorithm (~2000 lines): extract → merge → select → instantiate → validate
    trim.py                 # Perpendicular overlap detection + slab-based trim resolution
    audit.py                # AuditTrail: phase checkpoints, decision logs, SHA256 hash chains
    scad_writer.py          # Renders PanelInstance list to parametric OpenSCAD code
  materials.py              # SendCutSend material catalog (Material dataclass, MATERIALS dict)
  run_protocol.py           # Run directory management: prepare_run_dir(), manifest/metrics writing

scripts/
  generate_openscad_step1.py  # CLI entry point: parses args, calls pipeline, writes all artifacts

app/
  streamlit_app.py          # Streamlit review UI for browsing runs
  data.py                   # Pure data helpers for the UI (no Streamlit imports)

tests/
  conftest.py                       # Shared fixtures (box_mesh_file, cylinder_mesh_file)
  test_openscad_step1_pipeline.py   # Core pipeline unit tests
  test_openscad_step1_trim.py       # Trim resolution logic tests
  test_openscad_step1_audit.py      # Audit trail and checkpoint tests
  test_openscad_step1_cli.py        # CLI integration test
  test_streamlit_step1_smoke.py     # Streamlit smoke test

benchmarks/
  meshes/                   # 13 tracked STL test meshes (01_box through 13_trapezoidal_tray)
  generate_meshes.py        # Script to regenerate benchmark meshes from code
```

### Import Convention

All imports use **bare module names** (e.g., `from materials import MATERIALS`, not `from src.materials`). Scripts and app files add `src/` to `sys.path` before importing.

### Units & Conventions

- All dimensions in **millimeters**
- Positions as **numpy arrays** / `Vec3` tuples
- Panel outlines are 2D (u, v) polygons in the panel's local frame
- `origin_3d` = anti-normal face; `origin_3d + basis_n * thickness` = outward surface
- A 2D point `(u, v)` in outline maps to world as: `origin_3d + u * basis_u + v * basis_v`
- Right-handed Z-up coordinate frame

## Development Workflow

```
1. Edit pipeline          → src/openscad_step1/pipeline.py (or trim.py)
2. Run tests              → venv/bin/python3 -m pytest tests/ -v
3. Generate test meshes   → venv/bin/python3 scripts/generate_openscad_step1.py --mesh <mesh>
4. Inspect outputs        → runs/<run_id>/metrics.json, design_step1_openscad.json
5. Iterate
```

## Run Output Structure

```
runs/<run_id>/
  input/<mesh>
  artifacts/
    design_step1_openscad.json    # Full design payload (families, panels, violations, trim decisions)
    model_step1.scad              # Parametric OpenSCAD assembly
    spatial_capsule_step1.json    # World-space AABBs/OBBs + pairwise relations
    checkpoints/phase_*.json      # Per-phase audit checkpoints with hashes
    decision_log.jsonl            # Ordered decision log (all selection/trim choices)
    decision_hash_chain.json      # SHA256 hash chain for decision integrity
  manifest.json
  metrics.json
  summary.md
```

## Debug Artifacts For Coding Agents

When diagnosing regressions, read in this order:

1. `runs/<run_id>/metrics.json` — status, violation counts, panel/family counts
2. `runs/<run_id>/artifacts/design_step1_openscad.json` — full panel geometry + selection debug
3. `runs/<run_id>/artifacts/spatial_capsule_step1.json` — pairwise spatial relations
4. `runs/<run_id>/artifacts/checkpoints/phase_*.json` — per-phase counts, metrics, invariants
5. `runs/<run_id>/artifacts/decision_log.jsonl` — ordered trace of every pipeline decision

Key fields in `design_step1_openscad.json`:

- `panels[*].origin_3d`, `basis_u/v/n`, `outline_2d` — full 3D placement
- `panels[*].metadata.panel_role` — "shell" or "interior"
- `panels[*].metadata.cavity_id`, `axis_role` — cavity assignment
- `families[*].estimated_gap_mm`, `selected_layer_count` — stacking decisions
- `trim_decisions[*]` — which panel was cut at each junction and why
- `violations[*]` — DFM issues (sheet size, minimum feature, overlaps)
