# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This project decomposes 3D meshes into a discrete set of flat parts manufacturable by SendCutSend's CNC cutting services. Parts are arbitrary 2D profiles — any shape that can be cut from sheet stock (not limited to rectangles). The pipeline extracts planar regions from a mesh, selects optimal parts within a budget, synthesizes joints, validates against CNC manufacturing constraints, and outputs SVG/DXF cut files.

**Key principle:** SendCutSend CNC machines can cut any arbitrary 2D shape from sheet material. The only constraints are DFM rules (minimum feature sizes, internal radii, sheet dimensions) — not geometric simplicity.

## Environment & Commands

```bash
# Python virtual environment (Python 3.12)
venv/bin/python3          # Always use this, not system python
venv/bin/pip install -r requirements.txt

# Run pipeline on a single mesh
venv/bin/python3 scripts/generate_furniture.py --mesh benchmarks/meshes/01_box.stl

# Run benchmark suite
venv/bin/python3 scripts/run_mesh_suite.py

# Analyze latest suite results
venv/bin/python3 scripts/analyze_suite.py
venv/bin/python3 scripts/analyze_suite.py --compare  # vs previous run

# Run tests
venv/bin/python3 -m pytest tests/

# Format code
venv/bin/python3 -m black src/

# Launch review UI
venv/bin/streamlit run app/streamlit_app.py
```

## Architecture

### Data Flow

```
Mesh file (.stl/.obj/.glb/.ply)
  → step3_first_principles.decompose_first_principles()
    → extract planar regions from mesh surface
    → generate candidate parts (arbitrary 2D profiles)
    → score and select best candidates (within part budget)
    → synthesize joints between adjacent parts
    → validate DFM rules (SendCutSend CNC constraints)
  → Design output
    → SVG export (red=cut, blue=engrave)
    → DXF export
    → metrics.json (quality scores, violations)
```

### Source Layout

**Core modules (`src/`):**
- **step3_first_principles.py** — Core decomposition algorithm. `Step3Input` (all tunable params), `decompose_first_principles()`, `Step3Output`.
- **pipeline.py** — Orchestrator: `run_pipeline_from_mesh()` → creates run directory, calls decomposition, exports artifacts, writes metrics. Returns `PipelineResult`.
- **run_protocol.py** — Run directory management: `prepare_run_dir()`, manifest/metrics writing.
- **furniture.py** — Foundation types: `Component`, `Joint`, `AssemblyGraph`, `FurnitureDesign`.
- **materials.py** — SendCutSend material catalog (`MATERIALS` dict, `Material` dataclass).
- **dfm_rules.py** — Design-for-manufacturing validation against SendCutSend CNC constraints.
- **geometry_primitives.py** — Geometric helpers for profiles, polygons, transforms.
- **svg_exporter.py** — CNC-ready SVG with nested layout packing.
- **dxf_exporter.py** — DXF export for CAM systems.
- **overlay_viewer.py** — 3D overlay scene builder for UI visualization.

**Scripts (`scripts/`):**
- **generate_furniture.py** — CLI for single-mesh pipeline runs.
- **run_mesh_suite.py** — Benchmark suite runner with incremental progress output.
- **analyze_suite.py** — Suite result analysis for structured review.

**UI (`app/`):**
- **streamlit_app.py** — Streamlit review UI (Suite Review + Run Detail tabs, suite launch with live progress).
- **data.py** — Pure data helpers for the UI (no Streamlit imports).

**Benchmarks (`benchmarks/`):**
- **mesh_suite.json** — Suite definition.
- **meshes/** — Tracked STL test meshes.
- **generate_meshes.py** — Script to regenerate benchmark meshes.

### Import Convention

All imports use **bare module names** (e.g., `import furniture`, not `import src.furniture`). Scripts and app files add `src/` to `sys.path` before importing.

### Units & Conventions

- All dimensions in **millimeters**
- Positions as **numpy arrays**
- Component profiles are 2D (x,y) polygons — arbitrary shapes, not just rectangles
- Thickness is a separate field per part
- Rotations in **radians** (Euler XYZ, intrinsic convention: Rz @ Ry @ Rx)

## Development Workflow

```
1. Edit algorithm       →  src/step3_first_principles.py
2. Run benchmark suite  →  venv/bin/python3 scripts/run_mesh_suite.py
3. Analyze results      →  venv/bin/python3 scripts/analyze_suite.py
4. Fix issues based on analysis
5. Repeat
```

Suite results live in `runs/suites/<suite_run_id>/`. The analyze tool outputs priority cases (worst first), failure mode breakdown, and violation codes — designed for direct consumption by a coding agent.

## Run Output Structure

```
runs/<timestamp>_<design_name>/
  input/<mesh>
  artifacts/
    debug_trace.json
    design_first_principles.json
    snapshots/phase_*.json
    svg/
    dxf/
  manifest.json
  metrics.json
  summary.md

runs/suites/<suite_run_id>/
  manifest.json
  results.json
  results.csv
  summary.md
```

## Debug Artifacts For Coding Agents

When diagnosing regressions, read in this order:

1. `runs/<run_id>/metrics.json`
2. `runs/<run_id>/artifacts/debug_trace.json`
3. `runs/<run_id>/artifacts/snapshots/phase_*.json`
4. `runs/<run_id>/artifacts/design_first_principles.json`
5. `runs/suites/<suite_run_id>/results.csv`

Key tracked fields:

- `debug.plane_overlap_details` and `debug.plane_overlap_regions` (final overlap pairs + region geometry)
- `debug.step2_plane_overlap_details` and `debug.step2_plane_overlap_regions` (Step 2 overlap)
- `debug.step2_trim_debug` (trim pair classification, mask/search decisions, exhaustive/fallback mode)
- `debug.intersection_events` and `debug.intersection_part_decisions` (intersection filter reasoning)
- `phase_diagnostics[*].diagnostics.part_geometry` (part-level area/bbox stats at each phase)

Suite CSV contains aggregate debug counters including:

- `plane_overlap_region_count`
- `step2_plane_overlap_region_count`
- `step2_trim_search_mode`
- `step2_trim_minor_pairs_count`
- `step2_trim_significant_pairs_count`

Field-level schema notes: `notes/debug_artifacts.md`.
