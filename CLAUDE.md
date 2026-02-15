# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Text-to-furniture decomposes 3D meshes into flat-pack furniture designs for CNC cutting. The pipeline takes a mesh file, extracts planar and bendable regions, selects optimal parts within a budget, synthesizes joints, validates DFM rules, and outputs SVG/DXF cut files ready for SendCutSend manufacturing.

## Environment & Commands

```bash
# Python virtual environment (Python 3.12)
venv/bin/python3          # Always use this, not system python
venv/bin/pip install -r requirements.txt

# Run pipeline on a single mesh
venv/bin/python3 scripts/generate_furniture.py --mesh benchmarks/meshes/01_box.stl

# Run benchmark suite (10 cases)
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
    → extract planar/bend regions from mesh
    → score and select best candidates (within part budget)
    → synthesize joints between adjacent parts
    → validate DFM rules
  → FurnitureDesign
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
- **dfm_rules.py** — Design-for-manufacturing validation rules.
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
- **mesh_suite.json** — Suite definition (10 cases).
- **meshes/** — Tracked STL test meshes.
- **generate_meshes.py** — Script to regenerate benchmark meshes.

### Import Convention

All imports use **bare module names** (e.g., `import furniture`, not `import src.furniture`). Scripts and app files add `src/` to `sys.path` before importing.

### Units & Conventions

- All dimensions in **millimeters**
- Positions as **numpy arrays**
- Component profiles are 2D (x,y) polygons; thickness is a separate field
- Rotations in **radians**

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
    design_first_principles.json
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
