# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-furniture generates flat-pack furniture designs from text descriptions or images. Cloud APIs (Tripo3D, Meshy) generate 3D meshes, which are decomposed into flat rectangular components suitable for CNC cutting. The pipeline outputs SVG cut files ready for SendCutSend manufacturing.

## Environment & Commands

```bash
# Python virtual environment (Python 3.12)
venv/bin/python3          # Always use this, not system python
venv/bin/pip install -r requirements.txt

# Generate furniture from text (requires TRIPO_API_KEY or MESHY_API_KEY env var)
venv/bin/python3 scripts/generate_furniture.py --text "a modern coffee table" --provider tripo

# Generate furniture from image
venv/bin/python3 scripts/generate_furniture.py --image photo.jpg --provider meshy

# Generate from existing mesh file (no API key needed)
venv/bin/python3 scripts/generate_furniture.py --mesh model.glb

# Decompose a mesh directly (lower-level)
venv/bin/python3 scripts/decompose_mesh.py --input model.stl

# Format code
venv/bin/python3 -m black src/

# Run tests
venv/bin/python3 -m pytest tests/

# Use MPLBACKEND=Agg for headless matplotlib rendering
MPLBACKEND=Agg venv/bin/python3 scripts/visualize_decomposition.py --input model.stl
```

## Architecture

### Data Flow

```
Text  → [Tripo3D / Meshy API] → 3D Mesh ┐
Image → [Tripo3D / Meshy API] → 3D Mesh ┤→ mesh_decomposer → FurnitureDesign → SVG export
Local mesh file ─────────────────────────┘                          ↓
                                                              Physics sim
                                                              (PyBullet)
```

### Core Modules (src/)

- **mesh_provider.py** — Abstract `MeshProvider` base class, `ProviderConfig`, `GenerationResult`, exception hierarchy (`ProviderError`, `ProviderTimeoutError`, `ProviderAPIError`, `ProviderRateLimitError`).
- **tripo_provider.py** — Tripo3D provider using official `tripo3d` SDK. Async wrapped in sync interface. API key: `TRIPO_API_KEY` env var.
- **meshy_provider.py** — Meshy provider using REST API via `requests`. Preview mode for text-to-3D. API key: `MESHY_API_KEY` env var.
- **pipeline.py** — Orchestrator: `run_pipeline(provider, prompt/image)` and `run_pipeline_from_mesh(path)`. Returns `PipelineResult` with design, simulation, SVG paths.
- **mesh_decomposer.py** — Converts 3D meshes to flat-pack: cluster face normals → generate candidate slabs → greedy voxel set-cover → infer joints. Entry: `decompose(filepath, config)`.
- **materials.py** — SendCutSend material catalog (`MATERIALS` dict, `Material` dataclass, `MIN_OVERLAP_MM=25`). Real thicknesses, densities, max sheet sizes.
- **furniture.py** — Foundation types: `Component`, `Joint`, `AssemblyGraph`, `FurnitureDesign`.
- **simulator.py** — PyBullet physics: stability testing, load testing, tip-over detection. Returns `SimulationResult`.
- **urdf_generator.py** — Converts `FurnitureDesign` → URDF for PyBullet.
- **svg_exporter.py** — CNC-ready SVG output with nested layout packing. Red=cut, blue=engrave.

### Import Convention

All imports use **bare module names** (e.g., `import furniture`, not `import src.furniture`). Scripts in `scripts/` add `src/` to `sys.path` before importing.

### Units & Conventions

- All dimensions in **millimeters**
- Positions as **numpy arrays**
- Component profiles are 2D (x,y) polygons; thickness is a separate field
- Rotations in **radians**
- `FurnitureDesign.validate()` checks component name uniqueness and joint reference integrity

### Deprecated Modules

Old genome-based pipeline modules are in `src/deprecated/` for reference (genome.py, evolution.py, fitness.py, model.py, etc.). These are no longer part of the active pipeline.
