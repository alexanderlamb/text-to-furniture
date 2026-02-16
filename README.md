# Mesh-to-Parts

Decomposes 3D meshes into a discrete set of flat parts that can be CNC-cut from sheet stock by [SendCutSend](https://sendcutsend.com/). Parts can be any arbitrary 2D shape — not limited to rectangles. The assembled parts approximate the original 3D form.

## Goals

**End-to-end goal:** Take any 3D mesh and produce a set of CNC-ready cut files, where each file defines an arbitrary 2D profile to be cut from sheet material. The cut parts assemble into a physical object that approximates the original 3D shape.

### Sequential steps

1. **Accept a 3D shape** -- Load a mesh file (.stl, .obj, .glb, .ply) representing the target object.

2. **Normalize geometry** -- Scale the mesh to a target physical size and center it for consistent processing.

3. **Decompose into flat parts** -- Extract planar regions from the mesh surface and generate candidate parts. Each part is an arbitrary 2D profile (triangles, L-shapes, curves — whatever the geometry requires) with a material thickness. Select the best set of parts that covers the mesh surface within a part budget.

4. **Synthesize joints** -- Determine where parts meet in 3D space and define connection geometry at those intersections.

5. **Validate manufacturability** -- Check each part against SendCutSend's CNC constraints: minimum feature sizes, internal radii, material thicknesses, and maximum sheet dimensions.

6. **Export cut files** -- Output SVG and DXF files with the 2D profiles for each part, plus nested sheet layouts. Red lines = cut, blue lines = engrave.

### The core hard problem

Step 3 is where the complexity lives: approximating an arbitrary 3D surface as a minimal set of flat parts with arbitrary 2D outlines, cut from real sheet materials, that assemble back into the original shape.

## Quick Start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Run on a single mesh:

```bash
venv/bin/python3 scripts/generate_furniture.py --mesh benchmarks/meshes/01_box.stl
```

Output goes to `runs/<timestamp>_<name>/` with design JSON, SVG cut files, DXF files, and metrics.

## Development Workflow

```
1. Edit algorithm       →  src/step3_first_principles.py (core decomposition)
2. Run benchmark suite  →  venv/bin/python3 scripts/run_mesh_suite.py
3. Analyze results      →  venv/bin/python3 scripts/analyze_suite.py
4. Review in UI         →  venv/bin/streamlit run app/streamlit_app.py
5. Repeat
```

The analyze tool prints a structured summary (priority cases, failure modes, violation codes) that can be handed directly to a coding agent.

Compare against the previous suite run:

```bash
venv/bin/python3 scripts/analyze_suite.py --compare
```

## Benchmark Suite

`benchmarks/mesh_suite.json` defines test meshes of increasing complexity:

| Case | Shape | Faces |
|------|-------|-------|
| 01_box | Simple rectangular box | 12 |
| 02_tall_cabinet | High aspect ratio box | 12 |
| 03_l_bracket | L-shaped bracket | 24 |
| 04_t_beam | T cross section | 28 |
| 05_u_channel | U-shaped channel | 28 |
| 06_step_stool | Two-step staircase | 32 |
| 07_h_beam | H cross section | 60 |
| 08_shelf_unit | Open shelf (sides + shelves) | 84 |
| 09_desk | Top + sides + back | 48 |
| 10_table_with_stretchers | Table with legs + stretchers | 116 |
| 11_v_bracket | V-shaped bracket | — |
| 12_angled_flange | Angled flange | — |
| 13_trapezoidal_tray | Trapezoidal tray | — |

Regenerate meshes: `venv/bin/python3 benchmarks/generate_meshes.py`

## Run Output

Each pipeline run writes:

```
runs/<run_id>/
  input/<mesh>
  artifacts/
    design_first_principles.json
    svg/
    dxf/
  manifest.json
  metrics.json
  summary.md
```

Suite runs aggregate under `runs/suites/<suite_run_id>/` with `results.json`, `results.csv`, and `summary.md`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_furniture.py` | Run pipeline on a single mesh |
| `scripts/run_mesh_suite.py` | Run full benchmark suite |
| `scripts/analyze_suite.py` | Analyze latest suite results |

## Testing

```bash
venv/bin/python3 -m pytest tests/
```

## UI

```bash
venv/bin/streamlit run app/streamlit_app.py
```

Two tabs: **Suite Review** (launch suites, compare runs, drill into cases) and **Run Detail** (inspect individual runs).
