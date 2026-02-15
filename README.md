# Text-to-Furniture

Decomposes 3D meshes into flat-pack furniture designs suitable for CNC cutting (SendCutSend). Outputs SVG and DXF cut files.

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

`benchmarks/mesh_suite.json` defines 10 single-volume meshes of increasing complexity:

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
