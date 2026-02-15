# Text-to-Furniture (Streamlined)

This repository is a focused iteration harness for **one strategy only**:

- First-principles Step 3 decomposition
- Mesh input only
- DFM + geometry metrics by default
- JSON + SVG + DXF output every run
- Streamlit UI for rapid debug and feedback

## Quick Start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Run strategy from CLI:

```bash
venv/bin/python3 scripts/generate_furniture.py \
  --mesh uploads/generated_mesh.glb \
  --name my_run
```

Launch Streamlit app:

```bash
venv/bin/python3 scripts/run_app.py
```

Use the **Suite Lab** tab for iterative development:

- run the standard suite with current parameter settings
- compare candidate vs baseline suite runs
- inspect case-level diagnostics and failure modes with direct run drill-down

## Run Protocol

Every run writes a timestamped folder under `runs/`:

```text
runs/<run_id>/
  input/
    <mesh>
  artifacts/
    design_first_principles.json
    svg/
    dxf/
  manifest.json
  metrics.json
  summary.md
  logs.txt
```

`runs/latest` points to the newest run.

## CLI Flags

```text
--mesh                 required mesh path
--name                 design/run name
--runs-dir             run root (default: runs)
--material             material key
--step3-fidelity-weight
--step3-part-budget
--step3-bending / --step3-no-bending
--step3-no-planar-stacking
--step3-max-stack-layers
--step3-stack-roundup-bias
--step3-stack-extra-layer-gain
--step3-no-thin-side-suppression
--step3-thin-side-dim-multiplier
--step3-thin-side-aspect-limit
--step3-thin-side-coverage-start
--step3-thin-side-coverage-drop
--step3-no-intersection-filter
--step3-no-joint-intent-crossings
--step3-intersection-clearance-mm
--step3-joint-contact-tolerance-mm
--step3-joint-parallel-dot-threshold
--target-height-mm
--no-auto-scale
--no-svg
--no-dxf
-v
```

## Testing

```bash
venv/bin/python3 -m pytest tests/
```

Focused test suite covers:

- strategy core behavior
- single-path pipeline run protocol
- SVG/DXF export
- mesh-only CLI
- Streamlit smoke and feedback helpers

## Mesh Progress Suite

Run the standard mesh benchmark suite after each strategy change:

```bash
venv/bin/python3 scripts/run_mesh_suite.py
```

Outputs are written under `runs/suites/<suite_run_id>/` with:

- `summary.md` for quick scan
- `results.json` and `results.csv` for tracking deltas

`runs/suites/latest` points to the newest suite run.
