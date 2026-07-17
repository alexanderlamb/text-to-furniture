# Hybrid Fabrication Iteration Workflow

Use this when tuning the mesh-to-furniture hybrid compositor. The goal is to make each iteration evidence-driven: run the benchmark, inspect assignment previews, inspect burden metrics, then tune one hypothesis at a time.

## Current Reference Run

Latest healthy run:

```bash
venv/bin/python3 scripts/evaluate_hybrid_benchmarks.py \
  --mesh-dir benchmarks/meshes \
  --name hybrid-benchmark-eval-componentregions1 \
  --runs-dir runs \
  --no-auto-scale \
  --part-budget 24 \
  --max-regions 4
```

Artifacts:

- HTML report: `runs/20260515_044155_hybrid-benchmark-eval-componentregions1/artifacts/hybrid_evaluation/report.html`
- Evaluation JSON: `runs/20260515_044155_hybrid-benchmark-eval-componentregions1/artifacts/hybrid_evaluation/evaluation.json`
- Evaluation CSV: `runs/20260515_044155_hybrid-benchmark-eval-componentregions1/artifacts/hybrid_evaluation/evaluation.csv`

Comparison against previous healthy run `hybrid-benchmark-eval-observedbounds2`:

- Status: `ok`
- Meshes: `13/13`
- Mixed strategy meshes: `13/13`
- Boundary-jointed meshes: `13/13`
- Total selected parts: `153 -> 153`
- Waffle selected parts: `49 -> 49`
- Accidental duplicated source parts: `0`
- Shared source parts spanning regions: `4`

## What To Inspect

1. Open the HTML report first. The useful sections are `Strategy Mix`, `Strategy Burden`, `Flags`, and the per-mesh region preview thumbnails.
2. Use `Strategy Mix` to see region assignment counts. This answers: which strategy owns spatial regions?
3. Use `Strategy Burden` to see selected physical parts and material volume. This answers: which strategy actually dominates the BOM?
4. Use `Reuse/Shared` in the mesh table. The format is `accidental_reuse/shared_physical_parts`. Healthy runs should keep accidental reuse at `0`.
5. If a mesh has flags, open its `hybrid_plan.json` and inspect `debug.region_assignment_debug`, `debug.source_part_reuse`, and `debug.source_part_sharing`.
6. For waffle-heavy meshes, also inspect `source_strategies/waffle_ribs/waffle_ribs_debug.json` to check rib station allocation and spans.

## 3D Compare UI

Launch the Streamlit dashboard:

```bash
venv/bin/streamlit run app/streamlit_app.py --server.port 8501
```

Open `http://localhost:8501`, choose `Hybrid 3D Compare`, then pick baseline/candidate runs and a mesh. The viewer renders generated region or selected-part AABBs colored by source strategy. This is a geometric debugging view, not exact final cut geometry yet.

## Current Hypotheses

- The compositor is now correctly treating a source part that spans multiple regions as one shared physical part instead of cloning it into multiple BOM entries.
- Interior shell bands now need structural evidence before becoming `rib_band`. Region boxes are also tightened to observed geometry on transverse axes, so strategy selection is driven by the occupied local footprint instead of the full mesh AABB.
- Regioning now tries meaningful connected components first and falls back to observed-geometry axis bands for single-component meshes. The current benchmark is mostly single-component, so the component path is covered by regression tests rather than aggregate benchmark movement.

## Fast Commands

Focused tests after report/compositor changes:

```bash
venv/bin/python3 -m pytest \
  tests/test_fabrication_hybrid.py \
  tests/test_fabrication_hybrid_eval_cli.py \
  tests/test_fabrication_visual_report.py \
  -q
```

Full stabilization:

```bash
venv/bin/python3 -m black --check scripts src tests
venv/bin/python3 -m pytest tests/ -q
```

Quick run comparison:

```bash
venv/bin/python3 - <<'PY'
import json
from pathlib import Path

paths = {
    'baseline': Path('runs/20260511_083611_hybrid-benchmark-eval-regiongate3/artifacts/hybrid_evaluation/evaluation.json'),
    'latest': Path('runs/20260515_044155_hybrid-benchmark-eval-componentregions1/artifacts/hybrid_evaluation/evaluation.json'),
}
for label, path in paths.items():
    data = json.loads(path.read_text())
    print(label)
    print('  status:', data['status'])
    print('  flags:', data.get('flag_counts'))
    print('  total parts:', sum(row['parts'] for row in data['rows']))
    print('  region use:', data.get('strategy_region_use'))
    print('  part use:', data.get('strategy_part_use'))
    print('  shared:', sum(int(row.get('source_part_shared_count', 0) or 0) for row in data['rows']))
    print('  reuse:', sum(int(row.get('source_part_reuse_count', 0) or 0) for row in data['rows']))
PY
```
