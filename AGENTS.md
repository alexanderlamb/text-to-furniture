# Repository Guidelines

## Project Structure

- `src/` — Core modules (pipeline, decomposition, export, materials, DFM rules)
- `app/` — Streamlit review UI and data helpers
- `scripts/` — CLI entry points (generate_furniture, run_mesh_suite, analyze_suite)
- `tests/` — pytest test suite with shared fixtures in `tests/conftest.py`
- `benchmarks/` — Suite definition, tracked test meshes, mesh generation script
- `runs/` — Runtime output (gitignored)

## Build & Dev Commands

- `python3 -m venv venv` creates the virtual environment
- `venv/bin/pip install -r requirements.txt` installs dependencies
- `venv/bin/python3 scripts/generate_furniture.py --mesh <path>` runs the pipeline on a mesh
- `venv/bin/python3 scripts/run_mesh_suite.py` runs the benchmark suite
- `venv/bin/python3 scripts/analyze_suite.py` analyzes latest suite results
- `venv/bin/streamlit run app/streamlit_app.py` launches the review UI
- `venv/bin/python3 -m pytest tests/` runs all tests
- `venv/bin/python3 -m black src/ tests/ scripts/` formats code

## Coding Style

Python with 4-space indentation and Black formatting. Modules/files in `snake_case`, classes in `PascalCase`, functions/variables in `snake_case`, constants in `UPPER_SNAKE_CASE`. Bare imports from `src/` modules (e.g., `from furniture import FurnitureDesign`); scripts add `src/` to `sys.path`. Geometry units in millimeters, numpy arrays for 3D positions.

## Testing

Use pytest. Name files `tests/test_<module>.py`, functions `test_<behavior>`. Reuse fixtures from `tests/conftest.py`. Run targeted tests during iteration, full suite before committing.

## Commits

Short, descriptive subjects. Single-purpose commits with clear scope.

## Security

Never commit secrets (`.env`). Keep API keys in environment variables. Generated run data goes to `runs/` which is gitignored.

## Debug Artifacts (For Coding Agents)

Use these outputs when debugging pipeline behavior:

1. `runs/<run_id>/metrics.json`
2. `runs/<run_id>/artifacts/debug_trace.json`
3. `runs/<run_id>/artifacts/snapshots/phase_*.json`
4. `runs/<run_id>/artifacts/design_first_principles.json`
5. `runs/suites/<suite_run_id>/results.csv`

Primary tracked diagnostics:

- Final overlap: `debug.plane_overlap_*`, `debug.plane_overlap_details`, `debug.plane_overlap_regions`
- Step 2 overlap: `debug.step2_plane_overlap_*`, `debug.step2_plane_overlap_details`, `debug.step2_plane_overlap_regions`
- Step 2 trim logic: `debug.step2_trim_debug`
- Intersection filter reasoning: `debug.intersection_events`, `debug.intersection_part_decisions`, `debug.intersection_reindex_map`
- Phase-level diagnostics: `debug_trace.json -> phase_diagnostics[*].diagnostics`

Suite CSV includes aggregate fields for quick triage:

- `plane_overlap_pairs`
- `plane_overlap_region_count`
- `plane_overlap_total_mm`
- `step2_plane_overlap_pairs`
- `step2_plane_overlap_region_count`
- `step2_trim_search_mode`
- `step2_trim_minor_pairs_count`
- `step2_trim_significant_pairs_count`

Detailed schema and examples live in `notes/debug_artifacts.md`.
