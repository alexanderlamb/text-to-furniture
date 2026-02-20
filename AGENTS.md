# Repository Guidelines

## Project Structure

- `src/openscad_step1/` — Core Step 1 pipeline, contracts, audit trail, OpenSCAD emitter
- `src/materials.py` — Material catalog and thickness options
- `src/run_protocol.py` — Run folder creation and artifact writing helpers
- `app/` — Streamlit dashboard for Step 1 runs
- `scripts/` — CLI entrypoint for Step 1 generation
- `tests/` — pytest suite for Step 1 CLI/pipeline/audit/dashboard smoke
- `runs/` — Runtime output (gitignored)

## Build & Dev Commands

- `python3 -m venv venv` creates the virtual environment
- `venv/bin/pip install -r requirements.txt` installs dependencies
- `venv/bin/python3 scripts/generate_openscad_step1.py --mesh <path>` runs Step 1 on a mesh
- `venv/bin/streamlit run app/streamlit_app.py` launches the Step 1 dashboard
- `venv/bin/python3 -m pytest tests/` runs all tests
- `venv/bin/python3 -m black src/ tests/ scripts/ app/` formats code

## Coding Style

Python with 4-space indentation and Black formatting. Modules/files in `snake_case`, classes in `PascalCase`, functions/variables in `snake_case`, constants in `UPPER_SNAKE_CASE`.

Use bare imports from `src/` modules after adding `src/` to `sys.path` in scripts/tests as needed. Geometry units are millimeters.

## Testing

Use pytest. Name files `tests/test_<module>.py`, functions `test_<behavior>`. Reuse fixtures from `tests/conftest.py`. Run targeted tests during iteration and full suite before committing.

## Commits

Short, descriptive subjects. Single-purpose commits with clear scope.

## Security

Never commit secrets (`.env`). Keep API keys in environment variables. Generated run data goes to `runs/` which is gitignored.

## Debug Artifacts (for coding agents)

Use these outputs first when debugging Step 1 behavior:

1. `runs/<run_id>/metrics.json`
2. `runs/<run_id>/artifacts/design_step1_openscad.json`
3. `runs/<run_id>/artifacts/spatial_capsule_step1.json`
4. `runs/<run_id>/artifacts/checkpoints/phase_*.json`
5. `runs/<run_id>/artifacts/decision_log.jsonl`
6. `runs/<run_id>/artifacts/decision_hash_chain.json`

Key diagnostics:

- Panel counts and selection: `metrics.counts.*`
- Candidate/family stats: `metrics.debug.*`
- Per-phase invariants and hashes: `checkpoints/phase_*.json`
- Decision sequence integrity: `decision_log.jsonl` + `decision_hash_chain.json`
