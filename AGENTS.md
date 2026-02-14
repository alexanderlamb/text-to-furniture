# Repository Guidelines

## Project Structure & Module Organization
Core implementation lives in `src/` (pipeline, decomposition, CAD export, simulation). The web app is in `src/ui/`. CLI entry points are in `scripts/` (for example, `generate_furniture.py`, `decompose_mesh.py`, `run_ui.py`). Tests live in `tests/` with shared fixtures in `tests/conftest.py`. Design notes are in `docs/` (see `docs/manufacturing-pipeline.md`). Generated data and examples are under paths like `output/`, `examples/`, `training_data/`, and `uploads/`.

## Build, Test, and Development Commands
- `python3.12 -m venv venv` creates the local virtual environment.
- `venv/bin/pip install -r requirements.txt` installs runtime and dev dependencies.
- `cp .env.example .env` then set provider keys (`TRIPO_API_KEY`, `MESHY_API_KEY`).
- `venv/bin/python3 scripts/generate_furniture.py --mesh model.glb` runs the pipeline from an existing mesh.
- `venv/bin/python3 scripts/run_ui.py` starts the NiceGUI app on `http://localhost:8080`.
- `venv/bin/python3 -m pytest tests/` runs the full test suite.
- `venv/bin/python3 -m black src/ tests/ scripts/` formats Python code.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and Black formatting. Follow existing naming patterns: modules/files in `snake_case`, classes in `PascalCase`, functions/variables in `snake_case`, constants in `UPPER_SNAKE_CASE`. Project convention is bare imports from `src/` modules (for example, `from furniture import FurnitureDesign`); scripts should add `src/` to `sys.path`. Keep geometry units in millimeters and use numpy arrays for 3D positions.

## Testing Guidelines
Use `pytest`. Name files `tests/test_<module>.py` and test functions `test_<behavior>`. Reuse fixtures from `tests/conftest.py` where possible. Add or update tests for any behavior changes in decomposition, DFM checks, joints, exporters, or UI workers. Run targeted tests during iteration (for example, `venv/bin/python3 -m pytest tests/test_dfm_rules.py`) and run the full suite before opening a PR.

## Commit & Pull Request Guidelines
Git history uses short, descriptive subjects (for example, `Phase 1: Core infrastructure...`, `Initial project setup...`). Prefer concise, single-purpose commits with clear scope. PRs should include: what changed, why, test commands run, and related issue links. Include screenshots for `src/ui/` changes and sample output artifacts when export behavior changes.

## Security & Configuration Tips
Never commit secrets or local settings (`.env`, `.ui_settings.json`). Keep API keys in environment variables. Avoid committing generated binaries/meshes and large output files; use ignored output directories for local runs.
