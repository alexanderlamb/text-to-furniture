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
