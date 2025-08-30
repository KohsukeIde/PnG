# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/` (core packages: `camera/`, `optimizer/`, `oracle_study/`, `rasterizer/`, `reconstructor/`, `utils/`, `evaluator/`, `primitive/`, `results/`). Keep reusable logic here; keep scripts thin.
- Experiments & scripts: `pipelines/` (e.g., `optimize_fundamental.py`, `png.py`). Prefer calling into `src` modules.
- Data & artifacts: `data/`, `models/`, `outputs/`, `gaussian_mixture_results/`. Avoid committing large binaries.
- Dev environment: `environments/` (Dockerfiles and compose for `cpu/`, `gpu/`, `ci/`).
- CI & templates: `.github/`.
- Tests: `tests/` (create if missing). Mirror `src/` structure.

## Build, Test, and Development Commands
- Install: `poetry install` (inside container or locally).
- Lint + type check: `make lint` (Ruff lint/format check, mdformat check, mypy).
- Auto-fix formatting: `make format` (Ruff format, Ruff fix, mdformat).
- Run tests: `make test` (pytest with coverage for `src/`).
- Full gate: `make test-all` (lint + tests).
- Container workflow: `cd environments/gpu && docker compose up -d && docker compose exec core bash` then run Make targets.

## Coding Style & Naming Conventions
- Language: Python 3.9. Formatter/linter: Ruff (line length 88). Docstrings: Google style. Type checking: mypy (strict settings in `pyproject.toml`).
- Indentation: 4 spaces. Imports sorted (Ruff `I`). Avoid unused code/vars.
- Naming: modules/files `snake_case.py`; functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE`.
- Layout: place modules under `src/<package>/<module>.py`; do not import across pipelines.

## Testing Guidelines
- Framework: pytest (+ `pytest-cov`).
- Structure: `tests/` mirrors `src/` (e.g., `tests/reconstructor/test_triangulate.py`).
- Naming: files `test_*.py`; functions `test_*`.
- Scope: add unit tests for algorithms and smoke tests for pipeline scripts using small fixtures in `data/` or synthetic arrays.
- Run: `make test` (shows coverage with missing lines).

## Commit & Pull Request Guidelines
- Commits: use a short type prefix and summary, matching history (e.g., `feat : add PnG initializer`, `fix : handle degenerate E-matrix`). Reference issues (`#123`) when applicable.
- Pull Requests: follow `.github/PULL_REQUEST_TEMPLATE.md`; include a clear overview, test steps, and relevant plots/screenshots (store large outputs under `outputs/`). Keep diffs focused and CI green.

## Security & Configuration Tips
- Do not commit secrets. Use `environments/envs.env` (gitignored) and `env_file` in compose if needed.
- When using Docker, set `HOST_UID`/`HOST_GID` to avoid permission issues. For GPU, use `environments/gpu/` and ensure NVIDIA drivers/toolkit are available.
