---
applyTo: '**/*.ts'
---
# Project Overview
This is a Python 3.11+ project using standard library + common packages (fastapi, sqlalchemy, pydantic, etc.).
Follow PEP 8 and use type hints everywhere.

# Persistent TODO / Issue Tracking
- There is a root-level file called `TODO.md` that contains the authoritative list of bugs, features, and refactors.
- Before suggesting any code, always read the current contents of `TODO.md`.
- When you fix or implement something listed there:
  - Add a comment in the code: `# TODO: Resolves item X from TODO.md`
  - Suggest marking the item as done in `TODO.md` (change `[ ]` → `[x]`) and optionally add a short note.
- If you discover a new bug or improvement while working, suggest adding it to `TODO.md`.

# Python-specific rules
- Use `ruff` for linting/formatting (`ruff check .` and `ruff format .`)
- Tests are written with `pytest`. Always add or update tests for new/changed logic.
- Use `pyproject.toml` (managed by poetry or pdm — do not touch requirements.txt)
- Prefer pathlib over os.path
- Async code uses async/await (no callbacks)
- Logging via the standard `logging` module, never print() for production code

# Common commands (for context)
- Install: `poetry install` or `pdm sync`
- Run tests: `pytest`
- Lint/format: `ruff check . && ruff format .`
- Run app locally: `uvicorn main:app --reload` (or whatever the project uses)

---
applies_to: "**/*.py"
---
- Always reference `TODO.md` when touching Python files.
- Keep functions short (< 50 lines when possible).
- Use descriptive variable names and docstrings (Google or NumPy style).