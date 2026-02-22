# AGENTS.md — Developer & Agent Reference

This file documents conventions, tooling commands, and code style guidelines for the
**SegData** repository. It is intended for human contributors and agentic coding tools.

---

## Repository Overview

SegData is a pure-Python (3.9–3.12) toolkit for preparing, validating, and analysing
medical/scientific image segmentation datasets in **nnU-Net format**. Key components:

| Path | Purpose |
|---|---|
| `src/` | Shared utility scripts: `plot.py`, `stats.py`, `verify.py` |
| `raw/Dataset###_Name/` | Per-dataset directory: `setup.py`, `dataset.json`, `get.md` |
| `tests/` | Pytest test suite |
| `pyproject.toml` | Project metadata, dependencies, ruff/formatter config |
| `uv.lock` | Pinned dependency lock (managed by `uv`) |

---

## Environment Setup

This project uses **`uv`** (Astral) as the package and environment manager.
Do **not** use `pip`, `poetry`, or `pdm` directly.

```bash
# Install all dependencies including dev extras
uv sync --all-extras

# Activate the managed venv (optional — prefix commands with `uv run` instead)
source .venv/bin/activate
```

---

## Build / Lint / Format / Type-Check Commands

```bash
# Lint (ruff — rules E, W, F, I, C, B)
uv run ruff check src/ tests/ raw/

# Lint with auto-fix
uv run ruff check --fix src/ tests/ raw/

# Format (ruff-format)
uv run ruff format src/ tests/ raw/

# Type check (ty)
uv run ty check --ignore unresolved-import

# Run all pre-commit hooks at once (lint + format + type-check)
pre-commit run --all-files
```

---

## Testing

The test framework is **pytest**.

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_setup_scripts.py -v

# Run a single test class
uv run pytest tests/test_src_scripts.py::TestVerifyScript -v

# Run a single test method  ← most common for focused work
uv run pytest tests/test_src_scripts.py::TestVerifyScript::test_validator_initialization -v

# Run tests for one specific dataset (parametrised)
uv run pytest "tests/test_datasets.py::TestDatasets::test_dataset_json_valid[Dataset006_BTXRD]" -v

# Filter by keyword across all test files
uv run pytest tests/ -k "Dataset005" -v
```

> When adding or modifying a dataset, run `uv run pytest tests/test_datasets.py -v` to verify
> structural correctness before committing.

---

## Code Style Guidelines

### Formatting

- **Line length**: 100 characters (enforced by ruff-format; `E501` is ignored so ruff check
  will not re-flag overruns that the formatter permits).
- **Quotes**: double quotes (`"`) everywhere — enforced by ruff-format.
- **Indentation**: 4 spaces, no tabs.
- **Trailing commas**: use them in multi-line collections/function signatures.
- Do **not** manually run `black` or `autopep8`; `ruff format` is the sole formatter.

### Imports

Follow `isort` ordering as enforced by ruff rule `I`:

```python
# 1. Standard library (alphabetical)
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 2. Third-party (alphabetical)
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# 3. Local / first-party (alphabetical, after a blank line)
from verify import DatasetValidator
```

- No star imports (`from module import *`).
- In `tests/`, use `sys.path.insert(0, str(Path(__file__).parent.parent / "src"))` before
  importing local modules (the `src/` scripts are not installed as a proper package).

### Type Annotations

- Annotate all public function signatures (parameters + return type).
- Use `from typing import Dict, List, Optional, Tuple` for Python 3.9 compatibility.
  Prefer `Dict[str, int]` over `dict[str, int]` in files that target 3.9; modern
  `list[tuple[Path, Path]]` syntax is acceptable in `raw/*/setup.py` scripts.
- Do **not** use bare `Any` unless unavoidable; prefer specific types or `object`.
- No `TypeVar` or `Protocol` unless genuinely needed.

### Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Functions / methods | `snake_case` | `load_dataset_meta`, `binarize_mask` |
| Private helpers | `_snake_case` | `_find_mask_for_image` |
| Classes | `PascalCase` | `DatasetValidator`, `DatasetAnalyzer` |
| Constants | `SCREAMING_SNAKE_CASE` | `GREEN`, `REQUIRED_FILES` |
| Test classes | `TestXxxScript` / `TestDataset###Name` | `TestVerifyScript` |
| Case IDs | zero-padded integer strings | `f"{idx:03d}"` |
| Channel suffixes | 4-digit zero-padded | `f"{ch:04d}"` |

### Module Structure

Every `src/` script and `raw/*/setup.py` must follow this template:

```python
#!/usr/bin/env python3
"""One-line summary.

Extended description and CLI usage if applicable.
"""

# --- imports (stdlib, third-party, local) ---

# --- constants ---

# --- helper functions / classes ---

def main() -> None:
    """Main entry point."""
    ...

if __name__ == "__main__":
    main()
```

### Path Handling

- Use `pathlib.Path` exclusively — never `os.path`.
- Resolve user-provided paths: `Path(arg).resolve()`.
- Derive paths relative to the script file: `Path(__file__).parent`.

### String Formatting

- Use **f-strings** exclusively. Do not use `%` formatting or `.format()`.

### Error Handling

- **Non-fatal per-item errors** (e.g., a single case failing during setup): catch with
  `except Exception as e:`, print a warning (`print(f"Warning: {e}")`), and continue.
- **Fatal / unrecoverable errors**: print a clear message and call `sys.exit(1)` (or
  `return` early from `main()`).
- **JSON parsing**: catch `json.JSONDecodeError` explicitly.
- **Unexpected top-level failures** in `main()`: use `import traceback; traceback.print_exc()`
  to surface the full stack trace.
- **Validators**: accumulate all failures into a list and report them together at the end
  rather than raising on the first error.
- In tests, use `pytest.skip(reason)` for conditions that make a test irrelevant
  (e.g., `"Not a 2D dataset"`, `"No training samples found"`).

### Dataset `setup.py` Contract

Each `raw/Dataset###_Name/setup.py` must expose a `main()` function that:
1. Accepts zero CLI arguments (data root is discovered via `Path(__file__).parent`).
2. Converts raw source files into the nnU-Net directory layout
   (`imagesTr/`, `labelsTr/`, `imagesTs/`, `labelsTs/`).
3. Writes or updates `dataset.json` according to the nnU-Net v2 schema.
4. Prints progress with `tqdm` for long loops.
5. Is importable without side effects (guarded by `if __name__ == "__main__"`).

---

## Dataset JSON Schema (required fields)

```json
{
  "channel_names": {"0": "modality_name"},
  "labels": {"background": 0, "class_name": 1},
  "numTraining": 42,
  "file_ending": ".png"
}
```

---

## No CI Pipeline

There is no automated CI. Quality gates are enforced locally via **pre-commit** hooks
(`ruff check --fix`, `ruff format`, `ty check`). Always run `pre-commit run --all-files`
before opening a PR or pushing a significant change.
