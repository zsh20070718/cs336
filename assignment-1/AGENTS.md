# CS336 Assignment 1 Agent Instructions

## Do not change code outside the current task

- Only edit what the user explicitly asked for. Do not refactor, "improve", or touch other files or functions.
- Leave already-working or previously completed code unchanged unless the user explicitly asks to change it.
- If the user says "implement X" or "fix Y", change only the code required for X or Y. Do not rewrite unrelated parts of the file or other files.

## Working agreement

- Default implementation target is `cs336_basics/`.
- Match the code style of existing files in `cs336_basics/` (e.g. `linear.py`: one param per line, spaces around `=` in kwargs, minimal comments, compact expressions).
- `tests/` is allowed for reading to understand expected behavior and interfaces.
- Do not edit `tests/` unless I explicitly ask.

## Project commands

- Run code/tests via `uv run ...` (see `README.md`).
- Prefer `uv run pytest` for verification.

## Keep context lean

- Avoid pulling `.venv/`, `data/`, caches, logs, and model checkpoints into context.

## Code style (cs336_basics)

- Signatures: One parameter per line for multi-arg functions; trailing comma on the last parameter. Type hints on main args and return types; optional args can use `param = None` without type.
- Spacing: Spaces around `=` in keyword arguments (e.g. `mean = 0.0`, `std = 2.0 / (in_features + out_features)`).
- Structure: Single blank line between methods; blank line between logical blocks inside a method. No docstrings or narrative comments unless necessary.
- Expression style: Prefer compact, readable expressions (e.g. `x @ self.weight.T`). Keep functions small and focused.

## Testing / formatting

- After edits, run the smallest relevant test(s) when possible; otherwise run `uv run pytest`.
- Respect `pyproject.toml` settings (e.g. ruff line length 120).

