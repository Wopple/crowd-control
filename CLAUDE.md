Read `structure.md`.
Keep `structure.md` up-to-date as files are added, removed, and updated.
Read `README.md` for project goals and background.
Read all files in `@docs/plans/`.
Document all planning in `@docs/plans`.

## Development

This is a uv project. Use `uv run` to execute all commands.

```
uv sync              # Install/update dependencies (run after changing pyproject.toml)
uv run pytest        # Run tests
uv run pytest -v     # Run tests with verbose output
uv run ruff check    # Lint
uv run ruff format   # Format
uv run crowd-control --help   # Run the CLI
```

Do NOT use `.venv/bin/python`, `python -m pytest`, or bare `pytest`. Always use `uv run`.

Dev dependencies (pytest, ruff) are in `[dependency-groups] dev` in pyproject.toml, not
`[project.optional-dependencies]`. This is important — uv auto-installs dependency groups
but not optional dependencies.
