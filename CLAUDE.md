Read `structure.md`.
Keep `structure.md` up-to-date as files are added, removed, and updated.
Read `README.md` for project goals and background.
Read all files in `@docs/` (excluding `docs/plans/`) for documentation on what is implemented.

## Direction

Do not implement features. Focus on improving code quality and documentation. We are going to find bugs and address architectural weaknesses.

## Documentation

There are two kinds of docs in this project:

- `docs/plans/` — ephemeral planning documents. These exist only to support implementation
  and should not be read to understand what is already built. They may be outdated or
  describe things that haven't been implemented yet.
- `docs/` — durable implementation documentation. This describes what exists, how it works,
  and how the pieces connect. An agent should be able to understand the system from these
  docs without reading source code.

When implementing a phase, write or update docs in `docs/` (not `docs/plans/`). This is
part of completing the phase, not a separate task.

## Planning

Document all planning in `@docs/plans/`.

## Development

This is a uv project. Use `uv run` to execute project commands.

```
uv sync              # Install/update dependencies (run after changing pyproject.toml)
uv run pytest        # Run tests
uv run pytest -v     # Run tests with verbose output
uv run ruff check    # Lint
uv run ruff format   # Format
uv run crowd-control --help   # Run the CLI
```

## Coding Advice

- Single Responsibility Principle
    - Each responsibility is handled in only one software component.
    - Each software component handles only one responsibility.
    - These goals are ideal, not hard requirements.
- Favor Pure Functions
    - Complex logic must be encapsulated in a pure function.
    - Pure functions have no side effects.
    - Pure functions do not mutate their inputs.
    - Pure functions do not mutate their outputs after returning (e.g. threads).
    - Pure functions do not access global state.
    - Pure functions do not access external resources.
- Clean Code
    - Code is easy to understand.
    - Software components operate at a consistent level of abstraction.
    - Code is straightforward.
    - Code nests for necessity, not convenience, the less nesting the better.
- Design Patterns
    - Each pattern usage provides benefit, it is not superfluous.
- Code Smells
    - Address with design patterns.
- Ease of future maintenance.
- Defensive Coding
- Performance Bottlenecks

## Tests

Tests must not call claude code or query any LLM. Tests cannot assume a connection to an embedding model either. You may
use claude code or an embedding model for generating test data. Tests can be written against that test data.
