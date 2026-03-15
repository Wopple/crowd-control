# Bootstrapping

1. Read `docs/introduction.md`.
2. Use the `search_learnings` tool from crowd-control to gather relevant information for the task at hand.

# Documentation

There are two kinds of docs in this project:

- `docs/plans/` — ephemeral planning documents. These exist only to support implementation
  and should not be read to understand what is already built. They may be outdated or
  describe things that haven't been implemented yet.
- `docs/` — durable implementation documentation. This describes what exists, how it works,
  and how the pieces connect. An agent should be able to understand the system from these
  docs without reading source code.

When implementing a phase, write or update docs in `docs/` (not `docs/plans/`). This is
part of completing the phase, not a separate task.

# Commands

This is a uv project that uses `just`.

```
just install      # Install/update dependencies (run after changing pyproject.toml)
just test         # Run tests
just test -v      # Run tests with verbose output
just lint         # Lint
just fmt          # Format
just run --help   # Run the CLI
```

# Coding Guidelines

Treat these guidelines as extremely important. Violations are only allowed with strong justification.

## Source

- Single Responsibility Principle
    - Each responsibility is handled in only one software component.
    - Each software component handles only one responsibility.
- Pure Functions
    - Always favor pure functions.
    - Complex logic must be encapsulated in a pure function.
    - Pure functions have no side effects.
    - Pure functions do not mutate their inputs.
    - Pure functions do not mutate their outputs after returning (e.g. threads).
    - Pure functions do not access global state.
    - Pure functions do not access external resources.
- Clean Code
    - Code is straightforward.
    - Software components operate at a consistent level of abstraction.
    - Code nests for necessity, not convenience, the less nesting the better.
- Design Patterns
    - Each pattern usage provides benefit, it is not superfluous.
- Code Smells
    - Aggressively avoid code smells.
    - Fix them with design patterns.
- Maintainability
    - Always consider the future maintainability of the code you write.
- Defensive Coding
    - Always handle error paths.
- Performance Bottlenecks
    - Only optimize bottlenecks that impact the user experience.
- Singletone
    - Avoid singletons.
    - Instantiate objects in main code and pass in as dependencies to other code.
- Type Hints
    - Always use fully parametrized type hints.

## Tests

- Never test implementation details, only test behavior.
- Never test trivial code, each test is a liability, so every test needs to be valuable.
- Tests must not call claude code or query any LLM.
- Tests must not create or depend on external state.
- Tests cannot assume a connection to an embedding model.
- Models and connections may be used for generating test data.
