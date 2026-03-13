# Create Plan Skill

## Purpose

Create a new plan for work. This includes planning for any kind of work that creates or updates non-plan files.

## Invocation

User or agent invokes `/create-plan description-of-plan`.

## Request

Consider any relevant history when this skill is invoked. Often the user will want you to create a plan from some
recent discovery work. This is initial planning work, so keep things high level and details will be broken down later.

## Rules

- Write the details in a new file in the `docs/plans/` directory.
- Consult the user to resolve any ambiguity in decision-making.
