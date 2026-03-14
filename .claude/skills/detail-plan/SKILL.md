# Detail Plan Skill

## Purpose

Break down a plan into much more detail so the user can ensure implementation aligns with their expectations. Also to
produce a plan that can be easily followed without any room for ambiguity.

## Invocation

User invokes `/detail-plan description`.

## Request

Break down the planning referenced by the user-provided description into very high detail.

In addition to your best effort, make sure to include:
- architectural decisions
- rationales for decisions
- documentation
- tests
- logging (once logging is implemented)
- how the user can verify correct implementation by running the project on real data

## Steps

1. Create the detailed plan.
2. Only after the detailed plan is complete, invoke the `/review-plan` skill on the plan.

## Rules

- Write the details in a new file in the `docs/plans/` directory.
- Consult the user to resolve any ambiguity in decision-making.
