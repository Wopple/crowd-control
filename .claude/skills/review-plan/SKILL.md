# Review Plan Skill

## Purpose

Ensure the plan is comprehensive, aligned with requirements, and ready for implementation.

## Invocation

User invokes `/review-plan description`.

## Request

Review and update the plan referenced by the user-provided description:
- check for alignment with requirements
- check for maintainability of the architecture
- check implementation supports future use cases
- check for corner cases that need to be handled
- check all possible failure scenarios are handled effectively
- check for security vulnerabilities
- check for friction points for the end user
- check for scalability and performance bottlenecks
- check for integration errors with the existing code
- check application behavior is made visible through logging (once logging is implemented)
- check documentation is included

## Rules

- Consult the user to resolve any ambiguity in decision-making.
- Make sure every update to the plan has a good reason. Do not change the plan just to do work or please the user. Quality is the goal.
