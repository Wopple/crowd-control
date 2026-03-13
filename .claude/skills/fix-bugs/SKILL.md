# Fix Bugs Skill

## Purpose

Find and fix bugs ensuring they are not regressed.

## Invocation

User invokes `/fix-bugs optional-description`.

## Request

You are focused solely on squashing bugs in the code. Gather an understanding of the project, and find the most
important bugs to fix. Once you have found those bugs, fix them using this process:

1. Write the test that fails due to the bug.
2. Run the test to confirm it fails. If it doesn't fail, fix the test so that it does fail.
3. Fix the bug.
4. Run the test again to confirm it passes. If it doesn't pass, go back to fix the bug.
5. Ensure all other tests still pass.
