# First prompt for Claude Code

Paste this as your first message after opening the folder in Claude Code:

---

Read CLAUDE.md, then docs/SANDBOX_PLAN.md, and skim docs/Project_Skell_MVP_Spec_v1.2.txt sections 5, 6 and 8. Set up the venv, install requirements, and run `python tools/generate_fixtures.py` and `pytest -q` to confirm the starting point is green.

Then build Step 1 and Step 2 of the sandbox plan only. Write the tests first from the fixture file, then the code. Stop when both steps pass and show me the test output, plus anything in the spec or fixtures you think is wrong or unclear. Don't start Step 3 until I say so.

---

After that, continue one step at a time: "Build Step 3", "Build Step 4", and so on. At Step 4, ask it to save the fixture camera images to `out/` so you can compare them with the design notes figures.
