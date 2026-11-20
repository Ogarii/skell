# Project Skell

Proportion-locked 2D pose assistant for animators. Draw a character's proportions once; pose it from text or a rough sketch; view it from any angle; the bone lengths never drift.

Current phase: Python sandbox (`docs/SANDBOX_PLAN.md`).

## Start

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/generate_fixtures.py  # prints the reference checks and writes fixtures/
pytest -q
```

## Contents

| Path | What it is |
|---|---|
| `CLAUDE.md` | Rules and conventions for Claude Code |
| `docs/Project_Skell_MVP_Spec_v1.2.txt` | Working specification |
| `docs/Project_Skell_Design_Notes.pdf` | Original design notes |
| `docs/SANDBOX_PLAN.md` | Step-by-step sandbox build with tests |
| `docs/DECISIONS.md` | Decisions, defaults, open questions |
| `data/` | Shared rule data: skeleton, limits, camera, intents, phrases |
| `fixtures/` | Golden expected outputs for the reference pose |
| `tools/generate_fixtures.py` | Independent reference maths that writes the fixtures |
| `sandbox/` | The sandbox code (to be built) |
