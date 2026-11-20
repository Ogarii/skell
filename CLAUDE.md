# Project Skell

Skeleton-based 2D animation assistant. An animator locks a character's proportions once; Skell poses a 3D skeleton (from natural language or joints placed over a sketch), validates it with deterministic geometry, and draws it as a clean 2D technical sketch from any camera angle. Skell never generates images.

## Where things are

- `docs/Project_Skell_MVP_Spec_v1.2.txt` — the working specification. Read the relevant section before building anything.
- `docs/Project_Skell_Design_Notes.pdf` — original design notes. The spec is reconciled with them; spec Appendix A lists every departure and why.
- `docs/SANDBOX_PLAN.md` — the current phase: the Python sandbox, step by step.
- `docs/DECISIONS.md` — settled decisions, reversible defaults, open questions.
- `data/` — shared rule data (skeleton, limits, masses, camera, intents, camera phrases). Rules live here, never hard-coded.
- `fixtures/` — golden expected outputs. `tools/generate_fixtures.py` regenerates them from `data/`.
- `sandbox/` — the Python sandbox (being built).

## Source of truth

1. The spec v1.2, which already folds in the design notes.
2. If the spec and the design notes disagree and Appendix A doesn't explain it, stop and ask; don't pick one silently.
3. If you find a problem in the spec itself, say so before working around it, and record the outcome in `docs/DECISIONS.md`.

## Frozen conventions (do not change without asking)

- World axes: X right in the front camera image, Y up, Z away from the front camera. Ground plane y = 0.
- Sides are anatomical: `l_` is the character's own left, which is +X (image right) when it faces the front camera (facing 0, front toward -Z).
- Skeleton format `skell_v1`: 15 joints, 14 bones named `from>to`, 10 posable + 4 rigid offsets (`data/skell_v1_reference_skeleton.json`). Names and layout are frozen because future models train on them.
- Bone direction from yaw/pitch, in the character's heading frame: `dir = (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))`. Yaw 0 = character's front, +90 = its left, pitch = degrees above horizontal.
- Facing uses the same sign: front direction = `(sin φ, 0, -cos φ)`.
- Camera azimuth uses the same sign as yaw: the camera sits at `C = T + R * dir(azimuth, elevation)`. Character-relative azimuth: `azimuth = facing + azimuth_rel`. Formulas in spec 8.1.
- Screen: `x' = cx + s*u'`, `y' = cy - s*v'` (screen y down). The design notes' section 9 coordinates are screen y-down; convert before using them.
- Stored pose = unit quaternions (local per bone, world for the root). Directions are derived. Never store joint positions as the pose.
- Interpolate rotations (quaternion Slerp, shortest path), never positions.
- Bone lengths never change after setup. Foreshortening comes from projection only.
- AI proposes, geometry validates. The AI never sets positions directly.

## Working rules

- Verify before claiming done: run the tests and compare against `fixtures/`. A feature with an interface but failing numbers is not done.
- Tolerances: world 1e-6 units, screen 0.01 units, angles 1e-4 degrees (see the fixture file).
- When you change anything in `data/`, rerun `python tools/generate_fixtures.py` and say what changed in the fixtures.
- Keep the geometry kernel free of rendering and UI code so it can be ported to TypeScript and Kotlin later.
- Build only what the current step in `docs/SANDBOX_PLAN.md` needs. New ideas go in `docs/DECISIONS.md` under "Ideas", tagged core / later / separate product.
- Keep the sandbox pure Python + numpy + matplotlib until step 8 (AI), which adds `anthropic`. Exception: the joint-model training code may use PyTorch, kept in `requirements-train.txt` and never imported by the sandbox runtime or its tests (see `docs/DECISIONS.md`). Read the API key from the `ANTHROPIC_API_KEY` environment variable; never write it to a file.

## Commands

```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/generate_fixtures.py     # regenerate golden fixtures
pytest -q                              # run all tests
python -m sandbox.app                  # interactive viewer (from step 5)
pip install -r requirements-train.txt  # joint-model training only (PyTorch)
python tools/kaggle_train.py push     # train on a Kaggle GPU; then: status, fetch
```

## Later stack (not yet)

React + TypeScript + Konva/PixiJS frontend, Spring Boot (Java/Kotlin) backend, PostgreSQL with JSONB, Claude API, Docker Compose, GitHub Actions. See spec Section 12. Don't start these until the sandbox passes its plan.
