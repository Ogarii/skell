# Python sandbox plan

Goal: prove the core of Skell before building the real app. Lock proportions, pose the skeleton, view it from any angle, check it, and see whether it reads right to an animator. Spec sections are given as "§".

Work one step at a time. Each step ends with its tests passing. Don't start the next step until then.

## Layout

```
sandbox/
  skell/
    __init__.py
    data.py         # load data/*.json (skeleton, rules, intents, phrases)
    quat.py         # quaternion maths: multiply, rotate vector, from axis-angle, shortest arc, slerp, swing-twist
    pose.py         # yaw/pitch <-> rotations, forward kinematics, grounding
    limits.py       # cones, twist, hinges, bend-direction rule, clamping
    camera.py       # 360-degree camera, projection, near plane, floor guard, shot sizes
    checks.py       # lengths, balance, overlap, continuity
    render.py       # matplotlib drawing: head, bones, joints, grid, depth order, overlays
    reconstruct.py  # joint placement over a sketch -> 3D (ray-sphere)        (step 6, optional)
    turnaround.py   # 8-view sheet
    ai.py           # Claude API call, schema, validation pipeline            (step 8)
  app.py            # interactive viewer with sliders
tests/
  test_step1_data.py ... one file per step
out/                # generated images (gitignored)
```

## Step 1 — Skeleton and reference pose

Build: `data.py`; load `data/skell_v1_reference_skeleton.json` and `data/reference_kneeling_pose.json`.

Tests:
- 15 joints, 14 bones, 10 posable, 4 rigid; every bone's `from` appears before it is used (parent before child).
- Names exactly as in the spec (§5.1). Lengths match the design notes: spine 90, neck-head 30, upper arm 47, forearm 39, thigh 56, shin 55.
- The pose has all 10 posable bones and no rigid ones.

## Step 2 — Posing maths (§5.2, §5.3, §6)

Build: `quat.py`, `pose.py`.
- Convert each yaw/pitch (heading frame) to a world direction, then to a local rotation relative to the parent: shortest-arc rotation from the bone's rest axis, expressed in the parent frame (§6.3). Twist rules from §5.4.
- Forward kinematics from the local rotations: child = parent + L * (world rotation applied to rest axis).
- Grounding: lowest joint to y = 0 unless airborne.

Tests (against `fixtures/expected_reference_kneeling.json`):
- World positions within 1e-6. This proves yaw/pitch → quaternion → FK round-trips to the same geometry the fixture got from directions directly.
- Every bone length within 1e-6 of its rest length.
- Rotations → yaw/pitch → rotations returns the same pose.
- Vertical bone (pitch 90, any yaw): converts without NaN; reading it back returns the previous yaw (§6.3).
- 180° swing (direction exactly opposite the rest axis): uses the parent's lateral axis (§6.3).
- Turned character: facing 90 rotates every position 90° about the vertical axis through the root; bone yaw/pitch unchanged.
- Rotating the spine carries the arms and head with it.

## Step 3 — 360° camera (§8)

Build: `camera.py`.
- Camera from target, azimuth, elevation, roll, distance, focal (§8.1). World and character-relative azimuth.
- Projection, roll, near plane (w < 20 not drawn), floor guard (§8.3), shot sizes, camera validation (§8.5).

Tests:
- All 12 camera rows in the fixture: every joint's dx, dy, s within 0.01.
- Basis orthonormal at every 15° of azimuth and elevation, including ±90.
- Facing 90 seen from azimuth_rel 0 renders exactly like the front eye-level row.
- Floor guard: the low-angle row resolves to R 214.314814, F 261.359529, camera 5.0 above the floor.
- Full shot of a standing 250-unit character with a 380-unit canvas: R = 411.184211.
- Near/far scale ratio r_hand / l_foot: 1.560658 at eye level, 1.268611 at double distance.

## Step 4 — Drawing (§9)

Build: `render.py`. Matplotlib, equal aspect, white background.
- Head circle (radius from the skeleton, scaled by s at the head), bone lines, joint dots, root in a distinct colour.
- Perspective floor grid (finite, clipped at the near plane) and horizon line at eye level; dashed grid when seen from below.
- Depth order: draw farthest first, by w at each bone's midpoint.
- Overlays (toggle): teal for bones nearly flat to the screen, amber for tilted bones, tilt-angle badge, dashed ghost line at true length.
- Boxes as props.

Tests:
- Renders every fixture camera to `out/` without error.
- Image-level check by eye: compare `out/front_eye_level.png`, `elevated_30`, `overhead_90`, `distant_eye_level` with design notes Figures 6–9. Write down what matches and what doesn't.

## Step 5 — Interactive viewer

Build: `app.py` with matplotlib sliders: azimuth (-180..180), elevation (-90..90), roll, distance, focal, character facing; buttons for the presets in `data/skell_rules_v1.json`; toggles for floor guard, overlays, world vs character-relative.

Done when: you can spin the kneeling pose all the way around, go over the top and under the floor (guard off) without the image flipping, and every preset lands where the fixture says.

## Step 6 — Checks (§5.4, §10.5)

Build: `limits.py`, `checks.py`.
- Cones, twist ranges, hinge bend ranges and the bend-direction rule (§5.4), with clamping that records the correction per bone.
- Balance with the mass fractions and contact rules in `data/skell_rules_v1.json`.
- Overlap (non-adjacent bones closer than 1% of height) and continuity (more than 45° per frame at 24 fps).

Tests:
- Reference pose: contacts `l_knee`, `r_foot`; bend angles and cone angles as in the fixture; no clamps (all four limbs bend the correct way); balance `unstable`, about 16 units from the contact line; no overlap or continuity warnings.
- A deliberately backward elbow and a backward knee get clamped to the nearest allowed bend.
- A standing rest pose is balanced.

The bend-direction rule is new and the fixture generator does not implement it. This is its first real test. If the reference limbs fail it, the sign convention is wrong, not the pose.

## Step 7 — Turnaround sheet and export (§8.7)

Build: `turnaround.py`. Eight character-relative views every 45° at the current elevation and distance, on one PNG, with a height ruler.

Done when: the sheet renders, and each view matches turning the character by the same angle.

## Step 7b — Sketch input (§4.2 Method B) — core, not optional

Joint placement over a sketch is one of the two primary input methods (§4.2) and has its own milestone, so the sandbox must prove it (decision 2026-11-18).

Build: `reconstruct.py`. Ray from the camera through each placed 2D joint, intersected with the sphere around the parent joint. Two solutions → default to the one nearest the previous pose, with a per-bone flip; none → nearest point plus a "drawn too long" flag above 2%. In the viewer: drag joints over the view (and an optional underlay image), flip a bone's depth, see the validated result.

Test: project the reference pose at the elevated camera, reconstruct from the 2D points, recover every joint within 1e-6; all 10 bones offer two depth solutions, and every flip works.

Added: automatic joints with no AI (`sandbox/sketch_detect.py`, outside the kernel). The viewer's "auto joints" button and `tools/sketch_to_pose.py` without `--joints` find the 11 joints in a stick-figure drawing, pick the start pose and facing, fit the camera to the drawing, and rebuild with the gesture fit. The result is a first guess to correct by dragging. Sketch pad (`sandbox/sketch_pad.py`): "draw sketch" turns the view into a page to draw on with the mouse or a finger, and "use drawing" saves the drawing to `out/sketches/pad/` (PNG + strokes JSON), puts it behind the figure and runs auto joints. Nearly closed strokes are joined, so a hand-drawn head encloses a hole. Tests: `tests/test_sketch_pad.py`, `tests/test_sketch_detect.py` (a synthetic figure within 6% of its height, front read as front with anatomical sides, and the viewer button puts the pose on the drawing).

## Step 8 — AI suggestions (§10), after steps 1–7 pass

Build: `ai.py`. Claude API with tool calling / JSON schema matching spec §10.4; system prompt with skeleton, rules and conventions; user message with current pose, previous poses, intent, scene and current camera (§10.2). Low temperature, 3–4 options. Run the validation pipeline in §10.5 (retry once on hard errors). Draw the options side by side with their warning badges. Log each request and result to `out/logs/*.jsonl` with the fields in spec §12.3.

Done when: a small set of requests ("kneel and pick up the box", "guard stance", "throw a right hook", "run toward the camera", "from behind, low angle") returns validated options. Track how many are valid after clamping (target ≥ 90%).

## Pitfalls to watch

- Screen y is down; world Y is up.
- `l_` is +X at facing 0. The design notes' drawings label sides the other way.
- Camera azimuth +90 is the camera on the +X side (the character's left at facing 0). Version 1.1 of the spec used the opposite sign.
- Quaternion Slerp: negate one quaternion when the dot product is negative.
- Yaw is undefined at pitch ±90; never divide by cos(pitch).
- Clip lines at the near plane in 3D before projecting, or the floor grid explodes near the camera.
