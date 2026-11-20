# Decisions log

Add an entry whenever a decision is made or changed. Newest at the bottom of each list. Spec references are "§".

## Settled

- Skeleton is the source of truth; bone lengths fixed after setup (§1.2).
- All poses exist in 3D and are projected to 2D; no image generation (§1).
- Continuous representation is authoritative; discrete sphere points are an optional layer for AI, classification and snapping only (§6).
- Stored pose = unit quaternions, local per bone, world for the root; directions derived (§6.1, spec v1.1).
- AI talks yaw/pitch per posable bone in the character's heading frame, plus facing and root position (§10.4).
- `skell_v1`: 15 joints, 14 bones, anatomical left/right (§5.1, v1.1).
- Two characters per scene in the MVP; contact pins and collision deferred (§11, v1.1).
- Sketch input is deterministic joint placement with ray-sphere depth; automatic joint detection deferred (§4.2, v1.1).
- 360-degree camera set by hand or described in words; presets are shortcuts (§8, v1.2).
- Camera azimuth uses the same sign as bone yaw (§8.1, v1.2).
- Rules are data in `data/`, checked by shared golden fixtures; the server is the authority in the full app (§12.2).
- Python sandbox first, then React/TypeScript + Spring Boot (this handoff).
- **Spec change (user decision, 2026-11-18): automatic joint detection from a rough drawing moves into scope as a pre-fill.** In step 8, Claude's image understanding proposes the 2D joint positions on the sketch. The artist corrects them with guided clicks, and the same deterministic ray-sphere reconstruction and validation then run. The AI only proposes 2D points; geometry still decides the pose. This departs from §4.2 ("automatic detection … is outside the MVP") and §14. Record it in the next spec revision's Appendix A.
- Guided joint placement: a "place joints" mode prompts for the joints in parent-before-child order — the root, then each posable bone's end joint, 11 clicks; rigid shoulders and hips follow. Each click can be undone or skipped (skipping keeps that joint's current position), and the pose is rebuilt from the clicks. A side-view inset (§9.2) shows depth while placing (sandbox step 7b).
- Sketch input (§4.2 Method B: joint placement with ray-sphere depth) is core. The sandbox builds it as step 7b, with viewer dragging and depth flips; the plan had marked it optional (user decision, 2026-11-18).
- **Face cross on the head (user decision, 2026-11-20).** The renderer draws the artist's face cross on the head sphere: a centre line and an eye line, each running 75° from the face centre over the front of the head. Only the parts facing the camera are drawn, so it reads as a "+" from the front, a curve at the edge in profile, and nothing from behind. It follows the world rotation of the bone that ends at the head (`Subject.head_rotation`, from `pose.head_rotation`), so head turns show. Without that rotation it falls back to the character's facing. `RenderOptions(face=False)` turns it off. This departs from §9.1's plain head circle and answers the "eyes would need head facing" note under the step 4 visual check. Record it in the next spec revision's Appendix A. The synthetic training sketches (`sandbox/synth.py`) and `tools/make_sample_sketch.py` draw their own figures and are unchanged.
- **PyTorch for training only (user decision, 2026-11-19).** This is an exception to the numpy + matplotlib rule. Training code and its dependencies stay in their own files (`requirements-train.txt`), and they run on Kaggle. The sandbox runtime and its tests do not import torch. To run the trained model in the sandbox, export the weights and do inference in numpy, so `requirements.txt` stays unchanged.

## Defaults that can be reversed

- Average uneven left/right lengths (symmetric on), rather than mirror one side (§4.1).
- Reference character height 250 units; reference canvas height 380 units (§4.1, §8.3).
- Balance, overlap and continuity are warnings, not rejections (§10.5).
- Default joint limits in `data/skell_v1_reference_skeleton.json` (§5.4).
- Chest twist range ±45° relative to the pelvis. Added in the skeleton data during the Claude Code handoff; the spec gives no value. Confirm or change.
- Rest axes for arms and legs point straight down. The spec asks animators to draw the setup sketch with arms slightly away from the body; setup should measure lengths from that drawing but store straight-down rest axes. Confirm when building character setup.
- Floor guard on by default, keeping the angle and on-screen size (§8.3).
- Camera fixed within a shot for the MVP (§8.6, §14).
- Camera phrase values, e.g. low angle -30°, Dutch 20° (§8.4, `data/camera_phrases_v1.json`).
- Continuity threshold 45° per frame at 24 fps (§10.5).
- Chest and head facing values are absolute yaw angles, as §5.4's "Default: the root facing" implies. Spine twist = chest − root facing; head twist = head − chest facing. Each twist is applied about the bone's own direction, after the shortest-arc swing (sandbox step 2).
- Because chest and head facings are absolute, turning a character (changing the root facing) must add the same angle to its chest and head facings. Otherwise the turn becomes a chest twist, which the step 6 limits clamp, swinging the arms. The viewer does this. The step 8 AI output and the later editor must do the same (sandbox step 6).
- Twist roles come from the skeleton's structure, not from bone names. A bone attached to the root is the chest. A ball bone whose parent is the chest is the head. A bone with a hinge child is an upper limb. Hinge bones have no twist (sandbox step 2).
- An airborne frame carries its pelvis height as `root.y`. The §10.4 schema has no field for it. Confirm before step 8 (sandbox step 2).
- Upper-limb twist sign: twist ψ about the bone rotates the neutral axis (fold sign × carried lateral axis) onto the bend normal n = a × b. Elbows fold front (+1), knees back (−1). With this, every forearm and shin's local rotation is a pure rotation about the upper bone's lateral axis (sandbox step 2).
- **Bend-direction rule revised (sandbox step 2, replaces the §5.4 wording).** All three values live in data:
  - The neutral side axis is the parent frame's lateral axis carried by the shortest arc from the reference axis (yaw 0, pitch −45, parent frame) to the bone, not from the rest axis. The value is `limits.limb_twist_reference_yaw_pitch` in `skell_rules_v1.json`.
  - The hinge-plane threshold is 20°, up from 10° (`limits.hinge_plane_min_bend_deg`).
  - The hip rotation range is ±60°, up from ±45° (thigh `rotation_range`).

  Reference limbs now: both elbows are under the threshold (twist 0), the left thigh reads −21.8° and the right thigh −51.0°. Spec §5.4 text and its limit table need the same change in the next spec revision.
- Canvas width 570 units (3:2 with the 380-unit height), added as `camera.canvas_width`. The spec gives only the height. It is used for the "more than half outside the frame" warning (§8.5) (sandbox step 3).
- "Character height" in the wide (2×) and extreme-wide (4×) shots means the current pose's bounding-box height, like the full shot, following §8.3's "from the current pose's bounding box". A kneeling figure therefore gets closer wide shots too. Medium and close-up use vertical extents, so a lying-down figure frames badly; revisit with props and scenes (sandbox step 3).
- Floor guard with the camera above the target's level: when the target itself is below the clearance height and the camera looks down at it, the guard moves the camera *farther* along the same line, scaling the focal length by the same factor. §8.3 describes only the "closer" case (sandbox step 3).
- Camera targets in the sandbox are a joint of the selected character, the other character's head (over-the-shoulder), or a fixed world point. Prop centre and scene centre wait for props and two-character scenes (sandbox step 3).
- Renderer style follows the design notes figures (§9.1). Bones are teal, joints and head pale with a dark teal outline, and the root coral. Joint markers are 4.5 units, about a quarter of the head radius as in the figures. With overlays on, bones flat to the screen stay teal and tilted ones turn amber, as in Figure 3. The teal/amber threshold of 15° and the grid step of 50 are in `render` in the rules (sandbox step 4).
- Step 4 visual check against design notes Figures 6–9. The PDF was re-rendered from its vector content and compared side by side in `out/compare_design_notes_fig6-9.png`, which is not committed.
  - Matches in all four figures: the pose shape, the horizon through the eyes, nearer parts drawn larger, the head hiding the neck and spine stub (Figures 7 and 8), the right shin trailing to the top in the overhead view, and the box under the reaching right hand.
  - Differs:
    - The figures draw eyes on the head; Skell draws a plain circle (§9.1). Eyes would need head facing; possible later. (Done as a face cross, 2026-11-20: see "Face cross on the head" under Settled.)
    - The figures show a small floor patch; Skell draws the spec's ±1000 grid.
    - The figures are zoomed in further than the 380-unit canvas.
- Step 6 check definitions (sandbox step 6):
  - **Contacts:** joints within 2% of standing height of the floor. Joints placed by zero-length offsets (zero-width shoulders and hips) are skipped because they duplicate the neck and pelvis.
  - **Balance:** the centre of mass must lie inside the outline of the contacts, widened by discs of 4% of standing height around feet and 2% around other contacts. The reported distance is to the unwidened outline, which matches the fixture's 16.03. The mass segments, head joint and foot joints are in `balance` in the rules.
  - **Overlap:** two bones count as adjacent, and are never flagged, when they meet at a joint or at joints joined by rigid offsets. So arms hanging from zero-width shoulders never "overlap" the spine.
  - **Continuity:** compares each bone's local rotation, and the root's world rotation, between frames. The threshold scales with the frames between keys.
  - **Clamping order:** parent before child. A cone clamp keeps children's local rotations, so they move with the bone. A bend-direction clamp turns the upper bone about its own length, which carries the lower bone onto the nearest allowed plane.
- Full and wide shots fit the pose as seen from the camera. The full-shot distance is the smallest at which every joint and the head circle fall inside 80% of the canvas from that view; wide and extreme wide are 2× and 4× of it. For a pose flat to the camera this equals the §8.3 formula (the standing reference still gives 411.184211). Deep poses such as kneeling or sitting, seen from three-quarter, no longer run off the canvas. Medium and close-up still use the §8.3 formula (sandbox step 6).
- Viewer pose library: `data/sandbox_poses_v1.json` holds 11 poses besides the reference: standing, T-pose, kneel and point, guard, right hook, walk, run, squat, sit on the floor, reach up and jump. Every one passes the step 6 limits unclamped with no overlaps, and a test enforces this. The viewer shows the validated pose and its checks, and has a "full shot" toggle, off by default so the presets still match the fixture (sandbox step 6).
- Turnaround sheet (sandbox step 7):
  - **Views:** the eight views come from the character-relative azimuth presets in the rules, ordered 0, +45, … , −45.
  - **Framing modes:**
    - "current" (§8.7 as written) uses the head target and the given distance.
    - "full" targets the character's vertical axis at the pose's mid-height and uses the largest fit distance over all eight views, so every view shares one scale, as a model sheet needs. The viewer uses "full" when its full-shot toggle is on.
  - **Ruler:** a perspective pole beside the character, from the floor to standing height, marked every head unit (2 × head radius) and labelled with the head count.
  - **Turning:** `pose.turn_frame` turns a character by turning its root, chest and head facings together; the viewer and the turnaround share it.
- Sketch input details (sandbox step 7b):
  - **Root depth:** the root slides along its ray, from the current depth, to where the lowest reconstructed joint is on the floor. Then grounding moves nothing and the figure sits exactly on the drawing. Only depths where every bone keeps the same kind of solution count, so a flip or depth choice is never undone by grazing a sphere. With no such depth (for example, the drawn feet are above this camera's horizon), the current depth is kept, grounding moves the figure, and a note says the drawing does not fit this camera. §4.2 says only that the root's depth comes from the current position; this refines it.
  - **Default depth:** of two solutions, the default prefers one that keeps the joint limits. For an upper arm or thigh this looks one bone ahead, so the elbow or knee can still bend the right way. Then it prefers the one nearest the current pose. §4.2 says "closest to the previous pose"; the limits preference was added after a rough sketch reconstructed with a backward knee.
  - **Two ways to read a drawing** (`reconstruct(..., fit=...)`):
    - **"exact"** (default): drawn lengths are true, for drawings in the character's own proportions. Every joint lands on its drawn point.
    - **"gesture"**: for stylised drawings such as stock stick figures, with long arms, equal upper and lower limbs and floating heads. Each bone keeps its drawn 2D direction exactly but gets the character's length, nearly flat to the view. A bone drawn clearly short (under `sketch.gesture_keep_short_frac` = 0.6 of the drawing's median scale) keeps that foreshortening as depth. The size follows the camera and the floor, not the drawing. On six stock stick figures, exact reading flagged up to 9 of 10 bones as drawn too long, 7–70% each; gesture reading flagged none, gave every bone two depth options, and produced no clamps.
  - **A drawing implies a camera.** `tools/sketch_to_pose.py --fit gesture` fits an eye-level camera to the drawing:
    - **Height:** the lowest drawn joint stands on the floor.
    - **Scale (which sets the distance):** the scale at which the rebuilt gesture lands closest, on average, to all the drawn joints.
    - **Rejected anchors:** each of these alone left some figures too big or too small: the median bone ratio, the start pose's height, and a pelvis-height anchor (which also failed for sitting and running figures).
    - **Search:** now `reconstruct.fit_drawing_camera`, searched between two first guesses: the median bone scale, and the scale at which the start pose is as tall as the drawing. The thorough mode (CLI) takes 27–52 s and matches the earlier optimum. The quick mode (viewer, 5 rebuilds) takes 9–15 s, with RMS misfit within 0–3.5 screen units of the optimum on four stock figures.
  - **Automatic joints without AI** (`sandbox/sketch_detect.py`, outside the kernel; the user has no API credits, so this comes before the Step 8 vision route):
    - **Head:** the highest round enclosed hole. Fill above 0.82 of a disc and aspect above 0.7 rule out leg loops.
    - **Rest of the figure:** the Zhang-Suen centre-line of the remaining strokes.
    - **Fit:** an 11-joint stick figure is fitted to the centre-line, started from every grounded library pose at facings 0, ±45 and ±90. It is ICP for a jointed figure:
      - ink → nearest segment and segment → nearest ink, with a robust scale annealed from broad to sharp;
      - stroke tips ↔ hands and feet;
      - joints sit on ink;
      - the neck hangs from the head;
      - lengths are free within 0.3–1.3 of the character's.
    - **Escaping local minima:** discrete limb re-seatings, kept when they lower the cost: lay a limb on its twin (a hidden limb), or put a limb onto a stroke tip no hand or foot has reached.
    - **Tips are found geometrically:** all nearby ink lies on one side. A pixel-neighbour test fails because thinning forks thick tips into prongs.
    - **Side views must earn it:** a side-view start must fit 25% better to win. Symmetric front drawings otherwise read as profiles, with left and right swapped.
    - **Airborne library poses are not starts.** A drawing can't say it is off the ground, and an airborne start skips the floor.
    - **Accuracy on the six stock figures**, against joints read by eye, as mean error over the figure's height (left and right may swap per limb pair in profile):
      - 2.1%, 2.1%, 2.5% and 2.5% on the standing, running, cheering and lunging figures;
      - 6.5% on reaching (the arm hidden along the body was placed on the reaching arm);
      - 13% on sitting with knees up (hidden limbs and crossing loops).
    - **Time:** 10–15 s per drawing.
    - **The result is a first guess:** the CLI writes `<out>_joints.json` to correct, and the viewer's "auto joints" button leaves sketch mode on for dragging.
  - **Sketch pad, free drawing:** the viewer's pad takes any drawing, not only stick figures. "use drawing" saves it, puts it behind the figure and starts joint clicking; "auto joints" remains for stick figures.
    - Once every joint is clicked, the start pose is the library pose and facing whose front view best matches the clicks (`sketch_detect.best_start`: a scale and shift fitted to all 11 joints, the same side-view penalty). The rebuild then uses the gesture fit and the quick camera fit.
    - Before this, clicks were rebuilt from whatever pose was selected. On a standing test drawing, the kneeling default landed 42 screen units RMS off the clicks; with `best_start` it lands 9.8 off.
    - Fast strokes come out as straight segments on slow machines: the whole window redraws on every pen movement, so most points are lost. Planned fix: blitting plus recording every point (not done yet).
  - **Synthetic training drawings** (`sandbox/synth.py`, `tools/make_training_set.py`; phase 1 data for the learned detector):
    - **Poses:** real Skell poses (validated and grounded): library poses with every bone jittered by about 20°, plus random upright, bent and lying spines with limbs pointing anywhere.
    - **Cameras:** random azimuth, elevation and distance, re-framed in 2D.
    - **Drawing style:** per-drawing proportions, a 0.9–1.9× head, wobble, open or overshooting head circles, curved limbs (30% of arms, 25% of legs), line width 1.2–7.5 px, occasional ground line and noise.
    - **Labels:** the 11 joints in pixels, exact except for wobble at the middle joints (at most about 1.2% of the image).
    - **Speed:** 42 ms per drawing at 192 px; shards of 5000 in parallel, each with its own seed.
    - **Real drawings** are framed the same way (`synth.frame_drawing`: crop to the ink, square, box-filter resample), and `tools/make_test_set.py` builds `out/training/test_real.npz` from the six stock figures plus anything labelled with `tools/label_drawing.py`.
  - **Practice sketch:** `tools/make_sample_sketch.py` draws a rough right hook, limbs 6% short and wobbly, as seen from the viewer's default camera, plus the joint positions to click. A test runs the whole workflow on it: the result lands exactly on every click with exact bone lengths and no clamps.
  - **Flips:** they are one-shot. A flip takes the other solution relative to the current pose, and the result becomes the current pose.
  - **Drawn too long:** with no solution, the nearest point on the sphere is used. "Drawn too long by X%" uses X = (ray distance from the parent joint ÷ bone length − 1) × 100 and is flagged above `limits.setup_drawn_length_tolerance_frac` (2%).
  - **Rigid joints:** shoulders and hips are not placed; they follow their parent bone. With non-zero widths, the chest frame's twist is carried over from the current pose, so shoulders may need a follow-up placement. Exact for the zero-width reference character.
  - **Validation:** the result is validated, so a depth that bends a limb backward is clamped and reported.
  - **Viewer:** the camera is frozen while a joint is dragged, so a head-targeted camera doesn't chase the head.
- Fixture rounding fixed: the generator now computes `checks.near_far_scale_ratio_*` from unrounded scales and rounds once. Before, it divided two s values already rounded to 6 decimals. The fixture changed in those two values only: eye level 1.560659 → 1.560658, double distance 1.26861 → 1.268611. The step 3 test compares at 1e-6 (sandbox step 3).
- Gesture fit fixed (2026-11-20): upper arms and thighs were aimed from where the *start pose* had the shoulders and hips on screen, because those rigid joints are never placed. A drawing far from the start pose came out with drooping arms and legs, which the limits then clamped further (seen on a side-kick sketch: thighs 45–75° off). `reconstruct` now carries each rigid joint with its drawn parent, so with zero-width shoulders and hips they sit on the drawn neck and pelvis. `drawing_scale` uses the same points and is corrected too. The exact fit and the fixtures are unchanged. Test: `test_gesture_fit_aims_limbs_from_the_drawn_neck_and_pelvis`.

## Open questions

- Shoulder and hip widths for the reference character. The fixture uses 0 to match the design notes; a rest-pose fixture with real widths is still to be defined (spec Appendix B.5).
- Whether the bend-direction rule's sign convention holds; first real test in sandbox step 6.
- **(Resolved, see "Bend-direction rule revised" under defaults.)** Spec problem found in sandbox step 2: the reference pose fails the bend-direction rule as written in §5.4. The limb rotation (angle about a from the shortest-arc-carried lateral axis to n) comes out as follows:
  - left arm 0°: elbow bend 5.9°, under the 10° threshold, so the rule doesn't apply
  - right arm 120.8° in the chest frame, against a ±90° range
  - left thigh −7.1°, within ±45°
  - right thigh −81.7°, against a ±45° range

  Three alternative readings were also tried, and none passes all four limbs:
  - lateral axis turned by the bone's yaw
  - lateral axis mirrored on the right side
  - lateral axis carried from the thigh cone axis

  The right arm's 11.8° bend is barely over the threshold, so its bend plane is weakly determined. The right thigh's large value is mostly the swing-twist effect of a thigh pointing 52° to the side. Positions and the fixture are unaffected; only the stored upper-limb twist depends on this.

  How it was resolved. Seven measures were tried on 34 test limbs per side (natural and backward arms and legs, both sides) plus the reference pose:
  - Spec as written: fails the reference right thigh (−82°) and the overhead and wind-up arms (144° to 164°).
  - Flexion-then-abduction: best leg values, but jumps 45° when the thigh points straight sideways, which side kicks reach. The jump would also make the stored twist jump.
  - Yaw-frame, mirrored and abduction-then-flexion measures: fail natural or backward limbs.
  - **Forward-down reference (chosen):** 0 test failures. Natural limbs read at most 63° (arms) and 51° (legs). Backward limbs read at least 135° (arms) and 159° (legs). Its singular direction is back-and-up, outside the thigh cone and rare for arms. Trade-off: arms passing within a few degrees of straight back-and-up get an unreliable reading.

  With 5° AI rounding (§10.4), the worst-case bend-plane error is 36° at a 10° bend, 24° at 15° and 16° at 20°. That gives the 20° threshold. The right arm's 11.8° bend reads anywhere from 71° to 134° under that rounding, so it can't be judged.

  Hip range: under 5° rounding the right thigh spans −58° to −44°, so it needs ±60°. Range widening alone was rejected: it needs about ±100° under the old measure, which leaves a 28° margin to the nearest backward knee.

  The analysis scripts were scratch files and were not kept in the repo.

  Also tried and rejected: legs on the hip-style measure, blended into forward-down near the straight-sideways poles, keeping the ±45° hip range. It is not better than the current measure:
  - Its jump rate per degree of thigh movement is 3.9 to 5.2, against 1.4 for the current measure, because the two axes differ by up to 122° inside the blend zone.
  - The right thigh reaches −44.7° under 5° rounding, against a ±45° limit.
  - It adds a second measure and a blend-zone value to port.
  - It gains only a little on natural legs (42° of a ±45° range used, against 51° of ±60°), and backward knees read 138° either way.

  Known edge: a turned-out plié (thigh straight out and 30° down, shin down) reads 63° and is flagged at ±60°. Per-character limits can widen it.
- Does the kneeling pose's "unstable" balance warning match animators' judgement? Ask in the first animator tests.

## Ideas (tag: core / later / separate product)

- (later) Animated camera moves within a shot.
- (core) Automatic joint detection from rough drawings: the offline fit is built (see sketch input details). Later: AI vision (Step 8) as a second opinion where credits allow; a filled-disc head; several figures per image; better handling of hidden limbs.
- (later) Contact pins between characters.
- (core) Learned joint detector, in phases. Phase 1 trains on synthetic stick figures (`tools/make_training_set.py`) on Kaggle. Later phases: (a) detect the shoulders and hips too, not just the 11 placed joints. These are already joints in `skell_v1`, so only the detector output grows and the skeleton format does not change. (b) Synthetic character sketches with varied body types (child, girl, boy, adult, and so on), drawn with body outlines instead of sticks, to look like a real artist's sketch. (c) Fine-tune on real artist sketches labelled with `tools/label_drawing.py`. Keep one data format (image + joints in pixels + head size) throughout so the phases share a pipeline.
  Method (built 2026-11-19: `ml/model.py`, `ml/train_joints.py`, `sandbox/joint_net.py` for numpy inference, `tools/kaggle_train.py`; tests in `tests/test_joint_net.py` and `tests/test_joint_net_torch.py`. Not yet connected to `sketch_detect`):
  - **Model:** a small CNN that outputs one heatmap per joint, read with soft-argmax for sub-pixel accuracy. Its output feeds the existing geometry fit, so the AI proposes and the geometry validates. It is exported to ONNX for the later ports. PyTorch is used for training only (approved 2026-11-19, see Settled).
  - **Left/right:** the loss accepts either side labelling, and the geometry fit decides which side is which.
  - **Mirroring:** generate every viewing angle (it's cheap). In training, mirror about half the samples: flip x, swap `l_` and `r_`, and in 3D negate X, yaw, facing and azimuth. At prediction time, average the predictions for the drawing and its mirror (test-time augmentation). Training only on views from one side was rejected, because deciding whether to flip is the same left/right ambiguity. Needs a test that mirroring twice gives back the original.
  - **Network:** a U-Net with widths 32/64/128/192/256, from 192 px down to 6×6 and back up to 48×48 heatmaps (4 px cells). Each batch norm is folded into its conv when exported. The loss is cross-entropy against a Gaussian (σ = 5 px). Decoding uses a soft-argmax within 2 cells of the peak, and the probability mass in that window is the confidence.
  - **Default run:** 100k drawings generated on Kaggle, 30 epochs, batch 128, AdamW with one-cycle at a learning rate of 2e-3. It keeps the best epoch by synthetic validation error, and scores `test_real.npz` with the numpy model (both plain and with mirror averaging). Errors are given as % of figure size with sides aligned, plus PCK at 3/5/10%.
  - **Style variation:** randomise line thickness, wobble, gaps, overdrawn strokes, scale, rotation and noise, so the model learns body structure rather than one drawing style.
  - **First model (2026-11-19, Kaggle T4, 66 min of training, 30 epochs):**
    - Synthetic validation: 1.7% mean error (0.55% median), 95.5% of joints within 5% of figure size.
    - Real drawings: 4.1% mean. Stock figures 1, 2, 5 and 6: 1.5–1.8%. Figure 3: 5.5%. Figure 4 (sitting, thick strokes): 17.6%. Both pad drawings: 1.6–1.7% (the offline fit failed on both). These are on different scales: the fit's figures are % of figure height, these are % of the larger bounding-box side.
    - Speed: about 0.13 s per drawing with numpy on the laptop, against 10–15 s for the offline fit.
    - Mirror averaging at prediction time made the real drawings worse (4.1% → 4.7%; running pad drawing 1.7% → 8.7%), and so did keeping the more confident pass where the two disagree. It is now off by default (`mirror_too=False`). Retest when more drawings are labelled.
  - **Connected to `sketch_detect` (2026-11-19):**
    - `detect()` uses the trained model when `out/models/joint_net.npz` exists (`model="auto"`); `model=None` forces the geometric fit. The viewer's "auto joints" button and `tools/sketch_to_pose.py` say which method was used. Joints the model is unsure of (confidence < 0.5) are named in the notes.
    - The model's joints are used as they are. A short polish with the geometric fitter made them worse on the real drawings (4.1% → 6.4% with sides aligned), so it isn't used.
    - The head is the round hole's centre where one is found near the model's head.
    - **Sides:** the model leaves the arm and leg sides open. They start from the front-view reading (the limb whose hand or foot is further to image right is the character's left) and swap only if the swap fits a library pose clearly better (`SWAP_PENALTY` = 0.1 per pair, like `PROFILE_PENALTY` for side views). On the 8 real drawings plus the test figure this works best: anatomical error 10.5%, against 12.3% for the front rule alone and 10.0% for free swaps. Free swaps read a plain front figure as a side view. For side-on figures (the running pad drawing) the sides stay a guess, as with the geometric fit.
    - **Time:** about 1–3 s for joint finding, against 18–260 s for the geometric fit on the same drawings. The full `sketch_to_pose` run takes about 10 s.
    - **Still open:** the 3D rebuild after detection. On both pad drawings the joints are right but the rebuilt pose doesn't sit on the drawing: the running figure starts from "squat", and the crawling figure has no horizontal start pose and its scale assumes an upright figure. Fixes: add crawl/plank/lying start poses and a direction-independent size (see the sketch input details). More real labelled drawings, and synthetic sitting poses with thick strokes, should help with stock figure 4.
