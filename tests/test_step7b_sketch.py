"""Step 7b: sketch input (spec 4.2 Method B). Joints placed in 2D -> 3D by ray-sphere intersection."""
import copy
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sandbox.skell import data  # noqa: E402
from sandbox.skell.camera import Camera, Subject, resolve_camera  # noqa: E402
from sandbox.skell.checks import validate  # noqa: E402
from sandbox.skell.pose import pose_from_frame, solve  # noqa: E402
from sandbox.skell.reconstruct import ray_sphere, reconstruct, screen_ray  # noqa: E402

WORLD_TOL = 1e-6


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


def positions_of(skel, frame):
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    return pos


def camera_row(name, skel, frame):
    r = next(c for c in data.load_fixture()["cameras"] if c["name"] == name)
    i = r["input"]
    pos = positions_of(skel, frame)
    req = {k: i[k] for k in ("target", "azimuth", "elevation", "roll", "distance", "focal", "floor_guard")}
    return resolve_camera(req, Subject(skel, pos, 0.0)).camera


def placements(cam, pos):
    out = {}
    for j, P in pos.items():
        p = cam.project(P)
        out[j] = (p.dx, p.dy)
    return out


# ---------------------------------------------------------------- rays and spheres
def test_screen_ray_goes_through_the_point():
    cam = Camera(target=np.array([10.0, 80, -20]), azimuth=37, elevation=-22, roll=15, distance=300, focal=420)
    rng = np.random.default_rng(3)
    for _ in range(50):
        dx, dy = rng.uniform(-250, 250, 2)
        C, D = screen_ray(cam, dx, dy)
        assert np.allclose(C, cam.position) and abs(np.linalg.norm(D) - 1) < 1e-12
        p = cam.project(C + rng.uniform(50, 900) * D)
        assert abs(p.dx - dx) < 1e-9 and abs(p.dy - dy) < 1e-9


def test_ray_sphere_cases():
    C, D = np.zeros(3), np.array([0.0, 0, 1])
    assert np.allclose(sorted(ray_sphere(C, D, np.array([0, 0, 100.0]), 10)), [90, 110])   # two
    assert np.allclose(ray_sphere(C, D, np.array([10.0, 0, 100]), 10), [100])               # tangent: one
    assert ray_sphere(C, D, np.array([20.0, 0, 100]), 10) == []                             # none
    assert ray_sphere(C, D, np.array([0, 0, -100.0]), 10) == []                             # behind the camera


# ---------------------------------------------------------------- round trip (Appendix B.5)
def test_round_trip_at_the_elevated_camera(skel):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("elevated_30", skel, frame)
    rec = reconstruct(skel, cam, placements(cam, pos), frame)
    for j, P in pos.items():
        assert np.allclose(rec.positions[j], P, atol=WORLD_TOL), j
    for bid in skel.posable_ids:
        assert rec.bones[bid].solutions == 2, bid                      # every bone offers two depths
        assert rec.bones[bid].too_long_pct is None
    assert not rec.flags
    for j, P in pos.items():                                            # validated and grounded: unchanged
        assert np.allclose(rec.validation.positions[j], P, atol=WORLD_TOL), j
    for bid, yp in frame["pose"].items():
        assert abs(rec.frame["pose"][bid]["yaw"] - yp["yaw"]) < 1e-4
        assert abs(rec.frame["pose"][bid]["pitch"] - yp["pitch"]) < 1e-4


@pytest.mark.parametrize("name", ["front_eye_level", "right_side", "back_left_three_quarter", "dutch_20",
                                  "low_angle_30_floor_guard_on"])
def test_round_trip_from_other_cameras(skel, name):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row(name, skel, frame)
    rec = reconstruct(skel, cam, placements(cam, pos), frame)
    for j, P in pos.items():
        assert np.allclose(rec.positions[j], P, atol=WORLD_TOL), (name, j)


def test_every_flip_works(skel):
    """Flipping a bone picks its other depth: same 2D point, same bone length, different 3D joint."""
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("elevated_30", skel, frame)
    pl = placements(cam, pos)
    for bid in skel.posable_ids:
        b = skel.bone(bid)
        rec = reconstruct(skel, cam, pl, frame, flips={bid: True})
        moved = rec.positions[b.to_joint]
        assert not np.allclose(moved, pos[b.to_joint], atol=1e-3), bid
        p = cam.project(moved)
        assert abs(p.dx - pl[b.to_joint][0]) < 1e-9 and abs(p.dy - pl[b.to_joint][1]) < 1e-9, bid
        assert abs(np.linalg.norm(moved - rec.positions[b.from_joint]) - b.length) < WORLD_TOL, bid
        assert rec.bones[bid].flipped
        # flipping again returns the original
        back = reconstruct(skel, cam, pl, frame, flips={bid: False})
        assert np.allclose(back.positions[b.to_joint], pos[b.to_joint], atol=WORLD_TOL)


def test_flipping_a_parent_keeps_the_childrens_drawn_points(skel):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("elevated_30", skel, frame)
    pl = placements(cam, pos)
    rec = reconstruct(skel, cam, pl, frame, flips={"r_shoulder>r_elbow": True})
    for j in ("r_elbow", "r_hand"):
        p = cam.project(rec.positions[j])
        assert abs(p.dx - pl[j][0]) < 1e-9 and abs(p.dy - pl[j][1]) < 1e-9


# ---------------------------------------------------------------- editing and flags
def test_moving_one_joint_moves_only_that_joint(skel):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("front_eye_level", skel, frame)
    pl = placements(cam, pos)
    pl["r_hand"] = (pl["r_hand"][0] - 12, pl["r_hand"][1] - 8)
    rec = reconstruct(skel, cam, {"r_hand": pl["r_hand"]}, frame)          # only the changed joint is given
    for j, P in pos.items():
        if j != "r_hand":
            assert np.allclose(rec.positions[j], P, atol=WORLD_TOL), j
    assert not np.allclose(rec.positions["r_hand"], pos["r_hand"])
    assert abs(np.linalg.norm(rec.positions["r_hand"] - rec.positions["r_elbow"]) - 39) < WORLD_TOL


def test_drawn_too_long_is_flagged_above_2_percent(skel):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("front_eye_level", skel, frame)
    f, r, _ = cam.basis()
    for factor, flagged in ((1.01, False), (1.06, True)):
        # a hand drawn in the plane of the elbow, factor x forearm length away: its ray misses the sphere
        P = pos["r_elbow"] + factor * 39.0 * (-r)
        P = P + (cam.depth(pos["r_elbow"]) - cam.depth(P)) * f
        q = cam.project(P)
        rec = reconstruct(skel, cam, {"r_hand": (q.dx, q.dy)}, frame)
        info = rec.bones["r_elbow>r_hand"]
        assert info.solutions == 0
        assert info.too_long_pct is not None and info.too_long_pct > 0
        assert (info.too_long_pct > 2) == flagged, info.too_long_pct
        assert any("r_elbow>r_hand" in msg for msg in rec.flags) == flagged
        assert abs(np.linalg.norm(rec.positions["r_hand"] - rec.positions["r_elbow"]) - 39) < WORLD_TOL


def test_result_is_validated(skel):
    """A forearm drawn bending backward comes out of validation clamped, with the correction recorded."""
    standing = next(p for p in data.pose_library() if p["name"] == "standing")["frame"]
    pos = positions_of(skel, standing)
    cam = resolve_camera({"relative_to": "character", "azimuth": -90, "shot": "full"},
                         Subject(skel, pos, 0.0)).camera                          # right side view
    # draw the right hand behind and below the elbow: from the side that is a backward elbow
    target = pos["r_elbow"] + 39 * np.array([0.0, -0.6, 0.8])
    q = cam.project(target)
    rec = reconstruct(skel, cam, {"r_hand": (q.dx, q.dy)}, standing)
    assert rec.bones["r_elbow>r_hand"].solutions >= 1
    corr = [c for c in rec.validation.limits.corrections if c.bone == "r_elbow>r_hand"]
    assert corr and corr[0].kind == "bend_direction"


def test_rigid_joints_follow_their_parent(skel):
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("front_eye_level", skel, frame)
    rec = reconstruct(skel, cam, {"l_shoulder": (999.0, 999.0)}, frame)   # placing a zero-width offset is ignored
    assert np.allclose(rec.positions["l_shoulder"], rec.positions["neck"], atol=WORLD_TOL)
    assert any("l_shoulder" in n for n in rec.notes)


# ---------------------------------------------------------------- viewer
def test_viewer_drag_and_flip(tmp_path):
    from sandbox.app import App, ViewerState
    import matplotlib.pyplot as plt
    under = tmp_path / "sketch.png"
    fig = plt.figure(figsize=(3, 2))
    fig.gca().plot([0, 1], [0, 1])
    fig.savefig(under)
    plt.close(fig)
    app = App(ViewerState(underlay=str(under)), show=False)
    try:
        assert app.state.underlay is not None
        app.toggle("sketch: drag joints")
        assert app.state.sketch_mode
        before = {j: p.copy() for j, p in app.state.subject.positions.items()}
        cam = app.state.resolved().camera
        hand = cam.project(before["r_hand"])
        app.drag_joint("r_hand", hand.dx - 10, hand.dy - 6)
        after = app.state.subject.positions
        assert not np.allclose(after["r_hand"], before["r_hand"])
        assert np.allclose(after["neck"] - after["pelvis"], before["neck"] - before["pelvis"], atol=1e-6)
        assert "sketch" in app.status_text()
        moved = after["r_hand"].copy()
        app.flip("r_elbow>r_hand")
        rec = app.state.last_rec
        flipped = rec.positions["r_hand"]                           # raw reconstruction: the other depth
        assert rec.bones["r_elbow>r_hand"].flipped and not np.allclose(flipped, moved, atol=1e-3)
        q = cam.project(flipped)
        assert abs(q.dx - (hand.dx - 10)) < 1e-6 and abs(q.dy - (hand.dy - 6)) < 1e-6   # same drawn point
        # that depth bends the elbow backward, so validation clamps it and says so
        corr = [c for c in rec.validation.limits.corrections if c.bone == "r_elbow>r_hand"]
        assert corr and corr[0].kind == "bend_direction"
        assert "clamped r_elbow>r_hand" in app.status_text()
        app.select_pose("reference_kneeling")                       # re-selecting restores the library pose
        assert np.allclose(app.state.subject.positions["r_hand"], before["r_hand"], atol=1e-9)
        app.save(data.ROOT / "out" / "viewer_sketch.png")
    finally:
        app.close()


# ---------------------------------------------------------------- guided placement
def test_placement_order(skel):
    from sandbox.skell.reconstruct import placement_order
    order = placement_order(skel)
    assert order == ["pelvis", "neck", "head", "l_elbow", "l_hand", "r_elbow", "r_hand",
                     "l_knee", "l_foot", "r_knee", "r_foot"]


def _guided_app(start_pose="walk"):
    from sandbox.app import App
    app = App(show=False)
    app.select_pose(start_pose)
    app.toggle("full shot")
    return app


def test_guided_clicks_rebuild_the_pose_from_its_drawing(skel):
    """Clicking every joint where a pose's drawing shows it rebuilds that pose exactly."""
    app = _guided_app("walk")
    try:
        cam = app.state.resolved().camera
        target = {j: cam.project(P) for j, P in app.state.subject.positions.items()}
        before = {j: P.copy() for j, P in app.state.subject.positions.items()}
        app.start_placement()
        assert "click the pelvis" in app.status_text()
        for j in app.placement_order:
            assert app.next_joint == j
            app.place_click(target[j].dx, target[j].dy)
        assert app.next_joint is None and not app.placing
        for j, P in before.items():
            assert np.allclose(app.state.subject.positions[j], P, atol=1e-6), j
        assert app.state.edited
    finally:
        app.close()


def test_guided_clicks_from_another_pose(skel):
    """Starting from one pose and clicking another pose's drawing: every joint lands on its click,
    every bone keeps its length."""
    app = _guided_app("standing")
    try:
        cam = app.state.resolved().camera
        guard = next(p for p in data.pose_library() if p["name"] == "guard")["frame"]
        gpos = positions_of(skel, guard)
        clicks = {j: cam.project(P) for j, P in gpos.items()}
        app.start_placement()
        for j in app.placement_order:
            app.place_click(clicks[j].dx, clicks[j].dy)
        rec = app.state.last_rec
        for j in app.placement_order:
            if j == "pelvis":
                continue
            q = cam.project(rec.positions[j])
            info = next(b for b in rec.bones.values() if skel.bone(b.bone).to_joint == j)
            if info.solutions:                                             # reachable: exactly on the click
                assert abs(q.dx - clicks[j].dx) < 1e-6 and abs(q.dy - clicks[j].dy) < 1e-6, j
        for b in skel.bones:
            P0, P1 = rec.positions[b.from_joint], rec.positions[b.to_joint]
            assert abs(np.linalg.norm(P1 - P0) - b.length) < 1e-6, b.id
    finally:
        app.close()


def test_guided_undo_and_skip(skel):
    app = _guided_app("walk")
    try:
        before = {j: P.copy() for j, P in app.state.subject.positions.items()}
        app.start_placement()
        app.place_click(0.0, 0.0)
        assert app.next_joint == "neck"
        app.undo_click()
        assert app.next_joint == "pelvis"
        for _ in app.placement_order:
            app.skip_joint()                                   # skipping keeps each joint where it is
        assert not app.placing
        for j, P in before.items():
            assert np.allclose(app.state.subject.positions[j], P, atol=1e-6), j
    finally:
        app.close()


def test_guided_cancel_and_side_inset(skel):
    app = _guided_app("walk")
    try:
        app.toggle("sketch: drag joints")
        assert app.side_inset is not None                       # depth check while sketching
        app.start_placement()
        app.place_click(5.0, 5.0)
        app.cancel_placement()
        assert not app.placing and app.next_joint is None
        app.save(data.ROOT / "out" / "viewer_guided.png")
    finally:
        app.close()


def test_guided_result_sits_on_the_sketch(skel, tmp_path):
    """The real workflow: load a rough sketch, press place joints, click its joints. The finished figure is
    drawn exactly over the clicks (camera held, root slid to the floor so grounding moves nothing), with
    the character's own bone lengths, and passes validation."""
    import importlib.util
    from sandbox.app import App, ViewerState
    spec = importlib.util.spec_from_file_location("mss", data.ROOT / "tools" / "make_sample_sketch.py")
    mss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mss)
    clicks = mss.make(tmp_path / "sketch.png")                    # a right hook, drawn 6% short, wobbly
    app = App(ViewerState(underlay=str(tmp_path / "sketch.png")), show=False)   # starts on the reference pose
    try:
        cam = app.state.resolved().camera
        app.start_placement()
        for j in app.placement_order:
            app.place_click(*clicks[j])
        held = app.state.resolved().camera
        assert np.allclose(held.position, cam.position)                   # the view did not move
        rec = app.state.last_rec
        assert not rec.flags and not rec.notes
        v = app.state.validation
        assert v.limits.ok, [str(c) for c in v.limits.corrections]
        for j, (x, y) in clicks.items():
            q = held.project(app.state.subject.positions[j])
            assert abs(q.dx - x) < 1e-6 and abs(q.dy - y) < 1e-6, j
        assert min(p[1] for p in app.state.subject.positions.values()) == pytest.approx(0, abs=1e-9)
        for b in skel.bones:
            P0, P1 = app.state.subject.positions[b.from_joint], app.state.subject.positions[b.to_joint]
            assert abs(np.linalg.norm(P1 - P0) - b.length) < 1e-6
        assert "camera held still" in app.status_text()
        app.save(data.ROOT / "out" / "viewer_guided_done.png")
        app.set_slider("azimuth", 30)                                      # a camera control releases the hold
        assert app.state.camera_override is None
    finally:
        app.close()


def test_sketch_that_cannot_fit_this_camera_says_so(skel):
    """A drawing whose feet are above the horizon cannot stand on this camera's floor: grounding moves the
    figure, and Skell says why. A drawing that is merely lower is fitted by bringing the figure closer."""
    from sandbox.skell.reconstruct import placement_order
    frame = data.reference_frame()
    pos = positions_of(skel, frame)
    cam = camera_row("front_eye_level", skel, frame)
    pl = {j: xy for j, xy in placements(cam, pos).items() if j in placement_order(skel)}
    lower = reconstruct(skel, cam, {j: (x, y + 60) for j, (x, y) in pl.items()}, frame)
    assert not lower.notes                                               # fitted: feet on the floor, no shift
    for j, (x, y) in pl.items():
        q = cam.project(lower.validation.positions[j])
        assert abs(q.dx - x) < 1e-6 and abs(q.dy - (y + 60)) < 1e-6, j
    raised = reconstruct(skel, cam, {j: (x, y - 200) for j, (x, y) in pl.items()}, frame)
    assert any("does not reach the floor" in n for n in raised.notes)
    assert min(p[1] for p in raised.validation.positions.values()) == pytest.approx(0, abs=1e-9)


def test_gesture_fit_reads_a_stylised_drawing(skel):
    """A stick figure in its own proportions (long arms, long legs): gesture fit keeps every drawn bone
    direction, uses the character's lengths, and flags nothing as drawn too long."""
    hook = next(p for p in data.pose_library() if p["name"] == "right_hook")["frame"]
    pos = positions_of(skel, hook)
    cam = camera_row("front_eye_level", skel, data.reference_frame())
    stretch = {"l_shoulder>l_elbow": 1.5, "l_elbow>l_hand": 1.9, "r_shoulder>r_elbow": 1.5,
               "r_elbow>r_hand": 1.9, "l_hip>l_knee": 1.3, "l_knee>l_foot": 1.35, "r_hip>r_knee": 1.3,
               "r_knee>r_foot": 1.35, "neck>head": 1.4}
    drawn = {skel.root: np.array([cam.project(pos[skel.root]).dx, cam.project(pos[skel.root]).dy])}
    for b in skel.bones:                                 # draw each bone stretched along its own 2D direction
        a, c = cam.project(pos[b.from_joint]), cam.project(pos[b.to_joint])
        v = np.array([c.dx - a.dx, c.dy - a.dy])
        drawn[b.to_joint] = drawn[b.from_joint] + stretch.get(b.id, 1.0) * v
    from sandbox.skell.reconstruct import placement_order
    pl = {j: tuple(drawn[j]) for j in placement_order(skel)}
    exact = reconstruct(skel, cam, pl, hook)
    assert exact.flags                                   # read literally, the long limbs are "too long"
    rec = reconstruct(skel, cam, pl, hook, fit="gesture")
    assert not rec.flags
    for b in skel.bones:
        if b.kind != "posable":
            continue
        P0, P1 = rec.positions[b.from_joint], rec.positions[b.to_joint]
        assert abs(np.linalg.norm(P1 - P0) - b.length) < 1e-6, b.id
        q0, q1 = cam.project(P0), cam.project(P1)
        got = math.degrees(math.atan2(q1.dy - q0.dy, q1.dx - q0.dx))
        want = math.degrees(math.atan2(*(drawn[b.to_joint] - drawn[b.from_joint])[::-1]))
        assert abs((got - want + 180) % 360 - 180) < 1e-6, (b.id, got, want)   # same drawn direction


def test_gesture_fit_aims_limbs_from_the_drawn_neck_and_pelvis(skel):
    """Upper arms and thighs hang off rigid joints (shoulders, hips) that are never placed. Gesture fit must
    measure them from where the drawing puts those joints, not from where the start pose had them: a
    drawing far from the start pose (another pose, shifted on screen) still keeps every drawn direction."""
    hook = next(p for p in data.pose_library() if p["name"] == "right_hook")["frame"]
    standing = next(p for p in data.pose_library() if p["name"] == "standing")["frame"]
    pos = positions_of(skel, hook)
    cam = camera_row("front_eye_level", skel, data.reference_frame())
    shift = np.array([-90.0, 40.0])
    drawn = {j: np.array([cam.project(P).dx, cam.project(P).dy]) + shift for j, P in pos.items()}
    from sandbox.skell.reconstruct import placement_order
    pl = {j: tuple(drawn[j]) for j in placement_order(skel)}
    rec = reconstruct(skel, cam, pl, standing, fit="gesture")
    for b in skel.bones:
        if b.kind != "posable":
            continue
        q0, q1 = cam.project(rec.positions[b.from_joint]), cam.project(rec.positions[b.to_joint])
        got = math.degrees(math.atan2(q1.dy - q0.dy, q1.dx - q0.dx))
        want = math.degrees(math.atan2(*(drawn[b.to_joint] - drawn[b.from_joint])[::-1]))
        assert abs((got - want + 180) % 360 - 180) < 1e-6, (b.id, got, want)
