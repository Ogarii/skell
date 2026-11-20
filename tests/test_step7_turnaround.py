"""Step 7: turnaround sheet (spec 8.7). Eight character-relative views every 45 degrees at the current
elevation and distance, one PNG, with a height ruler. Each view must match turning the character instead."""
import copy
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sandbox.skell import data  # noqa: E402
from sandbox.skell.camera import Camera, Subject  # noqa: E402
from sandbox.skell.checks import validate  # noqa: E402
from sandbox.skell.pose import pose_from_frame, pose_to_frame, turn_frame  # noqa: E402
from sandbox.skell.turnaround import VIEWS, render_turnaround, turnaround_cameras  # noqa: E402

OUT = data.ROOT / "out"


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


def subject_from(skel, frame):
    pose = pose_from_frame(skel, frame)
    return Subject(skel, validate(skel, pose).positions, pose.facing)


def lib(name):
    return next(p for p in data.pose_library() if p["name"] == name)


# ---------------------------------------------------------------- turning helper
def test_turn_frame_turns_root_chest_and_head(skel):
    fr = lib("right_hook")["frame"]                     # chest facing -30: not the same as the root
    t = turn_frame(fr, 90)
    assert t["root"]["facing"] == 90 and t["facing"] == {"chest": 60, "head": 90}
    assert t["pose"] == fr["pose"]                      # bone yaw/pitch are in the heading frame: unchanged
    assert fr["root"]["facing"] == 0                    # the input is not modified
    back = pose_to_frame(skel, pose_from_frame(skel, t))
    assert math.isclose(back["facing"]["chest"], 60, abs_tol=1e-6)


def test_turned_pose_is_the_same_pose_rotated(skel):
    for p in data.pose_library():
        a = subject_from(skel, p["frame"])
        b = subject_from(skel, turn_frame(p["frame"], 135))
        phi = math.radians(135)
        for j, P in a.positions.items():
            x, y, z = P
            rot = (x * math.cos(phi) - z * math.sin(phi), y, z * math.cos(phi) + x * math.sin(phi))
            assert np.allclose(b.positions[j], rot, atol=1e-9), (p["name"], j)


# ---------------------------------------------------------------- cameras
@pytest.mark.parametrize("framing", ["full", "current"])
def test_eight_views_every_45_degrees(skel, framing):
    subj = subject_from(skel, data.reference_frame())
    views = turnaround_cameras(subj, elevation=15, distance=500, framing=framing)
    assert [v.azimuth_rel for v in views] == [0, 45, 90, 135, 180, -135, -90, -45]
    presets = data.load_rules()["camera"]["presets"]
    for v in views:
        assert presets[v.name]["azimuth_rel"] == v.azimuth_rel
        assert v.camera.elevation == 15
    cams = [v.camera for v in views]
    assert len({round(c.distance, 9) for c in cams}) == 1 and len({c.focal for c in cams}) == 1
    assert all(np.allclose(c.target, cams[0].target) for c in cams)          # one orbit, one scale
    if framing == "current":
        assert cams[0].distance == 500 and np.allclose(cams[0].target, subj.positions["head"])


@pytest.mark.parametrize("name", ["reference_kneeling", "right_hook", "sit_floor", "run"])
@pytest.mark.parametrize("framing", ["full", "current"])
def test_each_view_matches_turning_the_character(skel, name, framing):
    """Camera at alpha_rel = a on the unturned character == front camera on the character turned by -a."""
    fr = lib(name)["frame"]
    subj = subject_from(skel, fr)
    views = turnaround_cameras(subj, elevation=20, distance=450, framing=framing, floor_guard=False)
    for v in views:
        turned = subject_from(skel, turn_frame(fr, -v.azimuth_rel))
        # a fixed camera at world azimuth 0, same target point on the character, distance and focal length
        target = turned.positions["head"] if framing == "current" else v.camera.target   # the axis point stays put
        fixed = Camera(target=np.asarray(target), azimuth=0, elevation=20, distance=v.camera.distance,
                       focal=v.camera.focal)
        for j, P in subj.positions.items():
            a, b = v.camera.project(P), fixed.project(turned.positions[j])
            assert abs(a.dx - b.dx) < 1e-9 and abs(a.dy - b.dy) < 1e-9 and abs(a.s - b.s) < 1e-12, (v.name, j)


def test_turnaround_is_character_relative(skel):
    """A character already turned to face 90 gets the same sheet."""
    fr = data.reference_frame()
    a = turnaround_cameras(subject_from(skel, fr), framing="full")
    t = subject_from(skel, turn_frame(fr, 90))
    b = turnaround_cameras(t, framing="full")
    s = subject_from(skel, fr)
    for va, vb in zip(a, b):
        for j in s.positions:
            pa, pb = va.camera.project(s.positions[j]), vb.camera.project(t.positions[j])
            assert abs(pa.dx - pb.dx) < 1e-9 and abs(pa.dy - pb.dy) < 1e-9


def test_full_framing_keeps_every_view_in_frame(skel):
    cr = data.load_rules()["camera"]
    for p in data.pose_library():
        subj = subject_from(skel, p["frame"])
        for v in turnaround_cameras(subj, elevation=10, framing="full"):
            for j, P in subj.positions.items():
                q = v.camera.project(P)
                assert q.visible and abs(q.dx) <= 0.8 * cr["canvas_width"] / 2 + 1e-6, (p["name"], v.name, j)
                assert abs(q.dy) <= 0.8 * cr["canvas_height"] / 2 + 1e-6, (p["name"], v.name, j)


# ---------------------------------------------------------------- sheet
@pytest.mark.parametrize("name", ["reference_kneeling", "standing", "right_hook"])
def test_sheet_renders(skel, name):
    p = lib(name)
    subj = subject_from(skel, p["frame"])
    path = OUT / f"turnaround_{name}.png"
    sheet = render_turnaround(path, subj, title=p["label"], elevation=10)
    assert path.exists() and path.stat().st_size > 50000
    assert len(sheet.views) == 8
    for v in sheet.views:
        assert sum(it.kind == "bone" for it in v.drawing.items) == 10, v.name
        assert sum(it.kind == "head" for it in v.drawing.items) == 1


def test_height_ruler(skel):
    subj = subject_from(skel, lib("standing")["frame"])
    sheet = render_turnaround(None, subj, elevation=0)
    head_unit = 2 * skel.head_radius
    assert sheet.ruler.head_unit == head_unit
    assert [t for t in sheet.ruler.ticks] == [k * head_unit for k in range(int(250 // head_unit) + 1)] + [250.0]
    assert math.isclose(sheet.ruler.heads_tall, 250 / 38, abs_tol=1e-12)
    # in the front view at eye level, the ruler's top lines up with the top of the head
    front = sheet.views[0]
    top = front.camera.project(sheet.ruler.points(front.camera)[1])
    head_top = front.camera.project(subj.positions["head"] + np.array([0, skel.head_radius, 0]))
    assert abs(top.dy - head_top.dy) < 0.5


def test_viewer_exports_turnaround(skel):
    from sandbox.app import App
    app = App(show=False)
    try:
        app.select_pose("guard")
        app.press("elevated")
        path = app.export_turnaround()
        assert path.exists() and path.name == "turnaround_guard.png"
        assert "turnaround" in app.status_text()
    finally:
        app.close()
