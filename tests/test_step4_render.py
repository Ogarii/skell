"""Step 4: drawing (spec 9). Renders every fixture camera to out/ and checks what was drawn."""
import copy
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sandbox.skell import data  # noqa: E402
from sandbox.skell.camera import Camera, Subject, resolve_camera  # noqa: E402
from sandbox.skell.pose import pose_from_frame, solve  # noqa: E402
from sandbox.skell.render import Box, RenderOptions, bone_tilt, render  # noqa: E402

OUT = data.ROOT / "out"


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


def subject_from(skel, frame):
    pose = pose_from_frame(skel, frame)
    pos, _ = solve(skel, pose)
    return Subject(skel=skel, positions=pos, facing=pose.facing)


@pytest.fixture(scope="module")
def kneeling(skel):
    return subject_from(skel, data.reference_frame())


@pytest.fixture(scope="module")
def standing(skel):
    rest = {"pelvis>neck": (0, 90), "neck>head": (0, 90)}
    frame = {"t": 1, "root": {"x": 0, "z": 0, "facing": 0}, "facing": {"chest": 0, "head": 0}, "airborne": False,
             "pose": {b: {"yaw": rest.get(b, (0, -90))[0], "pitch": rest.get(b, (0, -90))[1]}
                      for b in skel.posable_ids}}
    return subject_from(skel, frame)


def camera_for(name, subject):
    r = next(c for c in data.load_fixture()["cameras"] if c["name"] == name)
    i = r["input"]
    req = {k: i[k] for k in ("target", "azimuth", "elevation", "roll", "distance", "focal", "floor_guard")}
    return resolve_camera(req, subject).camera


def assert_depth_order(drawing):
    items = sorted(drawing.items, key=lambda it: it.zorder)
    keys = [it.sort_depth for it in items]
    assert all(a >= b for a, b in zip(keys, keys[1:])), "items must be drawn farthest first"


# ---------------------------------------------------------------- fixture cameras
def test_renders_every_fixture_camera(kneeling):
    OUT.mkdir(exist_ok=True)
    for row in data.load_fixture()["cameras"]:
        cam = camera_for(row["name"], kneeling)
        path = OUT / f"{row['name']}.png"
        drawing = render(path, cam, [kneeling], options=RenderOptions(title=row["name"].replace("_", " ")))
        assert path.exists() and path.stat().st_size > 5000, row["name"]
        assert sum(it.kind == "bone" for it in drawing.items) == 10, row["name"]   # 14 bones, 4 zero-width
        assert sum(it.kind == "joint" for it in drawing.items) == 10, row["name"]  # 15 joints, 4 coincide, head
        assert sum(it.kind == "head" for it in drawing.items) == 1
        assert_depth_order(drawing)


def test_overlay_versions_render(kneeling):
    for name in ("front_eye_level", "elevated_30", "overhead_90", "distant_eye_level"):
        cam = camera_for(name, kneeling)
        render(OUT / f"{name}_overlays.png", cam, [kneeling],
               options=RenderOptions(overlays=True, title=name.replace("_", " ") + " (overlays)"))


# ---------------------------------------------------------------- grid and horizon
def test_grid_dashed_only_from_below(kneeling):
    below = render(None, camera_for("worms_eye_75_floor_guard_off", kneeling), [kneeling])
    above = render(None, camera_for("front_eye_level", kneeling), [kneeling])
    assert below.grid_dashed and not above.grid_dashed
    assert below.grid_lines > 0 and above.grid_lines > 0


def test_horizon(kneeling):
    front = render(None, camera_for("front_eye_level", kneeling), [kneeling])
    assert front.horizon is not None
    (x0, y0), (x1, y1) = front.horizon
    assert abs(y0) < 1e-9 and abs(y1) < 1e-9            # eye level: through the image centre, level
    elevated = render(None, camera_for("elevated_30", kneeling), [kneeling])
    (_, y0), (_, y1) = elevated.horizon
    assert y0 < 0 and abs(y0 - y1) < 1e-9                 # looking down: horizon above the centre
    assert abs(y0 + 500 * math.tan(math.radians(30))) < 1e-6
    assert render(None, camera_for("overhead_90", kneeling), [kneeling]).horizon is None


def test_dutch_rotates_image_clockwise(kneeling):
    d = render(None, camera_for("dutch_20", kneeling), [kneeling])
    (x0, y0), (x1, y1) = d.horizon
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180  # screen y down: + is clockwise; a line has no direction
    assert abs(angle - 20) < 1e-6


# ---------------------------------------------------------------- overlays
def test_overlay_colours_standing_front(standing):
    cam = resolve_camera({}, standing).camera
    d = render(None, cam, [standing], options=RenderOptions(overlays=True))
    bones = [it for it in d.items if it.kind == "bone"]
    assert bones and all(it.color == "teal" for it in bones)    # vertical bones lie in the image plane
    assert all(abs(it.tilt) < 1e-9 for it in bones)
    assert d.ghosts == 0


def test_overlay_colours_follow_tilt(kneeling):
    cam = camera_for("front_eye_level", kneeling)
    flat = data.load_rules()["render"]["flat_tilt_deg"]
    d = render(None, cam, [kneeling], options=RenderOptions(overlays=True))
    amber = 0
    for it in (i for i in d.items if i.kind == "bone"):
        b = kneeling.skel.bone(it.name)
        tilt = bone_tilt(cam, kneeling.positions[b.from_joint], kneeling.positions[b.to_joint])
        assert abs(it.tilt - tilt) < 1e-9
        assert it.color == ("teal" if abs(tilt) < flat else "amber"), it.name
        amber += it.color == "amber"
    assert amber > 0 and d.ghosts == amber and d.badges == amber


def test_bone_tilt_sign():
    cam = Camera(target=np.array([0.0, 0, 0]), azimuth=0, elevation=0, distance=400)   # looks along +Z
    assert abs(bone_tilt(cam, [0, 0, 0], [0, 0, 10]) - 90) < 1e-9     # straight away from the camera
    assert abs(bone_tilt(cam, [0, 0, 0], [0, 0, -10]) + 90) < 1e-9    # straight toward it
    assert abs(bone_tilt(cam, [0, 0, 0], [10, 0, 10]) - 45) < 1e-9
    assert abs(bone_tilt(cam, [0, 0, 0], [10, 5, 0])) < 1e-9


def test_overlays_off_by_default(kneeling):
    d = render(None, camera_for("front_eye_level", kneeling), [kneeling])
    assert all(it.color == "bone" for it in d.items if it.kind == "bone")
    assert d.ghosts == 0 and d.badges == 0


# ---------------------------------------------------------------- props, clipping, two characters
def test_box_props(kneeling):
    box = Box(x=-40, z=-160, width=60, depth=50, height=40)
    for name in ("front_eye_level", "elevated_30", "front_right_three_quarter"):
        d = render(OUT / f"{name}_with_box.png", camera_for(name, kneeling), [kneeling], boxes=[box],
                   options=RenderOptions(title=f"{name.replace('_', ' ')} with box"))
        faces = [it for it in d.items if it.kind == "box_face"]
        assert len(faces) == 6
        assert_depth_order(d)
    corners = box.corners()
    assert np.allclose(corners[:, 1].min(), 0) and np.allclose(corners[:, 1].max(), 40)   # stands on the floor


def test_near_plane_clipping_does_not_break(kneeling):
    # camera 30 units in front of the head: the reaching right hand is behind the camera
    cam = Camera(target=kneeling.positions["head"], azimuth=0, elevation=0, distance=30)
    d = render(OUT / "near_plane_clip.png", cam, [kneeling], options=RenderOptions(title="near plane clip"))
    bones = [it for it in d.items if it.kind == "bone"]
    assert 0 < len(bones) <= 10
    assert all(it.w >= cam.near - 1e-9 for it in d.items if it.kind == "joint")


def test_two_characters(skel, kneeling):
    frame = copy.deepcopy(data.reference_frame())
    frame["root"].update({"x": 160, "z": 40, "facing": -60})
    other = subject_from(skel, frame)
    cam = resolve_camera({"elevation": 15, "distance": 600}, kneeling).camera
    d = render(OUT / "two_characters.png", cam, [kneeling, other], options=RenderOptions(title="two characters"))
    assert sum(it.kind == "bone" for it in d.items) == 20
    assert sum(it.kind == "head" for it in d.items) == 2
    assert_depth_order(d)
