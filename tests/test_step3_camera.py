"""Step 3: 360-degree camera (spec 8.1-8.5), checked against the golden fixture."""
import copy
import math

import numpy as np
import pytest

from sandbox.skell import data
from sandbox.skell.camera import (
    Camera,
    Subject,
    basis,
    over_the_shoulder,
    resolve_camera,
)
from sandbox.skell.pose import pose_from_frame, solve

SCREEN_TOL = 0.01
WORLD_TOL = 1e-6


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


@pytest.fixture(scope="module")
def fixture():
    return data.load_fixture()


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


def row(fixture, name):
    return next(c for c in fixture["cameras"] if c["name"] == name)


def request_from_row(r):
    i = r["input"]
    return {"target": i["target"], "azimuth": i["azimuth"], "elevation": i["elevation"], "roll": i["roll"],
            "distance": i["distance"], "focal": i["focal"], "floor_guard": i["floor_guard"]}


def assert_offsets(cam, subject, expected):
    for joint, exp in expected.items():
        p = cam.project(subject.positions[joint])
        for k in ("dx", "dy", "s"):
            assert abs(getattr(p, k) - exp[k]) < SCREEN_TOL, (joint, k, getattr(p, k), exp[k])
        assert abs(p.w - exp["w"]) < WORLD_TOL, (joint, "w")


# ---------------------------------------------------------------- basis
def test_basis_orthonormal_and_right_handed_full_sphere():
    for az in range(0, 360, 15):
        for el in range(-90, 91, 15):
            f, r, up = basis(az, el)
            for x in (f @ r, f @ up, r @ up, f @ f - 1, r @ r - 1, up @ up - 1):
                assert abs(x) < 1e-12, (az, el)
            assert np.allclose(np.cross(r, up), f, atol=1e-12), (az, el)  # never mirrored


def test_no_image_flip_at_the_poles():
    for az in range(0, 360, 15):
        for el in (-90, 90):
            near_pole = el - 15 * np.sign(el)
            _, r0, up0 = basis(az, near_pole)
            _, r1, up1 = basis(az, el)
            assert up0 @ up1 > 0 and r0 @ r1 > 0.999999, (az, el)


# ---------------------------------------------------------------- fixture rows
def test_all_fixture_camera_rows(fixture, kneeling):
    assert len(fixture["cameras"]) == 12
    for r in fixture["cameras"]:
        res = resolve_camera(request_from_row(r), kneeling)
        assert np.allclose(res.camera.position, r["camera_position"], atol=WORLD_TOL), r["name"]
        assert_offsets(res.camera, kneeling, r["screen_offsets"])


def test_head_lands_on_centre_with_scale_1_22(fixture, kneeling):
    for r in fixture["cameras"]:
        p = resolve_camera(request_from_row(r), kneeling).camera.project(kneeling.positions["head"])
        assert abs(p.dx) < 1e-9 and abs(p.dy) < 1e-9
        expected = 0.61 if r["input"]["distance"] == 820 else 1.22
        assert abs(p.s - expected) < 0.005, r["name"]


def test_near_far_scale_ratio(fixture, kneeling):
    expected = fixture["checks"]["near_far_scale_ratio_r_hand_over_l_foot"]
    for name in ("front_eye_level", "distant_eye_level"):
        cam = resolve_camera(request_from_row(row(fixture, name)), kneeling).camera
        ratio = cam.project(kneeling.positions["r_hand"]).s / cam.project(kneeling.positions["l_foot"]).s
        assert abs(ratio - expected[name]) < 1e-6   # stored value is rounded once to 6 decimals
    # spec 17.A: about 1.56 at eye level and 1.27 at double distance
    assert abs(expected["front_eye_level"] - 1.56) < 0.005
    assert abs(expected["distant_eye_level"] - 1.27) < 0.005


# ---------------------------------------------------------------- character-relative azimuth
def test_turned_character_seen_from_alpha_rel_0_matches_front_row(skel, fixture):
    frame = copy.deepcopy(data.reference_frame())
    frame["root"]["facing"] = 90
    turned = subject_from(skel, frame)
    res = resolve_camera({"relative_to": "character", "azimuth": 0, "elevation": 0, "distance": 410}, turned)
    assert math.isclose(res.camera.azimuth, 90, abs_tol=1e-9)
    assert_offsets(res.camera, turned, row(fixture, "front_eye_level")["screen_offsets"])


def test_wrap_and_clamp(kneeling):
    cases = [
        ({"azimuth": 270}, ("azimuth", -90)),
        ({"azimuth": -180}, ("azimuth", 180)),
        ({"azimuth": 540}, ("azimuth", 180)),
        ({"elevation": 120, "floor_guard": False}, ("elevation", 90)),
        ({"elevation": -100, "floor_guard": False}, ("elevation", -90)),
        ({"roll": 200}, ("roll", -160)),
    ]
    for req, (key, value) in cases:
        cam = resolve_camera(req, kneeling).camera
        assert math.isclose(getattr(cam, key), value, abs_tol=1e-9), (req, getattr(cam, key))


def test_character_relative_wraps(skel):
    frame = copy.deepcopy(data.reference_frame())
    frame["root"]["facing"] = 30
    subj = subject_from(skel, frame)
    cam = resolve_camera({"relative_to": "character", "azimuth": 170}, subj).camera
    assert math.isclose(cam.azimuth, -160, abs_tol=1e-9)


def test_null_keeps_current_value(kneeling):
    current = resolve_camera({"azimuth": 45, "elevation": 20, "roll": 10, "distance": 600, "focal": 700}, kneeling).camera
    cam = resolve_camera({"azimuth": None, "elevation": -10, "roll": None, "distance": None, "focal": None},
                         kneeling, current=current).camera
    assert (cam.azimuth, cam.roll, cam.distance, cam.focal) == (45, 10, 600, 700)
    assert cam.elevation == -10


def test_distance_and_focal_limits(kneeling):
    res = resolve_camera({"distance": 5, "focal": 100}, kneeling)
    assert res.camera.distance == 30 and res.camera.focal == 150 and res.notes
    assert resolve_camera({"focal": 5000}, kneeling).camera.focal == 2000


# ---------------------------------------------------------------- floor guard
def test_floor_guard_low_angle(fixture, kneeling):
    res = resolve_camera({"azimuth": 0, "elevation": -30, "distance": 410, "focal": 500}, kneeling)
    ex = fixture["floor_guard_example"]["resolved"]
    assert abs(res.camera.distance - ex["distance"]) < WORLD_TOL
    assert abs(res.camera.focal - ex["focal"]) < WORLD_TOL
    assert abs(res.camera.position[1] - ex["camera_height"]) < WORLD_TOL
    assert abs(res.camera.distance - 214.314814) < 1e-6 and abs(res.camera.focal - 261.359529) < 1e-6
    assert res.camera.elevation == -30 and res.camera.azimuth == 0      # the angle is kept
    assert any("floor guard" in n for n in res.notes)                   # the editor says what changed
    assert_offsets(res.camera, kneeling, row(fixture, "low_angle_30_floor_guard_on")["screen_offsets"])
    # the target keeps its on-screen size
    unguarded = resolve_camera({"elevation": -30, "distance": 410, "floor_guard": False}, kneeling).camera
    head = kneeling.positions["head"]
    assert abs(res.camera.project(head).s - unguarded.project(head).s) < 1e-12


def test_floor_guard_off_allows_camera_below_floor(fixture, kneeling):
    r = row(fixture, "worms_eye_75_floor_guard_off")
    res = resolve_camera(request_from_row(r), kneeling)
    assert res.camera.distance == 410 and res.camera.position[1] < -283
    assert not res.notes


def test_floor_guard_never_leaves_camera_below_floor(kneeling):
    h = 0.02 * 250
    for az in range(-180, 181, 30):
        for el in range(-90, 91, 15):
            res = resolve_camera({"azimuth": az, "elevation": el, "distance": 410}, kneeling)
            assert res.camera.position[1] >= h - 1e-9, (az, el)


def test_floor_guard_target_on_floor_warns(kneeling):
    # r_foot is on the floor: no point on the line looking up at it is above the floor
    res = resolve_camera({"target": "r_foot", "elevation": -30, "distance": 410}, kneeling)
    assert res.camera.elevation == -30 and res.camera.distance == 410
    assert any("raise the target" in w for w in res.warnings)


# ---------------------------------------------------------------- shots
def test_full_shot_standing_reference(standing, fixture):
    res = resolve_camera({"shot": "full"}, standing)
    assert abs(res.camera.distance - fixture["checks"]["full_shot_distance_standing_250_canvas_380"]) < 1e-6
    assert abs(res.camera.distance - 411.184211) < 1e-6
    assert np.allclose(res.camera.target, [0, 125, 0], atol=1e-9)    # bounding-box centre


def test_shot_sizes(standing, kneeling):
    order = ["close_up", "medium", "full", "wide", "extreme_wide"]
    dists = [resolve_camera({"shot": s}, standing).camera.distance for s in order]
    assert dists == sorted(dists)
    assert np.allclose(resolve_camera({"shot": "close_up"}, standing).camera.target, standing.positions["head"])
    # a kneeling figure gets a closer full shot than a standing one
    assert resolve_camera({"shot": "full"}, kneeling).camera.distance < dists[2]


def test_over_the_shoulder():
    H = 250.0
    other_head = np.array([0.0, 170.0, -200.0])
    shoulder = np.array([10.0, 150.0, 0.0])
    az, el, R, C = over_the_shoulder(other_head, shoulder, H, data.load_rules())
    away = (shoulder - other_head) / np.linalg.norm(shoulder - other_head)
    expected_C = shoulder + 0.15 * H * away + np.array([0, 0.05 * H, 0])
    assert np.allclose(C, expected_C, atol=1e-9)
    cam = Camera(target=other_head, azimuth=az, elevation=el, distance=R)
    assert np.allclose(cam.position, expected_C, atol=1e-9)


def test_over_the_shoulder_resolves_with_two_characters(skel, kneeling):
    frame = copy.deepcopy(data.reference_frame())
    frame["root"].update({"x": 0, "z": -300, "facing": 180})
    other = subject_from(skel, frame)
    res = resolve_camera({"over_the_shoulder": "r_shoulder"}, kneeling, other=other)
    assert np.allclose(res.camera.target, other.positions["head"])
    p = res.camera.project(other.positions["head"])
    assert abs(p.dx) < 1e-9 and abs(p.dy) < 1e-9


# ---------------------------------------------------------------- near plane and warnings
def test_near_plane_and_clipping(kneeling):
    cam = Camera(target=np.array([0.0, 100.0, 0.0]), azimuth=0, elevation=0, distance=100)
    behind = np.array([0.0, 100.0, -95.0])        # w = 5: inside the near plane
    front = np.array([0.0, 100.0, 50.0])          # w = 150
    assert not cam.project(behind).visible and cam.project(front).visible
    seg = cam.clip_segment(behind, front)
    assert seg is not None
    assert abs(cam.project(seg[0]).w - 20) < 1e-9 and np.allclose(seg[1], front)
    assert cam.clip_segment(behind, behind + np.array([5.0, 0, 0])) is None
    assert np.allclose(cam.clip_segment(front, front + 1)[0], front)


def test_camera_inside_character_warns(kneeling):
    res = resolve_camera({"elevation": -90, "distance": 30}, kneeling)   # just under the head, by the neck
    assert any("inside" in w for w in res.warnings)
    assert not resolve_camera({}, kneeling).warnings


def test_character_mostly_out_of_frame_warns(kneeling):
    assert any("outside the frame" in w for w in resolve_camera({"focal": 2000}, kneeling).warnings)


def test_full_shot_keeps_every_pose_in_frame(skel):
    """Full shots fit the pose as seen from the camera, so deep poses (kneeling, sitting) stay in frame."""
    cr = data.load_rules()["camera"]
    for p in data.pose_library():
        subj = subject_from(skel, p["frame"])
        for az, el in ((0, 0), (-45, 10), (90, 30), (180, -20), (30, 89)):
            res = resolve_camera({"shot": "full", "azimuth": az, "elevation": el, "floor_guard": False}, subj)
            for j, P in subj.positions.items():
                q = res.camera.project(P)
                assert q.visible and abs(q.dx) <= 0.8 * cr["canvas_width"] / 2 + 1e-6, (p["name"], az, el, j)
                assert abs(q.dy) <= 0.8 * cr["canvas_height"] / 2 + 1e-6, (p["name"], az, el, j)
