"""Step 2: posing maths (spec 5.2, 5.3, 5.4 twist rules, 6.3), checked against the golden fixture."""
import copy
import math

import numpy as np
import pytest

from sandbox.skell import data, quat
from sandbox.skell.pose import (
    direction_to_yaw_pitch,
    forward_kinematics,
    heading_direction,
    pose_from_frame,
    pose_to_frame,
    solve,
)

WORLD_TOL = 1e-6   # fixture tolerance, world units
ANGLE_TOL = 1e-4   # fixture tolerance, degrees
QUAT_TOL = 1e-9


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


@pytest.fixture(scope="module")
def fixture():
    return data.load_fixture()


@pytest.fixture()
def frame():
    return copy.deepcopy(data.reference_frame())


def assert_same_rotation(a, b, tol=QUAT_TOL):
    assert abs(abs(float(np.dot(a, b))) - 1.0) < tol, f"{a} != {b}"


# ---------------------------------------------------------------- quaternion basics
def test_quat_rotate_and_axis_angle():
    q = quat.from_axis_angle([0, 0, 1], 90)
    assert np.allclose(quat.rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)
    q2 = quat.mul(q, q)
    assert np.allclose(quat.rotate(q2, [1, 0, 0]), [-1, 0, 0], atol=1e-12)
    axis, ang = quat.to_axis_angle(q2)
    assert math.isclose(ang, 180, abs_tol=1e-9) and np.allclose(abs(axis), [0, 0, 1])


def test_shortest_arc_general_and_opposite():
    a, b = np.array([0, -1.0, 0]), quat.normalize3([0.3, 0.4, -0.5])
    assert np.allclose(quat.rotate(quat.shortest_arc(a, b), a), b, atol=1e-12)
    q = quat.shortest_arc(a, -a, fallback_axis=[1, 0, 0])
    axis, ang = quat.to_axis_angle(q)
    assert math.isclose(ang, 180, abs_tol=1e-9)
    assert np.allclose(abs(axis), [1, 0, 0], atol=1e-12)
    assert np.allclose(quat.rotate(q, a), -a, atol=1e-12)


def test_slerp_endpoints_unit_and_shortest_path():
    q0 = quat.identity()
    q1 = quat.from_axis_angle([0, 1, 0], 120)
    assert_same_rotation(quat.slerp(q0, q1, 0), q0)
    assert_same_rotation(quat.slerp(q0, q1, 1), q1)
    mid = quat.slerp(q0, q1, 0.5)
    assert math.isclose(np.linalg.norm(mid), 1.0, abs_tol=1e-12)
    assert_same_rotation(mid, quat.from_axis_angle([0, 1, 0], 60))
    # -q1 is the same rotation; shortest path must give the same midpoint
    assert_same_rotation(quat.slerp(q0, -q1, 0.5), mid)


def test_twist_angle_recovers_twist():
    rest = np.array([0, -1.0, 0])
    d = quat.normalize3([0.2, -0.3, -0.9])
    swing = quat.shortest_arc(rest, d)
    for t in (-170, -40, 0, 25, 179):
        q = quat.mul(quat.from_axis_angle(d, t), swing)
        assert math.isclose(quat.twist_angle(q, rest), t, abs_tol=1e-9)


# ---------------------------------------------------------------- yaw / pitch
def test_heading_direction_convention():
    assert np.allclose(heading_direction(0, 0), [0, 0, -1])     # front, toward the front camera
    assert np.allclose(heading_direction(90, 0), [1, 0, 0])     # character's left = +X at facing 0
    assert np.allclose(heading_direction(0, 90), [0, 1, 0])
    for yaw, pitch in [(-6, 32), (146, -40), (179, 5), (-179, -89)]:
        y, p = direction_to_yaw_pitch(heading_direction(yaw, pitch))
        assert math.isclose(y, yaw, abs_tol=1e-9) and math.isclose(p, pitch, abs_tol=1e-9)


# ---------------------------------------------------------------- reference fixture
def test_world_positions_match_fixture(skel, fixture, frame):
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    assert set(pos) == set(fixture["world_positions"])
    for joint, expected in fixture["world_positions"].items():
        assert np.allclose(pos[joint], expected, atol=WORLD_TOL), (joint, pos[joint], expected)


def test_grounding_matches_fixture(skel, fixture, frame):
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    assert math.isclose(pos["pelvis"][1], fixture["pelvis_height_after_grounding"], abs_tol=WORLD_TOL)
    assert math.isclose(min(p[1] for p in pos.values()), 0.0, abs_tol=1e-12)


def test_derived_directions_equal_yaw_pitch_directions(skel, frame):
    """Quaternion pipeline reproduces the direction the generator gets straight from yaw/pitch."""
    _, dirs = solve(skel, pose_from_frame(skel, frame))
    for bid, yp in frame["pose"].items():
        assert np.allclose(dirs[bid], heading_direction(yp["yaw"], yp["pitch"]), atol=1e-12), bid


def test_bone_lengths_exact(skel, frame):
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    for b in skel.bones:
        length = np.linalg.norm(pos[b.to_joint] - pos[b.from_joint])
        assert abs(length - b.length) < WORLD_TOL, b.id


def test_stored_rotations_are_unit_quaternions(skel, frame):
    pose = pose_from_frame(skel, frame)
    assert set(pose.rotations) == set(skel.posable_ids)
    for q in list(pose.rotations.values()) + [pose.root_rotation]:
        assert math.isclose(np.linalg.norm(q), 1.0, abs_tol=1e-12)


def test_rotations_round_trip(skel, frame):
    pose = pose_from_frame(skel, frame)
    back = pose_to_frame(skel, pose)
    for bid, yp in frame["pose"].items():
        assert math.isclose(back["pose"][bid]["yaw"], yp["yaw"], abs_tol=ANGLE_TOL), bid
        assert math.isclose(back["pose"][bid]["pitch"], yp["pitch"], abs_tol=ANGLE_TOL), bid
    assert math.isclose(back["root"]["facing"], 0, abs_tol=ANGLE_TOL)
    assert math.isclose(back["facing"]["chest"], 0, abs_tol=ANGLE_TOL)
    assert math.isclose(back["facing"]["head"], 0, abs_tol=ANGLE_TOL)
    again = pose_from_frame(skel, back, prev=pose)
    for bid in skel.posable_ids:
        assert_same_rotation(again.rotations[bid], pose.rotations[bid])


def test_hinge_children_bend_in_their_fold_plane(skel, frame):
    """Upper-limb twist comes from the bend plane (spec 5.4): the forearm/shin local rotation is then
    a pure rotation about the upper bone's lateral axis, + for elbows (front), - for knees (back)."""
    pose = pose_from_frame(skel, frame)
    _, dirs = solve(skel, pose)
    min_bend = data.load_rules()["limits"]["hinge_plane_min_bend_deg"]
    # the reference elbows bend 6 and 12 degrees, under the threshold; add a bent elbow to cover arms
    frame["pose"]["l_elbow>l_hand"] = {"yaw": 60, "pitch": 30}
    arm_pose = pose_from_frame(skel, frame)
    _, arm_dirs = solve(skel, arm_pose)
    for p, d, child, sign in [(arm_pose, arm_dirs, "l_elbow>l_hand", 1),
                              (pose, dirs, "l_knee>l_foot", -1), (pose, dirs, "r_knee>r_foot", -1)]:
        upper = skel.bone(child).parent_bone
        bend = math.degrees(math.acos(np.clip(np.dot(d[upper], d[child]), -1, 1)))
        assert bend > min_bend
        q = p.rotations[child]
        axis, ang = quat.to_axis_angle(q)
        signed = ang * np.sign(axis[0])
        assert np.allclose(abs(axis), [1, 0, 0], atol=1e-9), child
        assert math.isclose(signed, sign * bend, abs_tol=1e-9), child


def test_reference_limb_rotations_within_ranges(skel, frame):
    """Early guard for the bend-direction rule (spec 5.4, fixed in docs/DECISIONS.md): the stored limb
    rotation of every reference limb lies inside its shoulder or hip rotation range."""
    from sandbox.skell.pose import limb_rotation
    pose = pose_from_frame(skel, frame)
    expected = {"l_shoulder>l_elbow": 0.0, "r_shoulder>r_elbow": 0.0}  # elbows under the threshold: twist 0
    for bid in ("l_shoulder>l_elbow", "r_shoulder>r_elbow", "l_hip>l_knee", "r_hip>r_knee"):
        b = skel.bone(bid)
        lo, hi = b.limit["rotation_range"]
        v = limb_rotation(b, pose.rotations[bid])
        assert lo <= v <= hi, (bid, v)
        if bid in expected:
            assert math.isclose(v, expected[bid], abs_tol=1e-9)
    assert math.isclose(limb_rotation(skel.bone("l_hip>l_knee"), pose.rotations["l_hip>l_knee"]), -21.8, abs_tol=0.05)
    assert math.isclose(limb_rotation(skel.bone("r_hip>r_knee"), pose.rotations["r_hip>r_knee"]), -51.0, abs_tol=0.05)


def test_backward_knee_and_elbow_read_outside_range(skel, frame):
    """A knee bent forward and an elbow bent backward give limb rotations near 180: outside any range."""
    from sandbox.skell.pose import limb_rotation
    frame["pose"]["l_hip>l_knee"] = {"yaw": 0, "pitch": -90}
    frame["pose"]["l_knee>l_foot"] = {"yaw": 0, "pitch": -30}       # shin forward from a vertical thigh
    frame["pose"]["r_shoulder>r_elbow"] = {"yaw": 0, "pitch": 0}
    frame["pose"]["r_elbow>r_hand"] = {"yaw": 0, "pitch": -40}      # forearm down from a forward arm
    pose = pose_from_frame(skel, frame)
    for bid in ("l_hip>l_knee", "r_shoulder>r_elbow"):
        v = limb_rotation(skel.bone(bid), pose.rotations[bid])
        lo, hi = skel.bone(bid).limit["rotation_range"]
        assert not (lo <= v <= hi), (bid, v)
        assert abs(v) > 130, (bid, v)


# ---------------------------------------------------------------- twist and facing rules
def test_chest_and_head_facing_set_twist(skel, frame):
    frame["pose"]["pelvis>neck"] = {"yaw": 0, "pitch": 90}
    frame["pose"]["neck>head"] = {"yaw": 0, "pitch": 90}
    frame["facing"] = {"chest": 30, "head": 50}
    pose = pose_from_frame(skel, frame)
    from sandbox.skell.pose import world_rotations
    W = world_rotations(skel, pose)
    front = np.array([0, 0, -1.0])
    assert np.allclose(quat.rotate(W["pelvis>neck"], front), heading_direction(30, 0), atol=1e-12)
    assert np.allclose(quat.rotate(W["neck>head"], front), heading_direction(50, 0), atol=1e-12)
    back = pose_to_frame(skel, pose)
    assert math.isclose(back["facing"]["chest"], 30, abs_tol=ANGLE_TOL)
    assert math.isclose(back["facing"]["head"], 50, abs_tol=ANGLE_TOL)


def test_straight_limb_keeps_previous_twist(skel, frame):
    frame["pose"]["l_shoulder>l_elbow"] = {"yaw": 90, "pitch": 0}
    frame["pose"]["l_elbow>l_hand"] = {"yaw": 90, "pitch": 0}
    pose = pose_from_frame(skel, frame)
    _, dirs = solve(skel, pose)
    a = dirs["l_shoulder>l_elbow"]
    twisted = copy.deepcopy(pose)
    # twist the upper arm 40 degrees about its own length (in its parent frame); a straight
    # forearm's local rotation is unaffected, so the geometry is unchanged
    from sandbox.skell.pose import world_rotations
    Wp = world_rotations(skel, pose)[skel.bone("l_shoulder>l_elbow").parent_bone]
    a_local = quat.rotate(quat.conj(Wp), a)
    twisted.rotations["l_shoulder>l_elbow"] = quat.mul(quat.from_axis_angle(a_local, 40),
                                                       pose.rotations["l_shoulder>l_elbow"])
    p1, _ = solve(skel, pose)
    p2, _ = solve(skel, twisted)
    for j in p1:
        assert np.allclose(p1[j], p2[j], atol=1e-9)
    fr = pose_to_frame(skel, twisted)
    kept = pose_from_frame(skel, fr, prev=twisted)
    assert_same_rotation(kept.rotations["l_shoulder>l_elbow"], twisted.rotations["l_shoulder>l_elbow"])
    reset = pose_from_frame(skel, fr)  # no previous pose: twist 0
    assert_same_rotation(reset.rotations["l_shoulder>l_elbow"], pose.rotations["l_shoulder>l_elbow"])


# ---------------------------------------------------------------- special cases (spec 6.3)
@pytest.mark.parametrize("yaw", [0, 37, -120, 180])
def test_vertical_bone(skel, frame, yaw):
    frame["pose"]["pelvis>neck"] = {"yaw": yaw, "pitch": 90}
    frame["pose"]["r_shoulder>r_elbow"] = {"yaw": yaw, "pitch": -90}
    pose = pose_from_frame(skel, frame)
    for q in pose.rotations.values():
        assert np.all(np.isfinite(q))
    pos, dirs = solve(skel, pose)
    assert np.allclose(dirs["pelvis>neck"], [0, 1, 0], atol=1e-12)
    assert np.allclose(dirs["r_shoulder>r_elbow"], [0, -1, 0], atol=1e-12)
    for b in skel.bones:
        assert abs(np.linalg.norm(pos[b.to_joint] - pos[b.from_joint]) - b.length) < WORLD_TOL
    # read back: previous frame's yaw, or 0 if there is none
    prev = copy.deepcopy(frame)
    prev["pose"]["pelvis>neck"]["yaw"] = 37
    prev["pose"]["r_shoulder>r_elbow"]["yaw"] = -12
    with_prev = pose_to_frame(skel, pose, prev_frame=prev)
    assert with_prev["pose"]["pelvis>neck"] == {"yaw": 37, "pitch": pytest.approx(90)}
    assert with_prev["pose"]["r_shoulder>r_elbow"] == {"yaw": -12, "pitch": pytest.approx(-90)}
    no_prev = pose_to_frame(skel, pose)
    assert no_prev["pose"]["pelvis>neck"]["yaw"] == 0
    assert no_prev["pose"]["r_shoulder>r_elbow"]["yaw"] == 0


def test_180_degree_swing_uses_parent_lateral_axis(skel, frame):
    # thigh straight up: exactly opposite its rest axis (down); straight knee so no bend-plane twist
    frame["pose"]["l_hip>l_knee"] = {"yaw": 0, "pitch": 90}
    frame["pose"]["l_knee>l_foot"] = {"yaw": 0, "pitch": 90}
    pose = pose_from_frame(skel, frame)
    q = pose.rotations["l_hip>l_knee"]
    axis, ang = quat.to_axis_angle(q)
    assert math.isclose(ang, 180, abs_tol=1e-6)
    assert np.allclose(abs(axis), [1, 0, 0], atol=1e-6)
    _, dirs = solve(skel, pose)
    assert np.allclose(dirs["l_hip>l_knee"], [0, 1, 0], atol=1e-12)


# ---------------------------------------------------------------- turning and parenting
def test_turned_character(skel, fixture, frame):
    phi = math.radians(90)
    frame["root"]["facing"] = 90
    pose = pose_from_frame(skel, frame)
    pos, _ = solve(skel, pose)
    for joint, (x, y, z) in fixture["world_positions"].items():
        expected = (x * math.cos(phi) - z * math.sin(phi), y, z * math.cos(phi) + x * math.sin(phi))
        assert np.allclose(pos[joint], expected, atol=WORLD_TOL), joint
    back = pose_to_frame(skel, pose)
    assert math.isclose(back["root"]["facing"], 90, abs_tol=ANGLE_TOL)
    for bid, yp in data.reference_frame()["pose"].items():
        assert math.isclose(back["pose"][bid]["yaw"], yp["yaw"], abs_tol=ANGLE_TOL)
        assert math.isclose(back["pose"][bid]["pitch"], yp["pitch"], abs_tol=ANGLE_TOL)


def test_root_position_moves_character(skel, fixture, frame):
    frame["root"]["x"], frame["root"]["z"] = 100, -40
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    for joint, (x, y, z) in fixture["world_positions"].items():
        assert np.allclose(pos[joint], (x + 100, y, z - 40), atol=WORLD_TOL)


def test_airborne_uses_explicit_pelvis_height(skel, frame):
    frame["airborne"] = True
    frame["root"]["y"] = 120
    pos, _ = solve(skel, pose_from_frame(skel, frame))
    assert math.isclose(pos["pelvis"][1], 120, abs_tol=1e-12)
    frame["root"].pop("y")
    with pytest.raises(ValueError):
        pose_from_frame(skel, frame)


def test_rotating_spine_carries_arms_and_head(skel, frame):
    pose = pose_from_frame(skel, frame)
    before, _ = forward_kinematics(skel, pose)
    delta = quat.from_axis_angle([1, 0.3, -0.2], 35)
    pose.rotations["pelvis>neck"] = quat.mul(delta, pose.rotations["pelvis>neck"])
    after, _ = forward_kinematics(skel, pose)
    pelvis = before["pelvis"]
    upper = ["neck", "head", "l_shoulder", "l_elbow", "l_hand", "r_shoulder", "r_elbow", "r_hand"]
    lower = ["l_hip", "l_knee", "l_foot", "r_hip", "r_knee", "r_foot"]
    for j in upper:
        assert np.allclose(after[j] - pelvis, quat.rotate(delta, before[j] - pelvis), atol=1e-9), j
    for j in lower:
        assert np.allclose(after[j], before[j], atol=1e-12), j


def test_random_poses_keep_lengths_and_directions(skel):
    rng = np.random.default_rng(1)
    for _ in range(300):
        fr = {
            "t": 1,
            "root": {"x": float(rng.uniform(-200, 200)), "z": float(rng.uniform(-200, 200)),
                     "facing": float(rng.uniform(-180, 180))},
            "facing": {"chest": float(rng.uniform(-180, 180)), "head": float(rng.uniform(-180, 180))},
            "airborne": False,
            "pose": {b: {"yaw": float(rng.uniform(-180, 180)), "pitch": float(rng.uniform(-90, 90))}
                     for b in skel.posable_ids},
        }
        pose = pose_from_frame(skel, fr)
        pos, dirs = solve(skel, pose)
        heading = quat.from_axis_angle([0, 1, 0], -fr["root"]["facing"])
        for b in skel.bones:
            assert abs(np.linalg.norm(pos[b.to_joint] - pos[b.from_joint]) - b.length) < WORLD_TOL
            if b.kind == "posable":
                yp = fr["pose"][b.id]
                assert np.allclose(dirs[b.id], quat.rotate(heading, heading_direction(yp["yaw"], yp["pitch"])),
                                   atol=1e-9)
        back = pose_to_frame(skel, pose)
        for b in skel.posable_ids:
            if abs(fr["pose"][b]["pitch"]) < 89.9:
                dy = (back["pose"][b]["yaw"] - fr["pose"][b]["yaw"] + 180) % 360 - 180
                assert abs(dy) < ANGLE_TOL
            assert math.isclose(back["pose"][b]["pitch"], fr["pose"][b]["pitch"], abs_tol=ANGLE_TOL)
