"""Step 6: joint limits with clamping, balance, overlap and continuity (spec 5.4, 10.5)."""
import copy
import math

import numpy as np
import pytest

from sandbox.skell import data
from sandbox.skell.checks import balance, contacts, overlaps, rotation_change, validate
from sandbox.skell.limits import apply_limits
from sandbox.skell.pose import pose_from_frame, solve
from sandbox.skell.render import Box

ANGLE_TOL = 1e-4
WORLD_TOL = 1e-6


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


@pytest.fixture(scope="module")
def fixture():
    return data.load_fixture()


@pytest.fixture()
def frame():
    return copy.deepcopy(data.reference_frame())


def standing_frame(spread=True):
    out = 4 if spread else 0
    pose = {
        "pelvis>neck": (0, 90), "neck>head": (0, 90),
        "l_shoulder>l_elbow": (90, -80), "l_elbow>l_hand": (90, -85),
        "r_shoulder>r_elbow": (-90, -80), "r_elbow>r_hand": (-90, -85),
        "l_hip>l_knee": (90, -90 + out), "l_knee>l_foot": (0, -90),
        "r_hip>r_knee": (-90, -90 + out), "r_knee>r_foot": (0, -90),
    }
    return {"t": 1, "root": {"x": 0, "z": 0, "facing": 0}, "facing": {"chest": 0, "head": 0}, "airborne": False,
            "pose": {k: {"yaw": y, "pitch": p} for k, (y, p) in pose.items()}}


def check(skel, frame, prev=None, **kw):
    return validate(skel, pose_from_frame(skel, frame), prev=prev, **kw)


def assert_lengths(skel, pos):
    for b in skel.bones:
        assert abs(np.linalg.norm(pos[b.to_joint] - pos[b.from_joint]) - b.length) < WORLD_TOL, b.id


# ---------------------------------------------------------------- reference fixture
def test_reference_pose_matches_fixture(skel, fixture, frame):
    v = check(skel, frame)
    exp = fixture["validation"]
    assert v.contacts == exp["contacts"] == ["l_knee", "r_foot"]
    for joint, deg in exp["hinge_bend_deg"].items():
        assert abs(v.limits.hinge_bend[joint] - deg) < ANGLE_TOL, joint
    names = {"pelvis>neck_from_up": "pelvis>neck", "neck>head_from_spine": "neck>head",
             "l_hip>l_knee_from_thigh_axis": "l_hip>l_knee", "r_hip>r_knee_from_thigh_axis": "r_hip>r_knee"}
    for key, deg in exp["ball_angles_deg"].items():
        assert abs(v.limits.cone_angle[names[key]] - deg) < ANGLE_TOL, key
    assert v.limits.clamped_bones == exp["clamped_bones"] == []            # all four limbs bend the right way
    assert v.balance.status == exp["balance"]["status"] == "unstable"
    assert abs(v.balance.distance_to_support - exp["balance"]["distance_to_contact_line"]) < WORLD_TOL
    assert np.allclose(v.balance.centre_of_mass, exp["balance"]["centre_of_mass"], atol=WORLD_TOL)
    assert v.overlaps == exp["overlap_warnings"] == []
    assert check(skel, frame, prev=v.pose).continuity == exp["continuity_warnings"] == []


def test_reference_limb_rotations(skel, frame):
    rot = check(skel, frame).limits.limb_rotation
    assert rot["l_shoulder>l_elbow"] is None and rot["r_shoulder>r_elbow"] is None   # bends under 20 deg
    assert abs(rot["l_hip>l_knee"] + 21.8) < 0.05 and abs(rot["r_hip>r_knee"] + 51.0) < 0.05


# ---------------------------------------------------------------- clamping
def test_backward_elbow_is_clamped_to_nearest_allowed_bend(skel):
    frame = standing_frame()                                             # upright chest: an exactly backward bend
    frame["pose"]["r_shoulder>r_elbow"] = {"yaw": 0, "pitch": 0}        # arm straight forward
    frame["pose"]["r_elbow>r_hand"] = {"yaw": 0, "pitch": -40}          # forearm bends down: backward
    raw = apply_limits(skel, pose_from_frame(skel, frame), clamp=False)
    assert abs(abs(raw.limb_rotation["r_shoulder>r_elbow"]) - 180) < 1e-6
    v = check(skel, frame)
    corr = [c for c in v.limits.corrections if c.bone == "r_elbow>r_hand"]
    assert len(corr) == 1 and corr[0].kind == "bend_direction"
    assert abs(abs(corr[0].after) - 90) < 1e-9 and abs(corr[0].degrees - 90) < 1e-6
    again = apply_limits(skel, v.pose)                                   # the clamped pose is valid
    assert again.ok
    assert abs(abs(again.limb_rotation["r_shoulder>r_elbow"]) - 90) < 1e-6
    assert abs(again.hinge_bend["r_elbow"] - 40) < 1e-6                  # bend angle kept
    assert_lengths(skel, v.positions)
    a = v.directions["r_shoulder>r_elbow"]
    assert np.allclose(a, [0, 0, -1], atol=1e-9)                         # the upper arm itself did not move


def test_backward_knee_is_clamped_to_nearest_allowed_bend(skel):
    frame = standing_frame()
    frame["pose"]["l_hip>l_knee"] = {"yaw": 0, "pitch": -90}            # thigh straight down
    frame["pose"]["l_knee>l_foot"] = {"yaw": 0, "pitch": -30}           # shin kicks forward: backward knee
    v = check(skel, frame)
    corr = [c for c in v.limits.corrections if c.bone == "l_knee>l_foot"]
    assert len(corr) == 1 and corr[0].kind == "bend_direction" and abs(abs(corr[0].after) - 60) < 1e-9
    again = apply_limits(skel, v.pose)
    assert again.ok and abs(abs(again.limb_rotation["l_hip>l_knee"]) - 60) < 1e-6
    assert_lengths(skel, v.positions)


def test_bend_beyond_range_is_clamped(skel, frame):
    frame["pose"]["l_shoulder>l_elbow"] = {"yaw": 0, "pitch": -90}
    frame["pose"]["l_elbow>l_hand"] = {"yaw": 0, "pitch": 80}           # folds 170 deg, limit 155
    v = check(skel, frame)
    corr = [c for c in v.limits.corrections if c.kind == "bend"]
    assert len(corr) == 1 and corr[0].bone == "l_elbow>l_hand" and abs(corr[0].degrees - 15) < 1e-6
    assert abs(apply_limits(skel, v.pose).hinge_bend["l_elbow"] - 155) < 1e-6
    assert apply_limits(skel, v.pose).ok


def test_limb_folded_exactly_back_is_clamped_in_its_natural_plane(skel, frame):
    """Bones exactly opposite have no bend plane (their cross product is zero): the clamp folds them back
    in the limb's neutral plane, so the elbow ends up bending forward and the knee backward, with no NaNs."""
    frame["pose"]["r_shoulder>r_elbow"] = {"yaw": -41.4, "pitch": -90}
    frame["pose"]["r_elbow>r_hand"] = {"yaw": 18.0, "pitch": 90}        # 180 deg fold
    frame["pose"]["r_hip>r_knee"] = {"yaw": 0, "pitch": -90}
    frame["pose"]["r_knee>r_foot"] = {"yaw": 30, "pitch": 90}           # 180 deg fold
    v = check(skel, frame)
    assert all(np.isfinite(p).all() for p in v.positions.values())
    rep = apply_limits(skel, v.pose)
    assert rep.ok
    assert abs(rep.hinge_bend["r_elbow"] - 155) < 1e-6
    assert abs(rep.limb_rotation["r_shoulder>r_elbow"]) < 1e-6 and abs(rep.limb_rotation["r_hip>r_knee"]) < 1e-6
    assert (v.positions["r_hand"] - v.positions["r_elbow"])[2] < 0     # forearm folds toward the front (-Z)
    assert (v.positions["r_foot"] - v.positions["r_knee"])[2] > 0      # shin folds behind (+Z)
    assert_lengths(skel, v.positions)


def test_cone_limits_are_clamped(skel, frame):
    frame["pose"]["l_hip>l_knee"] = {"yaw": 180, "pitch": 40}           # thigh back and up: outside 105 deg
    frame["pose"]["l_knee>l_foot"] = {"yaw": 180, "pitch": 40}
    frame["pose"]["pelvis>neck"] = {"yaw": 0, "pitch": -20}             # spine 110 deg from up: limit 100
    v = check(skel, frame)
    kinds = {(c.bone, c.kind) for c in v.limits.corrections}
    assert ("l_hip>l_knee", "cone") in kinds and ("pelvis>neck", "cone") in kinds
    again = apply_limits(skel, v.pose)
    assert again.ok
    assert abs(again.cone_angle["l_hip>l_knee"] - 105) < 1e-6
    assert abs(again.cone_angle["pelvis>neck"] - 100) < 1e-6
    assert_lengths(skel, v.positions)


def test_twist_limits_are_clamped(skel, frame):
    frame["facing"] = {"chest": 70, "head": 175}                        # chest +70 (limit 45), head +105 (limit 80)
    v = check(skel, frame)
    tw = {c.bone: c for c in v.limits.corrections if c.kind == "twist"}
    assert abs(tw["pelvis>neck"].before - 70) < 1e-6 and abs(tw["pelvis>neck"].after - 45) < 1e-9
    assert abs(tw["neck>head"].before - 105) < 1e-6 and abs(tw["neck>head"].after - 80) < 1e-9
    assert apply_limits(skel, v.pose).ok


def test_corrections_are_recorded_per_bone(skel, frame):
    frame["pose"]["r_shoulder>r_elbow"] = {"yaw": 0, "pitch": 0}
    frame["pose"]["r_elbow>r_hand"] = {"yaw": 0, "pitch": -40}
    frame["pose"]["l_hip>l_knee"] = {"yaw": 0, "pitch": -90}
    frame["pose"]["l_knee>l_foot"] = {"yaw": 0, "pitch": -30}
    v = check(skel, frame)
    assert v.limits.clamped_bones == ["l_knee>l_foot", "r_elbow>r_hand"]
    assert all(c.degrees > 0 for c in v.limits.corrections)
    assert any("clamped" in w for w in v.warnings())


# ---------------------------------------------------------------- balance
def test_standing_rest_pose_is_balanced(skel):
    v = check(skel, standing_frame())
    assert v.contacts == ["l_foot", "r_foot"]
    assert v.balance.status == "stable" and v.balance.distance_to_support < 1e-9
    assert v.limits.ok and not v.overlaps


def test_leaning_far_out_is_unstable(skel):
    fr = standing_frame()
    fr["pose"]["pelvis>neck"] = {"yaw": 0, "pitch": 20}                 # torso far forward over straight legs
    fr["pose"]["neck>head"] = {"yaw": 0, "pitch": 20}
    assert check(skel, fr).balance.status == "unstable"


def test_airborne_skips_balance(skel):
    fr = standing_frame()
    fr["airborne"], fr["root"]["y"] = True, 200
    v = check(skel, fr)
    assert v.balance.status == "airborne" and v.contacts == []


def test_contacts_ignore_zero_width_offsets(skel, frame):
    v = check(skel, frame)
    pos = dict(v.positions)
    assert "l_hip" not in contacts(skel, pos)
    b = balance(skel, pos, ["r_foot"])                                   # one contact: a disc around the foot
    assert b.status == "unstable"


# ---------------------------------------------------------------- overlap
def test_overlap_of_non_adjacent_bones(skel):
    v = check(skel, standing_frame(spread=False))                       # shins exactly side by side
    assert any("l_knee>l_foot and r_knee>r_foot" in o for o in v.overlaps)
    assert not check(skel, standing_frame()).overlaps


def test_adjacent_bones_never_overlap(skel):
    # arms hanging from zero-width shoulders start exactly at the neck: they meet the spine, so no warning
    v = check(skel, standing_frame())
    assert not any("pelvis>neck" in o and "shoulder>" in o for o in v.overlaps)


def test_bone_through_box(skel, frame):
    pos = check(skel, frame).positions
    hand = pos["r_hand"]
    box = Box(x=float(hand[0]), z=float(hand[2]), width=20, depth=20, height=40)   # the hand is inside it
    out = overlaps(skel, pos, boxes=[box])
    assert any("r_elbow>r_hand passes through box 0" in o for o in out)
    far = Box(x=300, z=300, width=20, depth=20, height=40)
    assert not overlaps(skel, pos, boxes=[far])


# ---------------------------------------------------------------- continuity
def test_continuity(skel, frame):
    prev = pose_from_frame(skel, frame)
    nxt = copy.deepcopy(frame)
    nxt["pose"]["l_hip>l_knee"]["yaw"] += 60
    v = check(skel, nxt, prev=prev)
    assert any(c.startswith("l_hip>l_knee ") for c in v.continuity)
    assert not check(skel, nxt, prev=prev, frames_apart=2).continuity or \
        all("l_hip>l_knee " not in c for c in check(skel, nxt, prev=prev, frames_apart=2).continuity)
    small = copy.deepcopy(frame)
    small["pose"]["l_hip>l_knee"]["yaw"] += 20
    assert not check(skel, small, prev=prev).continuity


def test_rotation_change():
    from sandbox.skell import quat
    q = quat.from_axis_angle([0, 1, 0], 50)
    assert abs(rotation_change(quat.identity(), q) - 50) < 1e-9
    assert abs(rotation_change(quat.identity(), -q) - 50) < 1e-9            # q and -q are the same rotation


def test_random_poses_clamp_to_valid(skel):
    """Any pose, however wild, comes out of the limits valid, with lengths intact."""
    rng = np.random.default_rng(7)
    for _ in range(150):
        fr = standing_frame()
        fr["facing"] = {"chest": float(rng.uniform(-120, 120)), "head": float(rng.uniform(-180, 180))}
        for b in skel.posable_ids:
            fr["pose"][b] = {"yaw": float(rng.uniform(-180, 180)), "pitch": float(rng.uniform(-89, 89))}
        v = check(skel, fr)
        again = apply_limits(skel, v.pose)
        assert again.ok, [str(c) for c in again.corrections]
        pos, _ = solve(skel, v.pose)
        assert_lengths(skel, pos)


# ---------------------------------------------------------------- viewer pose library
def test_pose_library_poses_pass_the_checks(skel):
    lib = data.pose_library()
    assert lib[0]["name"] == "reference_kneeling" and len(lib) >= 10
    assert len({p["name"] for p in lib}) == len(lib)
    for p in lib:
        assert set(p["frame"]["pose"]) == set(skel.posable_ids), p["name"]
        v = check(skel, p["frame"])
        assert v.limits.ok, (p["name"], [str(c) for c in v.limits.corrections])
        assert not v.overlaps, (p["name"], v.overlaps)
        assert_lengths(skel, v.positions)
        if p["frame"].get("airborne"):
            assert v.balance.status == "airborne"
        else:
            assert v.contacts, p["name"]
