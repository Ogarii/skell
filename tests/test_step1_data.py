"""Step 1: skeleton and reference pose load correctly (spec 5.1, Appendix B.1)."""
import copy
import math

import pytest

from sandbox.skell import data

# Spec 5.1, written out literally so a change in data/ cannot silently change the test.
JOINTS = [
    "pelvis", "neck", "head",
    "l_shoulder", "l_elbow", "l_hand",
    "r_shoulder", "r_elbow", "r_hand",
    "l_hip", "l_knee", "l_foot",
    "r_hip", "r_knee", "r_foot",
]
POSABLE = {
    "pelvis>neck", "neck>head",
    "l_shoulder>l_elbow", "l_elbow>l_hand",
    "r_shoulder>r_elbow", "r_elbow>r_hand",
    "l_hip>l_knee", "l_knee>l_foot",
    "r_hip>r_knee", "r_knee>r_foot",
}
RIGID = {"neck>l_shoulder", "neck>r_shoulder", "pelvis>l_hip", "pelvis>r_hip"}
LENGTHS = {
    "pelvis>neck": 90, "neck>head": 30,
    "l_shoulder>l_elbow": 47, "r_shoulder>r_elbow": 47,
    "l_elbow>l_hand": 39, "r_elbow>r_hand": 39,
    "l_hip>l_knee": 56, "r_hip>r_knee": 56,
    "l_knee>l_foot": 55, "r_knee>r_foot": 55,
}


@pytest.fixture(scope="module")
def skel():
    return data.load_skeleton()


def test_counts(skel):
    assert skel.format == "skell_v1"
    assert len(skel.joints) == 15
    assert len(skel.bones) == 14
    assert len(skel.posable_ids) == 10
    assert len(skel.rigid_ids) == 4
    assert skel.root == "pelvis"


def test_names_match_spec(skel):
    assert skel.joints == JOINTS
    assert set(skel.posable_ids) == POSABLE
    assert set(skel.rigid_ids) == RIGID
    for b in skel.bones:
        assert b.id == f"{b.from_joint}>{b.to_joint}"


def test_parent_before_child(skel):
    known = {skel.root}
    seen_bones = set()
    for b in skel.bones:
        assert b.from_joint in known, f"{b.id}: 'from' joint used before it exists"
        assert b.to_joint not in known, f"{b.id}: joint {b.to_joint} reached twice"
        if b.parent_bone is not None:
            assert b.parent_bone in seen_bones, f"{b.id}: parent bone listed later"
        known.add(b.to_joint)
        seen_bones.add(b.id)
    assert known == set(JOINTS)


def test_lengths_match_design_notes(skel):
    for bid, length in LENGTHS.items():
        assert skel.bone(bid).length == length
    for bid in RIGID:  # widths are 0 in the reference character
        assert skel.bone(bid).length == 0


def test_reference_character(skel):
    assert skel.standing_height == 250
    assert skel.head_radius == 19
    assert skel.symmetric is True


def test_rest_axes_are_unit_vectors(skel):
    for b in skel.bones:
        assert math.isclose(float(sum(b.rest_axis ** 2)), 1.0, abs_tol=1e-12), b.id


def test_limits_present_only_on_posable_bones(skel):
    for b in skel.bones:
        if b.kind == "posable":
            assert b.limit is not None, b.id
        else:
            assert b.limit is None, b.id
    hinges = {b.id for b in skel.bones if b.limit and b.limit["type"] == "hinge"}
    assert hinges == {"l_elbow>l_hand", "r_elbow>r_hand", "l_knee>l_foot", "r_knee>r_foot"}


def test_reference_pose_has_all_posable_bones_and_no_rigid():
    frame = data.reference_frame()
    assert set(frame["pose"]) == POSABLE
    assert not (set(frame["pose"]) & RIGID)
    for v in frame["pose"].values():
        assert math.isfinite(v["yaw"]) and math.isfinite(v["pitch"])


def test_reference_pose_request():
    req = data.load_reference_pose()
    assert req["intent"]["tag"] in data.all_intent_tags()
    assert len(req["options"]) == 1
    assert req["options"][0]["frames"][0]["t"] == 1


def test_rules_and_other_data_load():
    rules = data.load_rules()
    mf = rules["balance"]["mass_fractions"]
    total = mf["head"] + mf["trunk"] + 2 * (mf["upper_arm"] + mf["forearm_hand"] + mf["thigh"] + mf["shin_foot"])
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    assert rules["camera"]["near_plane"] == 20
    assert rules["limits"]["hinge_plane_min_bend_deg"] == 20
    assert rules["limits"]["limb_twist_reference_yaw_pitch"] == [0, -45]
    assert len(data.load_camera_phrases()["phrases"]) > 0
    assert "kneel" in data.all_intent_tags()


def test_malformed_skeleton_is_rejected():
    raw = data.load_json("skell_v1_reference_skeleton.json")
    bad = copy.deepcopy(raw)
    bad["bones"][0], bad["bones"][1] = bad["bones"][1], bad["bones"][0]  # child before parent
    with pytest.raises(ValueError):
        data.Skeleton.from_dict(bad)
    bad = copy.deepcopy(raw)
    bad["bones"][2]["id"] = "wrong>name"
    with pytest.raises(ValueError):
        data.Skeleton.from_dict(bad)
