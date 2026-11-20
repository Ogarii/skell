"""Posing maths: yaw/pitch <-> rotations, forward kinematics, grounding (spec 5.2, 5.3, 5.4, 6.3).

The stored pose is unit quaternions: one local rotation per posable bone (relative to its parent
bone's world rotation) and one world rotation for the root. Directions and positions are derived.

Frames:
- Heading frame = world turned by the root facing. Facing phi is a rotation of -phi about +Y, so the
  front (0, 0, -1) goes to (sin phi, 0, -cos phi) as the spec requires.
- A bone's parent frame is its parent bone's world rotation, or the root rotation if it has no parent.
- Rigid offset bones have the identity local rotation: they are fixed in their parent's frame.

Twist rules (spec 5.4), with twist applied after the shortest-arc swing, about the bone's new direction:
- Chest (the bone attached to the root): twist from the chest facing, relative to the root facing.
- Head (a ball bone whose parent is the chest bone): twist from the head facing, relative to the chest facing.
- Upper limbs (a bone whose child is a hinge): twist from the bend plane when the bend exceeds the
  data threshold; otherwise the previous pose's twist (0 if none). The twist is measured from a base
  rotation that carries the rest axis to a forward-down reference axis and then to the bone direction
  (both shortest arcs). Its singular direction is the reference's opposite, back-and-up, which thighs
  cannot reach and arms rarely do. See docs/DECISIONS.md.
- Hinge children (forearms, shins): no twist of their own; they inherit the parent's.
"""
from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import quat
from .data import Bone, Skeleton, load_rules

LATERAL = np.array([1.0, 0.0, 0.0])  # lateral axis of every parent frame (+X at rest = character's left)
FRONT = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 1.0, 0.0])
VERTICAL_PITCH = 89.9                # spec 6.3: above this, yaw is undefined
FOLD_SIGN = {"front": 1.0, "back": -1.0}


# ---------------------------------------------------------------- yaw / pitch
def heading_direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """dir = (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch)) in the heading frame."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])


def direction_to_yaw_pitch(d, prev_yaw: float = 0.0) -> tuple[float, float]:
    """Inverse of heading_direction. Near straight up/down, report prev_yaw (never divides by cos(pitch))."""
    d = quat.normalize3(d)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(d[1])))))
    if abs(pitch) > VERTICAL_PITCH:
        return prev_yaw, pitch
    return math.degrees(math.atan2(float(d[0]), float(-d[2]))), pitch


def facing_quat(facing_deg: float) -> np.ndarray:
    return quat.from_axis_angle(UP, -facing_deg)


def facing_of(q) -> float:
    """Yaw of a rotation's front vector (the facing it gives)."""
    f = quat.rotate(q, FRONT)
    return math.degrees(math.atan2(float(f[0]), float(-f[2])))


# ---------------------------------------------------------------- pose
@dataclass
class Pose:
    rotations: dict[str, np.ndarray]          # local unit quaternion per posable bone
    root_rotation: np.ndarray = field(default_factory=quat.identity)  # world rotation of the root
    root_x: float = 0.0
    root_z: float = 0.0
    airborne: bool = False
    pelvis_height: float | None = None         # only for airborne poses

    @property
    def facing(self) -> float:
        return facing_of(self.root_rotation)


def twist_role(skel: Skeleton, b: Bone) -> str:
    """Which twist rule applies to a posable bone, read from the skeleton's structure and limit data."""
    if b.limit and b.limit.get("type") == "hinge":
        return "hinge"
    if any(c.kind == "posable" and c.limit and c.limit.get("type") == "hinge" for c in skel.children(b.id)):
        return "limb_upper"
    if b.parent_bone is None:
        return "chest"
    parent = skel.bone(b.parent_bone)
    if parent.kind == "posable" and parent.parent_bone is None:
        return "head"
    raise ValueError(f"no twist rule for bone {b.id}")


@functools.lru_cache(maxsize=1)
def _default_rules() -> dict:
    return load_rules()


def _hinge_child(skel: Skeleton, b: Bone) -> Bone:
    return next(c for c in skel.children(b.id) if c.limit and c.limit.get("type") == "hinge")


def _parent_world(world: dict, root_q, b: Bone):
    return world[b.parent_bone] if b.parent_bone is not None else root_q


def turn_frame(frame: dict, delta: float) -> dict:
    """The same frame with the character turned by delta degrees about its root's vertical axis.
    Root, chest and head facings all turn (they are absolute yaw angles); bone yaw/pitch are in the
    heading frame and do not change. Returns a new frame."""
    fr = copy.deepcopy(frame)
    root = fr.setdefault("root", {})
    base = float(root.get("facing", 0.0))
    faces = fr.get("facing") or {}
    chest = float(faces.get("chest", base))
    head = float(faces.get("head", chest))
    root["facing"] = quat.wrap_deg(base + delta)
    fr["facing"] = {"chest": quat.wrap_deg(chest + delta), "head": quat.wrap_deg(head + delta)}
    return fr


def pose_from_frame(skel: Skeleton, frame: dict, prev: Pose | None = None, rules: dict | None = None) -> Pose:
    """Convert a yaw/pitch frame (spec 10.4) into stored rotations (spec 6.3)."""
    rules = rules or _default_rules()
    min_bend = rules["limits"]["hinge_plane_min_bend_deg"]

    root = frame.get("root", {})
    facing = float(root.get("facing", 0.0))
    faces = frame.get("facing") or {}
    chest_facing = float(faces.get("chest", facing))
    head_facing = float(faces.get("head", chest_facing))
    airborne = bool(frame.get("airborne", False))
    pelvis_height = root.get("y")
    if airborne and pelvis_height is None:
        raise ValueError("airborne frame needs an explicit pelvis height (root.y)")

    missing = set(skel.posable_ids) - set(frame["pose"])
    if missing:
        raise ValueError(f"frame is missing posable bones: {sorted(missing)}")

    root_q = facing_quat(facing)
    world_dir = {bid: quat.rotate(root_q, heading_direction(v["yaw"], v["pitch"])) for bid, v in frame["pose"].items()
                 if bid in skel.posable_ids}

    world: dict[str, np.ndarray] = {}
    local: dict[str, np.ndarray] = {}
    for b in skel.bones:
        Rp = _parent_world(world, root_q, b)
        if b.kind == "rigid":
            world[b.id] = Rp
            continue
        d_local = quat.rotate(quat.conj(Rp), world_dir[b.id])
        role = twist_role(skel, b)
        if role == "limb_upper":
            swing = limb_base(b, d_local, rules)
        else:
            swing = quat.shortest_arc(b.rest_axis, d_local, LATERAL)
        if role == "chest":
            twist = -(chest_facing - facing)       # facing sense -> right-handed angle about the bone
        elif role == "head":
            twist = -(head_facing - chest_facing)
        elif role == "limb_upper":
            twist = _limb_twist(skel, b, d_local, swing, Rp, world_dir, min_bend, prev, rules)
        else:  # hinge: no twist of its own
            twist = 0.0
        q = quat.mul(quat.from_axis_angle(d_local, twist), swing) if twist else swing
        local[b.id] = quat.normalize(q)
        world[b.id] = quat.normalize(quat.mul(Rp, local[b.id]))

    return Pose(rotations=local, root_rotation=root_q, root_x=float(root.get("x", 0.0)),
                root_z=float(root.get("z", 0.0)), airborne=airborne,
                pelvis_height=None if pelvis_height is None else float(pelvis_height))


def limb_base(b: Bone, a_local, rules: dict | None = None) -> np.ndarray:
    """Untwisted rotation of an upper limb: rest axis -> reference axis -> a_local (parent frame)."""
    rules = rules or _default_rules()
    ref = heading_direction(*rules["limits"]["limb_twist_reference_yaw_pitch"])
    return quat.mul(quat.shortest_arc(ref, a_local, LATERAL), quat.shortest_arc(b.rest_axis, ref, LATERAL))


def limb_rotation(b: Bone, q_local, rules: dict | None = None) -> float:
    """Twist of an upper limb's stored local rotation, measured from limb_base. This is the limb
    rotation that spec 5.4 checks against the shoulder or hip rotation range."""
    a = quat.rotate(q_local, b.rest_axis)
    tw = quat.mul(q_local, quat.conj(limb_base(b, a, rules)))
    return quat.wrap_deg(math.degrees(2.0 * math.atan2(float(np.dot(tw[1:], a)), float(tw[0]))))


def _limb_twist(skel, b, a_local, base, Rp, world_dir, min_bend, prev, rules) -> float:
    """Twist of an upper arm or thigh from its hinge's bend plane (spec 5.4)."""
    child = _hinge_child(skel, b)
    b_local = quat.rotate(quat.conj(Rp), world_dir[child.id])
    n = np.cross(a_local, b_local)
    bend = math.degrees(math.atan2(float(np.linalg.norm(n)), float(np.dot(a_local, b_local))))
    if bend > min_bend:
        n = n / np.linalg.norm(n)
        neutral = FOLD_SIGN[child.limit["folds_toward"]] * quat.rotate(base, LATERAL)
        return quat.signed_angle(neutral, n, a_local)
    if prev is not None and b.id in prev.rotations:
        return limb_rotation(b, prev.rotations[b.id], rules)
    return 0.0


def world_rotations(skel: Skeleton, pose: Pose) -> dict[str, np.ndarray]:
    """World rotation of every bone (rigid bones carry their parent's)."""
    world: dict[str, np.ndarray] = {}
    for b in skel.bones:
        Rp = _parent_world(world, pose.root_rotation, b)
        world[b.id] = Rp if b.kind == "rigid" else quat.mul(Rp, pose.rotations[b.id])
    return world


def head_rotation(skel: Skeleton, pose: Pose) -> np.ndarray:
    """World rotation of the bone ending at the head (it orients the face cross in the renderer)."""
    bone = next(b for b in skel.bones if b.to_joint == "head")
    return world_rotations(skel, pose)[bone.id]


def forward_kinematics(skel: Skeleton, pose: Pose) -> tuple[dict, dict]:
    """child = parent + L * (world rotation applied to rest axis). Pelvis at (root_x, 0, root_z), not grounded."""
    world = world_rotations(skel, pose)
    pos = {skel.root: np.array([pose.root_x, 0.0, pose.root_z])}
    dirs = {}
    for b in skel.bones:
        d = quat.rotate(world[b.id], b.rest_axis)
        dirs[b.id] = d
        pos[b.to_joint] = pos[b.from_joint] + b.length * d
    return pos, dirs


def ground(skel: Skeleton, pose: Pose, pos: dict) -> dict:
    """Lowest joint to y = 0, unless airborne, when the pelvis sits at the explicit height (spec 5.3)."""
    if pose.airborne:
        shift = pose.pelvis_height - pos[skel.root][1]
    else:
        shift = -min(p[1] for p in pos.values())
    return {k: p + np.array([0.0, shift, 0.0]) for k, p in pos.items()}


def solve(skel: Skeleton, pose: Pose) -> tuple[dict, dict]:
    """Forward kinematics plus grounding: world joint positions and bone directions."""
    pos, dirs = forward_kinematics(skel, pose)
    return ground(skel, pose, pos), dirs


def pose_to_frame(skel: Skeleton, pose: Pose, prev_frame: dict | None = None, t: int = 1) -> dict:
    """Read stored rotations back as a yaw/pitch frame in the heading frame (spec 6.3)."""
    _, dirs = forward_kinematics(skel, pose)
    inv_heading = quat.conj(facing_quat(pose.facing))
    prev_pose = (prev_frame or {}).get("pose", {})
    out_pose = {}
    for bid in skel.posable_ids:
        prev_yaw = prev_pose.get(bid, {}).get("yaw", 0)
        yaw, pitch = direction_to_yaw_pitch(quat.rotate(inv_heading, dirs[bid]), prev_yaw)
        out_pose[bid] = {"yaw": yaw, "pitch": pitch}

    facing = pose.facing
    chest = head = facing
    for bid in skel.posable_ids:
        b = skel.bone(bid)
        role = twist_role(skel, b)
        if role == "chest":
            chest = quat.wrap_deg(facing - quat.twist_angle(pose.rotations[bid], b.rest_axis, LATERAL))
    for bid in skel.posable_ids:
        b = skel.bone(bid)
        if twist_role(skel, b) == "head":
            head = quat.wrap_deg(chest - quat.twist_angle(pose.rotations[bid], b.rest_axis, LATERAL))

    root = {"x": pose.root_x, "z": pose.root_z, "facing": facing}
    if pose.airborne:
        root["y"] = pose.pelvis_height
    return {"t": t, "root": root, "facing": {"chest": chest, "head": head}, "airborne": pose.airborne,
            "pose": out_pose}
