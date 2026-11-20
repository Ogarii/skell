"""Joint limits (spec 5.4, bend-direction rule as revised in docs/DECISIONS.md).

Ball limits: the bone direction stays inside a cone (axis in the parent frame, half-angle) and the twist
of the chest and head stays inside its range. Hinge limits: the bend angle stays inside its range, and
the bend goes the correct way: the limb rotation (angle about the upper bone from the neutral side axis
to the bend normal) stays inside the shoulder or hip rotation range.

apply_limits walks the bones parent before child, measures every limit, and (when clamping) moves each
offending bone to the nearest allowed value, recording the size of every correction.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import quat
from .data import Skeleton, load_rules
from .pose import FOLD_SIGN, LATERAL, Pose, heading_direction, limb_base

EPS = 1e-9


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


@dataclass(frozen=True)
class Correction:
    bone: str
    kind: str          # "cone", "twist", "bend", "bend_direction"
    before: float      # measured value, degrees
    after: float       # value after clamping, degrees
    degrees: float     # size of the rotation applied

    def __str__(self):
        return f"{self.bone}: {self.kind} {self.before:.1f} -> {self.after:.1f} ({self.degrees:.1f} deg)"


@dataclass
class LimitReport:
    pose: Pose                                              # the clamped pose (or the input, if clamp=False)
    corrections: list[Correction] = field(default_factory=list)
    hinge_bend: dict[str, float] = field(default_factory=dict)       # by hinge joint name, e.g. "l_elbow"
    cone_angle: dict[str, float] = field(default_factory=dict)       # by bone id, angle from the cone axis
    twist: dict[str, float] = field(default_factory=dict)            # chest and head twist, facing sense
    limb_rotation: dict[str, float | None] = field(default_factory=dict)  # by upper bone; None under threshold

    @property
    def clamped_bones(self) -> list[str]:
        return sorted({c.bone for c in self.corrections})

    @property
    def ok(self) -> bool:
        return not self.corrections


def _angle(u, v) -> float:
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))


def _cone_axis(skel: Skeleton, b) -> np.ndarray | None:
    lim = b.limit
    if lim.get("cone_half_angle", 180) >= 180:
        return None
    frame = lim.get("frame")
    if frame == "parent_direction":
        return skel.bone(b.parent_bone).rest_axis.astype(float)
    if "cone_axis_yaw_pitch" in lim:
        # pelvis heading frame: the parent frame of the spine and thighs (the root frame in the MVP)
        return heading_direction(*lim["cone_axis_yaw_pitch"])
    return None


def cone_ok(skel: Skeleton, b, d_local) -> bool:
    """Is a ball bone's direction (in its parent frame) inside its cone?"""
    axis = _cone_axis(skel, b) if (b.limit or {}).get("type") == "ball" else None
    return axis is None or _angle(d_local, axis) <= b.limit["cone_half_angle"] + EPS


def hinge_ok(upper, lower, a_l, b_l, rules: dict | None = None) -> bool:
    """Does a hinge bend inside its range and the correct way? a_l, b_l: upper and lower bone directions,
    both in the upper bone's parent frame. Same measures as apply_limits."""
    rules = rules or _rules()
    lim = lower.limit
    beta = _angle(a_l, b_l)
    if beta > lim["bend"][1] + EPS:
        return False
    if beta <= rules["limits"]["hinge_plane_min_bend_deg"]:
        return True
    n = np.cross(a_l, b_l)
    n /= np.linalg.norm(n)
    neutral = FOLD_SIGN[lim["folds_toward"]] * quat.rotate(limb_base(upper, a_l, rules), LATERAL)
    psi = quat.signed_angle(neutral, n, a_l)
    lo, hi = upper.limit["rotation_range"]
    return lo - EPS <= psi <= hi + EPS


def apply_limits(skel: Skeleton, pose: Pose, rules: dict | None = None, clamp: bool = True) -> LimitReport:
    rules = rules or _rules()
    min_bend = rules["limits"]["hinge_plane_min_bend_deg"]
    rot = {k: np.array(v, dtype=float) for k, v in pose.rotations.items()}
    world: dict[str, np.ndarray] = {}
    rep = LimitReport(pose=pose)

    def parent_world(b):
        return world[b.parent_bone] if b.parent_bone is not None else pose.root_rotation

    for b in skel.bones:
        Rp = parent_world(b)
        if b.kind == "rigid":
            world[b.id] = Rp
            continue
        lim = b.limit or {}
        q = rot[b.id]

        if lim.get("type") == "ball":
            a = quat.rotate(q, b.rest_axis)
            axis = _cone_axis(skel, b)
            if axis is not None:
                ang = _angle(a, axis)
                rep.cone_angle[b.id] = ang
                half = lim["cone_half_angle"]
                if ang > half + EPS and clamp:
                    k = np.cross(a, axis)
                    if np.linalg.norm(k) < 1e-12:          # exactly opposite the axis: turn about the lateral axis
                        k = LATERAL - np.dot(LATERAL, a) * a
                    q = quat.mul(quat.from_axis_angle(k, ang - half), q)
                    rep.corrections.append(Correction(b.id, "cone", ang, half, ang - half))
            if "twist" in lim:
                a = quat.rotate(q, b.rest_axis)
                t = -quat.twist_angle(q, b.rest_axis, LATERAL)          # facing sense, as in pose.py
                rep.twist[b.id] = t
                lo, hi = lim["twist"]
                if not lo - EPS <= t <= hi + EPS and clamp:
                    tc = min(max(t, lo), hi)
                    swing = quat.shortest_arc(b.rest_axis, a, LATERAL)
                    q = quat.mul(quat.from_axis_angle(a, -tc), swing)
                    rep.corrections.append(Correction(b.id, "twist", t, tc, abs(t - tc)))

        elif lim.get("type") == "hinge":
            upper = skel.bone(b.parent_bone)
            Rpu = parent_world(upper)
            a_w = quat.rotate(world[upper.id], upper.rest_axis)
            b_w = quat.rotate(quat.mul(Rp, q), b.rest_axis)
            beta = _angle(a_w, b_w)
            lo, hi = lim["bend"]
            if beta > hi + EPS and clamp:
                n = np.cross(a_w, b_w)
                if np.linalg.norm(n) < 1e-9:      # folded exactly back on itself: no bend plane, use the neutral one
                    a_l = quat.rotate(quat.conj(Rpu), a_w)
                    neutral = FOLD_SIGN[lim["folds_toward"]] * quat.rotate(limb_base(upper, a_l, rules), LATERAL)
                    n = quat.rotate(Rpu, neutral)
                    n = n - np.dot(n, a_w) * a_w
                corr = quat.from_axis_angle(n, -(beta - hi))           # turn the lower bone back toward the upper
                q = quat.mul(quat.mul(quat.conj(Rp), quat.mul(corr, Rp)), q)
                rep.corrections.append(Correction(b.id, "bend", beta, hi, beta - hi))
                beta = hi
                b_w = quat.rotate(quat.mul(Rp, q), b.rest_axis)
            rep.hinge_bend[b.from_joint] = beta

            # bend direction: limb rotation of the upper bone, measured in the upper bone's parent frame
            if beta > min_bend:
                a_l = quat.rotate(quat.conj(Rpu), a_w)
                b_l = quat.rotate(quat.conj(Rpu), b_w)
                n_l = np.cross(a_l, b_l)
                n_l /= np.linalg.norm(n_l)
                neutral = FOLD_SIGN[lim["folds_toward"]] * quat.rotate(limb_base(upper, a_l, rules), LATERAL)
                psi = quat.signed_angle(neutral, n_l, a_l)
                rep.limb_rotation[upper.id] = psi
                rlo, rhi = upper.limit["rotation_range"]
                if not rlo - EPS <= psi <= rhi + EPS and clamp:
                    pc = min(max(psi, rlo), rhi)
                    # turning the upper bone about its own length carries the lower bone onto the allowed plane
                    rot[upper.id] = quat.normalize(quat.mul(quat.from_axis_angle(a_l, pc - psi), rot[upper.id]))
                    world[upper.id] = quat.mul(Rpu, rot[upper.id])
                    Rp = world[upper.id]
                    rep.corrections.append(Correction(b.id, "bend_direction", psi, pc, abs(pc - psi)))
            else:
                rep.limb_rotation[upper.id] = None

        rot[b.id] = quat.normalize(q)
        world[b.id] = quat.mul(Rp, rot[b.id])

    if clamp:
        rep.pose = Pose(rotations=rot, root_rotation=pose.root_rotation, root_x=pose.root_x, root_z=pose.root_z,
                        airborne=pose.airborne, pelvis_height=pose.pelvis_height)
    return rep
