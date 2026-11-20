"""Pose checks (spec 5.3, 10.5): contacts, balance, overlap, continuity, and the validate() pipeline
that clamps to joint limits, runs forward kinematics and grounding, then checks the result.
Balance, overlap and continuity are warnings, never rejections.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from .data import Skeleton, load_rules
from .limits import LimitReport, apply_limits
from .pose import Pose, solve


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


# ---------------------------------------------------------------- contacts and balance
def duplicate_joints(skel: Skeleton) -> set[str]:
    """Joints placed by zero-length offsets: they coincide with their parent joint."""
    return {b.to_joint for b in skel.bones if b.length == 0}


def contacts(skel: Skeleton, positions: dict, rules: dict | None = None) -> list[str]:
    """Joints within 2% of standing height of the floor (spec 5.3)."""
    rules = rules or _rules()
    tol = rules["grounding"]["contact_tolerance_frac_height"] * skel.standing_height
    skip = duplicate_joints(skel)
    return sorted(j for j, p in positions.items() if p[1] <= tol + 1e-9 and j not in skip)


def centre_of_mass(skel: Skeleton, positions: dict, rules: dict | None = None) -> np.ndarray:
    """Segment masses at segment midpoints, the head at the head joint (spec 10.5)."""
    bal = (rules or _rules())["balance"]
    mf = bal["mass_fractions"]
    total, acc = 0.0, np.zeros(3)
    for seg, bones in bal["segments"].items():
        for bid in bones:
            b = skel.bone(bid)
            acc += mf[seg] * (np.asarray(positions[b.from_joint]) + np.asarray(positions[b.to_joint])) / 2
            total += mf[seg]
    acc += mf["head"] * np.asarray(positions[bal["head_joint"]])
    total += mf["head"]
    return acc / total


def _hull(points: list) -> list:
    """2D convex hull, counter-clockwise (monotone chain). Collinear points are dropped."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _dist_point_segment(p, a, b) -> float:
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def _inside(p, hull) -> bool:
    if len(hull) < 3:
        return False
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            return False
    return True


def _dist_to_hull(p, hull) -> float:
    if len(hull) == 1:
        return float(np.linalg.norm(np.asarray(p) - np.asarray(hull[0])))
    if _inside(p, hull):
        return 0.0
    n = len(hull)
    edges = [(hull[i], hull[(i + 1) % n]) for i in range(n if n > 2 else 1)]
    return min(_dist_point_segment(p, a, b) for a, b in edges)


@dataclass
class Balance:
    status: str                   # "stable", "unstable", "airborne" or "no_contact"
    centre_of_mass: np.ndarray
    distance_to_support: float    # from the centre of mass to the outline of the contact joints (0 inside)


def balance(skel: Skeleton, positions: dict, contact_joints: list[str], airborne: bool = False,
            rules: dict | None = None) -> Balance:
    """Centre of mass over the support area: the outline of the contacts, widened by 4% of standing
    height around feet and 2% around other contacts (spec 10.5)."""
    rules = rules or _rules()
    bal = rules["balance"]
    com = centre_of_mass(skel, positions, rules)
    if airborne:
        return Balance("airborne", com, math.nan)
    if not contact_joints:
        return Balance("no_contact", com, math.nan)
    P = (com[0], com[2])
    ground = [(positions[j][0], positions[j][2]) for j in contact_joints]
    dist = _dist_to_hull(P, _hull(ground))
    widened = []
    for j, (x, z) in zip(contact_joints, ground):
        r = bal["support_widen_frac_height"]["foot" if j in bal["foot_joints"] else "other"] * skel.standing_height
        widened += [(x + r * math.cos(t), z + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)]
    stable = _inside(P, _hull(widened))
    return Balance("stable" if stable else "unstable", com, dist)


# ---------------------------------------------------------------- overlap
def _segment_distance(p1, q1, p2, q2) -> float:
    """Closest distance between segments p1q1 and p2q2."""
    p1, q1, p2, q2 = map(lambda v: np.asarray(v, dtype=float), (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _clusters(skel: Skeleton) -> dict[str, int]:
    """Joints joined by rigid offsets form one cluster (chest: neck and shoulders; pelvis and hips)."""
    parent = {j: j for j in skel.joints}

    def find(j):
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j
    for b in skel.bones:
        if b.kind == "rigid":
            parent[find(b.to_joint)] = find(b.from_joint)
    roots = {}
    return {j: roots.setdefault(find(j), len(roots)) for j in skel.joints}


def _segment_hits_box(P0, P1, box) -> bool:
    """Segment against a box prop (render.Box-like: x, z, width, depth, height, yaw)."""
    a = math.radians(box.yaw)
    right = np.array([math.cos(a), 0.0, math.sin(a)])
    front = np.array([math.sin(a), 0.0, -math.cos(a)])
    c = np.array([box.x, box.height / 2, box.z])
    axes = (right, np.array([0.0, 1.0, 0.0]), front)
    half = (box.width / 2, box.height / 2, box.depth / 2)
    p0, d = np.asarray(P0, dtype=float) - c, np.asarray(P1, dtype=float) - np.asarray(P0, dtype=float)
    t0, t1 = 0.0, 1.0
    for ax_, h in zip(axes, half):
        o, v = float(p0 @ ax_), float(d @ ax_)
        if abs(v) < 1e-12:
            if abs(o) > h:
                return False
            continue
        ta, tb = sorted(((-h - o) / v, (h - o) / v))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def overlaps(skel: Skeleton, positions: dict, rules: dict | None = None, boxes=()) -> list[str]:
    """Non-adjacent bones closer than 1% of standing height, and bones passing through a box."""
    rules = rules or _rules()
    limit = rules["overlap"]["min_clearance_frac_height"] * skel.standing_height
    cl = _clusters(skel)
    bones = [b for b in skel.bones if b.length > 0]
    out = []
    for i, b1 in enumerate(bones):
        for b2 in bones[i + 1:]:
            if {cl[b1.from_joint], cl[b1.to_joint]} & {cl[b2.from_joint], cl[b2.to_joint]}:
                continue                                   # adjacent: they meet at a joint
            d = _segment_distance(positions[b1.from_joint], positions[b1.to_joint],
                                  positions[b2.from_joint], positions[b2.to_joint])
            if d < limit:
                out.append(f"{b1.id} and {b2.id} are {d:.1f} apart (minimum {limit:.1f})")
    for k, box in enumerate(boxes):
        for b in bones:
            if _segment_hits_box(positions[b.from_joint], positions[b.to_joint], box):
                out.append(f"{b.id} passes through box {k}")
    return out


# ---------------------------------------------------------------- continuity
def rotation_change(q0, q1) -> float:
    """Angle in degrees between two rotations (shortest way)."""
    d = min(1.0, abs(float(np.dot(q0, q1))))
    return math.degrees(2.0 * math.acos(d))


def continuity(skel: Skeleton, prev: Pose, pose: Pose, rules: dict | None = None, frames_apart: int = 1) -> list[str]:
    """A bone rotating more than 45 degrees per frame at 24 fps is flagged as a jump (spec 10.5).
    Bones are compared by their local rotation, the root by its world rotation."""
    rules = rules or _rules()
    limit = rules["continuity"]["max_bone_rotation_deg_per_frame"] * frames_apart
    out = []
    pairs = [("root", prev.root_rotation, pose.root_rotation)] + \
            [(b, prev.rotations[b], pose.rotations[b]) for b in skel.posable_ids]
    for name, q0, q1 in pairs:
        ang = rotation_change(q0, q1)
        if ang > limit:
            out.append(f"{name} turns {ang:.1f} deg in {frames_apart} frame(s) (maximum {limit:.0f})")
    return out


# ---------------------------------------------------------------- pipeline
@dataclass
class Validation:
    pose: Pose                         # clamped pose
    limits: LimitReport
    positions: dict
    directions: dict
    contacts: list[str]
    balance: Balance
    overlaps: list[str] = field(default_factory=list)
    continuity: list[str] = field(default_factory=list)

    def warnings(self) -> list[str]:
        out = [f"clamped {c}" for c in self.limits.corrections]
        if self.balance.status == "unstable":
            out.append(f"unstable: centre of mass {self.balance.distance_to_support:.1f} outside the contacts")
        if self.balance.status == "no_contact":
            out.append("no joint touches the floor")
        out += [f"overlap: {o}" for o in self.overlaps]
        out += [f"jump: {c}" for c in self.continuity]
        return out


def validate(skel: Skeleton, pose: Pose, prev: Pose | None = None, boxes=(), rules: dict | None = None,
             frames_apart: int = 1) -> Validation:
    """Limits (with clamping), forward kinematics and grounding, then contacts, balance, overlap and
    continuity (spec 10.5 steps 3, 4, 6, 7, 8)."""
    rules = rules or _rules()
    rep = apply_limits(skel, pose, rules)
    pos, dirs = solve(skel, rep.pose)
    cts = contacts(skel, pos, rules)
    bal = balance(skel, pos, cts, rep.pose.airborne, rules)
    ovl = overlaps(skel, pos, rules, boxes)
    cont = continuity(skel, prev, rep.pose, rules, frames_apart) if prev is not None else []
    return Validation(rep.pose, rep, pos, dirs, cts, bal, ovl, cont)
