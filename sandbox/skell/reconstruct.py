"""Sketch input (spec 4.2 Method B): joints placed over a drawing -> 3D pose, deterministically.

1. The root sits on its own ray. Its depth starts from the character's current position and slides along
   the ray to where the lowest joint lands on the floor, so grounding moves nothing and every joint stays
   on its drawn point. If no such depth exists for this camera, it keeps the current depth and says so.
2. Parent before child, cast a ray from the camera through each placed 2D joint and intersect it with
   the sphere around the (already placed) parent joint whose radius is the fixed bone length.
3. Two solutions (toward / away from the camera): by default take one that keeps the joint limits
   (looking one bone ahead, so a thigh is chosen with a knee that can bend the right way), then the one
   nearest the current pose; a flip takes the other. One solution (ray tangent): take it. None (drawn
   longer than it can ever look): take the nearest point on the sphere; flag it when more than 2% too long.
4. Convert to yaw/pitch, then rotations (twist rules), then validate like any other pose.

Joints the artist did not move keep their current 2D position. Rigid offset joints (shoulders, hips) are
not placed: they follow their parent bone.
"""
from __future__ import annotations

import copy
import functools
import math
from dataclasses import dataclass, field

import numpy as np

from . import quat
from .camera import Camera
from .checks import Validation, validate
from .data import Skeleton, load_rules
from .limits import cone_ok, hinge_ok
from .pose import direction_to_yaw_pitch, facing_quat, pose_from_frame, world_rotations


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


def sketch_camera(skel: Skeleton, rules: dict | None = None) -> Camera:
    """The standard sketch camera: the viewer's full shot of the standing pose, front, eye level."""
    from .camera import Subject, resolve_camera
    from .data import pose_library
    standing = next(p for p in pose_library() if p["name"] == "standing")["frame"]
    subj = Subject(skel, validate(skel, pose_from_frame(skel, standing), rules=rules).positions, 0.0)
    return resolve_camera({"shot": "full"}, subj, rules=rules).camera


def drawing_camera(skel: Skeleton, joints: dict, sigma: float | None = None, azimuth: float = 0.0,
                   focal: float = 500.0) -> Camera:
    """The eye-level camera in which a drawing has the character's size: sigma screen units per world unit
    (default: the drawing's own scale, drawing_scale) sets the distance, and the lowest drawn joint stands on
    the floor, which sets the height."""
    pts = {j: tuple(xy) for j, xy in joints.items()}
    for b in skel.bones:                       # zero-width shoulders and hips sit on their parent joint
        if b.kind == "rigid" and b.to_joint not in pts:
            pts[b.to_joint] = pts[b.from_joint]
    sigma = sigma or drawing_scale(skel, pts)
    floor_dy = max(y for _, y in joints.values())
    return Camera(target=np.array([0.0, floor_dy / sigma, 0.0]), azimuth=azimuth, elevation=0.0,
                  distance=focal / sigma, focal=focal)


def fit_drawing_camera(skel: Skeleton, joints: dict, frame: dict, quick: bool = False, azimuth: float = 0.0,
                       focal: float = 500.0, rules: dict | None = None) -> Camera:
    """The eye-level camera a (stylised) drawing implies: the scale at which the gesture-fit rebuild of the
    pose lands closest to the drawn joints (RMS over all joints). frame gives the depth choices.
    Searched between two first guesses: the drawing's median bone scale and the scale at which the start
    pose is as tall as the drawing. quick: 5 rebuilds; otherwise a grid and a golden-section refinement."""
    pts = {j: tuple(xy) for j, xy in joints.items()}
    for b in skel.bones:
        if b.kind == "rigid" and b.to_joint not in pts:
            pts[b.to_joint] = pts[b.from_joint]
    pos = validate(skel, pose_from_frame(skel, frame), rules=rules).positions
    tall = float(pos["head"][1] - min(p[1] for p in pos.values()))
    s_med = drawing_scale(skel, pts)
    s_tall = (max(y for _, y in joints.values()) - joints["head"][1]) / tall if tall > 1e-6 else s_med
    lo, hi = 0.95 * min(s_med, s_tall), 1.05 * max(s_med, s_tall)

    def camera(sigma):
        return drawing_camera(skel, joints, sigma, azimuth=azimuth, focal=focal)

    @functools.lru_cache(maxsize=None)
    def misfit(sigma):
        cam = camera(sigma)
        rec = reconstruct(skel, cam, joints, frame, rules=rules, fit="gesture")
        p = rec.validation.positions
        d = [math.hypot(cam.project(p[j]).dx - x, cam.project(p[j]).dy - y) for j, (x, y) in joints.items()]
        return float(np.sqrt(np.mean(np.square(d))))

    grid = list(np.geomspace(lo, hi, 5 if quick else 9))
    i = int(np.argmin([misfit(s) for s in grid]))
    if quick:
        return camera(grid[i])
    a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (5 ** 0.5 - 1) / 2                                 # golden-section refinement around the best
    for _ in range(8):
        c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
        if misfit(c_) < misfit(d_):
            b_ = d_
        else:
            a_ = c_
    return camera((a_ + b_) / 2)


def placement_order(skel: Skeleton) -> list[str]:
    """Joints an artist places over a sketch, parent before child: the root, then the end joint of each
    posable bone. Rigid offset joints (shoulders, hips) are not placed; they follow their parent."""
    return [skel.root] + [b.to_joint for b in skel.bones if b.kind == "posable"]


def screen_ray(cam: Camera, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and unit direction of the ray through screen offset (dx, dy) (screen y down)."""
    f, r, up = cam.basis()
    u2, v2 = dx / cam.focal, -dy / cam.focal          # per unit depth, after roll
    rho = math.radians(cam.roll)
    u = u2 * math.cos(rho) - v2 * math.sin(rho)       # undo the roll
    v = u2 * math.sin(rho) + v2 * math.cos(rho)
    d = f + u * r + v * up
    return cam.position, d / np.linalg.norm(d)


def ray_sphere(C, D, O, L: float) -> list[float]:
    """Ray parameters t > 0 where C + t D meets the sphere |P - O| = L. Two, one (tangent) or none."""
    oc = np.asarray(C, dtype=float) - np.asarray(O, dtype=float)
    b = float(D @ oc)
    c = float(oc @ oc) - L * L
    disc = b * b - c
    eps = 1e-9 * max(1.0, L * L)
    if disc < -eps:
        return []
    if disc <= eps:
        return [-b] if -b > 0 else []
    s = math.sqrt(disc)
    return [t for t in (-b - s, -b + s) if t > 0]


@dataclass
class BoneSolve:
    bone: str
    solutions: int                    # 2, 1 (tangent) or 0 (drawn too long)
    candidates: list                  # 3D points for the child joint (the two depths, or the nearest point)
    chosen: int                       # index into candidates
    flipped: bool = False
    too_long_pct: float | None = None  # how much longer the drawn bone is than it can ever look
    flagged: bool = False


@dataclass
class Reconstruction:
    positions: dict                   # reconstructed joints, before validation and grounding
    frame: dict                       # yaw/pitch frame (spec 10.4) of the reconstructed pose
    validation: Validation            # clamped, grounded and checked
    bones: dict[str, BoneSolve] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _floor_depth(trace, w0: float, gesture: bool = False) -> float | None:
    """Root depth near w0 at which the lowest reconstructed joint sits at y = 0, or None.

    Only depths where every bone keeps the same kind of solution as at w0 count: sliding the figure until
    a ray just grazes its sphere would silently undo a depth choice (or a flip)."""
    def probe(w):
        pos, bones = trace(w)
        # gesture bones sit near the flat limit, where solution counts flicker: there, only new flags count
        sig = tuple(i.flagged for i in bones.values()) if gesture else tuple(i.solutions for i in bones.values())
        return min(float(p[1]) for p in pos.values()), sig
    f0, sig0 = probe(w0)
    if abs(f0) < 1e-9:
        return w0
    ws = sorted(set(np.concatenate([w0 * np.geomspace(0.4, 2.5, 48), [w0]])))
    probes = [probe(w) for w in ws]
    brackets = [(a, b) for (a, (fa, sa)), (b, (fb, sb)) in zip(zip(ws, probes), zip(ws[1:], probes[1:]))
                if fa * fb <= 0 and sa == sb == sig0]
    if not brackets:
        return None
    a, b = min(brackets, key=lambda ab: abs(math.log(((ab[0] + ab[1]) / 2) / w0)))
    fa = probe(a)[0]
    for _ in range(60):
        m = (a + b) / 2
        fm = probe(m)[0]
        a, b, fa = (m, b, fm) if fa * fm > 0 else (a, m, fa)
    w = (a + b) / 2
    fw, sw = probe(w)
    return w if abs(fw) < 1e-6 and sw == sig0 else None


def drawing_scale(skel: Skeleton, pts: dict) -> float:
    """Median drawn length / true length over the posable bones: the drawing's own scale."""
    posable = [b for b in skel.bones if b.kind == "posable"]
    return float(np.median([np.hypot(*np.subtract(pts[b.to_joint], pts[b.from_joint])) / b.length
                            for b in posable]))


def _screen_dir_3d(cam: Camera, u) -> np.ndarray:
    """The 3D direction, perpendicular to the view, that runs along screen direction u (unit, y down)."""
    _, r, up = cam.basis()
    rho = math.radians(cam.roll)
    u2, v2 = float(u[0]), -float(u[1])
    a = u2 * math.cos(rho) - v2 * math.sin(rho)
    b = u2 * math.sin(rho) + v2 * math.cos(rho)
    e = a * r + b * up
    return e / np.linalg.norm(e)


def reconstruct(skel: Skeleton, cam: Camera, placements: dict, current_frame: dict, flips: dict | None = None,
                rules: dict | None = None, fit: str = "exact") -> Reconstruction:
    """Rebuild the pose from 2D joint placements (screen offsets, y down) seen through cam.

    placements: joint -> (dx, dy); joints left out keep their current screen position.
    current_frame: the pose being edited (gives depths, the default depth choice, facings).
    flips: bone id -> True to take the other of its two depth solutions.
    fit: "exact"   drawn lengths are true (a drawing in the character's own proportions): every joint
                   lands on its drawn point.
         "gesture" a stylised drawing (stick figures with their own proportions): every bone keeps its drawn
                   2D direction but gets the character's length, nearly flat to the view; a bone drawn
                   clearly short (under gesture_keep_short_frac of the drawing's scale) keeps that
                   foreshortening. The size follows the camera and the floor, not the drawing.
    """
    rules = rules or _rules()
    flips = flips or {}
    if fit not in ("exact", "gesture"):
        raise ValueError(f"unknown fit {fit!r}")
    tol_pct = 100.0 * rules["limits"]["setup_drawn_length_tolerance_frac"]
    keep_short = rules["sketch"]["gesture_keep_short_frac"]
    prev_pose = pose_from_frame(skel, current_frame)
    prev_val = validate(skel, prev_pose, rules=rules)
    prev_pos = prev_val.positions
    prev_world = world_rotations(skel, prev_val.pose)
    prev_dir = {b.id: quat.rotate(prev_world[b.id], b.rest_axis) for b in skel.bones}

    rec = Reconstruction(positions={}, frame={}, validation=None)
    rigid_joints = {b.to_joint for b in skel.bones if b.kind == "rigid"}
    pts = {}
    for j, P in prev_pos.items():
        p = cam.project(P)
        pts[j] = (p.dx, p.dy)
    for j, xy in placements.items():
        if j in rigid_joints:
            rec.notes.append(f"{j} follows its parent bone (rigid offset); its placement is ignored")
            continue
        pts[j] = (float(xy[0]), float(xy[1]))
    # Rigid joints (shoulders, hips) are never placed: carry them with their drawn parent, keeping their screen
    # offset from the current pose, so the limbs hanging off them are measured from the drawing (with zero-width
    # shoulders and hips they sit on the drawn neck and pelvis), not from where the old pose had them.
    for b in skel.bones:
        if b.kind == "rigid":
            a, c = cam.project(prev_pos[b.from_joint]), cam.project(prev_pos[b.to_joint])
            pts[b.to_joint] = (pts[b.from_joint][0] + (c.dx - a.dx), pts[b.from_joint][1] + (c.dy - a.dy))

    root = skel.root
    root_rot = prev_val.pose.root_rotation
    f, _, _ = cam.basis()
    w_start = cam.depth(prev_pos[root])
    sigma = drawing_scale(skel, pts)
    if fit == "gesture":
        w_start = cam.focal / sigma          # where the drawing's scale is the character's: a starting guess
        rec.notes.append("gesture fit: bone directions from the drawing, lengths from the character")
    C_root, D_root = screen_ray(cam, *pts[root])

    def target(b, O) -> tuple:
        """The 2D point to cast the ray through for bone b, whose parent joint is at O."""
        if fit == "exact":
            return pts[b.to_joint]
        v = np.subtract(pts[b.to_joint], pts[b.from_joint])
        drawn = float(np.hypot(*v))
        p0 = cam.project(O)
        if drawn < 1e-9:
            return (p0.dx, p0.dy)
        u = v / drawn
        q = cam.project(O + b.length * _screen_dir_3d(cam, u))          # the bone lying flat along u
        flat = 0.995 * float(np.hypot(q.dx - p0.dx, q.dy - p0.dy))
        length = min(drawn * p0.s / sigma, flat) if drawn < keep_short * sigma * b.length else flat
        return (p0.dx + u[0] * length, p0.dy + u[1] * length)

    def trace(w: float):
        """Place every joint with the root at depth w along its ray."""
        pos = {root: C_root + (w / float(D_root @ f)) * D_root}
        new_world: dict[str, np.ndarray] = {}
        bones: dict[str, BoneSolve] = {}
        for b in skel.bones:
            parent_rot = new_world[b.parent_bone] if b.parent_bone is not None else root_rot
            O = pos[b.from_joint]
            if b.kind == "rigid":
                new_world[b.id] = parent_rot
                pos[b.to_joint] = O + b.length * quat.rotate(parent_rot, b.rest_axis)
                continue
            C, D = screen_ray(cam, *target(b, O))
            ts = ray_sphere(C, D, O, b.length)
            if len(ts) == 2:
                cands = [C + t * D for t in ts]
                dirs = [(c - O) / b.length for c in cands]
                # default: a depth that keeps the joint limits, then the one nearest the current pose
                ok = [_limits_ok(b, d, parent_rot, new_world, pos) for d in dirs]
                near = [float(d @ prev_dir[b.id]) for d in dirs]
                chosen = max((0, 1), key=lambda i: (ok[i], near[i]))
                flipped = bool(flips.get(b.id, False))
                if flipped:
                    chosen = 1 - chosen
                info = BoneSolve(b.id, 2, cands, chosen, flipped)
            elif len(ts) == 1:
                info = BoneSolve(b.id, 1, [C + ts[0] * D], 0)
            else:
                t0 = max(0.0, -float(D @ (C - O)))         # closest point on the ray to the parent joint
                Q = C + t0 * D
                d = float(np.linalg.norm(Q - O))
                pct = 100.0 * (d / b.length - 1.0)
                info = BoneSolve(b.id, 0, [O + b.length * (Q - O) / d], 0, too_long_pct=pct, flagged=pct > tol_pct)
            bones[b.id] = info
            child = info.candidates[info.chosen]
            pos[b.to_joint] = child
            new_world[b.id] = quat.mul(quat.shortest_arc(prev_dir[b.id], (child - O) / b.length), prev_world[b.id])
        return pos, bones

    hinge_child = {b.parent_bone: b for b in skel.bones if (b.limit or {}).get("type") == "hinge"}

    def _limits_ok(b, d_world, parent_rot, new_world, pos) -> bool:
        if (b.limit or {}).get("type") == "hinge":
            upper = skel.bone(b.parent_bone)
            Rpu = new_world[upper.parent_bone] if upper.parent_bone is not None else root_rot
            a_w = (pos[b.from_joint] - pos[upper.from_joint]) / upper.length
            inv = quat.conj(Rpu)
            return hinge_ok(upper, b, quat.rotate(inv, a_w), quat.rotate(inv, d_world), rules)
        if not cone_ok(skel, b, quat.rotate(quat.conj(parent_rot), d_world)):
            return False
        lower = hinge_child.get(b.id)
        if lower is None:
            return True
        # look one bone ahead: an upper arm or thigh depth is only good if its elbow or knee can bend right
        E = pos[b.from_joint] + b.length * d_world
        C, D = screen_ray(cam, *target(lower, E))
        ts = ray_sphere(C, D, E, lower.length)
        inv = quat.conj(parent_rot)
        a_l = quat.rotate(inv, d_world)
        if not ts:
            t0 = max(0.0, -float(D @ (C - E)))
            Q = C + t0 * D
            ends = [E + lower.length * (Q - E) / np.linalg.norm(Q - E)]
        else:
            ends = [C + t * D for t in ts]
        return any(hinge_ok(b, lower, a_l, quat.rotate(inv, (P - E) / lower.length), rules) for P in ends)

    # root depth: slide the root along its ray until the lowest joint is on the floor (then grounding
    # moves nothing); if no such depth keeps the solutions as they are, keep the start depth and say so
    w = w_start
    if not current_frame.get("airborne"):
        w = _floor_depth(trace, w_start, gesture=(fit == "gesture"))
        if w is None:
            w = w_start
            rec.notes.append("the drawing does not reach the floor from this view; grounding moved the figure")
    pos, rec.bones = trace(w)
    for info in rec.bones.values():
        if info.flagged:
            rec.flags.append(f"{info.bone} drawn too long by {info.too_long_pct:.1f}%")

    # yaw/pitch in the character's heading frame, facings unchanged
    frame = copy.deepcopy(current_frame)
    inv_heading = quat.conj(facing_quat(prev_val.pose.facing))
    for b in skel.bones:
        if b.kind != "posable":
            continue
        d = (pos[b.to_joint] - pos[b.from_joint]) / b.length
        prev_yaw = current_frame["pose"][b.id]["yaw"]
        yaw, pitch = direction_to_yaw_pitch(quat.rotate(inv_heading, d), prev_yaw)
        frame["pose"][b.id] = {"yaw": yaw, "pitch": pitch}
    root_p = pos[root]
    frame.setdefault("root", {}).update({"x": float(root_p[0]), "z": float(root_p[2])})
    if frame.get("airborne"):
        frame["root"]["y"] = float(root_p[1])

    rec.positions = pos
    rec.frame = frame
    rec.validation = validate(skel, pose_from_frame(skel, frame, prev=prev_val.pose), rules=rules)
    return rec
