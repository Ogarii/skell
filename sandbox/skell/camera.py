"""360-degree camera (spec 8.1-8.5): orbit model, projection, near plane, floor guard, shot sizes,
over-the-shoulder placement and camera validation. Geometry only; no drawing here.

Conventions (spec 8.1):
    d  = ( sin a cos t,  sin t,  -cos a cos t )      camera direction from the target
    C  = T + R d                                       camera position
    f  = -d,  r = ( cos a, 0, sin a ),  up = ( -sin t sin a,  cos t,  sin t cos a )
    u = (P-C).r,  v = (P-C).up,  w = (P-C).f
    u' = u cos p + v sin p,   v' = -u sin p + v cos p  (roll p)
    s = F / w,   screen offset (dx, dy) = (s u', -s v')   (screen y down)
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from .data import Skeleton, load_rules
from .quat import wrap_deg


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


def _cam_rule(key):
    return lambda: float(_rules()["camera"][key])


def direction(azimuth: float, elevation: float) -> np.ndarray:
    a, t = math.radians(azimuth), math.radians(elevation)
    return np.array([math.sin(a) * math.cos(t), math.sin(t), -math.cos(a) * math.cos(t)])


def basis(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward, right and up vectors. Orthonormal for every azimuth and elevation, including the poles."""
    a, t = math.radians(azimuth), math.radians(elevation)
    f = -direction(azimuth, elevation)
    r = np.array([math.cos(a), 0.0, math.sin(a)])
    up = np.array([-math.sin(t) * math.sin(a), math.cos(t), math.sin(t) * math.cos(a)])
    return f, r, up


def world_azimuth(facing: float, azimuth_rel: float) -> float:
    """Character-relative azimuth to world azimuth: alpha = facing + alpha_rel, wrapped to (-180, 180]."""
    return wrap_deg(facing + azimuth_rel)


@dataclass(frozen=True)
class Projection:
    dx: float        # x' - cx
    dy: float        # y' - cy, screen y down
    s: float         # perspective scale F / w
    w: float         # depth along the view direction
    visible: bool    # False inside the near plane (w < near)


@dataclass(frozen=True)
class Camera:
    """A resolved camera: world values only."""
    target: np.ndarray
    azimuth: float = 0.0
    elevation: float = 0.0
    roll: float = 0.0
    distance: float = field(default_factory=lambda: float(_rules()["camera"]["presets"]["eye_level"]["distance"]))
    focal: float = field(default_factory=_cam_rule("default_focal"))
    near: float = field(default_factory=_cam_rule("near_plane"))

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self.target, dtype=float) + self.distance * direction(self.azimuth, self.elevation)

    def basis(self):
        return basis(self.azimuth, self.elevation)

    def depth(self, P) -> float:
        f, _, _ = self.basis()
        return float((np.asarray(P, dtype=float) - self.position) @ f)

    def project(self, P) -> Projection:
        f, r, up = self.basis()
        v = np.asarray(P, dtype=float) - self.position
        u, vv, w = float(v @ r), float(v @ up), float(v @ f)
        if w <= 1e-12:
            return Projection(math.nan, math.nan, math.nan, w, False)
        rho = math.radians(self.roll)
        u2 = u * math.cos(rho) + vv * math.sin(rho)
        v2 = -u * math.sin(rho) + vv * math.cos(rho)
        s = self.focal / w
        return Projection(s * u2, -s * v2, s, w, w >= self.near)

    def project_all(self, positions: dict) -> dict[str, Projection]:
        return {k: self.project(p) for k, p in positions.items()}

    def clip_segment(self, P0, P1):
        """Clip a 3D segment to the near plane before projection. Returns (P0', P1') or None."""
        P0, P1 = np.asarray(P0, dtype=float), np.asarray(P1, dtype=float)
        w0, w1 = self.depth(P0), self.depth(P1)
        if w0 >= self.near and w1 >= self.near:
            return P0, P1
        if w0 < self.near and w1 < self.near:
            return None
        P = P0 + (self.near - w0) / (w1 - w0) * (P1 - P0)
        return (P, P1) if w0 < self.near else (P0, P)


@dataclass
class Subject:
    """A posed character as the camera sees it: grounded world joint positions."""
    skel: Skeleton
    positions: dict
    facing: float = 0.0
    head_rotation: np.ndarray | None = None   # world rotation of the head bone; orients the face cross (render)


@dataclass
class ResolvedCamera:
    camera: Camera
    notes: list[str] = field(default_factory=list)      # what Skell changed (shown in the editor)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- framing
def bounding_box(subject: Subject) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box around every joint and the head circle (as a sphere)."""
    pts = np.array(list(subject.positions.values()), dtype=float)
    head = np.asarray(subject.positions["head"], dtype=float)
    rad = subject.skel.head_radius
    pts = np.vstack([pts, head - rad, head + rad])
    return pts.min(axis=0), pts.max(axis=0)


def shot_distance(framed_height: float, focal: float, rules: dict) -> float:
    """R = F x framed height / (shot_fill x canvas height)  (spec 8.3)."""
    cam = rules["camera"]
    return focal * framed_height / (cam["shot_fill"] * cam["canvas_height"])


def shot_framing(shot: str, subject: Subject, focal: float, rules: dict,
                 view: tuple[float, float, float] | None = None) -> tuple[np.ndarray, float]:
    """Target and distance for a shot size, from the current pose's bounding box (spec 8.3).
    With view = (azimuth, elevation, roll), full and wide shots fit the pose as seen from that view."""
    spec = rules["camera"]["shots"].get(shot)
    if spec is None:
        raise ValueError(f"unknown shot size {shot!r}")
    lo, hi = bounding_box(subject)
    pos = subject.positions
    head_top = np.asarray(pos["head"], dtype=float) + np.array([0.0, subject.skel.head_radius, 0.0])
    framed = spec["framed"]
    if framed == "bbox":
        height = hi[1] - lo[1]
    elif framed.endswith("x_height"):
        height = float(framed.split("x_")[0]) * (hi[1] - lo[1])
    elif framed == "pelvis_to_head_top":
        height = head_top[1] - pos[subject.skel.root][1]
    elif framed == "neck_to_head_top":
        height = head_top[1] - pos["neck"][1]
    else:
        raise ValueError(f"unknown framing {framed!r}")
    tgt = spec["target"]
    if tgt == "bbox_centre":
        target = (lo + hi) / 2
    elif tgt == "region_midpoint":
        target = (np.asarray(pos[subject.skel.root], dtype=float) + head_top) / 2
    else:
        target = np.asarray(pos[tgt], dtype=float)
    R = shot_distance(abs(height), focal, rules)
    if view is not None and (framed == "bbox" or framed.endswith("x_height")):
        # frame the pose as projected from this view, not only its height (see docs/DECISIONS.md)
        fit = fit_distance(subject, target, focal, rules, *view)
        R = fit * (1.0 if framed == "bbox" else float(framed.split("x_")[0]))
    return target, R


def fit_distance(subject: Subject, target, focal: float, rules: dict, azimuth: float, elevation: float,
                 roll: float = 0.0) -> float:
    """Smallest distance at which every joint and the head circle fall inside the shot fill (80%) of the
    canvas, seen from this azimuth, elevation and roll. For a pose lying flat to the camera this equals
    F x height / (0.8 x canvas height), the spec 8.3 formula."""
    cr = rules["camera"]
    half_w, half_h = cr["shot_fill"] * cr["canvas_width"] / 2, cr["shot_fill"] * cr["canvas_height"] / 2
    _, r, up = basis(azimuth, elevation)
    rad = subject.skel.head_radius
    head = np.asarray(subject.positions["head"], dtype=float)
    pts = [np.asarray(p, dtype=float) for j, p in subject.positions.items() if j != "head"]
    pts += [head + rad * r, head - rad * r, head + rad * up, head - rad * up]

    def fits(R):
        cam = Camera(target=np.asarray(target, dtype=float), azimuth=azimuth, elevation=elevation, roll=roll,
                     distance=R, focal=focal, near=cr["near_plane"])
        for P in pts:
            p = cam.project(P)
            if not p.visible or abs(p.dx) > half_w * (1 + 1e-12) or abs(p.dy) > half_h * (1 + 1e-12):
                return False
        return True

    lo, hi = 0.0, float(cr["min_distance"])
    while not fits(hi):
        lo, hi = hi, hi * 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def over_the_shoulder(other_head, shoulder, standing_height: float, rules: dict):
    """Camera on the line from the other character's head through this character's shoulder,
    15% of standing height beyond the shoulder and 5% above it (spec 8.3).
    Returns (azimuth, elevation, distance, camera position) with the other head as target."""
    ots = rules["camera"]["over_the_shoulder"]
    other_head, shoulder = np.asarray(other_head, dtype=float), np.asarray(shoulder, dtype=float)
    away = shoulder - other_head
    away /= np.linalg.norm(away)
    C = shoulder + ots["behind_shoulder_frac_height"] * standing_height * away \
        + np.array([0.0, ots["above_shoulder_frac_height"] * standing_height, 0.0])
    d = C - other_head
    R = float(np.linalg.norm(d))
    d /= R
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(d[1])))))
    az = math.degrees(math.atan2(float(d[0]), float(-d[2]))) if abs(el) < 90 - 1e-9 else 0.0
    return az, el, R, C


def floor_guard(cam: Camera, standing_height: float, rules: dict) -> tuple[Camera, list[str], list[str]]:
    """Keep the camera at least 2% of standing height above the floor without changing the angle:
    move it along the same line and scale the focal length by the same factor (spec 8.3)."""
    h = rules["camera"]["floor_clearance_frac_height"] * standing_height
    T = np.asarray(cam.target, dtype=float)
    if cam.position[1] >= h - 1e-12:
        return cam, [], []
    sin_t = math.sin(math.radians(cam.elevation))
    R_new = (h - T[1]) / sin_t if abs(sin_t) > 1e-12 else -1.0
    if R_new <= 0:
        return cam, [], ["floor guard: no point on this camera line is above the floor; "
                         "raise the target or switch the floor guard off"]
    k = R_new / cam.distance
    new = Camera(target=T, azimuth=cam.azimuth, elevation=cam.elevation, roll=cam.roll,
                 distance=R_new, focal=cam.focal * k, near=cam.near)
    note = (f"floor guard: camera moved along its line from R {cam.distance:.1f} to {R_new:.1f} and focal "
            f"from {cam.focal:.1f} to {new.focal:.1f}, so it stays {h:.1f} above the floor at the same angle "
            f"and the target keeps its size")
    return new, [note], []


# ---------------------------------------------------------------- validation
def _fraction_in_frame(cam: Camera, subject: Subject, rules: dict) -> float:
    lo, hi = bounding_box(subject)
    corners = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    proj = [cam.project(c) for c in corners]
    if not all(p.visible for p in proj):
        return 0.0
    xs, ys = [p.dx for p in proj], [p.dy for p in proj]
    W, H = rules["camera"]["canvas_width"] / 2, rules["camera"]["canvas_height"] / 2
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))
    if area <= 0:
        return 1.0
    ix = max(0.0, min(max(xs), W) - max(min(xs), -W))
    iy = max(0.0, min(max(ys), H) - max(min(ys), -H))
    return ix * iy / area


def resolve_camera(request: dict, subject: Subject, other: Subject | None = None,
                   current: Camera | None = None, rules: dict | None = None) -> ResolvedCamera:
    """Resolve and check a camera request, set by hand or proposed by the AI (spec 8.5).

    request keys (all optional; None or missing keeps the current value, or the default):
      relative_to ("world" | "character"), target (joint name of the subject, or [x, y, z]),
      azimuth, elevation, roll, distance, focal, shot (shot size name),
      over_the_shoulder (shoulder joint name of the subject; needs `other`), floor_guard (bool).
    """
    rules = rules or _rules()
    cr = rules["camera"]
    notes: list[str] = []
    warnings: list[str] = []

    def pick(key, default):
        v = request.get(key)
        if v is not None:
            return float(v)
        return float(getattr(current, key)) if current is not None else float(default)

    az = pick("azimuth", 0.0)
    if request.get("relative_to") == "character" and request.get("azimuth") is not None:
        az = subject.facing + float(request["azimuth"])                       # 1. relative -> world
    el, roll = pick("elevation", 0.0), pick("roll", 0.0)
    R = pick("distance", cr["presets"]["eye_level"]["distance"])
    F = pick("focal", cr["default_focal"])
    tgt = request.get("target")
    if tgt is None:
        target = np.asarray(current.target, dtype=float) if current is not None else subject.positions["head"]
    elif isinstance(tgt, str):
        target = subject.positions[tgt]
    else:
        target = tgt
    target = np.asarray(target, dtype=float)
    target_subject = subject

    az, roll = wrap_deg(az), wrap_deg(roll)                                   # 2. wrap and clamp
    el = max(-90.0, min(90.0, el))

    if request.get("over_the_shoulder"):                                     # 3. shot and over-the-shoulder
        if other is None:
            raise ValueError("over-the-shoulder needs a second character")
        az, el, R, _ = over_the_shoulder(other.positions["head"], subject.positions[request["over_the_shoulder"]],
                                         subject.skel.standing_height, rules)
        az = wrap_deg(az)
        target, target_subject = np.asarray(other.positions["head"], dtype=float), other
    elif request.get("shot"):
        target, R = shot_framing(request["shot"], subject, F, rules, view=(az, el, roll))

    if R < cr["min_distance"]:                                                # 4. distance and focal limits
        notes.append(f"distance {R:.1f} raised to the minimum {cr['min_distance']}")
        R = float(cr["min_distance"])
    f_lo, f_hi = cr["focal_range"]
    if not f_lo <= F <= f_hi:
        F_new = float(min(max(F, f_lo), f_hi))
        notes.append(f"focal length {F:.1f} limited to {F_new:.0f}")
        F = F_new

    cam = Camera(target=target, azimuth=az, elevation=el, roll=roll, distance=R, focal=F, near=cr["near_plane"])
    guard = request.get("floor_guard")
    if (cr["floor_guard_default"] if guard is None else guard):               # 5. floor guard
        cam, n, w = floor_guard(cam, subject.skel.standing_height, rules)
        notes += n
        warnings += w

    C = cam.position                                                          # 6. warnings
    for s in [subject] + ([other] if other is not None else []):
        if any(np.linalg.norm(np.asarray(p) - C) < cam.near for p in s.positions.values()):
            warnings.append("camera is inside a character (a joint is closer than the near plane)")
            break
    if _fraction_in_frame(cam, target_subject, rules) < 0.5:
        warnings.append("more than half of the target character's bounding box is outside the frame")

    return ResolvedCamera(cam, notes, warnings)
