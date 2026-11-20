"""Synthetic stick-figure drawings with exact joint labels: training data for a joint-finding model.

Every drawing is a real Skell pose (validated: joint limits, grounding) seen from a random camera, then drawn
the way people draw stick figures: their own proportions, wobbly strokes, open or floating heads, curved limbs.
The labels are the 11 placed joints (sketch_detect.PLACED order, skell_v1 names) in image pixels, with pixel
(row, col) covering [col, col + 1) x [row, row + 1), so a pixel centre is at (col + 0.5, row + 0.5).

Pure numpy (plus the kernel), so it runs anywhere, including a cloud notebook. Not part of the kernel.
Shared preprocessing for real drawings: frame_drawing (crop to the ink, square, resample).
"""
from __future__ import annotations

import copy
import math

import numpy as np

from sandbox.sketch_detect import PLACED, ink_mask
from sandbox.skell import data
from sandbox.skell.camera import Camera
from sandbox.skell.checks import validate
from sandbox.skell.pose import pose_from_frame

# 2D drawing tree: joint -> parent (the stroke from the parent ends at the joint)
PARENT = {"pelvis": None, "neck": "pelvis", "head": "neck", "l_elbow": "neck", "l_hand": "l_elbow",
          "r_elbow": "neck", "r_hand": "r_elbow", "l_knee": "pelvis", "l_foot": "l_knee", "r_knee": "pelvis",
          "r_foot": "r_knee"}
DRAW_ORDER = ["neck", "head", "l_elbow", "l_hand", "r_elbow", "r_hand", "l_knee", "l_foot", "r_knee", "r_foot"]
POSABLE = ["pelvis>neck", "neck>head", "l_shoulder>l_elbow", "l_elbow>l_hand", "r_shoulder>r_elbow",
           "r_elbow>r_hand", "l_hip>l_knee", "l_knee>l_foot", "r_hip>r_knee", "r_knee>r_foot"]


# ---------------------------------------------------------------- poses

def _jitter(frame: dict, rng, deg: float) -> dict:
    fr = copy.deepcopy(frame)
    for b in fr["pose"].values():
        b["yaw"] = float(b["yaw"] + rng.normal(0, deg))
        b["pitch"] = float(np.clip(b["pitch"] + rng.normal(0, deg), -90, 90))
    f = float(fr.get("root", {}).get("facing", 0.0))
    fr.setdefault("facing", {})
    fr["facing"]["chest"] = float(fr["facing"].get("chest", f) + rng.normal(0, deg / 2))
    fr["facing"]["head"] = float(fr["facing"].get("head", f) + rng.normal(0, deg / 2))
    fr["airborne"] = False
    return fr


def _uniform_dir(rng) -> tuple[float, float]:
    """A direction uniform on the sphere, as (yaw, pitch) degrees."""
    return float(rng.uniform(-180, 180)), float(math.degrees(math.asin(rng.uniform(-1, 1))))


def random_frame(rng, library: list | None = None) -> tuple[dict, str]:
    """A random frame (spec 10.4 format) and its family name. Families:
    library  a library pose with every bone jittered (~20 deg)
    upright  spine 55-90 deg above horizontal, limbs anywhere
    bent     spine 10-55 deg (bending, crawling, kneeling forward)
    lying    spine within 12 deg of horizontal
    The joint limits are applied later, by validation."""
    library = library if library is not None else [p for p in data.pose_library() if not p["frame"].get("airborne")]
    family = rng.choice(["library", "upright", "bent", "lying"], p=[0.4, 0.3, 0.18, 0.12])
    if family == "library":
        base = library[rng.integers(len(library))]["frame"]
        return _jitter(base, rng, 20.0), "library"
    spine_pitch = {"upright": (55, 90), "bent": (10, 55), "lying": (-12, 12)}[family]
    sy, sp = float(rng.uniform(-40, 40)), float(rng.uniform(*spine_pitch))
    pose = {"pelvis>neck": {"yaw": sy, "pitch": sp},
            "neck>head": {"yaw": sy + float(rng.normal(0, 20)), "pitch": float(np.clip(sp + rng.normal(0, 20), -90, 90))}}
    for b in POSABLE[2:]:
        y, p = _uniform_dir(rng)
        if "knee" in b.split(">")[1] or "foot" in b.split(">")[1]:
            if family == "upright" and rng.random() < 0.7:      # standing legs mostly point down
                p = float(rng.uniform(-90, -40))
        pose[b] = {"yaw": y, "pitch": p}
    frame = {"t": 1, "root": {"x": 0, "z": 0, "facing": 0.0},
             "facing": {"chest": float(rng.normal(0, 15)), "head": float(rng.normal(0, 25))},
             "airborne": False, "pose": pose}
    return frame, str(family)


def pose_views(frame: dict, rng, n_views: int, skel=None, rules=None):
    """Validate a frame and see it from n_views random cameras. Yields (joints2d (11, 2) screen offsets
    y down, head radius on screen, camera dict)."""
    skel = skel or data.load_skeleton()
    pose = pose_from_frame(skel, frame)
    pos = validate(skel, pose, rules=rules).positions
    P = np.array([pos[j] for j in PLACED], dtype=float)
    centre = (P.min(0) + P.max(0)) / 2                     # the drawing is re-framed in 2D, so aim anywhere
    for _ in range(n_views):
        az = float(rng.uniform(-180, 180))
        R = float(rng.uniform(350, 1000))                  # nearer = stronger perspective
        el_min = -math.degrees(math.asin(min(1.0, (centre[1] - 15.0) / R)))   # keep the camera above the floor
        el = float(np.clip(rng.normal(8, 15), max(-20.0, el_min), 70))
        cam = Camera(target=centre, azimuth=az, elevation=el, distance=R, focal=500.0)
        pr = [cam.project(p) for p in P]
        J = np.array([[p.dx, p.dy] for p in pr], dtype=float)
        yield J, skel.head_radius * pr[PLACED.index("head")].s, {"azimuth": az, "elevation": el, "distance": R}


# ---------------------------------------------------------------- the drawing style

def stylise(J: np.ndarray, head_r: float, rng) -> tuple[np.ndarray, float]:
    """People's own proportions: rescale each drawn segment from its parent (sides alike), a bigger or
    smaller head, a slight tilt. Labels follow."""
    idx = {j: i for i, j in enumerate(PLACED)}
    f = {"torso": rng.uniform(0.75, 1.25), "upper_arm": rng.uniform(0.8, 1.35), "forearm": rng.uniform(0.8, 1.35),
         "thigh": rng.uniform(0.8, 1.25), "shin": rng.uniform(0.8, 1.25), "neck": rng.uniform(0.6, 1.4)}
    group = {"neck": "torso", "head": "neck", "l_elbow": "upper_arm", "r_elbow": "upper_arm", "l_hand": "forearm",
             "r_hand": "forearm", "l_knee": "thigh", "r_knee": "thigh", "l_foot": "shin", "r_foot": "shin"}
    out = J.copy()
    for j in DRAW_ORDER:
        p = PARENT[j]
        out[idx[j]] = out[idx[p]] + f[group[j]] * rng.uniform(0.93, 1.07) * (J[idx[j]] - J[idx[p]])
    a = math.radians(rng.normal(0, 4))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = out[idx["pelvis"]]
    out = (out - c) @ R.T + c
    return out, float(head_r * rng.uniform(0.9, 1.9))


def _wobble(P: np.ndarray, amp: float, rng, step: float = 4.0) -> np.ndarray:
    """Resample a polyline every `step` px and push it sideways by a smooth random wave."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return P
    t = np.linspace(0.0, s[-1], max(2, int(s[-1] / step) + 1))
    Q = np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], axis=1)
    if amp <= 0 or len(Q) < 3:
        return Q
    d = np.gradient(Q, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    w = np.zeros(len(Q))
    for _ in range(2):
        lam = rng.uniform(20, 80)
        w += rng.uniform(0.3, 1.0) * np.sin(2 * math.pi * t / lam + rng.uniform(0, 2 * math.pi))
    w *= np.clip(np.sin(math.pi * t / s[-1]), 0.0, 1.0) ** 0.5               # ends stay put (labels sit on the ends)
    return Q + amp * w[:, None] * n


def _limb(S, M, E, rng, curved: bool) -> np.ndarray:
    if not curved:
        return np.array([S, M, E])
    C = 2 * M - (S + E) / 2                                # quadratic through M at t = 0.5
    t = np.linspace(0, 1, 24)[:, None]
    return (1 - t) ** 2 * S + 2 * (1 - t) * t * C + t ** 2 * E


def strokes_for(J: np.ndarray, head_r: float, rng) -> list[np.ndarray]:
    """The strokes of a stick figure through joints J (image px)."""
    j = {n: J[i] for i, n in enumerate(PLACED)}
    out = []
    # head: a wobbly circle, often open or overshooting
    a0 = rng.uniform(0, 2 * math.pi)
    sweep = 2 * math.pi + (math.radians(rng.uniform(0, 30)) if rng.random() < 0.5 else -math.radians(rng.uniform(0, 30)))
    ang = np.linspace(a0, a0 + sweep, 48)
    rr = head_r * (1 + 0.05 * np.sin(ang * rng.integers(2, 4) + rng.uniform(0, 6)))
    out.append(np.stack([j["head"][0] + rr * np.cos(ang), j["head"][1] + rr * np.sin(ang)], axis=1))
    # body up to the head circle (sometimes a little gap, sometimes into the circle)
    u = j["head"] - j["neck"]
    du = float(np.linalg.norm(u))
    edge = j["head"] - (u / du) * head_r * rng.uniform(0.85, 1.35) if du > head_r else j["neck"]
    out.append(np.array([j["pelvis"], j["neck"], edge]))
    torso = j["pelvis"] - j["neck"]
    for side in ("l", "r"):
        shoulder = j["neck"] + torso * (rng.uniform(0.0, 0.2) if rng.random() < 0.5 else 0.0)
        out.append(_limb(shoulder, j[f"{side}_elbow"], j[f"{side}_hand"], rng, rng.random() < 0.3))
        out.append(_limb(j["pelvis"], j[f"{side}_knee"], j[f"{side}_foot"], rng, rng.random() < 0.25))
    if rng.random() < 0.12:                                 # a ground line under the feet
        y = max(j["l_foot"][1], j["r_foot"][1]) + rng.uniform(1, 6)
        x0, x1 = sorted([j["l_foot"][0], j["r_foot"][0]])
        out.append(np.array([[x0 - rng.uniform(10, 40), y], [x1 + rng.uniform(10, 40), y]]))
    return out


def rasterise(strokes, size: int, width: float, ink: float = 1.0) -> np.ndarray:
    """Antialiased strokes, size x size floats: 1 white, 0 black."""
    canvas = np.zeros((size, size))
    r = width / 2.0
    for P in strokes:
        for A, B in zip(P[:-1], P[1:]):
            x0 = int(max(0, math.floor(min(A[0], B[0]) - r - 2)))
            x1 = int(min(size, math.ceil(max(A[0], B[0]) + r + 2)))
            y0 = int(max(0, math.floor(min(A[1], B[1]) - r - 2)))
            y1 = int(min(size, math.ceil(max(A[1], B[1]) + r + 2)))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
            AB = B - A
            L2 = float(AB @ AB)
            if L2 > 1e-12:
                t = np.clip(((xx - A[0]) * AB[0] + (yy - A[1]) * AB[1]) / L2, 0.0, 1.0)
            else:
                t = np.zeros_like(xx)
            d = np.hypot(xx - (A[0] + t * AB[0]), yy - (A[1] + t * AB[1]))
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], np.clip(r + 0.5 - d, 0.0, 1.0))
    return 1.0 - ink * canvas


def draw_sample(J: np.ndarray, head_r: float, rng, size: int = 192):
    """Stylise and draw one figure, framed in a size x size image. Returns (image uint8, joints (11, 2) px,
    head radius px, style dict)."""
    J, head_r = stylise(J, head_r, rng)
    head = J[PLACED.index("head")]
    lo = np.minimum(J.min(0), head - head_r)               # the whole figure, head circle included
    hi = np.maximum(J.max(0), head + head_r)
    margin = rng.uniform(0.04, 0.15)
    scale = size * (1 - 2 * margin) / max(float((hi - lo).max()), 1e-6)
    free = size - scale * (hi - lo)
    off = rng.uniform(0.25, 0.75, size=2) * free - scale * lo
    J = J * scale + off
    head_r *= scale
    width = float(rng.uniform(1.2, 7.5))                   # real small drawings framed up get thick strokes
    amp = float(rng.uniform(0, 0.012) * size)
    strokes = [_wobble(P, amp, rng) for P in strokes_for(J, head_r, rng)]
    img = rasterise(strokes, size, width, ink=float(rng.uniform(0.75, 1.0)))
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)
    return (img * 255).round().astype(np.uint8), J.astype(np.float32), float(head_r), \
        {"width": width, "wobble": amp, "margin": margin}


def generate(n: int, seed: int = 0, size: int = 192, views_per_pose: int = 3):
    """n samples: images (n, size, size) uint8, joints (n, 11, 2) float32, head radii (n,), meta list."""
    rng = np.random.default_rng(seed)
    skel, rules = data.load_skeleton(), data.load_rules()
    library = [p for p in data.pose_library() if not p["frame"].get("airborne")]
    imgs = np.zeros((n, size, size), dtype=np.uint8)
    joints = np.zeros((n, len(PLACED), 2), dtype=np.float32)
    radii = np.zeros(n, dtype=np.float32)
    meta = []
    k = 0
    skipped = 0
    while k < n:
        frame, family = random_frame(rng, library)
        try:
            with np.errstate(invalid="raise", divide="raise"):
                views = list(pose_views(frame, rng, views_per_pose, skel, rules))
        except (FloatingPointError, ValueError):      # a degenerate pose: skip it rather than stop a long run
            skipped += 1
            if skipped > max(10, n // 100):
                raise RuntimeError(f"generate: {skipped} poses failed, something is wrong")
            continue
        for J, hr, cam in views:
            if k >= n:
                break
            if not (np.isfinite(J).all() and np.isfinite(hr)):
                skipped += 1
                continue
            imgs[k], joints[k], radii[k], style = draw_sample(J, hr, rng, size)
            meta.append({"family": family, "camera": cam, "style": style, "pose": frame["pose"],
                         "facing": frame["facing"]})
            k += 1
    return imgs, joints, radii, meta


# ---------------------------------------------------------------- mirroring

def _other_side(j: str) -> str:
    return "r_" + j[2:] if j.startswith("l_") else "l_" + j[2:] if j.startswith("r_") else j


# MIRROR[i] = index of the joint that joint i becomes in a mirrored drawing (the character's left becomes its right)
MIRROR = [PLACED.index(_other_side(j)) for j in PLACED]
# Left/right pairs a stick figure can't tell apart from one drawing (arms and legs swap independently);
# the geometry fit decides the sides. Each entry: the PLACED indices of one side, then the other.
SIDE_PAIRS = [([PLACED.index(j) for j in ("l_elbow", "l_hand")], [PLACED.index(j) for j in ("r_elbow", "r_hand")]),
              ([PLACED.index(j) for j in ("l_knee", "l_foot")], [PLACED.index(j) for j in ("r_knee", "r_foot")])]


def mirror(img: np.ndarray, joints: np.ndarray):
    """Mirror a drawing left-right with its labels: x -> width - x (pixel centres col + 0.5 map to
    (width - 1 - col) + 0.5) and the l_/r_ labels swap. A symmetric skeleton makes the mirror a valid pose.
    img (..., H, W); joints (..., 11, 2). Mirroring twice gives the original back."""
    w = img.shape[-1]
    J = np.array(joints, dtype=np.float32, copy=True)
    J[..., 0] = w - J[..., 0]
    return img[..., ::-1].copy(), J[..., MIRROR, :]


# ---------------------------------------------------------------- shared preprocessing for real drawings

def frame_drawing(img: np.ndarray, size: int = 192, margin: float = 0.08):
    """A real drawing framed like the training images: crop to the ink, pad to a square with a margin,
    resample to size x size (box filter). Returns (image uint8, (scale, ox, oy)) with
    model px = (image px - (ox, oy)) * scale."""
    ink = ink_mask(img)
    if not ink.any():
        raise ValueError("no ink found in the drawing")
    a = np.asarray(img, dtype=float)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    rows, cols = np.nonzero(ink)
    x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    side = max(x1 - x0, y1 - y0) / (1 - 2 * margin)
    ox, oy = (x0 + x1) / 2 - side / 2, (y0 + y1) / 2 - side / 2
    scale = size / side
    # box-filter resampling: average the source pixels whose centres fall in each target pixel (4x4 samples)
    sub = 4
    g = (np.arange(size * sub) + 0.5) / (size * sub) * side
    X, Y = np.meshgrid(ox + g, oy + g)
    ci, ri = np.floor(X).astype(int), np.floor(Y).astype(int)
    inside = (ci >= 0) & (ci < a.shape[1]) & (ri >= 0) & (ri < a.shape[0])
    vals = np.ones_like(X)
    vals[inside] = a[ri[inside], ci[inside]]
    out = vals.reshape(size, sub, size, sub).mean(axis=(1, 3))
    return (out * 255).round().astype(np.uint8), (scale, ox, oy)
