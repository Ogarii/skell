"""Find a stick figure's joints in a drawing with no AI: the offline pre-fill for sketch input (spec 4.2
Method B). Pure numpy; image arrays in, pixel joints out. Not part of the geometry kernel.

Pipeline
  1. ink mask (dark pixels)
  2. head: the round enclosed hole (a drawn circle); its ring is removed from the ink
  3. centre-line of the remaining strokes (Zhang-Suen thinning)
  4. fit an 11-joint stick figure to the centre-line, started from every library pose at several facings:
     alternate "which segment explains this ink point / which ink point does this segment sit on" with a
     linear least-squares update of the joints (ICP for a jointed figure). Hidden limbs are allowed to lie
     on visible ones, so overlaps and loops (bent legs touching, an arm along the body) are fine.
  5. the best fit gives the joints and the start pose + facing for the depth choices.

The result is a first guess for the animator to correct with clicks, never a final answer.
"""
from __future__ import annotations

import functools
import math
import pathlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from sandbox.skell import data
from sandbox.skell.camera import Camera
from sandbox.skell.checks import validate
from sandbox.skell.pose import pose_from_frame, turn_frame

PLACED = ["head", "neck", "pelvis", "l_elbow", "l_hand", "r_elbow", "r_hand", "l_knee", "l_foot", "r_knee",
          "r_foot"]
FREE = PLACED[1:]                                   # the head is fixed at the circle's centre
IDX = {j: i for i, j in enumerate(FREE)}
SEGS = [("neck", "pelvis"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"), ("r_elbow", "r_hand"),
        ("pelvis", "l_knee"), ("l_knee", "l_foot"), ("pelvis", "r_knee"), ("r_knee", "r_foot")]
LIMBS = [("neck", "l_elbow", "l_hand"), ("neck", "r_elbow", "r_hand"),
         ("pelvis", "l_knee", "l_foot"), ("pelvis", "r_knee", "r_foot")]
FACINGS = (0.0, 45.0, -45.0, 90.0, -90.0)
PROFILE_PENALTY = 0.25
W_EXT = 0.25                    # pull of hands and feet toward stroke tips (hidden limbs share a visible tip)


@dataclass
class Detection:
    joints: dict                      # joint -> (x, y) image pixels (pixel centres at integer + 0.5)
    start: str                        # library pose whose shape fitted best
    facing: float
    cost: float                       # mean squared misfit, px^2 (truncated); nan from the learned model
    head_radius: float
    stroke_width: float
    ranking: list = field(default_factory=list)      # (cost, start, facing), best first
    notes: list = field(default_factory=list)
    confidence: dict = field(default_factory=dict)  # joint -> 0..1, from the learned model only
    method: str = "fit"                             # "fit" (geometric) or "model" (learned)


# ---------------------------------------------------------------- image basics

def ink_mask(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Dark pixels of an image (H x W, H x W x 3 or x 4; floats 0..1 or uint8). Alpha composites on white."""
    a = np.asarray(img, dtype=float)
    if a.max() > 1.0:
        a = a / 255.0
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])
        a = a[..., :3].mean(axis=2)
    return a < threshold


def _outside(white: np.ndarray) -> np.ndarray:
    """White pixels reachable from the border (4-connected)."""
    reach = np.zeros_like(white)
    reach[0], reach[-1], reach[:, 0], reach[:, -1] = white[0], white[-1], white[:, 0], white[:, -1]
    while True:
        g = reach.copy()
        g[1:] |= reach[:-1]
        g[:-1] |= reach[1:]
        g[:, 1:] |= reach[:, :-1]
        g[:, :-1] |= reach[:, 1:]
        g &= white
        if np.array_equal(g, reach):
            return reach
        reach = g


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a (small) mask, each as an (n, 2) array of (row, col)."""
    seen = np.zeros_like(mask)
    out = []
    for r0, c0 in zip(*np.nonzero(mask)):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        q, comp = deque([(r0, c0)]), []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(comp))
    return out


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: the one-pixel centre-line of the strokes."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(int) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(int) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (P[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def find_head(ink: np.ndarray):
    """The drawn head: the highest round enclosed hole. Returns (centre (x, y), inner radius) or None."""
    holes = ink.copy()
    holes[:] = ~ink & ~_outside(~ink)
    best = None
    for comp in _components(holes):
        n = len(comp)
        if n < 12:
            continue
        h = int(np.ptp(comp[:, 0])) + 1
        w = int(np.ptp(comp[:, 1])) + 1
        fill = n / (math.pi * w * h / 4.0)          # 1 for a disc, ~0.64 for a triangle
        if fill < 0.82 or min(w, h) / max(w, h) < 0.7:
            continue
        cy, cx = comp.mean(axis=0)
        if best is None or cy < best[0][1]:
            best = ((float(cx) + 0.5, float(cy) + 0.5), math.sqrt(n / math.pi))
    return best


# ---------------------------------------------------------------- templates from the pose library

def _flat_lengths(skel) -> dict:
    """True length of each stick segment (shoulders and hips are zero-width, so they sit on their parent)."""
    alias = {b.to_joint: b.from_joint for b in skel.bones if b.kind == "rigid"}
    out = {}
    for b in skel.bones:
        if b.kind != "posable":
            continue
        seg = (alias.get(b.from_joint, b.from_joint), b.to_joint)
        if seg in SEGS:
            out[seg] = b.length
        elif seg[::-1] in SEGS:
            out[seg[::-1]] = b.length
    return out


@functools.lru_cache(maxsize=4)
def templates(facings=FACINGS) -> list[dict]:
    """Every grounded library pose at every facing, drawn from the front at eye level: 2D joints (y down)
    and flat segment lengths."""
    skel = data.load_skeleton()
    cam = Camera(target=np.array([0.0, 90.0, 0.0]), azimuth=0.0, elevation=0.0, distance=2000.0, focal=1000.0)
    flat = _flat_lengths(skel)
    out = []
    for p in data.pose_library():
        if p["frame"].get("airborne"):        # a drawing can't say it is off the ground: start grounded
            continue
        for fa in facings:
            frame = turn_frame(p["frame"], fa - float(p["frame"]["root"].get("facing", 0.0)))
            pos = validate(skel, pose_from_frame(skel, frame)).positions
            pr = {j: cam.project(pos[j]) for j in PLACED}
            s = pr["pelvis"].s
            out.append({"start": p["name"], "facing": fa,
                        "pts": {j: np.array([pr[j].dx, pr[j].dy]) for j in PLACED},
                        "len": {k: v * s for k, v in flat.items()}})
    return out


# ---------------------------------------------------------------- the fit

def _seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Distances from points P (n, 2) to segments A->B (m, 2): d (n, m), t (n, m)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0.0, 1.0)
    Q = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - Q, axis=2), t


def stroke_ends(pts: np.ndarray, sw: float) -> np.ndarray:
    """Free ends of the strokes, as (x, y): centre-line points whose nearby ink all lies to one side.
    (Geometric, not topological: thinning forks thick stroke tips into little prongs, which a pixel-neighbour
    test would read as junctions.) A straight run scores ~0, a right-angle corner ~0.7, a tip ~1."""
    R = 2.0 * sw + 2.0
    D = pts[None, :, :] - pts[:, None, :]                      # (n, n, 2): from each point to the others
    dist = np.linalg.norm(D, axis=2)
    near = (dist > 0) & (dist <= R)
    cnt = near.sum(1)
    mean_vec = (D * near[..., None]).sum(1)
    mean_len = (dist * near).sum(1)
    score = np.where(cnt >= 3, np.linalg.norm(mean_vec, axis=1) / np.maximum(mean_len, 1e-9), 0.0)
    ends = []
    for i in np.argsort(-score):                                 # strongest tips first, one per tip
        if score[i] < 0.85:
            break
        out = -mean_vec[i] / max(float(np.linalg.norm(mean_vec[i])), 1e-9)     # away from the stroke
        cand = np.nonzero(near[i])[0]
        tip = pts[cand[np.argmax((pts[cand] - pts[i]) @ out)]] if len(cand) else pts[i]
        tip = tip if (tip - pts[i]) @ out > 0 else pts[i]
        if all(np.linalg.norm(tip - e) > 2 * R for e in ends):
            ends.append(tip)
    return np.array(ends).reshape(-1, 2)


EXTREMITIES = np.array([IDX[j] for j in ("l_hand", "r_hand", "l_foot", "r_foot")])


class _Fitter:
    def __init__(self, pts: np.ndarray, ends: np.ndarray, neck_anchor: np.ndarray, sw: float, size: float):
        self.pts, self.ends, self.anchor, self.sw = pts, ends, neck_anchor, sw
        self.c_final = 3.0 * sw + 3.0                 # robust scale (px) once converged
        self.c0 = max(self.c_final, 0.2 * size)       # start broad: the fit sees the whole figure, then sharpens
        self.ns = 9                                   # samples per segment for the model -> ink term
        self.a = np.array([IDX[a] for a, _ in SEGS])
        self.b = np.array([IDX[b] for _, b in SEGS])

    def terms(self, X):
        A, B = X[self.a], X[self.b]
        d, t = _seg_dist(self.pts, A, B)
        k = d.argmin(1)
        r = np.arange(len(d))
        ts = np.linspace(0.0, 1.0, self.ns)
        S = (A[:, None, :] + ts[None, :, None] * (B - A)[:, None, :]).reshape(-1, 2)    # (m * ns, 2)
        dd = np.linalg.norm(S[:, None, :] - self.pts[None], axis=2)
        nn = dd.argmin(1)
        return k, t[r, k], d[r, k], ts, nn, dd[np.arange(len(S)), nn]

    def cost(self, X, lengths, scale):
        _, _, d_in, _, _, d_out = self.terms(X)
        c2 = self.c_final ** 2
        e = np.minimum(d_in ** 2, c2).mean() + np.minimum(d_out ** 2, c2).mean()
        if len(self.ends):
            de = np.linalg.norm(self.ends[:, None, :] - X[EXTREMITIES][None], axis=2)
            e += 0.5 * np.minimum(de.min(1) ** 2, c2).mean()       # every stroke end is a hand or a foot
            e += W_EXT * np.minimum(de.min(0) ** 2, c2).mean()     # hands and feet sit on stroke ends
        dj = np.linalg.norm(X[:, None, :] - self.pts[None], axis=2).min(1)
        e += 0.5 * np.minimum(dj ** 2, c2).mean()                   # every joint sits on a stroke
        for (a, b), L in lengths.items():                           # implausible lengths cost too
            ell = np.linalg.norm(X[IDX[b]] - X[IDX[a]]) / (L * scale)
            e += 4.0 * c2 * (max(0.0, ell - 1.3) ** 2 + max(0.0, 0.3 - ell) ** 2)
        return float(e)

    def step(self, X, lengths, scale, c):
        """One linearised least-squares update of all joints (robust scale c)."""
        n = len(FREE)
        c2 = c * c
        blocks = []          # (joint index arrays, coefficient arrays, targets (k, 2), weights (k,))

        k, t_in, d_in, ts, nn, d_out = self.terms(X)
        w = (1.0 / len(self.pts)) / (1 + d_in ** 2 / c2)                           # ink -> model
        blocks.append(((self.a[k], self.b[k]), (1 - t_in, t_in), self.pts, w))
        T = np.tile(ts, len(SEGS))
        ja, jb = np.repeat(self.a, self.ns), np.repeat(self.b, self.ns)
        w = (1.0 / (len(SEGS) * self.ns)) / (1 + d_out ** 2 / c2)                  # model -> ink
        blocks.append(((ja, jb), (1 - T, T), self.pts[nn], w))
        if len(self.ends):                                                        # stroke ends <-> extremities
            Xe = X[EXTREMITIES]
            de = np.linalg.norm(self.ends[:, None, :] - Xe[None], axis=2)
            near = de.argmin(1)
            dmin = de[np.arange(len(self.ends)), near]
            w = (0.5 / len(self.ends)) / (1 + dmin ** 2 / c2)
            blocks.append(((EXTREMITIES[near],), (np.ones(len(near)),), self.ends, w))
        if len(self.ends) and c < 1.5 * self.c_final:
            # and each hand / foot toward a tip, once the shape has settled: a hidden limb lies on its twin
            ne = de.argmin(0)
            dm = de[ne, np.arange(len(EXTREMITIES))]
            w = (W_EXT / len(EXTREMITIES)) / (1 + dm ** 2 / c2)
            blocks.append(((EXTREMITIES,), (np.ones(len(EXTREMITIES)),), self.ends[ne], w))
        if c < 1.5 * self.c_final:                       # joints <-> strokes, once the shape has settled
            dj = np.linalg.norm(X[:, None, :] - self.pts[None], axis=2)
            nj = dj.argmin(1)
            w = (0.5 / n) / (1 + dj[np.arange(n), nj] ** 2 / c2)
            blocks.append(((np.arange(n),), (np.ones(n),), self.pts[nj], w))
        blocks.append(((np.array([IDX["neck"]]),), (np.ones(1),), self.anchor[None], np.array([0.15])))
        blocks.append(((np.arange(n),), (np.ones(n),), X, np.full(n, 1e-4)))      # damping

        H = np.zeros((2 * n, 2 * n))
        g = np.zeros(2 * n)
        for js, cs, tgt, w in blocks:
            for ax in (0, 1):
                M = np.zeros((len(w), 2 * n))
                rr = np.arange(len(w))
                for j, cf in zip(js, cs):
                    np.add.at(M, (rr, 2 * j + ax), cf)
                H += (M * w[:, None]).T @ M
                g += (M * w[:, None]).T @ tgt[:, ax]
        extra = []
        # lengths: only outside a broad band (stick figures keep their own proportions)
        for (a, b), L in lengths.items():
            v = X[IDX[b]] - X[IDX[a]]
            ell = float(np.linalg.norm(v))
            lo, hi = 0.3 * L * scale, 1.3 * L * scale
            if ell > 1e-6 and not lo <= ell <= hi:
                extra.append(({b: v / ell, a: -v / ell}, hi if ell > hi else lo, 0.2))
        # a straight limb leaves its middle joint free to slide: keep it near the middle, along the limb only
        for s_, m_, e_ in LIMBS:
            u = X[IDX[e_]] - X[IDX[s_]]
            nu = float(np.linalg.norm(u))
            if nu > 1e-6:
                u = u / nu
                extra.append(({m_: 2 * u, s_: -u, e_: -u}, 0.0, 0.002))
        for coefs, target, w in extra:
            r = np.zeros(2 * n)
            for j, cf in coefs.items():
                r[2 * IDX[j]:2 * IDX[j] + 2] += cf
            H += w * np.outer(r, r)
            g += w * r * target
        return np.linalg.solve(H, g).reshape(n, 2)


def _reseatings(X: np.ndarray, ends: np.ndarray, pts: np.ndarray, c: float):
    """Alternative placements of whole limbs, to escape local minima:
    - lay a limb on its twin (a hidden limb shares its visible twin's stroke);
    - put a limb onto a stroke tip no hand or foot has reached (hand/foot on the tip, the middle joint on
      the ink halfway between the limb's root and the tip)."""
    for (_, m1, e1), (_, m2, e2) in ((LIMBS[0], LIMBS[1]), (LIMBS[2], LIMBS[3])):
        for (ma, ea), (mb, eb) in (((m1, e1), (m2, e2)), ((m2, e2), (m1, e1))):
            Y = X.copy()
            Y[IDX[mb]], Y[IDX[eb]] = X[IDX[ma]], X[IDX[ea]]
            yield Y
    if not len(ends):
        return
    reach = np.linalg.norm(ends[:, None, :] - X[EXTREMITIES][None], axis=2).min(1)
    for tip in ends[reach > c]:
        for root, mid, end in LIMBS:
            Y = X.copy()
            Y[IDX[end]] = tip
            half = (X[IDX[root]] + tip) / 2
            Y[IDX[mid]] = pts[np.linalg.norm(pts - half, axis=1).argmin()]
            yield Y


def detect(img: np.ndarray, facings=FACINGS, iters: int = 40, keep: int = 8, model="auto") -> Detection:
    """Find the 11 placed joints of the one stick figure in img (image pixels, y down).
    model: "auto" uses the trained joint model if there is one (detect_with_model), else the geometric fit
    below; None forces the geometric fit; or pass a loaded sandbox.joint_net.JointNet."""
    if model == "auto":
        model = default_model()
    if model is not None:
        return detect_with_model(img, model, facings)
    ink = ink_mask(img)
    if not ink.any():
        raise ValueError("no ink found in the drawing")
    notes = []
    skel_all = thin(ink)
    sw = float(ink.sum()) / max(1, int(skel_all.sum()))
    head = find_head(ink)
    rows, cols = np.nonzero(ink)
    if head is None:
        notes.append("no round head found: the head is put at the top of the drawing; please place it")
        top = rows.min()
        hc = np.array([cols[rows == top].mean() + 0.5, top + 0.5 + 2 * sw])
        r_in = 2.0 * sw
        body = ink
    else:
        hc, r_in = np.array(head[0]), head[1]
        yy, xx = np.mgrid[0:ink.shape[0], 0:ink.shape[1]]
        ring = np.hypot(xx + 0.5 - hc[0], yy + 0.5 - hc[1]) <= r_in + 1.75 * sw
        body = ink & ~ring
    sk = thin(body)
    r_, c_ = np.nonzero(sk)
    pts = np.stack([c_ + 0.5, r_ + 0.5], axis=1)
    if len(pts) < 10:
        raise ValueError("too little drawing below the head to find a body")
    anchor = pts[np.linalg.norm(pts - hc, axis=1).argmin()]
    floor_y = float(rows.max()) + 0.5 - 0.5 * sw
    size = floor_y - hc[1]
    ends = stroke_ends(pts, sw)
    ends = ends[np.linalg.norm(ends - anchor, axis=1) > 3 * sw + 3] if len(ends) else ends   # not the neck
    fitter = _Fitter(pts, ends, anchor, sw, size)

    def run(X, lengths, scale, its):
        for it in its:
            c = max(fitter.c_final, fitter.c0 * 0.8 ** it)
            Xn = fitter.step(X, lengths, scale, c)
            moved = float(np.abs(Xn - X).max())
            X = Xn
            if moved < 0.05 and c == fitter.c_final:
                break
        return X

    # every start gets a short coarse fit; the most promising few are carried to the end
    short = min(12, iters)
    results = []
    for tp in templates(tuple(facings)):
        T = tp["pts"]
        t_h = max(v[1] for v in T.values()) - T["head"][1]
        scale = size / max(t_h, 1e-6)
        X = run(np.array([hc + scale * (T[j] - T["head"]) for j in FREE]), tp["len"], scale, range(short))
        results.append([fitter.cost(X, tp["len"], scale), tp, scale, X])
    results.sort(key=lambda r: r[0])
    final = []
    for _, tp, scale, X in results[:keep]:
        X = run(X, tp["len"], scale, range(short, iters))
        cost = fitter.cost(X, tp["len"], scale)
        # a limb can stay folded onto the wrong stroke (local minimum): try discrete re-seatings, keep what helps
        for _ in range(6):
            best = None
            for Y in _reseatings(X, ends, pts, fitter.c_final):
                Y = run(Y, tp["len"], scale, range(iters - 10, iters))
                cy = fitter.cost(Y, tp["len"], scale)
                if cy < cost - 1e-6 and (best is None or cy < best[0]):
                    best = (cy, Y)
            if best is None:
                break
            cost, X = best
        # most drawings are front-ish: a side view must fit clearly better to win (it also decides left/right)
        rank = cost * (1.0 + PROFILE_PENALTY * abs(math.sin(math.radians(tp["facing"]))))
        final.append((rank, cost, tp["start"], tp["facing"], X))
    final.sort(key=lambda r: r[0])
    _, cost, start, facing, X = final[0]
    results = [(c, s, f, None) for _, c, s, f, _ in final]
    joints = {"head": (float(hc[0]), float(hc[1]))}
    joints.update({j: (float(X[i][0]), float(X[i][1])) for i, j in enumerate(FREE)})
    if facing in (90.0, -90.0):
        notes.append("profile view: which limb is left and which is right is a guess; swap them if needed")
    return Detection(joints, start, facing, cost, r_in + sw / 2, sw,
                     [(round(c, 2), s, f) for c, s, f, _ in results[:8]], notes)


def best_start(joints: dict, facings=FACINGS) -> tuple[str, float]:
    """The grounded library pose and facing whose front view looks most like these 2D joints (any units,
    y down): least squares over a scale and a shift, all 11 joints. Side views must fit clearly better
    (as in detect). For joints placed by hand, where no detection ran."""
    start, facing, _ = _match_template(joints, facings, swap_sides=False)
    return start, facing


# arm and leg pairs whose sides one drawing can't tell (the learned model leaves them open)
_SIDE_SWAPS = [(("l_elbow", "l_hand"), ("r_elbow", "r_hand")), (("l_knee", "l_foot"), ("r_knee", "r_foot"))]


def _with_swaps(joints: dict, swaps) -> dict:
    out = dict(joints)
    for k in swaps:
        a, b = _SIDE_SWAPS[k]
        for ja, jb in zip(a, b):
            out[ja], out[jb] = joints[jb], joints[ja]
    return out


SWAP_PENALTY = 0.1      # a side swap away from the front-view reading must fit this much better per pair


def _front_sides(joints: dict) -> dict:
    """The front-view reading of the arm and leg sides: the limb whose hand or foot is further to image
    right is the character's left (l_ is image right at facing 0)."""
    out = dict(joints)
    for k, (a, b) in enumerate(_SIDE_SWAPS):
        if joints[a[1]][0] < joints[b[1]][0]:
            out = _with_swaps(out, (k,))
    return out


def _match_template(joints: dict, facings=FACINGS, swap_sides=True):
    """best_start's search. With swap_sides (the learned model leaves the sides open), the arm and leg
    pairs start from the front-view reading and each may swap if that fits a library pose clearly better
    (SWAP_PENALTY per swapped pair, like PROFILE_PENALTY for side views). Returns (start, facing, joints
    with the chosen sides). Tuned on 9 drawings (2026-11-19); for side-on figures the sides stay a guess."""
    if swap_sides:
        joints = _front_sides(joints)
    options = [()] + ([(0,), (1,), (0, 1)] if swap_sides else [])
    best = None
    for tp in templates(tuple(facings)):
        T = np.array([tp["pts"][j] for j in PLACED], dtype=float)
        Tc = T - T.mean(0)
        for sw in options:
            J = _with_swaps(joints, sw)
            X = np.array([J[j] for j in PLACED], dtype=float)
            Xc = X - X.mean(0)
            s = float((Tc * Xc).sum() / max((Tc * Tc).sum(), 1e-12))
            if s <= 0:
                continue
            rms = float(np.sqrt(((s * Tc - Xc) ** 2).sum(1).mean()))
            rank = rms * (1.0 + PROFILE_PENALTY * abs(math.sin(math.radians(tp["facing"]))))                 * (1.0 + SWAP_PENALTY * len(sw))
            if best is None or rank < best[0]:
                best = (rank, tp["start"], tp["facing"], J)
    return best[1], best[2], best[3]


# ---------------------------------------------------------------- the learned model (when trained)

MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "out" / "models" / "joint_net.npz"
LOW_CONFIDENCE = 0.5


@functools.lru_cache(maxsize=2)
def _load_model(path: str):
    from sandbox.joint_net import JointNet
    return JointNet.load(path)


def default_model():
    """The trained joint model if out/models/joint_net.npz exists (tools/kaggle_train.py fetch), else None."""
    return _load_model(str(MODEL_PATH)) if MODEL_PATH.is_file() else None


def detect_with_model(img: np.ndarray, net, facings=FACINGS) -> Detection:
    """Joints from the learned model (about a second, against 10-60 s for the geometric fit). The model
    proposes 2D points and leaves the arm and leg sides open; the sides, start pose and facing come from
    the library pose and facing they look most like, each pair allowed either way round. A round head hole, where there
    is one, gives the head's exact centre."""
    from sandbox.synth import frame_drawing
    small, (scale, ox, oy) = frame_drawing(img, 192)
    J, conf = net.predict(small)
    J = J / scale + np.array([ox, oy])
    joints = {j: (float(J[i][0]), float(J[i][1])) for i, j in enumerate(PLACED)}
    confidence = {j: float(conf[i]) for i, j in enumerate(PLACED)}
    ink_s = ink_mask(small)                           # stroke width from the small framed copy (thinning
    sw = float(ink_s.sum()) / max(1, int(thin(ink_s).sum())) / scale    # full size is slow)
    head = find_head(ink_mask(img))
    notes = []
    if head is not None and np.hypot(*np.subtract(head[0], joints["head"])) <= 1.5 * head[1] + 2 * sw:
        joints["head"] = (float(head[0][0]), float(head[0][1]))
        head_r = head[1] + sw / 2
    else:
        free = np.array([joints[j] for j in FREE])
        head_r = 0.1 * float(np.ptp(free, axis=0).max())   # no hole to measure: a typical stick-figure head
        notes.append("no round head found: the head is the model's guess; please check it")
    start, facing, joints = _match_template(joints, facings, swap_sides=True)
    for j, c in confidence.items():
        if c < LOW_CONFIDENCE:
            notes.append(f"the model is unsure of the {j.replace('_', ' ')}: please check it")
    if facing in (90.0, -90.0):
        notes.append("profile view: which limb is left and which is right is a guess; swap them if needed")
    return Detection(joints, start, facing, float("nan"), head_r, sw, [], notes, confidence, "model")


def pixels_to_screen(joints: dict, img_shape, canvas_w: float, canvas_h: float) -> dict:
    """Image pixels -> screen offsets of the sketch canvas (the image is stretched over the whole canvas,
    as the viewer shows an underlay)."""
    h, w = img_shape[:2]
    return {j: [x / w * canvas_w - canvas_w / 2, y / h * canvas_h - canvas_h / 2] for j, (x, y) in joints.items()}
