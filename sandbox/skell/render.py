"""Sketch renderer (spec 9): head circle, bone lines, joint markers, perspective floor grid, horizon,
boxes as props, depth order and the editor overlays. Matplotlib only; the geometry comes from the kernel.

This module never selects a matplotlib backend: draw_scene draws on any Axes, and render() writes a PNG
through a bare Figure with the Agg canvas, so the interactive viewer (step 5) keeps its own backend.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from . import quat
from .camera import Camera, Subject
from .data import load_rules

# Design notes figure style (spec 9.1): teal bones, pale joints with a dark teal edge, coral root.
COLORS = {
    "bone": "#1f9e7a",         # default bone colour (overlays off)
    "teal": "#1f9e7a",         # overlay: bone in the image plane
    "amber": "#e09a1a",        # overlay: bone tilted toward or away from the camera
    "edge": "#17694f",         # joint and head outlines
    "joint_fill": "#e3f4ee",
    "root_fill": "#fbe6de",
    "root_edge": "#c0573e",
    "grid": "#d6d6d6",
    "horizon": "#9a9a9a",
    "box_fill": "#f1eee6",
    "box_edge": "#9c978c",
    "ghost": "#8c96a1",
    "text": "#2b2f36",
    "paper": "#ffffff",
}
JOINT_RADIUS = 4.5       # world units; scaled by perspective like everything else
BONE_WIDTH = 2.6         # points
JOINT_BIAS = 0.5         # joints sort this much nearer than their bones, so dots sit on the line ends


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


@dataclass(frozen=True)
class Box:
    """A box prop standing on the floor: centre (x, z) on the ground, size, turn about the vertical axis."""
    x: float
    z: float
    width: float
    depth: float
    height: float
    yaw: float = 0.0

    def corners(self) -> np.ndarray:
        a = math.radians(self.yaw)
        right = np.array([math.cos(a), 0.0, math.sin(a)])      # same turn sign as facing
        front = np.array([math.sin(a), 0.0, -math.cos(a)])
        c = np.array([self.x, 0.0, self.z])
        out = []
        for y in (0.0, self.height):
            for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                out.append(c + sx * self.width / 2 * right - sz * self.depth / 2 * front + np.array([0.0, y, 0.0]))
        return np.array(out)

    FACES = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


@dataclass
class RenderOptions:
    overlays: bool = False     # editor overlays: tilt colours, angle badges, ghost lines (off for exports)
    grid: bool = True
    horizon: bool = True
    face: bool = True          # face cross on the head (centre line + eye line) so the front reads at a glance
    title: str | None = None


@dataclass
class DrawnItem:
    kind: str                  # "bone", "joint", "head", "box_face"
    name: str
    w: float                   # depth used for ordering (midpoint for bones, centroid for faces)
    sort_depth: float          # w with the small joint/head bias applied
    zorder: float = 0.0
    color: str | None = None   # "ink", "teal", "amber" for bones
    tilt: float | None = None  # degrees toward (-) or away from (+) the camera, bones only


@dataclass
class Drawing:
    items: list[DrawnItem] = field(default_factory=list)
    grid_lines: int = 0
    grid_dashed: bool = False
    horizon: tuple | None = None          # two screen points (y down) on the horizon line
    ghosts: int = 0
    badges: int = 0


def bone_tilt(cam: Camera, P0, P1) -> float:
    """Angle between a bone and the image plane: + pointing away from the camera, - toward it."""
    d = np.asarray(P1, dtype=float) - np.asarray(P0, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return 0.0
    f, _, _ = cam.basis()
    return math.degrees(math.asin(max(-1.0, min(1.0, float(d @ f) / n))))


def _xy(p):
    return p.dx, p.dy


def _clip_polygon(cam: Camera, pts):
    """Sutherland-Hodgman against the near plane (keep w >= near)."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        wa, wb = cam.depth(a), cam.depth(b)
        if wa >= cam.near:
            out.append(a)
        if (wa >= cam.near) != (wb >= cam.near):
            out.append(a + (cam.near - wa) / (wb - wa) * (b - a))
    return out


def _horizon(cam: Camera):
    """Two screen points on the ground plane's vanishing line, or None when looking straight up or down."""
    if abs(cam.elevation) > 89.9:
        return None
    f, r, up = cam.basis()
    fh = np.array([f[0], 0.0, f[2]])
    fh /= np.linalg.norm(fh)
    pts = []
    rho = math.radians(cam.roll)
    for turn in (-20.0, 20.0):
        a = math.radians(turn)
        D = np.array([fh[0] * math.cos(a) - fh[2] * math.sin(a), 0.0, fh[0] * math.sin(a) + fh[2] * math.cos(a)])
        u, v, w = float(D @ r), float(D @ up), float(D @ f)
        s = cam.focal / w
        u2, v2 = u * math.cos(rho) + v * math.sin(rho), -u * math.sin(rho) + v * math.cos(rho)
        pts.append((s * u2, -s * v2))
    return tuple(pts)


def draw_scene(ax, cam: Camera, subjects: list[Subject], boxes=(), options: RenderOptions | None = None,
               rules: dict | None = None) -> Drawing:
    """Draw the scene on a matplotlib Axes in screen units (x right, y down, origin at the image centre)."""
    options = options or RenderOptions()
    rules = rules or _rules()
    cr, rr = rules["camera"], rules["render"]
    W, H = cr["canvas_width"], cr["canvas_height"]
    drawing = Drawing()

    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(-W / 2, W / 2)
    ax.set_ylim(H / 2, -H / 2)          # screen y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9cfd6")
    if options.title:
        ax.set_title(options.title, fontsize=10, color=COLORS["text"])

    # floor grid: finite, clipped at the near plane in 3D, dashed when seen from below
    if options.grid:
        E, step = cr["grid_half_extent"], rr["grid_step"]
        drawing.grid_dashed = bool(cam.position[1] < 0)
        ls = (0, (4, 4)) if drawing.grid_dashed else "-"
        segments = []
        for k in np.arange(-E, E + step / 2, step):
            for P0, P1 in (((k, 0, -E), (k, 0, E)), ((-E, 0, k), (E, 0, k))):
                seg = cam.clip_segment(P0, P1)
                if seg is None:
                    continue
                a, b = cam.project(seg[0]), cam.project(seg[1])
                segments.append([(a.dx, a.dy), (b.dx, b.dy)])
        if segments:   # one artist for the whole grid keeps redraws fast in the viewer
            ax.add_collection(LineCollection(segments, colors=COLORS["grid"], linewidths=0.7, linestyles=ls,
                                             zorder=1))
        drawing.grid_lines = len(segments)
    if options.horizon:
        drawing.horizon = _horizon(cam)
        if drawing.horizon is not None:
            ax.axline(drawing.horizon[0], drawing.horizon[1], color=COLORS["horizon"], lw=0.8, ls="--", zorder=2)

    # collect depth-sorted items; each has a draw callback taking its zorder
    queue: list[tuple[DrawnItem, callable]] = []
    flat_deg = rr["flat_tilt_deg"]

    for subj in subjects:
        skel, pos = subj.skel, subj.positions
        hidden = {b.to_joint for b in skel.bones if b.length == 0}
        for b in skel.bones:
            if b.length == 0:
                continue
            seg = cam.clip_segment(pos[b.from_joint], pos[b.to_joint])
            if seg is None:
                continue
            p0, p1 = cam.project(seg[0]), cam.project(seg[1])
            w_mid = cam.depth((seg[0] + seg[1]) / 2)
            tilt = bone_tilt(cam, pos[b.from_joint], pos[b.to_joint])
            color = "bone"
            if options.overlays:
                color = "teal" if abs(tilt) < flat_deg else "amber"
            item = DrawnItem("bone", b.id, w_mid, w_mid, color=color, tilt=tilt)

            def draw_bone(z, p0=p0, p1=p1, color=color, tilt=tilt, b=b, seg=seg, w_mid=w_mid):
                ax.plot([p0.dx, p1.dx], [p0.dy, p1.dy], color=COLORS[color], lw=BONE_WIDTH,
                        solid_capstyle="round", zorder=z)
                if options.overlays and color == "amber":
                    _ghost_and_badge(ax, cam, p0, p1, b.length, w_mid, tilt, z, drawing)
            queue.append((item, draw_bone))

        for j, P in pos.items():
            if j in hidden or j == "head":
                continue
            p = cam.project(P)
            if not p.visible:
                continue
            item = DrawnItem("joint", j, p.w, p.w - JOINT_BIAS)
            is_root = j == skel.root
            face = COLORS["root_fill"] if is_root else COLORS["joint_fill"]
            edge = COLORS["root_edge"] if is_root else COLORS["edge"]

            def draw_joint(z, p=p, face=face, edge=edge):
                ax.add_patch(Circle(_xy(p), JOINT_RADIUS * p.s, facecolor=face, edgecolor=edge, lw=1.2, zorder=z))
            queue.append((item, draw_joint))

        ph = cam.project(pos["head"])
        if ph.visible:
            item = DrawnItem("head", "head", ph.w, ph.w - skel.head_radius)

            face = face_cross(cam, subj) if options.face else []

            def draw_head(z, p=ph, rad=skel.head_radius, face=face):
                ax.add_patch(Circle(_xy(p), rad * p.s, facecolor=COLORS["joint_fill"], edgecolor=COLORS["edge"],
                                    lw=1.6, zorder=z))
                for line in face:
                    ax.plot(line[:, 0], line[:, 1], color=COLORS["edge"], lw=1.1, solid_capstyle="round",
                            zorder=z + 0.1)
            queue.append((item, draw_head))

    for bi, box in enumerate(boxes):
        corners = box.corners()
        for fi, face in enumerate(Box.FACES):
            poly = _clip_polygon(cam, [corners[i] for i in face])
            if len(poly) < 3:
                continue
            w_c = cam.depth(np.mean(poly, axis=0))
            xy = [_xy(cam.project(q)) for q in poly]
            item = DrawnItem("box_face", f"box{bi}.face{fi}", w_c, w_c)

            def draw_face(z, xy=xy):
                ax.add_patch(Polygon(xy, closed=True, facecolor=COLORS["box_fill"], edgecolor=COLORS["box_edge"],
                                     lw=1.0, zorder=z))
            queue.append((item, draw_face))

    # painter's order: farthest first
    queue.sort(key=lambda q: -q[0].sort_depth)
    for i, (item, fn) in enumerate(queue):
        item.zorder = 10 + i
        fn(item.zorder)
        drawing.items.append(item)
    return drawing


FACE_SPAN = 75.0     # degrees each arm of the face cross runs from the face centre over the head sphere


def head_front(subj: Subject) -> np.ndarray | None:
    """Unit direction the face points: the head bone's world rotation applied to the rest front (-Z), or the
    character's facing when no rotation is given; made perpendicular to the neck->head axis."""
    pos = subj.positions
    up = np.asarray(pos["head"], dtype=float) - np.asarray(pos["neck"], dtype=float)
    if np.linalg.norm(up) < 1e-9:
        return None
    up = up / np.linalg.norm(up)
    if subj.head_rotation is not None:
        f = quat.rotate(subj.head_rotation, np.array([0.0, 0.0, -1.0]))
    else:
        a = math.radians(subj.facing)
        f = np.array([math.sin(a), 0.0, -math.cos(a)])
    f = f - (f @ up) * up
    n = np.linalg.norm(f)
    return f / n if n > 1e-6 else None


def face_cross(cam: Camera, subj: Subject, steps: int = 24) -> list[np.ndarray]:
    """The artist's face cross on the head sphere: a centre line (chin to brow) and an eye line (ear to ear),
    both on the front half of the head. Returns screen polylines (y down) of the parts facing the camera; a head
    seen from behind gets none."""
    f = head_front(subj)
    if f is None:
        return []
    pos = subj.positions
    H = np.asarray(pos["head"], dtype=float)
    up = H - np.asarray(pos["neck"], dtype=float)
    up = up / np.linalg.norm(up)
    side = np.cross(up, f)
    r = subj.skel.head_radius
    t = np.radians(np.linspace(-FACE_SPAN, FACE_SPAN, steps))
    out = []
    for axis in (up, side):
        run = []
        for a in t:
            n = math.cos(a) * f + math.sin(a) * axis
            P = H + r * n
            p = cam.project(P)
            if p.visible and float((cam.position - P) @ n) > 0:   # on the camera's side of the sphere
                run.append(_xy(p))
            else:
                if len(run) > 1:
                    out.append(np.array(run))
                run = []
        if len(run) > 1:
            out.append(np.array(run))
    return out


def _ghost_and_badge(ax, cam, p0, p1, length, w_mid, tilt, z, drawing):
    """Dashed ghost line at the true length the bone would have if flat, and a tilt badge."""
    v = np.array([p1.dx - p0.dx, p1.dy - p0.dy])
    n = np.linalg.norm(v)
    if n > 1e-9:
        u = v / n
        g = length * cam.focal / w_mid
        ax.plot([p0.dx, p0.dx + g * u[0]], [p0.dy, p0.dy + g * u[1]], color=COLORS["ghost"], lw=1.0,
                ls=(0, (3, 3)), zorder=z - 0.5)
        drawing.ghosts += 1
        normal = np.array([-u[1], u[0]])
    else:
        normal = np.array([0.0, -1.0])
    mid = np.array([(p0.dx + p1.dx) / 2, (p0.dy + p1.dy) / 2]) + 9 * normal
    ax.text(mid[0], mid[1], f"{abs(tilt):.0f}° {'away' if tilt > 0 else 'toward'}", fontsize=6.5,
            color=COLORS["amber"], ha="center", va="center", zorder=z + 0.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    drawing.badges += 1


def render(path, cam: Camera, subjects: list[Subject], boxes=(), options: RenderOptions | None = None,
           rules: dict | None = None, dpi: int = 130) -> Drawing:
    """Draw one view. Writes a PNG when path is given; always returns the Drawing summary."""
    rules = rules or _rules()
    cr = rules["camera"]
    fig = Figure(figsize=(cr["canvas_width"] / 100, cr["canvas_height"] / 100 + 0.3), facecolor=COLORS["paper"])
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.9 if (options and options.title) else 0.98])
    drawing = draw_scene(ax, cam, subjects, boxes, options, rules)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=COLORS["paper"])
    return drawing
