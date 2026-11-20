"""Turnaround sheet (spec 8.7): the current pose from the eight character-relative azimuths, every 45
degrees, at one elevation and distance, side by side on one sheet with a height ruler.

    python -m sandbox.skell.turnaround --pose standing --elevation 10
    python -m sandbox.skell.turnaround --pose guard --framing current --distance 410

Framing:
  full     one target on the character's vertical axis (mid-height of the pose) and one distance that fits
           the pose in all eight views, so every view shares one scale (a model sheet)
  current  the viewer's camera: head target, the given distance and focal length (spec 8.7 as written)
"""
from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .camera import Camera, Subject, bounding_box, fit_distance, resolve_camera, world_azimuth
from .data import load_rules
from .render import COLORS, Drawing, RenderOptions, draw_scene


@functools.lru_cache(maxsize=1)
def _rules() -> dict:
    return load_rules()


def _views(rules: dict) -> list[tuple[float, str]]:
    """The eight character-relative azimuth presets, ordered 0, 45, ... 315 (wrapped to (-180, 180])."""
    presets = [(float(p["azimuth_rel"]), name) for name, p in rules["camera"]["presets"].items() if "azimuth_rel" in p]
    return sorted(presets, key=lambda v: v[0] % 360)


VIEWS = _views(load_rules())


@dataclass
class View:
    azimuth_rel: float
    name: str
    camera: Camera
    drawing: Drawing | None = None


@dataclass
class Ruler:
    """A vertical pole beside the character, marked every head unit, from the floor to standing height."""
    height: float
    head_unit: float
    axis: np.ndarray            # the character's vertical axis on the floor (root x, 0, root z)
    offset: float               # distance from the axis, toward the left of every view
    ticks: list[float] = field(default_factory=list)

    @property
    def heads_tall(self) -> float:
        return self.height / self.head_unit

    def points(self, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
        _, r, _ = cam.basis()
        base = self.axis - self.offset * r
        return base, base + np.array([0.0, self.height, 0.0])


@dataclass
class TurnaroundSheet:
    views: list[View]
    ruler: Ruler
    path: Path | None


def _axis_target(subject: Subject) -> np.ndarray:
    lo, hi = bounding_box(subject)
    root = np.asarray(subject.positions[subject.skel.root], dtype=float)
    return np.array([root[0], (lo[1] + hi[1]) / 2, root[2]])


def turnaround_cameras(subject: Subject, elevation: float = 0.0, distance: float | None = None,
                       focal: float | None = None, roll: float = 0.0, framing: str = "full",
                       floor_guard: bool | None = None, rules: dict | None = None) -> list[View]:
    """The eight cameras. Each is resolved and checked like any camera (spec 8.5)."""
    rules = rules or _rules()
    cr = rules["camera"]
    F = float(focal if focal is not None else cr["default_focal"])
    if framing == "full":
        target = _axis_target(subject)
        R = max(fit_distance(subject, target, F, rules, world_azimuth(subject.facing, rel), elevation, roll)
                for rel, _ in VIEWS)
    elif framing == "current":
        target = np.asarray(subject.positions["head"], dtype=float)
        R = float(distance if distance is not None else cr["presets"]["eye_level"]["distance"])
    else:
        raise ValueError(f"unknown framing {framing!r}")
    views = []
    for rel, name in VIEWS:
        req = {"relative_to": "character", "azimuth": rel, "elevation": elevation, "roll": roll,
               "distance": R, "focal": F, "target": list(target), "floor_guard": floor_guard}
        views.append(View(rel, name, resolve_camera(req, subject, rules=rules).camera))
    return views


def make_ruler(subject: Subject) -> Ruler:
    skel = subject.skel
    unit = 2.0 * skel.head_radius
    root = np.asarray(subject.positions[skel.root], dtype=float)
    reach = max(float(np.hypot(p[0] - root[0], p[2] - root[2])) for p in subject.positions.values())
    ticks = [k * unit for k in range(int(skel.standing_height // unit) + 1)]
    if ticks[-1] < skel.standing_height:
        ticks.append(float(skel.standing_height))
    return Ruler(float(skel.standing_height), unit, np.array([root[0], 0.0, root[2]]),
                 reach + skel.head_radius + 25.0, ticks)


def draw_ruler(ax, cam: Camera, ruler: Ruler) -> None:
    base, top = ruler.points(cam)
    seg = cam.clip_segment(base, top)
    if seg is None:
        return
    a, b = cam.project(seg[0]), cam.project(seg[1])
    ax.plot([a.dx, b.dx], [a.dy, b.dy], color=COLORS["ghost"], lw=1.2, zorder=5)
    _, r, _ = cam.basis()
    for t in ruler.ticks:
        p0 = base + np.array([0.0, t, 0.0])
        p1 = p0 - 8.0 * r
        if cam.depth(p0) < cam.near or cam.depth(p1) < cam.near:
            continue
        q0, q1 = cam.project(p0), cam.project(p1)
        ax.plot([q0.dx, q1.dx], [q0.dy, q1.dy], color=COLORS["ghost"], lw=1.0, zorder=5)
        top_tick = abs(t - ruler.height) < 1e-9
        label = f"{ruler.height:.0f} = {ruler.heads_tall:.1f} heads" if top_tick else f"{t / ruler.head_unit:.0f}"
        if top_tick or t > 0:
            ax.text(q1.dx - 3, q1.dy, label, fontsize=6.5, color=COLORS["text"], ha="right", va="center", zorder=5,
                    clip_on=True)                    # a tall ruler beside a low pose runs off this view


def render_turnaround(path, subject: Subject, title: str | None = None, elevation: float = 0.0,
                      distance: float | None = None, focal: float | None = None, roll: float = 0.0,
                      framing: str = "full", overlays: bool = False, floor_guard: bool | None = None,
                      rules: dict | None = None, dpi: int = 90) -> TurnaroundSheet:
    rules = rules or _rules()
    cr = rules["camera"]
    views = turnaround_cameras(subject, elevation, distance, focal, roll, framing, floor_guard, rules)
    ruler = make_ruler(subject)
    w, h = cr["canvas_width"] / 100, cr["canvas_height"] / 100
    fig = Figure(figsize=(4 * w, 2 * (h + 0.3) + 0.6), facecolor=COLORS["paper"])
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 4, left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.03, hspace=0.12)
    for i, v in enumerate(views):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        label = f"{v.name.replace('_', ' ')}  ({v.azimuth_rel:+.0f}°)"
        v.drawing = draw_scene(ax, v.camera, [subject], options=RenderOptions(overlays=overlays, title=label),
                               rules=rules)
        draw_ruler(ax, v.camera, ruler)
    c = views[0].camera
    head = f"Turnaround{': ' + title if title else ''}"
    sub = (f"elevation {c.elevation:.0f}°, distance {c.distance:.0f}, focal {c.focal:.0f}, {framing} framing  ·  "
           f"height {ruler.height:.0f} units = {ruler.heads_tall:.1f} heads")
    fig.suptitle(f"{head}\n{sub}", fontsize=11, color=COLORS["text"])
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=COLORS["paper"])
    return TurnaroundSheet(views, ruler, path)


def main(argv=None):
    from .checks import validate
    from .data import ROOT, load_skeleton, pose_library
    from .pose import head_rotation, pose_from_frame

    lib = {p["name"]: p for p in pose_library()}
    ap = argparse.ArgumentParser(description="Render a turnaround sheet for a library pose")
    ap.add_argument("--pose", default="reference_kneeling", choices=sorted(lib))
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--focal", type=float, default=None)
    ap.add_argument("--framing", choices=["full", "current"], default="full")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    skel = load_skeleton()
    pose = pose_from_frame(skel, lib[args.pose]["frame"])
    v = validate(skel, pose)
    subj = Subject(skel, v.positions, pose.facing, head_rotation(skel, v.pose))
    out = Path(args.out) if args.out else ROOT / "out" / f"turnaround_{args.pose}.png"
    render_turnaround(out, subj, title=lib[args.pose]["label"], elevation=args.elevation, distance=args.distance,
                      focal=args.focal, framing=args.framing, overlays=args.overlays)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
