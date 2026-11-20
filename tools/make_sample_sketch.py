"""Draw a practice sketch for sketch input: a right hook seen from the viewer's default camera, drawn with a
wobbly hand and limbs a little short, as a rough drawing would be. Writes the image and the joint positions
(screen offsets) that a careful artist would click.

    python tools/make_sample_sketch.py            # writes out/sample_sketch.png and out/sample_sketch_joints.json
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sandbox.skell import data  # noqa: E402
from sandbox.skell.camera import Subject, resolve_camera  # noqa: E402
from sandbox.skell.checks import validate  # noqa: E402
from sandbox.skell.pose import pose_from_frame  # noqa: E402


def make(png_path, json_path=None, pose_name="right_hook", shrink=0.94, seed=11):
    skel = data.load_skeleton()
    rules = data.load_rules()
    lib = {p["name"]: p for p in data.pose_library()}
    # the viewer's default camera: head of the starting (reference) pose, eye level, R 410
    ref = Subject(skel, validate(skel, pose_from_frame(skel, data.reference_frame())).positions, 0.0)
    cam = resolve_camera({}, ref).camera
    pos = validate(skel, pose_from_frame(skel, lib[pose_name]["frame"])).positions
    # a rough drawing: every bone drawn a bit short, measured out from the pelvis
    drawn = {skel.root: np.asarray(pos[skel.root], dtype=float)}
    for b in skel.bones:
        drawn[b.to_joint] = drawn[b.from_joint] + shrink * (np.asarray(pos[b.to_joint]) - np.asarray(pos[b.from_joint]))
    floor_y = min(p[1] for p in drawn.values())
    for j in drawn:
        drawn[j] = drawn[j] - np.array([0.0, floor_y, 0.0])
    clicks = {j: (cam.project(P).dx, cam.project(P).dy) for j, P in drawn.items()}

    rng = np.random.default_rng(seed)
    W, H = rules["camera"]["canvas_width"], rules["camera"]["canvas_height"]
    fig = Figure(figsize=(W / 100, H / 100))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-W / 2, W / 2)
    ax.set_ylim(H / 2, -H / 2)
    ax.axis("off")

    def stroke(a, b, n=30):
        t = np.linspace(0, 1, n)[:, None]
        pts = (1 - t) * np.asarray(a) + t * np.asarray(b) + rng.normal(0, 0.8, (n, 2))
        ax.plot(pts[:, 0], pts[:, 1], color="#444444", lw=2.2, alpha=0.85, solid_capstyle="round")

    for b in skel.bones:
        if b.length > 0 and b.to_joint != "head":
            stroke(clicks[b.from_joint], clicks[b.to_joint])
    hx, hy = clicks["head"]
    r = skel.head_radius * shrink * cam.project(drawn["head"]).s
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(hx + r * np.cos(th) + rng.normal(0, 0.7, 60), hy + r * np.sin(th) + rng.normal(0, 0.7, 60),
            color="#444444", lw=2.2, alpha=0.85)
    neck = clicks["neck"]
    stroke(neck, (hx, hy + r))
    fy = max(xy[1] for xy in clicks.values()) + 4
    stroke((-230, fy), (230, fy), 60)                                     # a floor line
    png_path = pathlib.Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=100, facecolor="white")
    rigid = {b.to_joint for b in skel.bones if b.kind == "rigid"}
    out = {j: [float(x), float(y)] for j, (x, y) in clicks.items() if j not in rigid}
    if json_path:
        pathlib.Path(json_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    out_dir = ROOT / "out"
    make(out_dir / "sample_sketch.png", out_dir / "sample_sketch_joints.json")
    print(f"wrote {out_dir / 'sample_sketch.png'} and {out_dir / 'sample_sketch_joints.json'}")
