"""Turn a rough sketch into a proper pose (spec 4.2 Method B) and show the result.

    python tools/sketch_to_pose.py --underlay sketch.png --joints sketch_joints.json --start standing \
        --facing 0 --out result.png

joints JSON: {"pelvis": [dx, dy], "neck": [...], ...} in screen offsets (y down) of the camera below, for the
11 placed joints (pelvis, neck, head, elbows, hands, knees, feet). The camera is the viewer's full shot of
the standing pose from the front at eye level: in the viewer, pick "Standing" and tick "full shot" to see
the same view. The start pose (and facing) gives the default depth choices.

Leave out --joints to have them found in the --underlay drawing (no AI: sandbox/sketch_detect.py). They are
written to <out>_joints.json; correct any that are off and rerun with --joints. The start pose and facing
then default to the library pose the drawing looked most like.

    python tools/sketch_to_pose.py --underlay sketch.png --fit gesture --out result.png
"""
import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sandbox.skell import data  # noqa: E402
from sandbox.skell.camera import Camera, Subject, resolve_camera  # noqa: E402
from sandbox.skell.pose import head_rotation, turn_frame  # noqa: E402
from sandbox.skell.reconstruct import fit_drawing_camera, reconstruct  # noqa: E402
from sandbox.skell.reconstruct import sketch_camera as _sketch_camera  # noqa: E402
from sandbox.skell.render import RenderOptions, draw_scene  # noqa: E402


def sketch_camera(skel=None):
    """The viewer's full shot of the standing pose, front, eye level."""
    return _sketch_camera(skel or data.load_skeleton())


def fit_camera(joints: dict, skel=None, azimuth: float = 0.0, focal: float = 500.0, start: str = "standing",
               facing: float = 0.0) -> Camera:
    """The eye-level camera a drawing implies (sandbox/skell/reconstruct.py fit_drawing_camera): its lowest
    drawn joint stands on the floor, and the scale is the one at which the rebuilt gesture lands closest to
    the drawn joints."""
    skel = skel or data.load_skeleton()
    return fit_drawing_camera(skel, joints, _start_frame(start, facing), azimuth=azimuth, focal=focal)


def _start_frame(start: str, facing: float) -> dict:
    frame = next(p for p in data.pose_library() if p["name"] == start)["frame"]
    return turn_frame(frame, facing - float(frame["root"].get("facing", 0.0)))


def sketch_to_pose(joints: dict, start: str = "standing", facing: float = 0.0, cam: Camera | None = None,
                   fit: str = "exact"):
    skel = data.load_skeleton()
    cam = cam or sketch_camera(skel)
    frame = _start_frame(start, facing)
    rec = reconstruct(skel, cam, {j: tuple(xy) for j, xy in joints.items()}, frame, fit=fit)
    subj = Subject(skel, rec.validation.positions, facing, head_rotation(skel, rec.validation.pose))
    return rec, subj, cam


def render_result(path, rec, subj, cam, underlay=None, joints=None, title=""):
    rules = data.load_rules()
    W, H = rules["camera"]["canvas_width"], rules["camera"]["canvas_height"]
    fig = Figure(figsize=(3 * W / 100, H / 100 + 0.9))
    FigureCanvasAgg(fig)
    views = [("over the sketch (same camera)", cam),
             ("three-quarter, from above", resolve_camera({"shot": "full", "azimuth": cam.azimuth - 45,
                                                           "elevation": 20}, subj).camera),
             ("side view (depth)", resolve_camera({"shot": "full", "azimuth": cam.azimuth + 90,
                                                   "elevation": 0}, subj).camera)]
    for i, (name, c) in enumerate(views):
        ax = fig.add_axes([i / 3 + 0.005, 0.02, 1 / 3 - 0.01, 0.8])
        if i == 0 and underlay is not None:
            ax.imshow(underlay, extent=(-W / 2, W / 2, H / 2, -H / 2), alpha=0.45, zorder=0.5, aspect="auto")
        draw_scene(ax, c, [subj], options=RenderOptions(title=name))
        if i == 0 and joints:
            for j, (x, y) in joints.items():
                ax.plot(x, y, "x", color="#d64545", ms=5, mew=1.5, zorder=500)
    v = rec.validation
    status = [f"balance {v.balance.status}", f"contacts {', '.join(v.contacts) or 'none'}"]
    status += [str(c) for c in v.limits.corrections] + rec.flags + v.overlaps
    fig.suptitle(f"{title}\n" + " · ".join(status[:6]), fontsize=8.5)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor="white")


def auto_joints(image_path, out_json=None):
    """Find the joints in a stick-figure drawing (no AI) and write them where they can be corrected."""
    from sandbox.sketch_detect import detect, pixels_to_screen
    img = mpimg.imread(image_path)
    det = detect(img)
    cam_rules = data.load_rules()["camera"]
    joints = pixels_to_screen(det.joints, img.shape, cam_rules["canvas_width"], cam_rules["canvas_height"])
    if out_json:
        pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_json).write_text(json.dumps(joints, indent=2) + "\n", encoding="utf-8")
    return joints, det


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--joints", help="joint positions JSON; left out: found in the --underlay drawing "
                                     "(and written next to --out as <name>_joints.json, to correct)")
    ap.add_argument("--underlay")
    ap.add_argument("--start", default=None, help="library pose for the depth choices (default: standing, "
                                                  "or the one the drawing looked like when joints are found)")
    ap.add_argument("--facing", type=float, default=None)
    ap.add_argument("--fit", choices=["exact", "gesture"], default="exact",
                    help="exact: drawn lengths are true; gesture: a stylised drawing, use its directions only")
    ap.add_argument("--camera", choices=["standard", "fit"], default=None,
                    help="standard: the viewer's full shot of the standing pose; fit: the camera the drawing "
                         "implies (default for --fit gesture)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.joints:
        joints = json.loads(pathlib.Path(args.joints).read_text(encoding="utf-8"))
        title = pathlib.Path(args.joints).stem
        start, facing = args.start or "standing", args.facing or 0.0
    else:
        if not args.underlay:
            ap.error("give --joints, or an --underlay drawing to find the joints in")
        out = pathlib.Path(args.out)
        jpath = out.with_name(out.stem + "_joints.json")
        joints, det = auto_joints(args.underlay, jpath)
        start = args.start or det.start
        facing = det.facing if args.facing is None else args.facing
        title = f"{pathlib.Path(args.underlay).stem} (joints found automatically)"
        print(f"joints found {'by the trained model' if det.method == 'model' else 'by the geometric fit'} -> {jpath} (looks like {det.start}, facing {det.facing:+.0f})")
        for line in det.notes:
            print("  " + line)
    use_fit = args.camera == "fit" or (args.camera is None and args.fit == "gesture")
    cam = fit_camera(joints, start=start, facing=facing) if use_fit else None
    rec, subj, cam = sketch_to_pose(joints, start, facing, cam=cam, fit=args.fit)
    under = mpimg.imread(args.underlay) if args.underlay else None
    render_result(args.out, rec, subj, cam, under, joints, title)
    print(f"wrote {args.out}")
    for line in [str(c) for c in rec.validation.limits.corrections] + rec.flags + rec.notes:
        print("  " + line)


if __name__ == "__main__":
    main()
