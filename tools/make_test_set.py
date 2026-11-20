"""Collect real drawings with joint labels into a test set, framed exactly like the training images.

    python tools/make_test_set.py                       # stock figures + labelled pad drawings
    python tools/make_test_set.py --folders out/sketches/pad my_drawings

Sources:
  - out/sketches/figure_N.png with figure_N_joints.json (joints read by eye, screen offsets on the canvas)
  - any <name>.png with <name>_joints.json from tools/label_drawing.py (image pixels) in --folders
Writes out/training/test_real.npz (images, joints, valid, names, joint_names) and test_preview.png.
"""
import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from make_training_set import preview  # noqa: E402
from sandbox.sketch_detect import PLACED  # noqa: E402
from sandbox.skell import data  # noqa: E402
from sandbox.synth import frame_drawing  # noqa: E402


def labelled_drawings(folders, stock=True):
    """(name, image, joints dict in image pixels or None per joint)."""
    cam = data.load_rules()["camera"]
    W, H = cam["canvas_width"], cam["canvas_height"]
    sk = ROOT / "out" / "sketches"
    if stock:
        for n in range(1, 10):
            img_p, j_p = sk / f"figure_{n}.png", sk / f"figure_{n}_joints.json"
            if img_p.is_file() and j_p.is_file():
                img = mpimg.imread(img_p)
                h, w = img.shape[:2]
                J = json.loads(j_p.read_text(encoding="utf-8"))   # screen offsets on the canvas
                yield f"stock figure {n}", img, {k: [(x + W / 2) * w / W, (y + H / 2) * h / H] for k, (x, y) in J.items()}
    for folder in folders:
        for j_p in sorted(pathlib.Path(folder).glob("*_joints.json")):
            img_p = j_p.with_name(j_p.name[:-len("_joints.json")] + ".png")
            lab = json.loads(j_p.read_text(encoding="utf-8"))
            if not img_p.is_file() or lab.get("units") != "pixels":
                continue
            yield img_p.stem, mpimg.imread(img_p), lab["joints"]


def build(folders, size=192, stock=True):
    names, imgs, joints, valid = [], [], [], []
    for name, img, J in labelled_drawings(folders, stock):
        small, (scale, ox, oy) = frame_drawing(img, size)
        arr = np.zeros((len(PLACED), 2), dtype=np.float32)
        ok = np.zeros(len(PLACED), dtype=bool)
        for i, j in enumerate(PLACED):
            if J.get(j) is not None:
                arr[i] = [(J[j][0] - ox) * scale, (J[j][1] - oy) * scale]
                ok[i] = True
        names.append(name)
        imgs.append(small)
        joints.append(arr)
        valid.append(ok)
    return names, np.array(imgs), np.array(joints), np.array(valid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folders", nargs="*", default=[str(ROOT / "out" / "sketches" / "pad")])
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--out", default=str(ROOT / "out" / "training"))
    args = ap.parse_args(argv)
    names, imgs, joints, valid = build(args.folders, args.size, stock=not args.no_stock)
    if not names:
        print("no labelled drawings found (label some with tools/label_drawing.py)")
        return
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test_real.npz", images=imgs, joints=joints, valid=valid, names=np.array(names),
                        joint_names=np.array(PLACED))
    preview(out / "test_preview.png", imgs, joints, cols=min(8, len(names)), title="real test drawings")
    print(f"wrote {out / 'test_real.npz'}: {len(names)} drawings: {', '.join(names)}")


if __name__ == "__main__":
    main()
