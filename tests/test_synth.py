"""Synthetic training drawings (sandbox/synth.py) and the real-drawing test set tools."""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from sandbox.sketch_detect import PLACED  # noqa: E402
from sandbox.synth import frame_drawing, generate  # noqa: E402


@pytest.fixture(scope="module")
def batch():
    return generate(120, seed=7, size=192)


def test_same_seed_same_drawings():
    a, b = generate(6, seed=3), generate(6, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_shapes_and_labels_inside_the_image(batch):
    imgs, joints, radii, meta = batch
    assert imgs.shape == (120, 192, 192) and imgs.dtype == np.uint8
    assert joints.shape == (120, len(PLACED), 2)
    assert np.isfinite(joints).all() and (joints > 0).all() and (joints < 192).all()
    assert (radii > 1).all() and len(meta) == 120


def test_every_pose_family_appears(batch):
    assert {m["family"] for m in batch[3]} == {"library", "upright", "bent", "lying"}


def test_joints_sit_on_the_strokes(batch):
    """Every labelled joint is on ink, within the line's half width plus the wobble (the label noise)."""
    imgs, joints, radii, meta = batch
    head = PLACED.index("head")
    for img, J, m in zip(imgs, joints, meta):
        rows, cols = np.nonzero(img < 128)
        ink = np.stack([cols + 0.5, rows + 0.5], axis=1)
        tol = m["style"]["width"] / 2 + m["style"]["wobble"] + 1.5
        for i, p in enumerate(J):
            if i == head:
                continue
            assert np.min(np.hypot(*(ink - p).T)) <= tol


def test_head_label_is_the_circle_centre(batch):
    imgs, joints, radii, meta = batch
    head = PLACED.index("head")
    a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for img, J, r in zip(imgs[:40], joints[:40], radii[:40]):
        x = np.clip((J[head, 0] + r * np.cos(a)).astype(int), 0, 191)
        y = np.clip((J[head, 1] + r * np.sin(a)).astype(int), 0, 191)
        near = [img[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3].min() < 128 for xx, yy in zip(x, y)]
        assert np.mean(near) > 0.75             # the ring is there (open heads leave a gap of up to 30 deg)


def test_frame_drawing_crops_squares_and_maps_points():
    img = np.ones((300, 500, 3))
    img[100:102, 200:300] = 0.0                # a 100 x 2 bar
    small, (scale, ox, oy) = frame_drawing(img, size=96, margin=0.1)
    assert small.shape == (96, 96)
    # the bar's ends land where the mapping says, and the drawing spans 80% of the frame
    for x, y in ((200.5, 101.0), (299.5, 101.0)):
        u, v = (x - ox) * scale, (y - oy) * scale
        assert small[int(v), int(np.clip(u, 0, 95))] < 160
    assert abs(100 * scale - 96 * 0.8) < 1.0


def test_labeler_and_test_set(tmp_path):
    from label_drawing import ORDER, Labeler
    from make_test_set import build
    img = np.ones((120, 160, 3))
    img[20:100, 78:82] = 0.0
    path = tmp_path / "drawing.png"
    mpimg.imsave(path, img)
    lab = Labeler(path)
    for i, _ in enumerate(ORDER):
        lab.click(80, 20 + i * 7)
    lab.undo()
    lab.skip()                                  # r_foot can't be placed
    assert lab.next_joint is None
    saved = json.loads(lab.save().read_text())
    assert saved["units"] == "pixels" and saved["joints"]["r_foot"] is None
    names, imgs, joints, valid = build([tmp_path], size=64, stock=False)
    assert names == ["drawing"] and imgs.shape == (1, 64, 64)
    assert valid[0].sum() == len(PLACED) - 1 and not valid[0][PLACED.index("r_foot")]
