"""Offline joint finding in stick-figure drawings (sandbox/sketch_detect.py): the geometric fit, and the
learned model path when a trained model (out/models/joint_net.npz) is present."""
import numpy as np
import pytest

from sandbox.sketch_detect import (MODEL_PATH, best_start, detect, find_head, ink_mask, pixels_to_screen,
                                   stroke_ends, thin)

# a front-facing figure, arms and legs out (image pixels, y down); l_ is image right (facing the camera)
TRUTH = {"head": (150, 50), "neck": (150, 82), "pelvis": (152, 160),
         "l_elbow": (185, 115), "l_hand": (220, 145), "r_elbow": (115, 115), "r_hand": (80, 145),
         "l_knee": (180, 215), "l_foot": (205, 270), "r_knee": (122, 215), "r_foot": (98, 270)}
LINES = [("neck", "pelvis"), ("neck", "l_elbow"), ("l_elbow", "l_hand"), ("neck", "r_elbow"),
         ("r_elbow", "r_hand"), ("pelvis", "l_knee"), ("l_knee", "l_foot"), ("pelvis", "r_knee"),
         ("r_knee", "r_foot")]


def draw(joints, size=(300, 300), width=3.0, head_r=20.0):
    """Rasterise a stick figure: thick segments and a ring for the head (white background)."""
    yy, xx = np.mgrid[0:size[0], 0:size[1]] + 0.5
    P = np.stack([xx, yy], axis=-1)
    ink = np.zeros(size, dtype=bool)
    for a, b in LINES:
        A, B = np.array(joints[a], float), np.array(joints[b], float)
        t = np.clip(((P - A) @ (B - A)) / ((B - A) @ (B - A)), 0, 1)
        ink |= np.linalg.norm(P - (A + t[..., None] * (B - A)), axis=-1) <= width / 2
    d = np.linalg.norm(P - np.array(joints["head"], float), axis=-1)
    ink |= np.abs(d - head_r) <= width / 2
    return np.where(ink, 0.0, 1.0)[..., None].repeat(3, axis=2)


@pytest.fixture(scope="module")
def drawn():
    img = draw(TRUTH)
    return img, detect(img, model=None)          # the geometric fit, model or not


def test_head_is_the_round_hole():
    centre, r = find_head(ink_mask(draw(TRUTH)))
    assert np.hypot(centre[0] - 150, centre[1] - 50) < 1.0
    assert 17 < r < 20


def test_stroke_ends_are_the_hands_and_feet():
    img = draw(TRUTH)
    ink = ink_mask(img)
    yy, xx = np.mgrid[0:300, 0:300]
    body = ink & (np.hypot(xx + 0.5 - 150, yy + 0.5 - 50) > 24)
    r, c = np.nonzero(thin(body))
    ends = stroke_ends(np.stack([c + 0.5, r + 0.5], 1), 3.0)
    want = [TRUTH[j] for j in ("l_hand", "r_hand", "l_foot", "r_foot")]
    for w in want:      # thinning shortens a diagonal stroke's tip by a couple of stroke widths
        assert min(np.hypot(*(e - w)) for e in ends) < 10
    # the neck top is also a free end; nothing else
    assert len(ends) <= 5


def test_joints_found_within_a_few_percent(drawn):
    _, det = drawn
    height = TRUTH["r_foot"][1] - TRUTH["head"][1]
    err = {j: float(np.hypot(*np.subtract(det.joints[j], TRUTH[j]))) for j in TRUTH}
    assert max(err.values()) < 0.06 * height, err
    assert np.mean(list(err.values())) < 0.03 * height, err


def test_front_figure_reads_as_front_with_anatomical_sides(drawn):
    _, det = drawn
    assert det.facing == 0.0
    assert det.joints["l_hand"][0] > det.joints["r_hand"][0]       # character's left is image right


def test_pixels_map_onto_the_canvas():
    out = pixels_to_screen({"a": (0.0, 0.0), "b": (300.0, 300.0), "c": (150.0, 150.0)}, (300, 300, 3), 570, 380)
    assert out["a"] == [-285.0, -190.0] and out["b"] == [285.0, 190.0] and out["c"] == [0.0, 0.0]


def test_viewer_auto_joints_puts_the_pose_on_the_drawing(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    from sandbox.app import App, ViewerState
    path = tmp_path / "figure.png"
    mpimg.imsave(path, draw(TRUTH))
    app = App(ViewerState(underlay=str(path)), show=False)
    try:
        app.press("auto joints")
        st = app.state
        assert st.edited and st.sketch_mode and st.last_detection is not None
        cam = st.resolved().camera
        drawn = pixels_to_screen(st.last_detection.joints, (300, 300), 570, 380)
        pos = st.subject.positions
        err = [np.hypot(cam.project(pos[j]).dx - x, cam.project(pos[j]).dy - y) for j, (x, y) in drawn.items()]
        height = drawn["r_foot"][1] - drawn["head"][1]
        assert np.sqrt(np.mean(np.square(err))) < 0.12 * height       # the rebuilt pose sits on the drawing
    finally:
        app.close()


def test_viewer_auto_joints_without_a_drawing_says_so():
    import matplotlib
    matplotlib.use("Agg")
    from sandbox.app import App
    app = App(show=False)
    try:
        app.press("auto joints")
        assert "--underlay" in app.status_text() and not app.state.edited
    finally:
        app.close()


def test_blank_drawing_is_refused():
    with pytest.raises(ValueError):
        detect(np.ones((50, 50, 3)), model=None)
    with pytest.raises(ValueError):
        detect(np.ones((50, 50, 3)))


def test_best_start_reads_a_front_figure_as_front():
    start, facing = best_start(TRUTH)
    assert facing == 0.0


# ---------------------------------------------------------------- the learned model (if trained)

needs_model = pytest.mark.skipif(not MODEL_PATH.is_file(), reason="no trained joint model (tools/kaggle_train.py fetch)")


@pytest.fixture(scope="module")
def by_model():
    img = draw(TRUTH)
    return img, detect(img)


@needs_model
def test_model_is_used_when_present(by_model):
    _, det = by_model
    assert det.method == "model"
    assert set(det.confidence) == set(TRUTH) and all(0 <= c <= 1 for c in det.confidence.values())


@needs_model
def test_model_finds_joints_within_a_few_percent(by_model):
    _, det = by_model
    height = TRUTH["r_foot"][1] - TRUTH["head"][1]
    err = {j: float(np.hypot(*np.subtract(det.joints[j], TRUTH[j]))) for j in TRUTH}
    assert max(err.values()) < 0.06 * height, err
    assert np.mean(list(err.values())) < 0.03 * height, err


@needs_model
def test_model_head_is_the_hole_centre_and_front_sides(by_model):
    _, det = by_model
    assert np.hypot(det.joints["head"][0] - 150, det.joints["head"][1] - 50) < 1.0
    assert det.facing == 0.0
    assert det.joints["l_hand"][0] > det.joints["r_hand"][0]       # character's left is image right
    assert det.joints["l_foot"][0] > det.joints["r_foot"][0]
