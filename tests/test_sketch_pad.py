"""The viewer's sketch pad: strokes drawn with a mouse or finger -> sketch image -> joints -> pose."""
import json
import math

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sandbox.app import App  # noqa: E402
from sandbox.sketch_detect import find_head, ink_mask  # noqa: E402
from sandbox.sketch_pad import close_loops, load_strokes, rasterize  # noqa: E402

W, H = 570, 380


def wobbly_circle(cx, cy, r, gap_deg=25.0, n=40, seed=0):
    """A hand-drawn head: slightly uneven, and it doesn't quite close."""
    rng = np.random.default_rng(seed)
    a = np.linspace(math.radians(-80), math.radians(280 - gap_deg), n)
    rr = r * (1 + 0.04 * rng.standard_normal(n))
    return list(zip(cx + rr * np.cos(a), cy + rr * np.sin(a)))


def line(*pts, step=4.0):
    """A stroke through pts, sampled as a mouse would (a point every few screen units)."""
    out = []
    for A, B in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.hypot(B[0] - A[0], B[1] - A[1]) / step))
        out += [(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t) for t in np.linspace(0, 1, n, endpoint=False)]
    return out + [tuple(pts[-1])]


# a stick figure, standing with arms out, filling most of the page (screen offsets, y down)
FIGURE = [
    wobbly_circle(0, -135, 22),
    line((0, -110), (2, -10)),                             # body
    line((0, -95), (-40, -60), (-75, -30)),                # arm (image left)
    line((0, -95), (40, -60), (78, -32)),                  # arm (image right)
    line((2, -10), (-25, 70), (-45, 150)),                 # leg
    line((2, -10), (28, 70), (48, 150)),                   # leg
]


def test_a_hand_drawn_circle_is_closed_but_a_limb_is_not():
    head, body, arm = close_loops([FIGURE[0], FIGURE[1], FIGURE[2]])
    assert head[-1] == head[0]
    assert body[-1] != body[0] and arm[-1] != arm[0]


def test_rasterised_head_is_found():
    img = rasterize(FIGURE, W, H)
    assert img.shape == (H, W, 3)
    centre, r = find_head(ink_mask(img))
    # pixel (col, row) is screen (col - W/2, row - H/2) at pixel corners
    assert abs(centre[0] - W / 2 - 0) < 2 and abs(centre[1] - H / 2 - (-135)) < 2
    assert 17 < r < 23


@pytest.fixture
def app():
    a = App(show=False)
    yield a
    a.close()


def test_pad_mode_shows_a_blank_page_and_collects_strokes(app):
    app.press("draw sketch")
    assert app.pad_mode and "sketch pad" in app.status_text()
    for s in FIGURE:
        app.pad_stroke(s)
    app.press("undo stroke")
    assert len(app.pad_strokes) == len(FIGURE) - 1
    app.press("clear drawing")
    assert app.pad_strokes == []


# where an artist would click the joints of FIGURE (screen offsets)
CLICKS = {"pelvis": (2, -10), "neck": (0, -100), "head": (0, -135), "l_elbow": (40, -60), "l_hand": (78, -32),
          "r_elbow": (-40, -60), "r_hand": (-75, -30), "l_knee": (28, 70), "l_foot": (48, 150),
          "r_knee": (-25, 70), "r_foot": (-45, 150)}


def test_use_drawing_saves_it_then_joints_are_clicked_on_it(app, tmp_path):
    app.press("draw sketch")
    for s in FIGURE:
        app.pad_stroke(s)
    app.use_drawing(folder=tmp_path)
    png, js = app.last_pad_files
    assert png.is_file() and js.is_file()
    assert len(load_strokes(js)) == len(FIGURE)
    assert json.loads(js.read_text())["canvas"] == [W, H]
    st = app.state
    assert not app.pad_mode and st.underlay is not None and app.placing      # clicking starts at once
    assert "auto joints" in app.status_text()
    while app.placing:
        app.place_click(*CLICKS[app.next_joint])
    assert st.edited and st.sketch_mode and not st.last_rec.flags            # a free drawing: no "too long"
    cam = st.resolved().camera
    pos = st.subject.positions
    err = [np.hypot(cam.project(pos[j]).dx - x, cam.project(pos[j]).dy - y) for j, (x, y) in CLICKS.items()]
    assert np.sqrt(np.mean(np.square(err))) < 0.06 * 285                     # the pose sits on the drawing
    assert st.pose_name in ("standing", "t_pose")          # started from the pose the clicks look like


def test_stick_figure_drawing_can_use_auto_joints(app, tmp_path):
    app.press("draw sketch")
    for s in FIGURE:
        app.pad_stroke(s)
    app.use_drawing(folder=tmp_path)
    app.press("auto joints")
    st = app.state
    assert not app.placing and st.edited
    det = st.last_detection
    assert not any("no round head" in n for n in det.notes)
    # the found joints sit where the figure was drawn (screen offsets = pixel - canvas/2)
    j = {k: (x - W / 2, y - H / 2) for k, (x, y) in det.joints.items()}
    assert np.hypot(j["head"][0], j["head"][1] + 135) < 3
    feet = sorted([j["l_foot"], j["r_foot"]])
    assert np.hypot(feet[0][0] + 45, feet[0][1] - 150) < 15 and np.hypot(feet[1][0] - 48, feet[1][1] - 150) < 15


def test_use_drawing_with_nothing_drawn_says_so(app):
    app.press("use drawing")
    assert "nothing drawn" in app.status_text() and not app.state.edited
