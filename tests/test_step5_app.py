"""Step 5: interactive viewer, driven headless through its real widgets.

Done when: the kneeling pose spins all the way round, goes over the top and under the floor (guard off)
without the image flipping, and every preset lands where the fixture says.
"""
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sandbox.app import App, ViewerState  # noqa: E402
from sandbox.skell import data  # noqa: E402

SCREEN_TOL = 0.01
WORLD_TOL = 1e-6

# fixture row -> preset buttons pressed after a reset (and whether the floor guard is switched off first)
ROWS = {
    "front_eye_level": (["front", "eye_level"], True),
    "elevated_30": (["front", "elevated"], True),
    "overhead_90": (["front", "overhead"], True),
    "distant_eye_level": (["front", "distant_eye_level"], True),
    "right_side": (["right_side", "eye_level"], True),
    "front_right_three_quarter": (["front_right_three_quarter", "eye_level"], True),
    "left_side": (["left_side", "eye_level"], True),
    "back_left_three_quarter": (["back_left_three_quarter", "eye_level"], True),
    "back": (["back", "eye_level"], True),
    "dutch_20": (["front", "eye_level", "dutch"], True),
    "low_angle_30_floor_guard_on": (["front", "low_angle"], True),
    "worms_eye_75_floor_guard_off": (["front", "worms_eye"], False),
}


@pytest.fixture(scope="module")
def app():
    a = App(show=False)
    yield a
    a.close()


def screen(state):
    cam = state.resolved().camera
    return cam, {j: cam.project(p) for j, p in state.subject.positions.items()}


def test_every_preset_lands_where_the_fixture_says(app):
    fixture = {r["name"]: r for r in data.load_fixture()["cameras"]}
    assert set(ROWS) == set(fixture)
    for name, (presets, guard) in ROWS.items():
        app.press("reset")
        if not guard:
            app.toggle("floor guard")
        for p in presets:
            app.press(p)
        cam, proj = screen(app.state)
        row = fixture[name]
        assert np.allclose(cam.position, row["camera_position"], atol=WORLD_TOL), name
        for joint, exp in row["screen_offsets"].items():
            for k in ("dx", "dy", "s"):
                assert abs(getattr(proj[joint], k) - exp[k]) < SCREEN_TOL, (name, joint, k)
        assert app.state.floor_guard == guard
    app.press("reset")


def test_presets_move_the_sliders(app):
    app.press("reset")
    app.press("back_left_three_quarter")
    app.press("elevated")
    assert app.sliders["azimuth"].val == 135
    assert app.sliders["elevation"].val == 30 and app.sliders["distance"].val == 410
    app.press("dutch")
    assert app.sliders["roll"].val == 20
    app.press("reset")
    assert app.sliders["azimuth"].val == 0 and app.sliders["roll"].val == 0


def _max_jump(frames):
    """Largest screen movement of any visible joint between consecutive frames."""
    worst = 0.0
    for a, b in zip(frames, frames[1:]):
        for j in a:
            if a[j].visible and b[j].visible:
                worst = max(worst, math.hypot(a[j].dx - b[j].dx, a[j].dy - b[j].dy))
    return worst


def test_spin_all_the_way_round_without_flipping(app):
    app.press("reset")
    app.toggle("floor guard")   # guard off, R stays 410 (with the guard on, low angles pull the camera in close)
    with app.paused():   # drive the real sliders, but redraw once at the end instead of per step
        for el in (-60, -30, 0, 30, 60, 89, 90):
            app.set_slider("elevation", el)
            frames = []
            for az in list(range(-180, 181, 2)):
                app.set_slider("azimuth", az)
                frames.append(screen(app.state)[1])
            assert _max_jump(frames) < 40, el     # 2-degree steps: smooth motion is under ~20, a flip is hundreds
    app.press("reset")


def test_over_the_top_and_under_the_floor_without_flipping(app):
    app.press("reset")
    app.toggle("floor guard")                                    # guard off: allow the camera under the floor
    assert not app.state.floor_guard
    with app.paused():
        for az in (0, 45, 90, 180, -135):
            app.set_slider("azimuth", az)
            frames = []
            for el in range(-90, 91):
                app.set_slider("elevation", el)
                frames.append(screen(app.state)[1])
            assert _max_jump(frames) < 40, az
    app.set_slider("elevation", -75)
    assert app.state.resolved().camera.position[1] < 0
    app.set_slider("elevation", 120)                             # dragging past the pole stops at the pole
    assert app.sliders["elevation"].val == 90
    app.press("reset")


def test_floor_guard_toggle(app):
    app.press("reset")
    app.press("low_angle")
    assert abs(app.state.resolved().camera.distance - 214.314814) < 1e-6
    assert "floor guard" in app.status_text()
    app.toggle("floor guard")
    assert app.state.resolved().camera.distance == 410
    app.press("reset")


def test_character_relative_and_facing(app):
    app.press("reset")
    assert app.state.relative
    _, before = screen(app.state)
    app.set_slider("facing", 90)                                  # character-relative: camera turns with it
    cam, after = screen(app.state)
    assert math.isclose(cam.azimuth, 90, abs_tol=1e-9)
    for j in before:
        assert abs(before[j].dx - after[j].dx) < 1e-6 and abs(before[j].dy - after[j].dy) < 1e-6
    app.toggle("character-relative")                              # switching keeps the same view
    assert not app.state.relative
    assert app.sliders["azimuth"].val == 90
    assert math.isclose(app.state.resolved().camera.azimuth, 90, abs_tol=1e-9)
    app.set_slider("facing", 0)                                   # world mode: the character turns on screen
    cam, turned = screen(app.state)
    assert math.isclose(cam.azimuth, 90, abs_tol=1e-9)
    assert max(abs(turned[j].dx - before[j].dx) for j in before) > 10
    app.press("reset")
    assert app.state.relative and app.state.facing == 0


def test_overlays_toggle_and_redraw(app):
    app.press("reset")
    app.toggle("overlays")
    assert app.state.overlays
    assert any(it.color == "amber" for it in app.drawing.items if it.kind == "bone")
    app.toggle("overlays")
    assert all(it.color == "bone" for it in app.drawing.items if it.kind == "bone")


def test_viewer_screenshot(app):
    app.press("reset")
    app.press("front_right_three_quarter")
    app.press("elevated")
    path = data.ROOT / "out" / "viewer.png"
    app.save(path)
    assert path.exists() and path.stat().st_size > 20000
    app.press("reset")


def test_state_without_ui():
    st = ViewerState()
    st.apply_preset("back")
    assert math.isclose(st.resolved().camera.azimuth, 180, abs_tol=1e-9)
    st.set_relative(False)
    assert st.azimuth == 180


def test_pose_picker(app):
    app.press("reset")
    app.press("front_right_three_quarter")
    names = list(app.state.poses)
    assert names[0] == "reference_kneeling" and len(names) >= 10
    for name in names:
        app.select_pose(name)
        assert app.state.pose_name == name
        assert app.state.poses[name]["label"] in app.status_text()
        assert app.state.validation.limits.ok, name                         # library poses need no clamping
        assert sum(it.kind == "bone" for it in app.drawing.items) == 10
        assert app.sliders["azimuth"].val == -45                            # the camera stays put
    app.select_pose("jump")
    assert "airborne" in app.status_text()
    app.select_pose("standing")
    app.save(data.ROOT / "out" / "viewer_standing.png")
    app.select_pose("reference_kneeling")
    assert "unstable" in app.status_text()
    app.press("reset")


def test_full_shot_frames_the_whole_pose(app):
    app.press("reset")
    app.select_pose("standing")
    feet = lambda: [app.state.resolved().camera.project(app.state.subject.positions[j]).dy for j in ("l_foot", "r_foot")]
    assert max(feet()) > 190                                    # head-target default: feet below the canvas
    app.toggle("full shot")
    from sandbox.skell.camera import bounding_box
    lo, hi = bounding_box(app.state.subject)                   # legs 4 deg apart: 249.9 tall, not 250
    expected = 500 * (hi[1] - lo[1]) / (0.8 * 380)
    assert app.state.full_shot and abs(app.state.resolved().camera.distance - expected) < 1e-9
    assert max(feet()) < 190                                    # now the whole figure is in frame
    app.save(data.ROOT / "out" / "viewer_standing.png")
    app.press("reset")
    assert not app.state.full_shot
    app.select_pose("reference_kneeling")
