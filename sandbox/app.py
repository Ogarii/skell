"""Interactive viewer (sandbox step 5, with step 6 checks).

    python -m sandbox.app                     # starts on the reference kneeling pose
    python -m sandbox.app --pose my_pose.json # adds a frame, or a pose request with options/frames
    python -m sandbox.app --underlay sketch.png  # a rough drawing behind the figure, to place joints over

Pose list: the reference pose plus data/sandbox_poses_v1.json (and --pose, if given).
Sliders: azimuth, elevation, roll, distance, focal length, character facing.
Buttons: every camera preset in data/skell_rules_v1.json, reset, and turnaround (saves the 8-view sheet).
Toggles: floor guard, editor overlays, character-relative azimuth, full shot (frame the whole pose).
The pose shown is the validated one (clamped to joint limits); the status panel lists every check.

Sketch input (spec 4.2 Method B): tick "sketch: drag joints", then drag any joint in the view. Skell
rebuilds the 3D pose by ray-sphere depth (sandbox/skell/reconstruct.py) and validates it. Press f to flip
the depth of the bone ending at the last joint you dragged. Re-select the pose in the list to undo.
Guided placement: press "place joints", then click each joint on the sketch as the status panel asks
(backspace undoes a click, n skips a joint, Esc cancels). A side view in the corner shows depth.
Automatic: with --underlay, press "auto joints" (about 20 s, no AI). It finds the joints in the drawing,
starts from the library pose it looks like, fits the camera to the drawing's scale, and leaves sketch mode
on so you can drag any joint that is off.
Sketch pad: press "draw sketch" and draw a stick figure in the view with the mouse or a finger (backspace or
"undo stroke" removes the last stroke, "clear drawing" starts over). "use drawing" saves it to
out/sketches/pad/ (PNG + strokes JSON), puts it behind the figure and runs auto joints.

ViewerState holds the values and has no matplotlib in it; App is the matplotlib front end.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from sandbox.skell import data
from sandbox.skell.camera import ResolvedCamera, Subject, resolve_camera
from sandbox.skell.checks import validate
from sandbox.skell.pose import head_rotation, pose_from_frame, turn_frame
from sandbox.skell.quat import wrap_deg
from sandbox.skell.camera import Camera
from sandbox.skell.reconstruct import fit_drawing_camera, placement_order, reconstruct
from sandbox.skell.render import RenderOptions, draw_scene
from sandbox.skell.turnaround import render_turnaround


class ViewerState:
    """Camera and character values behind the viewer. The camera is resolved through spec 8.5 each time."""

    def __init__(self, frame: dict | None = None, skel=None, rules: dict | None = None,
                 underlay: str | None = None):
        self.skel = skel or data.load_skeleton()
        self.underlay = mpimg.imread(underlay) if underlay else None   # sketch drawn behind the figure
        self.sketch_mode = False
        self.edited = False            # the pose was changed by sketch input
        self.last_rec = None           # last reconstruction (depth solutions and flags per bone)
        self.last_detection = None     # last automatic joint finding on the underlay
        self.camera_override = None    # the camera stays put while a joint is being dragged
        self.rules = rules or data.load_rules()
        self.poses = {p["name"]: p for p in data.pose_library()}
        if frame is not None:
            self.poses = {"custom": {"name": "custom", "label": "Custom (--pose)", "frame": frame}, **self.poses}
        self.pose_name = next(iter(self.poses))
        self.base_frame = copy.deepcopy(self.poses[self.pose_name]["frame"])
        self.validation = None
        self.reset()

    def select_pose(self, name: str) -> None:
        """Show another pose; the camera stays where it is, the facing comes from the pose."""
        self.pose_name = name
        self.base_frame = copy.deepcopy(self.poses[name]["frame"])
        self.facing = float(self.base_frame.get("root", {}).get("facing", 0.0))
        self.edited = False
        self.last_rec = None
        self._subject = None

    def reset(self) -> None:
        cam = self.rules["camera"]
        eye = cam["presets"]["eye_level"]
        self.facing = float(self.base_frame.get("root", {}).get("facing", 0.0))
        self.relative = True                 # azimuth slider is character-relative
        self.azimuth = 0.0
        self.elevation = float(eye["elevation"])
        self.roll = 0.0
        self.distance = float(eye["distance"])
        self.focal = float(cam["default_focal"])
        self.floor_guard = bool(cam["floor_guard_default"])
        self.overlays = False
        self.full_shot = False             # frame the whole pose (spec 8.3) instead of the head at R
        self._subject = None

    # ---- character
    def current_frame(self) -> dict:
        """The pose as shown: the base frame turned to the facing slider (chest and head turn with it)."""
        base = float(self.base_frame.get("root", {}).get("facing", 0.0))
        return turn_frame(self.base_frame, self.facing - base)

    def sketch_edit(self, placements: dict, flips: dict | None = None, fit: str = "exact"):
        """Rebuild the pose from 2D joint placements in the current view (spec 4.2 Method B)."""
        rec = reconstruct(self.skel, self.resolved().camera, placements, self.current_frame(), flips, self.rules,
                          fit=fit)
        self.base_frame = rec.frame            # its root facing is the current facing
        self.edited = True
        self.last_rec = rec
        self._subject = None
        return rec

    def auto_joints(self):
        """Find the joints in the underlay drawing (no AI), start from the library pose it looks like, and
        rebuild the pose (gesture fit: drawn directions, true lengths) in the camera of the drawing's scale."""
        from sandbox.sketch_detect import detect, pixels_to_screen
        if self.underlay is None:
            raise ValueError("no underlay drawing to find joints in (start with --underlay sketch.png)")
        det = detect(self.underlay)
        cam = self.rules["camera"]
        joints = pixels_to_screen(det.joints, self.underlay.shape, cam["canvas_width"], cam["canvas_height"])
        if det.start in self.poses:
            self.select_pose(det.start)
        self.set_facing(det.facing)
        # the view in which the drawing has the character's size (a stick figure has its own proportions),
        # held still so the result sits on the drawing
        cam = fit_drawing_camera(self.skel, joints, self.current_frame(), quick=True, rules=self.rules)
        self.camera_override = ResolvedCamera(cam, notes=["camera fitted to the drawing's scale (front, eye level)"])
        rec = self.sketch_edit(joints, fit="gesture")
        self.last_detection = det
        return rec, det

    @property
    def subject(self) -> Subject:
        if self._subject is None:
            frame = self.current_frame()
            pose = pose_from_frame(self.skel, frame)
            self.validation = validate(self.skel, pose, rules=self.rules)      # clamps, grounds and checks
            self._subject = Subject(self.skel, self.validation.positions, pose.facing,
                                    head_rotation(self.skel, self.validation.pose))
        return self._subject

    def set_facing(self, facing: float) -> None:
        self.facing = float(facing)
        self._subject = None

    # ---- camera
    def world_azimuth(self) -> float:
        return wrap_deg(self.facing + self.azimuth) if self.relative else wrap_deg(self.azimuth)

    def set_relative(self, on: bool) -> None:
        """Switch the azimuth slider between world and character-relative without moving the camera."""
        world = self.world_azimuth()
        self.relative = bool(on)
        self.azimuth = wrap_deg(world - self.facing) if self.relative else world

    def apply_preset(self, name: str) -> None:
        p = self.rules["camera"]["presets"][name]
        if "azimuth_rel" in p:
            rel = float(p["azimuth_rel"])
            self.azimuth = rel if self.relative else wrap_deg(self.facing + rel)
        for key in ("elevation", "distance", "roll"):
            if key in p:
                setattr(self, key, float(p[key]))

    def request(self) -> dict:
        req = {"azimuth": self.world_azimuth(), "elevation": self.elevation, "roll": self.roll,
               "distance": self.distance, "focal": self.focal, "floor_guard": self.floor_guard}
        if self.full_shot:
            req["shot"] = "full"       # target the bounding-box centre; the distance comes from the pose
        return req

    def resolved(self):
        if self.camera_override is not None:
            return self.camera_override
        return resolve_camera(self.request(), self.subject, rules=self.rules)


class App:
    """Matplotlib front end. All widgets can also be driven from code (press, toggle, set_slider)."""

    _PAD_ACTIONS = {"draw sketch": "toggle_pad", "use drawing": "use_drawing", "undo stroke": "undo_stroke",
                    "clear drawing": "clear_drawing"}
    TOGGLES = ("floor guard", "overlays", "character-relative", "full shot", "sketch: drag joints")

    def __init__(self, state: ViewerState | None = None, show: bool = True):
        self.state = state or ViewerState()
        cam = self.state.rules["camera"]
        self._mute = False
        self.live = True          # redraw on every change; paused() turns it off for scripted sweeps
        self.fig = plt.figure(figsize=(15.5, 8.4))
        try:
            self.fig.canvas.manager.set_window_title("Skell sandbox viewer")
        except AttributeError:
            pass
        self.ax = self.fig.add_axes([0.01, 0.31, 0.52, 0.66])

        # pose list
        pax = self.fig.add_axes([0.545, 0.31, 0.15, 0.66])
        pax.set_title("pose", fontsize=9, loc="left")
        self._pose_names = list(self.state.poses)
        labels = [self.state.poses[n]["label"] for n in self._pose_names]
        self.pose_radio = RadioButtons(pax, labels, active=self._pose_names.index(self.state.pose_name))
        for t in self.pose_radio.labels:
            t.set_fontsize(8)
        self.pose_radio.on_clicked(self._on_pose)

        specs = [
            ("azimuth", "azimuth (rel)", -180, 180),
            ("elevation", "elevation", -90, 90),
            ("roll", "roll", -180, 180),
            ("distance", "distance", cam["min_distance"], 1500),
            ("focal", "focal", cam["focal_range"][0], cam["focal_range"][1]),
            ("facing", "char facing", -180, 180),
        ]
        self.sliders: dict[str, Slider] = {}
        for i, (key, label, lo, hi) in enumerate(specs):
            sax = self.fig.add_axes([0.08, 0.245 - i * 0.038, 0.42, 0.026])
            s = Slider(sax, label, lo, hi, valinit=self._value(key))
            s.on_changed(lambda v, key=key: self._on_slider(key, v))
            self.sliders[key] = s

        self.buttons: dict[str, Button] = {}
        names = list(cam["presets"]) + ["reset", "turnaround", "place joints", "auto joints",
                                        "draw sketch", "use drawing", "undo stroke", "clear drawing"]
        for i, name in enumerate(names):
            col, row = i % 2, i // 2
            bax = self.fig.add_axes([0.71 + col * 0.145, 0.945 - row * 0.0325, 0.135, 0.028])
            b = Button(bax, name.replace("_", " "))
            b.label.set_fontsize(8)
            b.on_clicked(lambda _e, name=name: self.press(name))
            self.buttons[name] = b

        cax = self.fig.add_axes([0.71, 0.375, 0.28, 0.155])
        self.checks = CheckButtons(cax, list(self.TOGGLES), self._toggle_values())
        self.checks.on_clicked(self._on_check)

        self.status_ax = self.fig.add_axes([0.71, 0.01, 0.28, 0.355])
        self.status_ax.axis("off")
        self._status = self.status_ax.text(0, 1, "", va="top", ha="left", fontsize=8.5, family="monospace",
                                           wrap=True)
        self.drawing = None
        self._last_export = ""
        self.placement_order = placement_order(self.state.skel)
        self.placing = False          # guided placement in progress
        self._placement_fit = "exact"
        self._clicks: dict = {}       # joint -> (dx, dy) clicked, or None when skipped
        self._step = 0
        self.side_inset = None
        self._drag = None             # joint being dragged
        self._selected_bone = None    # bone ending at the last dragged joint (f flips it)
        self.pad_mode = False         # sketch pad: the view is a blank page to draw on
        self.pad_strokes: list = []   # finished strokes, each a list of (dx, dy) screen offsets
        self._stroke = None           # the stroke being drawn
        self._stroke_line = None
        self.last_pad_files = None    # (png, json) of the last drawing used
        for ev, fn in (("button_press_event", self._on_press), ("motion_notify_event", self._on_motion),
                       ("button_release_event", self._on_release), ("key_press_event", self._on_key)):
            self.fig.canvas.mpl_connect(ev, fn)
        self.redraw()
        if show:
            plt.show()

    # ---- programmatic control (also used by the widgets)
    def press(self, name: str) -> None:
        if name == "turnaround":
            self.export_turnaround()
            return
        if name == "place joints":
            self.start_placement()
            return
        if name == "auto joints":
            self.auto_joints()
            return
        if name in self._PAD_ACTIONS:
            getattr(self, self._PAD_ACTIONS[name])()
            return
        self.state.camera_override = None
        if name == "reset":
            self.state.reset()
        else:
            self.state.apply_preset(name)
        self._sync_widgets()
        self.redraw()

    def export_turnaround(self, path=None):
        """Save the eight-view sheet of the current pose at the current elevation, distance and focal length
        (full shot on: one distance that fits all views; off: head target at the slider distance)."""
        st = self.state
        path = Path(path) if path else data.ROOT / "out" / f"turnaround_{st.pose_name}.png"
        render_turnaround(path, st.subject, title=st.poses[st.pose_name]["label"], elevation=st.elevation,
                          distance=st.distance, focal=st.focal, roll=st.roll,
                          framing="full" if st.full_shot else "current", overlays=st.overlays,
                          floor_guard=st.floor_guard, rules=st.rules)
        self._last_export = f"turnaround saved: {path}"
        self.redraw()
        return path

    def toggle(self, label: str) -> None:
        self.checks.set_active(self.TOGGLES.index(label))   # fires _on_check

    def select_pose(self, name: str) -> None:
        self.pose_radio.set_active(self._pose_names.index(name))   # fires _on_pose

    def set_slider(self, key: str, value: float) -> None:
        s = self.sliders[key]
        s.set_val(min(max(value, s.valmin), s.valmax))       # past the pole stops at the pole

    @contextlib.contextmanager
    def paused(self):
        """Change widgets without redrawing each step; redraw once at the end."""
        self.live = False
        with self._quiet_widgets():
            try:
                yield self
            finally:
                self.live = True
        self.redraw()

    @contextlib.contextmanager
    def _quiet_widgets(self):
        """Widgets still fire their callbacks but do not each redraw the canvas."""
        widgets = list(self.sliders.values()) + [self.checks]
        before = [w.drawon for w in widgets]
        for w in widgets:
            w.drawon = False
        try:
            yield
        finally:
            for w, d in zip(widgets, before):
                w.drawon = d

    def save(self, path) -> None:
        self.fig.savefig(path, dpi=100)

    def close(self) -> None:
        plt.close(self.fig)

    def status_text(self) -> str:
        return self._status.get_text()

    # ---- callbacks
    def _value(self, key):
        return self.state.facing if key == "facing" else getattr(self.state, key)

    def _on_slider(self, key, value):
        if self._mute:
            return
        self.state.camera_override = None
        if key == "facing":
            self.state.set_facing(value)
        else:
            setattr(self.state, key, float(value))
        if self.live:
            self.redraw()

    def _toggle_values(self):
        return [self.state.floor_guard, self.state.overlays, self.state.relative, self.state.full_shot,
                self.state.sketch_mode]

    # ---- guided placement (click each joint on the sketch, in order)
    @property
    def next_joint(self):
        return self.placement_order[self._step] if self.placing else None

    def start_placement(self, fit: str = "exact") -> None:
        """Click each joint in turn. fit "gesture" (a free drawing): drawn directions, the character's lengths,
        and the camera fitted to the drawing's size once every joint is placed."""
        self.placing, self._clicks, self._step = True, {}, 0
        self._placement_fit = fit
        self.state.camera_override = self.state.resolved()           # the view holds still while clicking
        self.redraw()

    def auto_joints(self) -> None:
        """Find the joints in the underlay (no AI) and rebuild the pose over it; then drag to correct."""
        if self.state.underlay is None:
            self._last_export = "auto joints needs a drawing: start the viewer with --underlay sketch.png"
            self.redraw()
            return
        self._last_export = ""
        self.placing, self._clicks, self._step = False, {}, 0
        self._selected_bone = None
        try:
            self.state.auto_joints()
        except ValueError as e:                   # nothing drawn, or no body under the head
            self._last_export = f"auto joints: {e}"
            self.redraw()
            return
        self.state.sketch_mode = True             # ready to drag corrections
        self._sync_widgets()
        self.redraw()

    # ---- sketch pad (draw with the mouse or a finger)
    def toggle_pad(self) -> None:
        """Switch the view to a blank page to draw on, or back to the figure."""
        self.pad_mode = not self.pad_mode
        self.placing = False
        self._stroke = None
        self.redraw()

    def pad_stroke(self, points) -> None:
        """Add one finished stroke (screen offsets), as drawing it with the mouse does."""
        pts = [(float(x), float(y)) for x, y in points]
        if pts:
            self.pad_strokes.append(pts)
        if self.pad_mode:
            self.redraw()

    def undo_stroke(self) -> None:
        if self.pad_strokes:
            self.pad_strokes.pop()
        self.redraw()

    def clear_drawing(self) -> None:
        self.pad_strokes = []
        self.redraw()

    def use_drawing(self, folder=None) -> None:
        """Save the drawing (PNG + strokes JSON), put it behind the figure, and start clicking its joints
        (any drawing). For a stick figure, "auto joints" finds them instead."""
        from sandbox.sketch_pad import rasterize, save_pad
        if not self.pad_strokes:
            self._last_export = "use drawing: nothing drawn yet (press draw sketch, then draw)"
            self.redraw()
            return
        W, H = self.state.rules["camera"]["canvas_width"], self.state.rules["camera"]["canvas_height"]
        folder = Path(folder) if folder else data.ROOT / "out" / "sketches" / "pad"
        self.last_pad_files = save_pad(self.pad_strokes, folder, W, H)
        self.state.underlay = rasterize(self.pad_strokes, W, H)
        self.pad_mode = False
        self._last_export = (f"drawing saved: {self.last_pad_files[0]}\n"
                             "stick figure? press auto joints instead of clicking")
        self.start_placement(fit="gesture")

    def _pad_begin(self, x, y) -> None:
        self._stroke = [(float(x), float(y))]
        (self._stroke_line,) = self.ax.plot([x], [y], "-", color="#1b1f24", lw=2.2, solid_capstyle="round",
                                            zorder=10)
        self.fig.canvas.draw_idle()

    def _pad_move(self, x, y) -> None:
        if self._stroke is None:
            return
        self._stroke.append((float(x), float(y)))
        xs, ys = zip(*self._stroke)
        self._stroke_line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def _pad_end(self) -> None:
        if self._stroke is None:
            return
        stroke, self._stroke, self._stroke_line = self._stroke, None, None
        self.pad_stroke(stroke)

    def _redraw_pad(self) -> None:
        W, H = self.state.rules["camera"]["canvas_width"], self.state.rules["camera"]["canvas_height"]
        ax = self.ax
        ax.set_facecolor("white")
        ax.set_xlim(-W / 2, W / 2)
        ax.set_ylim(H / 2, -H / 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("sketch pad: draw your character freely", fontsize=9, loc="left")
        for s in self.pad_strokes:
            xs, ys = zip(*s)
            ax.plot(xs, ys, "-", color="#1b1f24", lw=2.2, solid_capstyle="round", solid_joinstyle="round",
                    zorder=10)
        lines = ["sketch pad",
                 "  draw with the mouse or your finger, in the view:",
                 "  any drawing of one character; fill most of the",
                 "  page's height, feet near the bottom",
                 "",
                 f"  strokes: {len(self.pad_strokes)}",
                 "  undo stroke (or backspace) · clear drawing",
                 "  use drawing: save it, put it behind the figure,",
                 "  then click its joints as the panel asks",
                 "  (a stick figure? press auto joints instead)",
                 "  draw sketch again: back to the figure"]
        if self._last_export:
            lines += ["", self._last_export]
        self._status.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def place_click(self, dx: float, dy: float) -> None:
        self._record((float(dx), float(dy)))

    def skip_joint(self) -> None:
        self._record(None)

    def undo_click(self) -> None:
        if self.placing and self._step > 0:
            self._step -= 1
            self._clicks.pop(self.placement_order[self._step], None)
            self.redraw()

    def cancel_placement(self) -> None:
        self.placing, self._clicks, self._step = False, {}, 0
        self.redraw()

    def _record(self, point) -> None:
        if not self.placing:
            return
        self._clicks[self.placement_order[self._step]] = point
        self._step += 1
        if self._step < len(self.placement_order):
            self.redraw()
            return
        placements = {j: xy for j, xy in self._clicks.items() if xy is not None}
        self.placing, self._step = False, 0
        self._selected_bone = None
        fit = self._placement_fit
        if fit == "gesture" and len(placements) == len(self.placement_order):
            # start from the library pose the clicks look like (its depth choices and height suit the drawing)
            from sandbox.sketch_detect import best_start
            start, facing = best_start(placements)
            if start in self.state.poses:
                self.state.select_pose(start)
                self.state.set_facing(facing)
                self._sync_widgets()
            cam = fit_drawing_camera(self.state.skel, placements, self.state.current_frame(), quick=True,
                                     rules=self.state.rules)
            self.state.camera_override = ResolvedCamera(
                cam, notes=["camera fitted to the drawing's scale (front, eye level)"])
        self.state.sketch_edit(placements, fit=fit)
        self.state.sketch_mode = True             # ready to drag corrections
        self._sync_widgets()
        self._clicks = {}
        self.redraw()                 # the camera stays where it was, so the result sits on the sketch

    # ---- sketch input
    def drag_joint(self, joint: str, dx: float, dy: float) -> None:
        """Drag a joint to screen offset (dx, dy), as the mouse does: press, move, release."""
        self._begin_drag(joint)
        self._drag_to(dx, dy)
        self._end_drag()

    def flip(self, bone: str | None = None) -> None:
        """Take the other depth solution for a bone (default: the bone ending at the last dragged joint)."""
        bone = bone or self._selected_bone
        if bone is None:
            return
        self._selected_bone = bone
        self.state.sketch_edit({}, flips={bone: True})
        self.redraw()

    def _pickable(self):
        skel = self.state.skel
        rigid = {b.to_joint for b in skel.bones if b.kind == "rigid"}
        return [j for j in skel.joints if j not in rigid]

    def _begin_drag(self, joint: str) -> None:
        self._drag = joint
        self.state.camera_override = self.state.resolved()           # freeze the view while dragging
        ends = [b.id for b in self.state.skel.bones if b.to_joint == joint and b.kind == "posable"]
        self._selected_bone = ends[0] if ends else None

    def _drag_to(self, dx: float, dy: float) -> None:
        if self._drag is None:
            return
        self.state.sketch_edit({self._drag: (dx, dy)})
        self.redraw()

    def _end_drag(self) -> None:
        self._drag = None             # the camera stays where it was, so the result sits on the sketch
        self.redraw()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.pad_mode:
            self._pad_begin(event.xdata, event.ydata)
            return
        if self.placing:
            self.place_click(event.xdata, event.ydata)
            return
        if not self.state.sketch_mode:
            return
        cam = self.state.resolved().camera
        best, best_d = None, 14.0
        for j in self._pickable():
            p = cam.project(self.state.subject.positions[j])
            if p.visible:
                d = ((p.dx - event.xdata) ** 2 + (p.dy - event.ydata) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = j, d
        if best is not None:
            self._begin_drag(best)

    def _on_motion(self, event):
        if self._stroke is not None:
            if event.inaxes is self.ax and event.xdata is not None:
                self._pad_move(event.xdata, event.ydata)
            return
        if self._drag is not None and event.inaxes is self.ax and event.xdata is not None:
            self._drag_to(event.xdata, event.ydata)

    def _on_release(self, _event):
        if self._stroke is not None:
            self._pad_end()
            return
        if self._drag is not None:
            self._end_drag()

    def _on_key(self, event):
        if self.pad_mode:
            if event.key in ("backspace", "ctrl+z"):
                self.undo_stroke()
            return
        if self.placing:
            if event.key == "backspace":
                self.undo_click()
            elif event.key == "n":
                self.skip_joint()
            elif event.key == "escape":
                self.cancel_placement()
            return
        if event.key == "f" and self.state.sketch_mode:
            self.flip()

    def _draw_side_inset(self, cam) -> None:
        """A small view from 90 degrees to the side of the current camera, for checking depth (spec 9.2)."""
        side = Camera(target=cam.target, azimuth=cam.azimuth + 90, elevation=0.0, distance=cam.distance,
                      focal=cam.focal, near=cam.near)
        self.side_inset = self.ax.inset_axes([0.70, 0.60, 0.29, 0.39])
        draw_scene(self.side_inset, side, [self.state.subject],
                   options=RenderOptions(horizon=False, title=None))
        self.side_inset.text(0.03, 0.95, "side view (depth)", transform=self.side_inset.transAxes, fontsize=7,
                             va="top", color="#2b2f36")
        for sp in self.side_inset.spines.values():
            sp.set_color("#9aa5b1")

    def _on_pose(self, label):
        if self._mute:
            return
        name = self._pose_names[[self.state.poses[n]["label"] for n in self._pose_names].index(label)]
        self.state.select_pose(name)
        self.state.camera_override = None
        self._sync_widgets()
        self.redraw()

    def _on_check(self, label):
        if self._mute:
            return
        on = self.checks.get_status()[self.TOGGLES.index(label)]
        if label in ("floor guard", "full shot", "character-relative"):
            self.state.camera_override = None
        if label == "floor guard":
            self.state.floor_guard = on
        elif label == "overlays":
            self.state.overlays = on
        elif label == "full shot":
            self.state.full_shot = on
        elif label == "sketch: drag joints":
            self.state.sketch_mode = on
        else:
            self.state.set_relative(on)
            self._sync_widgets()
        self.redraw()

    def _sync_widgets(self):
        self._mute = True
        try:
            with self._quiet_widgets():
                self._set_widget_values()
        finally:
            self._mute = False

    def _set_widget_values(self):
        for key, s in self.sliders.items():
            s.set_val(self._value(key))
        for i, (have, want) in enumerate(zip(self.checks.get_status(), self._toggle_values())):
            if have != want:
                self.checks.set_active(i)
        i = self._pose_names.index(self.state.pose_name)
        if self.pose_radio.value_selected != self.pose_radio.labels[i].get_text():
            self.pose_radio.set_active(i)
        self.sliders["azimuth"].label.set_text("azimuth (rel)" if self.state.relative else "azimuth (world)")

    def redraw(self):
        res = self.state.resolved()
        if self.side_inset is not None:
            self.side_inset.remove()
            self.side_inset = None
        self.ax.clear()
        if self.pad_mode:
            self._redraw_pad()
            return
        W, H = self.state.rules["camera"]["canvas_width"], self.state.rules["camera"]["canvas_height"]
        on_drawing = self.placing and self._placement_fit == "gesture" and self.state.underlay is not None
        if on_drawing:
            # clicking joints on a free drawing: show the drawing alone, the figure would only get in the way
            self.ax.imshow(self.state.underlay, extent=(-W / 2, W / 2, H / 2, -H / 2), alpha=0.9, zorder=0.5,
                           aspect="auto")
            self.ax.set_xlim(-W / 2, W / 2)
            self.ax.set_ylim(H / 2, -H / 2)
            self.ax.set_aspect("equal")
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            if self.state.underlay is not None:
                self.ax.imshow(self.state.underlay, extent=(-W / 2, W / 2, H / 2, -H / 2), alpha=0.35,
                               zorder=0.5, aspect="auto")
            self.drawing = draw_scene(self.ax, res.camera, [self.state.subject],
                                      options=RenderOptions(overlays=self.state.overlays))
        if (self.state.sketch_mode or self.placing) and not on_drawing:
            self._draw_side_inset(res.camera)
        if self.placing:
            for j, xy in self._clicks.items():
                if xy is not None:
                    self.ax.plot(*xy, marker="x", color="#d64545", ms=8, mew=2, zorder=400)
                    self.ax.text(xy[0] + 6, xy[1] - 6, j, fontsize=7, color="#d64545", zorder=400)
        c = res.camera
        lines = [
            "resolved camera (world)",
            f"  azimuth   {c.azimuth:8.2f}   (rel {wrap_deg(c.azimuth - self.state.facing):7.2f})",
            f"  elevation {c.elevation:8.2f}",
            f"  roll      {c.roll:8.2f}",
            f"  distance  {c.distance:8.2f}",
            f"  focal     {c.focal:8.2f}",
            f"  height    {c.position[1]:8.2f}  above floor",
            f"character facing {self.state.facing:7.2f}",
        ]
        v = self.state.validation
        lines += ["", f"pose: {self.state.poses[self.state.pose_name]['label']}",
                  f"  contacts  {', '.join(v.contacts) or 'none'}",
                  f"  balance   {v.balance.status}" + (f" ({v.balance.distance_to_support:.1f} outside)"
                                                       if v.balance.status == 'unstable' else ""),
                  f"  limits    {'all within range' if v.limits.ok else str(len(v.limits.corrections)) + ' clamped'}"]
        lines += [f"  {w}" for w in v.warnings() if not w.startswith("unstable")]
        if self.placing:
            n = len(self.placement_order)
            lines += ["", f"placing joints ({self._step + 1}/{n}): click the {self.next_joint}",
                      "  l_ = the character's own left (image right if it faces you)",
                      "  backspace undo · n skip (keep where it is) · esc cancel"]
        if self.state.sketch_mode or self.state.edited:
            lines += ["", "sketch: drag joints in the view; f flips the selected bone's depth"]
            rec = self.state.last_rec
            if self.state.edited:
                lines += ["  edited by sketch (re-select the pose to undo)"]
            if self.state.camera_override is not None and not self.placing and self._drag is None:
                lines += ["  camera held still over the sketch (any camera control releases it)"]
            if rec is not None and self._selected_bone in rec.bones:
                info = rec.bones[self._selected_bone]
                kind = {2: "two depths", 1: "one depth (tangent)", 0: "no exact depth"}[info.solutions]
                lines += [f"  {self._selected_bone}: {kind}" + (", flipped" if info.flipped else "")]
            if rec is not None:
                lines += [f"  {f}" for f in rec.flags]
            det = self.state.last_detection
            if det is not None and self.state.edited:
                how = "by the trained model" if det.method == "model" else "by the geometric fit"
                lines += [f"  joints found {how} (looks like {det.start}, facing {det.facing:+.0f});",
                          "  drag any that are off"] + [f"  {n}" for n in det.notes]
        if self._last_export:
            lines += ["", self._last_export]
        for n in res.notes:
            lines += ["", "note: " + n]
        for w in res.warnings:
            lines += ["", "warning: " + w]
        self._status.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


def _load_frame(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if "options" in obj:
        return obj["options"][0]["frames"][0]
    if "frames" in obj:
        return obj["frames"][0]
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser(description="Skell sandbox viewer")
    ap.add_argument("--pose", help="JSON frame (or pose request) to add to the pose list and show first")
    ap.add_argument("--underlay", help="image of a rough drawing to show behind the figure (sketch input)")
    args = ap.parse_args(argv)
    frame = _load_frame(args.pose) if args.pose else None
    underlay = args.underlay
    if underlay and not Path(underlay).is_file():
        print(f"underlay image not found: {Path(underlay).resolve()}\n"
              f"starting without an underlay (give the path to a PNG or JPG of your sketch)")
        underlay = None
    App(ViewerState(frame, underlay=underlay))


if __name__ == "__main__":
    main()
