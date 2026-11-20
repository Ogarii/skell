"""Click the joints of a drawing to make a test label (ground truth for the joint model).

    python tools/label_drawing.py out/sketches/pad/pad_20261119-012457.png [more.png ...]

For each image a window opens; click the joints in the order the title asks. Backspace undoes a click,
n skips a joint you can't place, q skips the image. The labels are saved next to the image as
<name>_joints.json, in image pixels: {"units": "pixels", "image": [w, h], "joints": {"head": [x, y], ...}}.
l_ is the character's own left: image right when it faces you.
"""
import json
import pathlib
import sys

import matplotlib.image as mpimg

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORDER = ["head", "neck", "pelvis", "l_elbow", "l_hand", "r_elbow", "r_hand", "l_knee", "l_foot", "r_knee",
         "r_foot"]


class Labeler:
    """Joint clicks for one image (no window needed: click() / skip() / undo() drive it)."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.img = mpimg.imread(self.path)
        self.clicks: dict = {}
        self.step = 0

    @property
    def next_joint(self):
        return ORDER[self.step] if self.step < len(ORDER) else None

    def click(self, x, y):
        if self.next_joint:
            self.clicks[self.next_joint] = [round(float(x), 2), round(float(y), 2)]
            self.step += 1

    def skip(self):
        if self.next_joint:
            self.clicks[self.next_joint] = None
            self.step += 1

    def undo(self):
        if self.step > 0:
            self.step -= 1
            self.clicks.pop(ORDER[self.step], None)

    def save(self):
        out = self.path.with_name(self.path.stem + "_joints.json")
        h, w = self.img.shape[:2]
        out.write_text(json.dumps({"units": "pixels", "image": [w, h], "joints": self.clicks}, indent=2) + "\n",
                       encoding="utf-8")
        return out


def label(path) -> pathlib.Path | None:
    import matplotlib.pyplot as plt
    lab = Labeler(path)
    fig, ax = plt.subplots(figsize=(9, 7))
    h, w = lab.img.shape[:2]
    done = {"quit": False}

    def show():
        ax.clear()
        ax.imshow(lab.img, extent=(0, w, h, 0))
        for j, xy in lab.clicks.items():
            if xy is not None:
                c = "#1f77b4" if j.startswith("l_") else "#d62728" if j.startswith("r_") else "#2ca02c"
                ax.plot(*xy, "x", color=c, ms=9, mew=2)
                ax.text(xy[0] + 5, xy[1] - 5, j, color=c, fontsize=8)
        nxt = lab.next_joint
        ax.set_title(f"{lab.path.name}: click the {nxt}  (l_ = character's left)   backspace undo · n skip · q skip image"
                     if nxt else "done: close the window (or press any key)", fontsize=9)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is ax and ev.xdata is not None and lab.next_joint:
            lab.click(ev.xdata, ev.ydata)
            show()
            if not lab.next_joint:
                plt.close(fig)

    def on_key(ev):
        if ev.key == "backspace":
            lab.undo()
        elif ev.key == "n":
            lab.skip()
        elif ev.key == "q":
            done["quit"] = True
            plt.close(fig)
            return
        show()
        if not lab.next_joint:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    if done["quit"] or lab.next_joint:
        print(f"{lab.path.name}: not finished, nothing saved")
        return None
    out = lab.save()
    print(f"{lab.path.name}: saved {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    for p in sys.argv[1:]:
        label(p)
