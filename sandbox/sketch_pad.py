"""Strokes drawn with a mouse or a finger -> a sketch image (the viewer's sketch pad). Pure numpy.

Strokes are lists of (dx, dy) screen offsets on the sketch canvas (y down), the same units the joints use, so
the picture lines up with the view it was drawn in.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def close_loops(strokes: list, gap: float = 14.0) -> list:
    """Join the ends of a stroke that nearly meet (a hand-drawn circle seldom closes exactly), so a drawn
    head encloses a hole. Only strokes that went well away from their start count as loops."""
    out = []
    for s in strokes:
        s = [tuple(map(float, p)) for p in s]
        if len(s) >= 6:
            P = np.array(s)
            d_end = float(np.hypot(*(P[-1] - P[0])))
            reach = float(np.max(np.hypot(*(P - P[0]).T)))
            if d_end <= max(gap, 0.0) and reach > 2.5 * max(d_end, 1.0) and reach > gap:
                s = s + [s[0]]
        out.append(s)
    return out


def rasterize(strokes: list, width: int, height: int, line_width: float = 4.0) -> np.ndarray:
    """Black strokes on white, height x width x 3 floats; pixel (row, col) has its centre at screen offset
    (col + 0.5 - width / 2, row + 0.5 - height / 2)."""
    ink = np.zeros((height, width), dtype=bool)
    r = line_width / 2.0
    for s in close_loops(strokes):
        P = np.array(s, dtype=float) + [width / 2.0, height / 2.0]      # to pixel coordinates
        if len(P) == 1:
            P = np.vstack([P, P])
        for A, B in zip(P[:-1], P[1:]):
            x0, x1 = int(max(0, np.floor(min(A[0], B[0]) - r - 1))), int(min(width, np.ceil(max(A[0], B[0]) + r + 1)))
            y0, y1 = int(max(0, np.floor(min(A[1], B[1]) - r - 1))), int(min(height, np.ceil(max(A[1], B[1]) + r + 1)))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1] + 0.5
            Q = np.stack([xx, yy], axis=-1)
            AB = B - A
            L2 = float(AB @ AB)
            t = np.clip(((Q - A) @ AB) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros(Q.shape[:2])
            ink[y0:y1, x0:x1] |= np.linalg.norm(Q - (A + t[..., None] * AB), axis=-1) <= r
    return np.where(ink, 0.0, 1.0)[..., None].repeat(3, axis=2)


def save_pad(strokes: list, folder, width: int, height: int, stem: str | None = None) -> tuple[Path, Path]:
    """Write the drawing as a PNG (the underlay) and its strokes as JSON (to replay or check later)."""
    import matplotlib.image as mpimg
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = stem or time.strftime("pad_%Y%m%d-%H%M%S")
    png, js = folder / f"{stem}.png", folder / f"{stem}.json"
    mpimg.imsave(png, rasterize(strokes, width, height))
    js.write_text(json.dumps({"canvas": [width, height], "strokes": [[[round(x, 2), round(y, 2)] for x, y in s]
                                                                     for s in strokes]}, indent=1) + "\n",
                  encoding="utf-8")
    return png, js


def load_strokes(path) -> list:
    return json.loads(Path(path).read_text(encoding="utf-8"))["strokes"]
