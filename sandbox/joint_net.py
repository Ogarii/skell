"""Run the trained joint model (ml/model.py JointNet) in plain numpy: no PyTorch needed in the sandbox.

    net = JointNet.load("out/models/joint_net.npz")
    joints, conf = net.detect(img)          # any drawing: framed for the net, answers in the image's pixels

The model proposes 2D joints; the geometry fit (sketch_detect / reconstruct) still decides the pose.
Left/right is not decided here: the net was trained to treat each arm pair and each leg pair as
interchangeable (synth.SIDE_PAIRS), and the fit picks the sides from the 3D pose.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from sandbox.synth import MIRROR, SIDE_PAIRS, frame_drawing

BLOCK_SIZES = {"stem": 2, "d1": 2, "d2": 2, "d3": 2, "d4": 3, "u3": 2, "u2": 2, "u1": 2}


def _conv3(x, w, b, stride):
    """x (C, H, W), w (O, C, 3, 3), padding 1."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(p, (3, 3), axis=(1, 2))[:, ::stride, ::stride]   # (C, Ho, Wo, 3, 3)
    return np.tensordot(w, win, axes=([1, 2, 3], [0, 3, 4])) + b[:, None, None]


def _up2(x):
    return x.repeat(2, axis=1).repeat(2, axis=2)


class JointNet:
    def __init__(self, weights: dict):
        self.w = {k: np.asarray(v) for k, v in weights.items()}
        self.stride = int(self.w.get("stride", 4))

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(dict(z))

    def _block(self, name, x):
        for i in range(BLOCK_SIZES[name]):
            w = self.w[f"{name}.{i}.w"]
            stride = 2 if i == 0 and name in ("stem", "d1", "d2", "d3", "d4") else 1
            x = np.maximum(_conv3(x, w, self.w[f"{name}.{i}.b"], stride), 0.0)
        return x

    def logits(self, img: np.ndarray) -> np.ndarray:
        """A framed uint8 drawing (S, S) -> heatmap logits (11, S/4, S/4)."""
        x = (1.0 - np.asarray(img, dtype=np.float32) / 255.0)[None]
        x = self._block("stem", x)
        s1 = self._block("d1", x)
        s2 = self._block("d2", s1)
        s3 = self._block("d3", s2)
        x = self._block("d4", s3)
        x = self._block("u3", np.concatenate([_up2(x), s3]))
        x = self._block("u2", np.concatenate([_up2(x), s2]))
        x = self._block("u1", np.concatenate([_up2(x), s1]))
        return np.tensordot(self.w["head.w"][:, :, 0, 0], x, axes=([1], [0])) + self.w["head.b"][:, None, None]

    def predict(self, img: np.ndarray, mirror_too: bool = False):
        """A framed drawing -> joints (11, 2) in its pixels, confidence (11,) in [0, 1].
        mirror_too: also run the mirrored drawing and average (test-time augmentation). Off by default: on the
        first model (2026-11-19) it made the real drawings worse (4.1% -> 4.7% of figure size), because on some
        drawings the mirrored pass is confidently wrong and the average lands between two candidates."""
        J, c = decode(self.logits(img), self.stride)
        if not mirror_too:
            return J, c
        Jm, cm = decode(self.logits(np.ascontiguousarray(img[:, ::-1])), self.stride)
        Jm[:, 0] = img.shape[1] - Jm[:, 0]
        Jm, cm = Jm[MIRROR], cm[MIRROR]
        Jm, cm = align_sides(J, Jm, cm)
        return (J + Jm) / 2, (c + cm) / 2

    def detect(self, img: np.ndarray, size: int = 192, mirror_too: bool = False):
        """Any drawing (uint8 or float, grey or RGB) -> joints (11, 2) in its own pixels, confidence (11,)."""
        small, (scale, ox, oy) = frame_drawing(img, size)
        J, c = self.predict(small, mirror_too)
        return J / scale + np.array([ox, oy]), c


def decode(logits: np.ndarray, stride: int = 4, radius: int = 2):
    """Heatmap logits (J, h, w) -> joints (J, 2) px and confidence (J,): softmax per joint, then the
    probability-weighted mean of the cells within `radius` of the peak (sub-pixel, and a second blob
    elsewhere can't pull it off). Confidence = the probability mass in that window."""
    J, h, w = logits.shape
    z = logits.reshape(J, -1)
    p = np.exp(z - z.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).reshape(J, h, w)
    out = np.zeros((J, 2), dtype=np.float64)
    conf = np.zeros(J)
    for j in range(J):
        r, c = np.unravel_index(np.argmax(p[j]), (h, w))
        r0, r1, c0, c1 = max(0, r - radius), min(h, r + radius + 1), max(0, c - radius), min(w, c + radius + 1)
        win = p[j, r0:r1, c0:c1]
        m = win.sum()
        rr, cc = np.mgrid[r0:r1, c0:c1]
        out[j] = [((cc + 0.5) * win).sum() / m * stride, ((rr + 0.5) * win).sum() / m * stride]
        conf[j] = m
    return out, conf


def align_sides(ref: np.ndarray, J: np.ndarray, conf: np.ndarray | None = None):
    """Swap each arm/leg pair of J (and conf) where the swap lies closer to ref."""
    J = J.copy()
    conf = None if conf is None else conf.copy()
    for a, b in SIDE_PAIRS:
        keep = np.linalg.norm(J[a] - ref[a], axis=1).sum() + np.linalg.norm(J[b] - ref[b], axis=1).sum()
        swap = np.linalg.norm(J[b] - ref[a], axis=1).sum() + np.linalg.norm(J[a] - ref[b], axis=1).sum()
        if swap < keep:
            J[a + b] = J[b + a]
            if conf is not None:
                conf[a + b] = conf[b + a]
    return (J, conf) if conf is not None else J
