"""Unit quaternion maths. Quaternions are numpy arrays [w, x, y, z]; angles are in degrees.

Right-handed rotations: a positive angle about an axis turns counter-clockwise when the axis points at you.
"""
from __future__ import annotations

import math

import numpy as np

_EPS_PARALLEL = 1e-8


def identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def normalize3(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def mul(a, b) -> np.ndarray:
    """Hamilton product a*b: apply b first, then a."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate(q, v) -> np.ndarray:
    """Rotate vector v by unit quaternion q."""
    v = np.asarray(v, dtype=float)
    u = np.asarray(q[1:], dtype=float)
    t = 2.0 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def from_axis_angle(axis, angle_deg: float) -> np.ndarray:
    axis = normalize3(axis)
    h = math.radians(angle_deg) / 2.0
    return np.concatenate(([math.cos(h)], math.sin(h) * axis))


def to_axis_angle(q) -> tuple[np.ndarray, float]:
    """Axis and angle in [0, 180] degrees. Identity returns axis (1, 0, 0), angle 0."""
    q = normalize(q)
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-15:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return q[1:] / s, math.degrees(2.0 * math.atan2(s, q[0]))


def _any_perpendicular(a: np.ndarray) -> np.ndarray:
    other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return normalize3(np.cross(a, other))


def shortest_arc(a, b, fallback_axis=None) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b.

    When b is exactly opposite a the axis is not unique; rotate 180 degrees about fallback_axis
    (made perpendicular to a), or about any perpendicular axis if none is given (spec 6.3).
    """
    a, b = normalize3(a), normalize3(b)
    d = float(np.dot(a, b))
    c = np.cross(a, b)
    if d < 0 and np.linalg.norm(c) < _EPS_PARALLEL:
        axis = None
        if fallback_axis is not None:
            f = np.asarray(fallback_axis, dtype=float)
            f = f - np.dot(f, a) * a
            if np.linalg.norm(f) > _EPS_PARALLEL:
                axis = normalize3(f)
        if axis is None:
            axis = _any_perpendicular(a)
        return np.concatenate(([0.0], axis))
    return normalize(np.concatenate(([1.0 + d], c)))


def slerp(q0, q1, t: float) -> np.ndarray:
    """Spherical linear interpolation along the shortest path (negates q1 when the dot is negative)."""
    q0, q1 = normalize(q0), normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    theta = math.acos(min(1.0, d))
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * q0 + (math.sin(t * theta) / s) * q1


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def signed_angle(u, v, axis) -> float:
    """Angle in degrees that turns u onto v about axis (right-handed), for u, v perpendicular to axis."""
    return math.degrees(math.atan2(float(np.dot(np.cross(u, v), axis)), float(np.dot(u, v))))


def twist_angle(q, rest_axis, fallback_axis=(1.0, 0.0, 0.0)) -> float:
    """Twist of q about the direction it gives rest_axis.

    Decomposes q = twist * swing, where swing = shortest_arc(rest_axis, q * rest_axis, fallback_axis)
    and twist is a rotation about d = q * rest_axis. Returns the twist angle in (-180, 180].
    """
    d = rotate(q, rest_axis)
    swing = shortest_arc(rest_axis, d, fallback_axis)
    tw = mul(q, conj(swing))
    return wrap_deg(math.degrees(2.0 * math.atan2(float(np.dot(tw[1:], d)), float(tw[0]))))
