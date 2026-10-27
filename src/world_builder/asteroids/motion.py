"""
How a rock turns: a closed form of time, the same in numpy, the native code
and the shader.

Every rock moves as a free symmetric top::

    q(t) = rot(axis, precession * t) * q0 * rot(z, spin * t)

``axis`` is fixed in the field frame (the angular momentum's direction): the
body turns about it at ``precession`` while turning about its own z axis at
``spin``. A rock that only spins has spin = 0: a constant rotation about a
fixed axis. A tumbling one has its body z axis at a constant angle from
``axis`` (set by q0), sweeping a cone. For moments of inertia I1 = I2 and I3
about body z, Euler's equations give spin = precession * cos(cone) * (I1/I3 -
1): :class:`~world_builder.asteroids.spec.Tumble`'s ratio is that last factor.

Quaternions are (w, x, y, z), Panda's order; a quaternion q turns a vector v
into q v q*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """:returns: the Hamilton product a b, over the last axis"""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def axis_angle(axis: ArrayLike, angle: ArrayLike) -> np.ndarray:
    """:returns: the quaternion turning by *angle* about the unit *axis*"""
    axis = np.asarray(axis, dtype=np.float64)
    half = 0.5 * np.asarray(angle, dtype=np.float64)
    return np.concatenate((np.cos(half)[..., None], axis * np.sin(half)[..., None]), -1)


def rotate(q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """:returns: v turned by the unit quaternion q (both over the last axis)"""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def orientation(
    q0: ArrayLike,
    axis: ArrayLike,
    precession: ArrayLike,
    spin: ArrayLike,
    t: float | ArrayLike,
) -> np.ndarray:
    """A rock's orientation at time *t*.

    :param q0: (..., 4) at t = 0
    :param axis: (..., 3) unit, field frame
    :param precession: (...) rad/s about *axis*
    :param spin: (...) rad/s about body z
    :param t: seconds (the game's clock)
    :returns: (..., 4) unit quaternions, body to field frame
    """
    precession = np.asarray(precession, dtype=np.float64)
    spin = np.asarray(spin, dtype=np.float64)
    body_z = np.zeros(spin.shape + (3,))
    body_z[..., 2] = 1.0
    turn = axis_angle(axis, precession * t)
    own = axis_angle(body_z, spin * t)
    return quat_multiply(quat_multiply(turn, q0), own)


def angular_velocity(
    q0: ArrayLike,
    axis: ArrayLike,
    precession: ArrayLike,
    spin: ArrayLike,
    t: float | ArrayLike,
) -> np.ndarray:
    """A rock's angular velocity at time *t*: precession about *axis* plus
    spin about its body z axis where that axis points then.

    :returns: (..., 3) rad/s, field frame
    """
    q = orientation(q0, axis, precession, spin, t)
    body_z = rotate(q, np.array([0.0, 0.0, 1.0]))
    precession = np.asarray(precession, dtype=np.float64)[..., None]
    spin = np.asarray(spin, dtype=np.float64)[..., None]
    return precession * np.asarray(axis, dtype=np.float64) + spin * body_z
