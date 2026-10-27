"""How rocks turn (asteroids.motion): constant spin and free symmetric-top
tumbling, as closed forms of time."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from world_builder.asteroids.motion import (
    angular_velocity,
    axis_angle,
    orientation,
    quat_multiply,
    rotate,
)


def _conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def test_quaternion_helpers():
    q = axis_angle([0.0, 0.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)
    a = axis_angle([1.0, 0.0, 0.0], 0.3)
    b = axis_angle([0.0, 1.0, 0.0], -0.7)
    v = np.array([0.2, -0.5, 0.9])
    np.testing.assert_allclose(
        rotate(quat_multiply(a, b), v), rotate(a, rotate(b, v)), atol=1e-15
    )


def test_constant_spin_turns_about_the_fixed_axis():
    q0 = axis_angle(np.array([0.3, 0.4, 0.5]) / math.sqrt(0.5), 1.1)
    axis = np.array([0.0, 0.6, 0.8])
    rate = 0.2
    for t in (0.0, 1.0, 17.5):
        q = orientation(q0, axis, rate, 0.0, t)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        expected = quat_multiply(axis_angle(axis, rate * t), q0)
        np.testing.assert_allclose(q, expected, atol=1e-14)
        np.testing.assert_allclose(
            angular_velocity(q0, axis, rate, 0.0, t), rate * axis, atol=1e-15
        )
    np.testing.assert_allclose(orientation(q0, axis, rate, 0.0, 0.0), q0)


def test_angular_velocity_matches_the_orientation_change():
    q0 = axis_angle(np.array([1.0, 2.0, 2.0]) / 3.0, 0.4)
    axis = np.array([0.0, 0.0, 1.0])
    precession, spin, t, dt = 0.3, -0.45, 4.0, 1e-6
    q1 = orientation(q0, axis, precession, spin, t - dt)
    q2 = orientation(q0, axis, precession, spin, t + dt)
    # dq/dt = 1/2 (0, w) q, so w = 2 dq/dt q*
    dq = (q2 - q1) / (2 * dt)
    q = orientation(q0, axis, precession, spin, t)
    w = 2.0 * quat_multiply(dq, _conjugate(q))
    np.testing.assert_allclose(w[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(
        w[1:], angular_velocity(q0, axis, precession, spin, t), atol=1e-8
    )


@pytest.mark.parametrize("cone,ratio", [(0.3, 0.8), (1.0, 0.25), (0.6, -0.4)])
def test_tumbling_solves_eulers_equations(cone, ratio):
    """A body with I1 = I2 = 1 and I3 = 1 / (1 + ratio), set going as the
    closed form says, then integrated: it stays on the closed form."""
    i3 = 1.0 / (1.0 + ratio)
    inertia = np.array([1.0, 1.0, i3])
    axis = np.array([0.36, -0.48, 0.8])
    precession = 0.25
    spin = precession * math.cos(cone) * ratio
    # body z at the cone angle from the axis
    tilt = axis_angle([1.0, 0.0, 0.0], 0.0)
    z = np.array([0.0, 0.0, 1.0])
    perp = np.cross(axis, [1.0, 0.0, 0.0])
    perp /= np.linalg.norm(perp)
    zb = math.cos(cone) * axis + math.sin(cone) * perp
    align_axis = np.cross(z, zb)
    align = axis_angle(align_axis / np.linalg.norm(align_axis), math.acos(zb[2]))
    q0 = quat_multiply(align, tilt)
    w0 = angular_velocity(q0, axis, precession, spin, 0.0)
    w_body = rotate(_conjugate(q0), w0)
    # the closed form needs the momentum along the axis
    momentum = rotate(q0, inertia * w_body)
    np.testing.assert_allclose(momentum / np.linalg.norm(momentum), axis, atol=1e-12)

    def rates(_, y):
        q, w = y[:4] / np.linalg.norm(y[:4]), y[4:]
        dq = 0.5 * quat_multiply(q, np.concatenate(([0.0], w)))
        dw = np.cross(inertia * w, w) / inertia
        return np.concatenate((dq, dw))

    times = np.linspace(0.0, 60.0, 13)
    solved = solve_ivp(
        rates,
        (0.0, 60.0),
        np.concatenate((q0, w_body)),
        t_eval=times,
        rtol=1e-11,
        atol=1e-12,
    )
    for k, t in enumerate(times):
        q = solved.y[:4, k] / np.linalg.norm(solved.y[:4, k])
        closed = orientation(q0, axis, precession, spin, t)
        # the same rotation (q and -q are)
        assert abs(np.dot(q, closed)) == pytest.approx(1.0, abs=1e-8)


def test_vectorised_over_rocks():
    rng = np.random.default_rng(3)
    q0 = rng.normal(size=(5, 4))
    q0 /= np.linalg.norm(q0, axis=1)[:, None]
    axis = rng.normal(size=(5, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    precession, spin = rng.random(5), rng.random(5)
    q = orientation(q0, axis, precession, spin, 3.0)
    for k in range(5):
        np.testing.assert_allclose(
            q[k], orientation(q0[k], axis[k], precession[k], spin[k], 3.0)
        )
