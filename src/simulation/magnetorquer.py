from __future__ import annotations
import numpy as np


def magnetic_torque(dipole_body_am2, magnetic_field_body_t):
    """Torque produced by a magnetic dipole: tau = m x B."""
    return np.cross(np.asarray(dipole_body_am2, float), np.asarray(magnetic_field_body_t, float))


def dipole_for_desired_torque(desired_torque_body_nm, magnetic_field_body_t,
                              max_dipole_am2=0.4):
    """Least-squares magnetic dipole for the achievable part of a desired torque.

    Magnetorquers cannot generate torque parallel to B.  The returned command gives
    the orthogonal projection of desired torque onto the plane normal to B.
    ``max_dipole_am2`` may be a scalar or one limit per body axis.
    """
    tau = np.asarray(desired_torque_body_nm, float)
    B = np.asarray(magnetic_field_body_t, float)
    b2 = float(B @ B)
    if b2 < 1e-18:
        return np.zeros(3), np.zeros(3)
    m = np.cross(B, tau) / b2
    lim = np.asarray(max_dipole_am2, dtype=float)
    if lim.shape == ():
        lim = np.repeat(float(lim), 3)
    if lim.shape != (3,) or np.any(lim <= 0):
        raise ValueError('max_dipole_am2 must be positive scalar or length-3 vector')
    m = np.clip(m, -lim, lim)
    achieved = magnetic_torque(m, B)
    return m, achieved


def first_order_dipole_response(previous_am2, command_am2, dt_s, time_constant_s=0.008):
    """Simple first-order electrical response of the torquer dipole."""
    previous = np.asarray(previous_am2, float)
    command = np.asarray(command_am2, float)
    tau = float(time_constant_s)
    if tau <= 0:
        return command.copy()
    alpha = 1.0 - np.exp(-float(dt_s) / tau)
    return previous + alpha * (command - previous)
