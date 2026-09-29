from __future__ import annotations
import numpy as np
from .attitude import quat_to_dcm
from .constants import MU_EARTH


def gravity_gradient_torque_body(q_body_to_inertial, position_eci_m, inertia_diag):
    """Classical gravity-gradient torque in body coordinates."""
    r_i = np.asarray(position_eci_m, float)
    r = float(np.linalg.norm(r_i))
    if r <= 0:
        return np.zeros(3)
    C = quat_to_dcm(q_body_to_inertial)
    rhat_b = C.T @ (r_i / r)
    I = np.diag(np.asarray(inertia_diag, float))
    return 3.0 * MU_EARTH / (r**3) * np.cross(rhat_b, I @ rhat_b)
