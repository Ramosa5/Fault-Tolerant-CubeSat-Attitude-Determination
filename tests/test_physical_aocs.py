import numpy as np
from datetime import datetime, timezone

from src.simulation.spacecraft import box_inertia_diag
from src.simulation.magnetorquer import dipole_for_desired_torque, magnetic_torque
from src.simulation.magnetic_field import magnetic_field_eci


def test_uniform_box_inertia_3u_positive_and_expected():
    I = box_inertia_diag(4.0, [0.34, 0.10, 0.10])
    assert np.all(I > 0)
    assert np.allclose(I, [0.0066666667, 0.0418666667, 0.0418666667], rtol=1e-6)


def test_magnetorquer_torque_matches_cross_product_and_is_perpendicular_to_B():
    B = np.array([20e-6, -10e-6, 35e-6])
    tau_des = np.array([5e-6, 3e-6, -2e-6])
    m, tau = dipole_for_desired_torque(tau_des, B, 0.4)
    assert np.allclose(tau, magnetic_torque(m, B))
    assert abs(np.dot(tau, B)) < 1e-15
    assert np.all(np.abs(m) <= 0.4 + 1e-12)


def test_dipole_field_has_leo_scale_and_changes_along_orbit():
    r = 6878e3
    p = np.array([[r,0,0],[0,r,0],[0,0,r]], float)
    t = np.array([0.0, 1000.0, 2000.0])
    B = magnetic_field_eci(p, t, epoch_utc=datetime(2026,1,1,tzinfo=timezone.utc), model='dipole')
    mag = np.linalg.norm(B, axis=1)
    assert B.shape == (3,3)
    assert np.all((mag > 10e-6) & (mag < 100e-6))
    assert not np.allclose(B[0], B[1])
