import numpy as np
from src.estimation.mekf import MEKF


def test_mekf_remains_finite_under_severe_sensor_corruption():
    f = MEKF()
    sun_i = np.array([1.0, 0.0, 0.0])
    mag_i = np.array([0.0, 1.0, 0.0])
    gyro = np.array([0.25, -0.20, 0.15])
    # Deliberately inconsistent, stuck body-frame vectors.
    sun = np.array([0.0, 1.0, 0.0])
    mag = np.array([1.0, 0.0, 0.0])
    for _ in range(5000):
        q, b, _ = f.step(gyro, sun, mag, sun_i, mag_i, 0.1)
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(f.P))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
