import numpy as np
from src.simulation.adcs_modes import (
    SunPointingModeManager, DETUMBLING, SUN_ACQUISITION, SUN_POINTING,
    ECLIPSE_RATE_DAMPING, magnetic_rate_damping_enabled,
)
from src.simulation.magnetorquer import dipole_for_desired_torque


def test_detumbling_is_deployment_only_and_reacquisition_never_returns_to_it():
    m = SunPointingModeManager(
        threshold_deg=2.0, hold_s=0.2, dt_s=0.1,
        rate_threshold_deg_s=0.08, rate_hold_s=0.2,
        rate_recovery_trigger_deg_s=0.20, mode=DETUMBLING,
    )
    # High rate keeps the system in detumbling even with perfect pointing.
    for _ in range(3):
        assert m.update(in_full_eclipse=False, real_sun_available=True,
                        estimated_pointing_error_deg=0.2,
                        estimated_rate_deg_s=0.5) == DETUMBLING
    # Low rate held long enough enables acquisition.
    m.update(in_full_eclipse=False, real_sun_available=True,
             estimated_pointing_error_deg=20.0, estimated_rate_deg_s=0.02)
    assert m.update(in_full_eclipse=False, real_sun_available=True,
                    estimated_pointing_error_deg=20.0,
                    estimated_rate_deg_s=0.02) == SUN_ACQUISITION
    # Pointing is not declared until both angle and rate satisfy their holds.
    m.update(in_full_eclipse=False, real_sun_available=True,
             estimated_pointing_error_deg=0.5, estimated_rate_deg_s=0.02)
    assert m.update(in_full_eclipse=False, real_sun_available=True,
                    estimated_pointing_error_deg=0.5,
                    estimated_rate_deg_s=0.02) == SUN_POINTING
    # Once deployment detumbling has completed, excessive rate must never return
    # the spacecraft to DETUMBLING.  A degraded nominal state returns to Sun
    # acquisition instead.
    assert m.update(in_full_eclipse=False, real_sun_available=True,
                    estimated_pointing_error_deg=10.0,
                    estimated_rate_deg_s=0.25) == SUN_ACQUISITION

    # Eclipse exit always enters SUN_REACQUISITION.  Even a high body rate keeps
    # the spacecraft in reacquisition; rate damping is handled by the controller
    # inside this mode rather than by re-entering deployment detumbling.
    assert m.update(in_full_eclipse=True, real_sun_available=False,
                    estimated_pointing_error_deg=60.0,
                    estimated_rate_deg_s=0.3) == ECLIPSE_RATE_DAMPING
    from src.simulation.adcs_modes import SUN_REACQUISITION
    assert m.update(in_full_eclipse=False, real_sun_available=True,
                    estimated_pointing_error_deg=60.0,
                    estimated_rate_deg_s=0.3) == SUN_REACQUISITION
    for _ in range(5):
        assert m.update(in_full_eclipse=False, real_sun_available=True,
                        estimated_pointing_error_deg=30.0,
                        estimated_rate_deg_s=0.5) == SUN_REACQUISITION


def test_magnetic_rate_damping_torque_never_adds_rotational_energy_in_achievable_plane():
    B = np.array([25e-6, -15e-6, 35e-6])
    omega = np.array([0.004, -0.002, 0.003])
    desired = -3.0e-3 * omega
    _, achieved = dipole_for_desired_torque(desired, B, [0.4, 0.4, 0.4])
    # Power tau·omega must be non-positive for the projected damping torque.
    assert float(np.dot(achieved, omega)) <= 1e-15
    assert magnetic_rate_damping_enabled(ECLIPSE_RATE_DAMPING)
