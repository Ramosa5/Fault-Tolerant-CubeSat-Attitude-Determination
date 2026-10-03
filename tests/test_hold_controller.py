import numpy as np
from src.simulation.control import full_attitude_hold_torque
from src.simulation.adcs_modes import SunPointingModeManager, SUN_POINTING, SUN_ACQUISITION


def test_hold_controller_prioritizes_rate_damping():
    q=np.array([1.,0.,0.,0.])
    Cdes=np.eye(3)
    integ=np.array([1.,0.,0.])
    tau,e=full_attitude_hold_torque(q,np.array([0.01,0,0]),Cdes,integ,
        kp=5e-6,kd=4e-3,ki=2e-8,max_torque_nm=1.2e-5,rate_priority_rad_s=np.deg2rad(0.04))
    assert tau[0] < 0
    assert np.linalg.norm(tau) <= 1.2e-5 + 1e-15


def test_sun_pointing_exit_threshold_is_configurable():
    m=SunPointingModeManager(threshold_deg=2.0,hold_s=1.0,dt_s=0.1,
        rate_threshold_deg_s=0.1,rate_hold_s=1.0,pointing_exit_threshold_deg=4.0,mode=SUN_POINTING)
    assert m.update(in_full_eclipse=False,real_sun_available=True,
        estimated_pointing_error_deg=3.9,estimated_rate_deg_s=0.01)==SUN_POINTING
    assert m.update(in_full_eclipse=False,real_sun_available=True,
        estimated_pointing_error_deg=4.1,estimated_rate_deg_s=0.01)==SUN_ACQUISITION
