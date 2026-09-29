import numpy as np
from src.simulation.control import sun_pointing_initial_quaternion,sun_pointing_error_deg,sun_pointing_torque

def test_sun_pointing_initial_error_matches_request():
    sun=np.array([1.0,0.2,0.1]); sun/=np.linalg.norm(sun)
    q=sun_pointing_initial_quaternion('x',sun,30.0)
    assert abs(sun_pointing_error_deg(q,'x',sun)-30.0)<1e-8

def test_sun_pointing_controller_reduces_small_error_direction():
    sun=np.array([1.0,0.0,0.0]); q=sun_pointing_initial_quaternion('x',sun,10.0)
    tau=sun_pointing_torque(q,[0,0,0],'x',sun,0.002,0.01,0.01)
    assert np.linalg.norm(tau)>0

from src.simulation.adcs_modes import (
    SunPointingModeManager, SUN_ACQUISITION, SUN_POINTING,
    ECLIPSE_DRIFT, SUN_REACQUISITION, control_enabled,
)


def test_sun_mode_sequence_acquire_eclipse_reacquire():
    m=SunPointingModeManager(threshold_deg=1.0,hold_s=0.2,dt_s=0.1)
    assert m.mode==SUN_ACQUISITION
    m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=0.5)
    assert m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=0.5)==SUN_POINTING
    assert m.update(in_full_eclipse=True,real_sun_available=False,estimated_pointing_error_deg=0.5)==ECLIPSE_DRIFT
    assert not control_enabled(m.mode)
    assert m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=5.0)==SUN_REACQUISITION
    m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=0.4)
    assert m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=0.4)==SUN_POINTING


def test_reacquisition_waits_for_real_sun_sensor():
    m=SunPointingModeManager(threshold_deg=1.0,hold_s=0.1,dt_s=0.1)
    m.mode=ECLIPSE_DRIFT
    assert m.update(in_full_eclipse=False,real_sun_available=False,estimated_pointing_error_deg=0.1)==ECLIPSE_DRIFT
    assert m.update(in_full_eclipse=False,real_sun_available=True,estimated_pointing_error_deg=10.0)==SUN_REACQUISITION
