import numpy as np
from src.simulation.orbit_profiles import resolve_orbit_profile, beta_angle_deg
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.orbit_model import propagate_circular_two_body, circular_orbit_period
from src.simulation.control import desired_sun_orbit_dcm, full_attitude_error_vector_body
from src.simulation.magnetic_ltv import finite_horizon_ltv_gain


def test_high_beta_profile_is_high_beta_and_sunlit():
    epoch='2026-01-01T00:00:00+00:00'
    global_orbit={'altitude_m':500000.0,'inclination_deg':51.6,'raan_deg':0.0,'phase_deg':0.0}
    oc=resolve_orbit_profile('high_beta_sunlit',global_orbit,epoch_utc=epoch,high_beta_cfg={'altitude_m':550000.0,'inclination_deg':97.6,'raan_mode':'auto'})
    assert abs(oc['beta_angle_deg']) > 68.0
    T=circular_orbit_period(oc['altitude_m'])
    t=np.linspace(0,T,721)
    sun,spos=sun_direction_series(t,'analytic',epoch=epoch)
    states=propagate_circular_two_body(t,oc['altitude_m'],oc['inclination_deg'],oc['raan_deg'],oc['phase_deg'])
    eclipse,_=eclipse_geometry(states[:,:3],spos)
    assert np.mean(eclipse==2) < 0.01


def test_full_target_dcm_is_orthonormal():
    C=desired_sun_orbit_dcm([1,0.2,0.1],[7e6,0,0],[0,7500,1000],'x')
    assert np.allclose(C.T@C,np.eye(3),atol=1e-12)
    assert np.linalg.det(C) > 0.999999
    assert np.dot(C[:,0],np.array([1,.2,.1])/np.linalg.norm([1,.2,.1])) > 0.999999


def test_full_attitude_error_zero_at_target():
    C=desired_sun_orbit_dcm([1,0,0],[7e6,0,0],[0,7500,0],'x')
    assert np.allclose(C,np.eye(3),atol=1e-12)
    e=full_attitude_error_vector_body([1,0,0,0],C)
    assert np.linalg.norm(e) < 1e-10


def test_ltv_gain_is_finite_and_correct_shape():
    B=np.tile(np.array([25e-6,-10e-6,35e-6]),(12,1))
    K=finite_horizon_ltv_gain([0.0017,0.0017,0.0017],B,step_s=5.0)
    assert K.shape==(3,6)
    assert np.all(np.isfinite(K))
