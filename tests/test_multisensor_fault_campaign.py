import numpy as np
from src.scenarios.reaction_wheel_sun_pointing_comparison import _apply_extra_sensor_faults
from src.scenarios.registry import SCENARIOS


def test_campaign_registered():
    assert 'reaction_wheel_fault_campaign' in SCENARIOS


def test_gyro_bias_step_is_applied():
    rng=np.random.default_rng(1)
    g,m,gf,mf=_apply_extra_sensor_faults(np.zeros(3),np.array([1.,0.,0.]),10.0,
        {'start_s':0.0,'gyro':{'type':'bias_step','bias_rad_s':[0.003,-0.002,0.001]},'mag':{'type':'none'}},True,rng)
    assert np.allclose(g,[0.003,-0.002,0.001])
    assert gf and not mf


def test_magnetometer_loss_is_nan():
    rng=np.random.default_rng(1)
    g,m,gf,mf=_apply_extra_sensor_faults(np.zeros(3),np.array([1.,0.,0.]),10.0,
        {'mag':{'type':'loss'},'gyro':{'type':'none'}},True,rng)
    assert np.all(np.isnan(m))
    assert mf and not gf


def test_faults_inactive_outside_window():
    rng=np.random.default_rng(1)
    g0=np.array([.1,.2,.3]); m0=np.array([1.,0.,0.])
    g,m,gf,mf=_apply_extra_sensor_faults(g0,m0,10.0,
        {'gyro':{'type':'noise','noise_std_rad_s':1.0},'mag':{'type':'loss'}},False,rng)
    assert np.allclose(g,g0) and np.allclose(m,m0)
    assert not gf and not mf
