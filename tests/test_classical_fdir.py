import numpy as np

from src.fdir.classical_confidence import ClassicalConfidenceFDIR, HEALTHY, FAULTY, LOST


def _step(det, gyro=(0,0,0), sun=(1,0,0), mag=(0,1,0), sun_visible=True):
    return det.update(
        np.asarray(gyro,float), np.asarray(sun,float), np.asarray(mag,float),
        np.array([1.,0.,0.,0.]), np.zeros(3), np.array([1.,0.,0.]), np.array([0.,1.,0.]),
        sun_visible=sun_visible,
    )


def test_healthy_sensors_remain_high_confidence():
    d=ClassicalConfidenceFDIR({'confidence_smoothing_alpha':0.5},dt=0.1)
    r=None
    for _ in range(30): r=_step(d)
    assert r.state['gyro']==HEALTHY
    assert r.state['sun']==HEALTHY
    assert r.state['mag']==HEALTHY
    assert min(r.confidence.values()) > 0.9


def test_loss_is_detected_immediately():
    d=ClassicalConfidenceFDIR({},dt=0.1)
    _step(d)
    r=d.update(np.full(3,np.nan),np.full(3,np.nan),np.array([0.,1.,0.]),np.array([1.,0.,0.,0.]),np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),sun_visible=True)
    assert r.state['gyro']==LOST
    assert r.state['sun']==LOST
    assert r.is_bad('gyro') and r.is_bad('sun')


def test_persistent_gyro_bias_is_flagged():
    d=ClassicalConfidenceFDIR({'confidence_smoothing_alpha':0.6,'fault_persistence_samples':3,'gyro_rate_residual_warn_rad_s':0.001,'gyro_rate_residual_fault_rad_s':0.006},dt=0.1)
    for _ in range(10): _step(d)
    r=None
    for _ in range(80): r=_step(d,gyro=(0.02,0,0))
    assert r.state['gyro'] in (FAULTY, LOST)
    assert r.confidence['gyro'] < 0.45


def test_vector_misalignment_degrades_magnetometer_not_sun():
    d=ClassicalConfidenceFDIR({'confidence_smoothing_alpha':0.6,'fault_persistence_samples':3},dt=0.1)
    for _ in range(10): _step(d)
    ang=np.deg2rad(30.)
    bad_mag=np.array([-np.sin(ang),np.cos(ang),0.])
    r=None
    for _ in range(15): r=_step(d,mag=bad_mag)
    assert r.state['mag'] in (FAULTY, LOST)
    assert r.state['sun']==HEALTHY


def test_multiple_sensor_losses_are_multi_label():
    d=ClassicalConfidenceFDIR({},dt=0.1)
    _step(d)
    r=d.update(np.full(3,np.nan),np.array([1.,0.,0.]),np.full(3,np.nan),np.array([1.,0.,0.,0.]),np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),sun_visible=True)
    assert r.is_bad('gyro')
    assert not r.is_bad('sun')
    assert r.is_bad('mag')


def test_expected_sun_unavailability_is_not_a_fault():
    from src.fdir.classical_confidence import EXPECTED_UNAVAILABLE
    d=ClassicalConfidenceFDIR({},dt=0.1)
    _step(d)
    r=d.update(np.array([0.001,0,0]),np.full(3,np.nan),np.array([0.,1.,0.]),np.array([1.,0.,0.,0.]),np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),sun_visible=False)
    assert r.state['sun']==EXPECTED_UNAVAILABLE
    assert not r.is_bad('sun')
    assert r.expected_unavailable['sun']


def test_configured_zero_gyro_sample_requests_reconstruction_immediately():
    d=ClassicalConfidenceFDIR({'gyro_zero_is_unavailable':True,'gyro_zero_sample_threshold_rad_s':1e-10},dt=0.1)
    r=d.update(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([1.,0.,0.,0.]),np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),sun_visible=True)
    assert r.state['gyro']==LOST
    assert r.hard_unavailable['gyro']
    assert r.is_bad('gyro')
