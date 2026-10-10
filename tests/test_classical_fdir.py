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
    d=ClassicalConfidenceFDIR({
        'confidence_smoothing_alpha':0.6,'fault_persistence_samples':3,
        'gyro_vector_rate_warn_rad_s':0.001,'gyro_vector_rate_fault_rad_s':0.006,
        'gyro_reference_rate_warn_rad_s':0.001,'gyro_reference_rate_fault_rad_s':0.006,
        'vector_rate_window_samples':2,
        'soft_ewma_alpha':0.2,'soft_cusum_gate':0.05,
        'soft_cusum_warn_score_s':0.1,'soft_cusum_fault_score_s':0.4,
    },dt=0.1)
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


def test_v31_cusum_does_not_accumulate_below_gate():
    d=ClassicalConfidenceFDIR({
        'soft_ewma_alpha':0.1,
        'soft_cusum_gate':0.15,
        'soft_cusum_leak_tau_s':60.0,
        'soft_cusum_reset_tau_s':2.0,
        'soft_cusum_warn_score_s':1.0,
        'soft_cusum_fault_score_s':3.0,
    },dt=0.1)
    for _ in range(3000):
        q=d._update_soft_accumulator('gyro',0.10,active=True)
    assert d.soft_cusum['gyro'] < 1e-3
    assert q > 0.99
    assert not d.soft_gate_active['gyro']


def test_v31_cusum_accumulates_persistent_anomaly_and_leaks():
    d=ClassicalConfidenceFDIR({
        'soft_ewma_alpha':0.2,
        'soft_cusum_gate':0.10,
        'soft_cusum_leak_tau_s':60.0,
        'soft_cusum_reset_tau_s':1.0,
        'soft_cusum_warn_score_s':0.5,
        'soft_cusum_fault_score_s':2.0,
    },dt=0.1)
    q=1.0
    for _ in range(120):
        q=d._update_soft_accumulator('gyro',1.0,active=True)
    assert d.soft_cusum['gyro'] > 0.5
    assert q < 1.0
    peak=d.soft_cusum['gyro']
    for _ in range(100):
        d._update_soft_accumulator('gyro',0.0,active=True)
    assert d.soft_cusum['gyro'] < peak


def test_v31_hard_unavailability_resets_persistent_statistic():
    d=ClassicalConfidenceFDIR({
        'soft_ewma_alpha':0.2,'soft_cusum_gate':0.05,
        'soft_cusum_warn_score_s':0.2,'soft_cusum_fault_score_s':1.0,
    },dt=0.1)
    for _ in range(50):
        d._update_soft_accumulator('gyro',1.0,active=True)
    assert d.soft_cusum['gyro'] > 0.0
    d._update_soft_accumulator('gyro',1.0,active=False)
    assert d.soft_cusum['gyro'] == 0.0
    assert d.soft_ewma['gyro'] == 0.0
    assert d.soft_quality['gyro'] == 1.0


def test_vector_sensor_reacquisition_completes_after_hard_mag_loss():
    d=ClassicalConfidenceFDIR({
        'availability_recovery_grace_samples':5,
        'vector_reacquisition_consistency_samples':3,
        'confidence_smoothing_alpha':0.5,
    },dt=0.1)
    gyro=np.array([0.001,0.0,0.0])
    sun=np.array([1.0,0.0,0.0])
    mag=np.array([0.0,1.0,0.0])
    q=np.array([1.0,0.0,0.0,0.0])
    b=np.zeros(3)
    for _ in range(10):
        d.update(gyro,sun,mag,q,b,sun,mag,sun_visible=True)
    lost=d.update(gyro,sun,np.full(3,np.nan),q,b,sun,mag,sun_visible=True)
    assert lost.state['mag']==LOST
    r=None
    for _ in range(6):
        r=d.update(gyro,sun,mag,q,b,sun,mag,sun_visible=True)
    assert r.state['mag']==HEALTHY
    assert r.diagnostics['mag_reacquiring']==0.0
    assert r.confidence['mag'] >= d.degraded_threshold
