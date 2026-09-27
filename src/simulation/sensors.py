import numpy as np
from .attitude import quat_to_dcm

def simulate_sensors(times_s, truth, seed=42, gyro_bias=(2e-4,-1e-4,1.5e-4),
                     gyro_noise_std=5e-5, vector_noise_std=2e-3):
    """Synthetic idealized sensors. DCM maps body -> inertial, so body vector = C.T @ inertial vector."""
    rng=np.random.default_rng(seed); n=len(times_s)
    sun_i=np.array([1.0,0.2,0.1]); sun_i/=np.linalg.norm(sun_i)
    mag_i=np.array([0.25,-0.10,0.96]); mag_i/=np.linalg.norm(mag_i)
    gyro=truth[:,4:]+np.asarray(gyro_bias)+rng.normal(0,gyro_noise_std,(n,3))
    sun=np.zeros((n,3)); mag=np.zeros((n,3))
    for k,q in enumerate(truth[:,:4]):
        C=quat_to_dcm(q)
        for arr,ref in ((sun,sun_i),(mag,mag_i)):
            v=C.T@ref+rng.normal(0,vector_noise_std,3); arr[k]=v/np.linalg.norm(v)
    return {'gyro':gyro,'sun':sun,'mag':mag,'sun_i':sun_i,'mag_i':mag_i,
            'gyro_bias_true':np.asarray(gyro_bias,float)}
