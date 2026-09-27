import numpy as np

FAULT_NAMES={0:'healthy',1:'gyro_bias',2:'gyro_drift',3:'gyro_noise',4:'sun_noise',5:'sun_dropout',6:'sun_loss'}

def apply_fault(sensors, times_s, fault_type='healthy', severity='medium', start_s=200.0, end_s=None, seed=100):
    """Return a copy of sensor data with one controlled fault injected.

    Labels are 0 before/after the fault interval and the FAULT_NAMES integer during the fault.
    This separation lets ML learn onset instead of scenario identity.
    """
    t=np.asarray(times_s,float); end_s=t[-1] if end_s is None else float(end_s)
    mask=(t>=float(start_s))&(t<=end_s); rng=np.random.default_rng(seed)
    out={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in sensors.items()}
    levels={'low':1.0,'medium':2.0,'high':4.0}; s=levels[severity]
    fault_id={v:k for k,v in FAULT_NAMES.items()}[fault_type]
    labels=np.zeros(len(t),dtype=np.int16); labels[mask]=fault_id
    if fault_type=='healthy': return out,labels
    if fault_type=='gyro_bias': out['gyro'][mask]+=np.array([4e-4,-3e-4,2e-4])*s
    elif fault_type=='gyro_drift':
        elapsed=np.maximum(t-start_s,0.0); drift=elapsed[:,None]*np.array([1.0,-0.7,0.5])[None,:]*2e-6*s
        out['gyro'][mask]+=drift[mask]
    elif fault_type=='gyro_noise': out['gyro'][mask]+=rng.normal(0,2e-4*s,(mask.sum(),3))
    elif fault_type=='sun_noise':
        z=out['sun'][mask]+rng.normal(0,8e-3*s,(mask.sum(),3)); out['sun'][mask]=z/np.linalg.norm(z,axis=1,keepdims=True)
    elif fault_type=='sun_dropout':
        idx=np.where(mask)[0]; drop=rng.random(len(idx)) < min(0.15*s,0.75); out['sun'][idx[drop]]=np.nan
    elif fault_type=='sun_loss': out['sun'][mask]=np.nan
    else: raise ValueError(f'Unknown fault_type: {fault_type}')
    return out,labels
