import numpy as np
from src.simulation.attitude import quat_normalize, quat_multiply, quat_to_dcm, small_angle_quat

def skew(v):
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

class MEKF:
    """Stateful 6-state multiplicative EKF for online closed-loop simulations.

    Quaternion convention matches the rest of the project: DCM maps body -> inertial.
    The ``step`` method propagates with the previous gyro sample over ``dt`` and then
    updates with the current Sun/magnetic vector measurements. NaN vectors are skipped.
    """
    def __init__(self, q0=(1,0,0,0), bias0=(0,0,0), gyro_noise_std=5e-5,
                 bias_rw_std=2e-7, vector_noise_std=2e-3):
        self.q=quat_normalize(q0)
        self.b=np.asarray(bias0,float).copy()
        self.P=np.diag([np.deg2rad(5)**2]*3+[5e-4**2]*3)
        self.gyro_noise_std=float(gyro_noise_std)
        self.bias_rw_std=float(bias_rw_std)
        self.vector_noise_std=float(vector_noise_std)
        self.last_gyro=np.zeros(3)

    def step(self, gyro_prev, sun_current, mag_current, sun_i, mag_i, dt, sun_noise_std=None, mag_noise_std=None):
        I3=np.eye(3); g=np.asarray(gyro_prev,float)
        if np.all(np.isfinite(g)): self.last_gyro=g.copy()
        else: g=self.last_gyro.copy()
        w=g-self.b
        self.q=quat_normalize(quat_multiply(self.q, small_angle_quat(w*float(dt))))
        F=np.block([[-skew(w),-I3],[np.zeros((3,3)),np.zeros((3,3))]])
        Phi=np.eye(6)+F*float(dt)
        Q=np.diag([self.gyro_noise_std**2]*3+[self.bias_rw_std**2]*3)*float(dt)
        self.P=Phi@self.P@Phi.T+Q
        diagnostics={}
        for name,z,ref in (("sun",sun_current,sun_i),("mag",mag_current,mag_i)):
            z=np.asarray(z,float)
            if not np.all(np.isfinite(z)):
                diagnostics[f'{name}_innovation']=np.full(3,np.nan)
                diagnostics[f'{name}_available']=False
                continue
            pred=quat_to_dcm(self.q).T@np.asarray(ref,float)
            innovation=z-pred
            H=np.hstack((skew(pred),np.zeros((3,3))))
            meas_std=self.vector_noise_std if (sun_noise_std if name=='sun' else mag_noise_std) is None else float(sun_noise_std if name=='sun' else mag_noise_std)
            R=(meas_std**2)*I3
            S=H@self.P@H.T+R; K=self.P@H.T@np.linalg.inv(S)
            dx=K@innovation
            self.q=quat_normalize(quat_multiply(self.q,small_angle_quat(dx[:3])))
            self.b=self.b+dx[3:]
            KH=K@H
            self.P=(np.eye(6)-KH)@self.P@(np.eye(6)-KH).T+K@R@K.T
            diagnostics[f'{name}_innovation']=innovation
            diagnostics[f'{name}_available']=True
        return self.q.copy(), self.b.copy(), diagnostics


def run_mekf(times_s, gyro, sun_b, mag_b, sun_i, mag_i,
             q0=(1,0,0,0), bias0=(0,0,0), gyro_noise_std=5e-5,
             bias_rw_std=2e-7, vector_noise_std=2e-3, return_diagnostics=False):
    """Batch convenience wrapper around :class:`MEKF`."""
    t=np.asarray(times_s,float); n=len(t)
    filt=MEKF(q0=q0,bias0=bias0,gyro_noise_std=gyro_noise_std,
              bias_rw_std=bias_rw_std,vector_noise_std=vector_noise_std)
    qs=np.zeros((n,4)); bs=np.zeros((n,3)); qs[0]=filt.q; bs[0]=filt.b
    sun_innov=np.full((n,3),np.nan); mag_innov=np.full((n,3),np.nan)
    sun_available=np.all(np.isfinite(sun_b),axis=1); mag_available=np.all(np.isfinite(mag_b),axis=1)
    for k in range(1,n):
        q,b,d=filt.step(gyro[k-1],sun_b[k],mag_b[k],sun_i,mag_i,t[k]-t[k-1])
        qs[k]=q; bs[k]=b
        sun_innov[k]=d['sun_innovation']; mag_innov[k]=d['mag_innovation']
    if not return_diagnostics: return qs,bs
    diagnostics={'sun_innovation':sun_innov,'mag_innovation':mag_innov,
                 'sun_available':sun_available,'mag_available':mag_available}
    return qs,bs,diagnostics
