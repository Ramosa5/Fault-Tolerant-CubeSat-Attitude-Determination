import numpy as np
from src.simulation.attitude import quat_normalize, quat_multiply, quat_to_dcm, small_angle_quat

def skew(v):
    x,y,z=v; return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)

def run_mekf(times_s, gyro, sun_b, mag_b, sun_i, mag_i,
             q0=(1,0,0,0), bias0=(0,0,0), gyro_noise_std=5e-5,
             bias_rw_std=2e-7, vector_noise_std=2e-3):
    """6-state multiplicative EKF: attitude error (3) + gyro bias (3)."""
    t=np.asarray(times_s,float); n=len(t); q=quat_normalize(q0); b=np.asarray(bias0,float).copy()
    P=np.diag([np.deg2rad(5)**2]*3+[5e-4**2]*3)
    qs=np.zeros((n,4)); bs=np.zeros((n,3)); qs[0]=q; bs[0]=b
    I3=np.eye(3)
    for k in range(1,n):
        dt=t[k]-t[k-1]; w=gyro[k-1]-b
        q=quat_normalize(quat_multiply(q, small_angle_quat(w*dt)))
        F=np.block([[-skew(w),-I3],[np.zeros((3,3)),np.zeros((3,3))]])
        Phi=np.eye(6)+F*dt
        Q=np.diag([gyro_noise_std**2]*3+[bias_rw_std**2]*3)*dt
        P=Phi@P@Phi.T+Q
        for z,ref in ((sun_b[k],sun_i),(mag_b[k],mag_i)):
            pred=quat_to_dcm(q).T@ref
            H=np.hstack((skew(pred),np.zeros((3,3))))
            R=(vector_noise_std**2)*I3
            S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S)
            dx=K@(z-pred); q=quat_normalize(quat_multiply(q,small_angle_quat(dx[:3]))); b=b+dx[3:]
            KH=K@H; P=(np.eye(6)-KH)@P@(np.eye(6)-KH).T+K@R@K.T
        qs[k]=q; bs[k]=b
    return qs,bs
