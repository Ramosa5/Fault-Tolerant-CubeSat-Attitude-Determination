import numpy as np
from .attitude import quat_to_dcm, quat_normalize, quat_multiply

AXES = {
    'x': np.array([1.0,0.0,0.0]),
    'y': np.array([0.0,1.0,0.0]),
    'z': np.array([0.0,0.0,1.0]),
}

def unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if n>0 else v

def quat_from_two_vectors(a,b):
    a=unit(a); b=unit(b); d=float(np.dot(a,b))
    if d < -0.999999:
        ref=np.array([1.0,0,0]) if abs(a[0])<0.9 else np.array([0,1.0,0])
        axis=unit(np.cross(a,ref))
        return np.r_[0.0,axis]
    return quat_normalize(np.r_[1.0+d,np.cross(a,b)])

def axis_angle_quat(axis, angle_rad):
    axis=unit(axis); h=0.5*float(angle_rad)
    return np.r_[np.cos(h), np.sin(h)*axis]

def sun_pointing_initial_quaternion(pointing_axis, sun_i, initial_error_deg):
    b=AXES[str(pointing_axis).lower()]; s=unit(sun_i)
    q_des=quat_from_two_vectors(b,s)
    ref=np.array([0.0,0.0,1.0]) if abs(s[2])<0.9 else np.array([0.0,1.0,0.0])
    perturb_axis=unit(np.cross(s,ref))
    q_err=axis_angle_quat(perturb_axis,np.deg2rad(initial_error_deg))
    return quat_normalize(quat_multiply(q_err,q_des))

def sun_pointing_error_deg(q, pointing_axis, sun_i):
    b=AXES[str(pointing_axis).lower()]; d=quat_to_dcm(q)@b; s=unit(sun_i)
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(d),s),-1.0,1.0))))

def sun_pointing_torque(q, omega_body, pointing_axis, sun_i, kp, kd, max_torque_nm=None):
    b=AXES[str(pointing_axis).lower()]
    C=quat_to_dcm(q)
    sun_b=C.T@unit(sun_i)
    e=np.cross(b,sun_b)
    tau=float(kp)*e-float(kd)*np.asarray(omega_body,float)
    if max_torque_nm is not None:
        m=float(max_torque_nm); n=np.linalg.norm(tau)
        if n>m and n>0: tau=tau*(m/n)
    return tau
