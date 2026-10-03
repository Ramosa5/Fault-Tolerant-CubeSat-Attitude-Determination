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


def desired_sun_orbit_dcm(sun_i, position_eci_m, velocity_eci_mps, pointing_axis='x'):
    """Full desired body-to-inertial DCM for Sun pointing with roll constraint.

    Current operational convention: body +X points to the Sun while body +Z is
    chosen as close as possible to the orbit normal.  This removes the otherwise
    unconstrained roll degree of freedom around the Sun vector.
    """
    if str(pointing_axis).lower() != 'x':
        raise ValueError('full Sun/orbit attitude currently supports pointing_axis="x"')
    x=unit(sun_i)
    h=unit(np.cross(np.asarray(position_eci_m,float),np.asarray(velocity_eci_mps,float)))
    z0=h-np.dot(h,x)*x
    if np.linalg.norm(z0)<1e-8:
        nadir=unit(-np.asarray(position_eci_m,float))
        z0=nadir-np.dot(nadir,x)*x
    z=unit(z0)
    y=unit(np.cross(z,x))
    z=unit(np.cross(x,y))
    return np.column_stack((x,y,z))


def full_attitude_error_vector_body(q_body_to_inertial, desired_dcm_body_to_inertial):
    """Small-angle attitude error vector expressed in the current body frame."""
    C=quat_to_dcm(q_body_to_inertial)
    E=C.T@np.asarray(desired_dcm_body_to_inertial,float)
    return 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])


def full_attitude_pd_torque(q, omega_body, desired_dcm, kp, kd, max_torque_nm=None):
    e=full_attitude_error_vector_body(q,desired_dcm)
    tau=float(kp)*e-float(kd)*np.asarray(omega_body,float)
    if max_torque_nm is not None:
        m=float(max_torque_nm); n=np.linalg.norm(tau)
        if n>m and n>0: tau*=m/n
    return tau,e


def full_attitude_hold_torque(q, omega_body, desired_dcm, integral_error_body, kp, kd, ki=0.0,
                              max_torque_nm=None, rate_priority_rad_s=None, error_weights=None):
    """Rate-prioritized PI-D hold torque for magnetorquer Sun pointing.

    The integral state is maintained by the caller and is intended to compensate
    persistent projected-torque errors as the geomagnetic field rotates.  When the
    body rate exceeds ``rate_priority_rad_s`` the attitude/integral contribution is
    smoothly reduced so damping takes priority over chasing attitude error.
    """
    e=full_attitude_error_vector_body(q,desired_dcm)
    w=np.asarray(omega_body,float)
    integ=np.asarray(integral_error_body,float)
    weights=np.ones(3) if error_weights is None else np.asarray(error_weights,float)
    attitude_term=float(kp)*(weights*e) + float(ki)*(weights*integ)
    if rate_priority_rad_s is not None and float(rate_priority_rad_s)>0:
        wn=float(np.linalg.norm(w))
        if wn>float(rate_priority_rad_s):
            attitude_term *= float(rate_priority_rad_s)/wn
    tau=attitude_term-float(kd)*w
    if max_torque_nm is not None:
        lim=float(max_torque_nm); n=float(np.linalg.norm(tau))
        if n>lim and n>0: tau*=lim/n
    return tau,e
