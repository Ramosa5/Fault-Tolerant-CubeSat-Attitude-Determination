from __future__ import annotations
import numpy as np


def _skew(v):
    x,y,z=np.asarray(v,float)
    return np.array([[0.0,-z,y],[z,0.0,-x],[-y,x,0.0]])


def finite_horizon_ltv_gain(inertia_diag, future_B_body_t, *, step_s=5.0,
                            q_attitude=8.0, q_rate=1.0, r_dipole=2.0e-5,
                            terminal_scale=4.0):
    """Finite-horizon LTV magnetic regulator gain for dipole command.

    Linearized state x=[small attitude error, body rate].  The input is magnetic
    dipole moment m and the time-varying input matrix follows tau=m x B.  Future
    geomagnetic vectors therefore enter the Riccati recursion explicitly.
    """
    I=np.diag(np.asarray(inertia_diag,float)); Iinv=np.diag(1.0/np.asarray(inertia_diag,float))
    h=float(step_s)
    A=np.block([[np.eye(3), -h*np.eye(3)],[np.zeros((3,3)),np.eye(3)]])
    Q=np.diag([float(q_attitude)]*3+[float(q_rate)]*3)
    R=float(r_dipole)*np.eye(3)
    P=float(terminal_scale)*Q
    K0=np.zeros((3,6))
    Bs=list(np.asarray(future_B_body_t,float))
    if not Bs:
        return K0
    for idx in range(len(Bs)-1,-1,-1):
        Bv=Bs[idx]
        # m x B = -[B]x m
        Gtau=-_skew(Bv)
        Bd=np.vstack([np.zeros((3,3)), h*(Iinv@Gtau)])
        S=R+Bd.T@P@Bd
        K=np.linalg.solve(S, Bd.T@P@A)
        P=Q+A.T@P@(A-Bd@K)
        if idx==0:
            K0=K
    return K0


def magnetic_ltv_dipole(error_body, omega_body, inertia_diag, future_B_body_t,
                         max_dipole_am2, *, step_s=5.0, q_attitude=8.0,
                         q_rate=1.0, r_dipole=2.0e-5, terminal_scale=4.0):
    x=np.r_[np.asarray(error_body,float),np.asarray(omega_body,float)]
    K=finite_horizon_ltv_gain(inertia_diag,future_B_body_t,step_s=step_s,
                              q_attitude=q_attitude,q_rate=q_rate,r_dipole=r_dipole,
                              terminal_scale=terminal_scale)
    m=-K@x
    lim=np.asarray(max_dipole_am2,float)
    if lim.shape==(): lim=np.repeat(float(lim),3)
    return np.clip(m,-lim,lim),K


def authority_scheduled_dipole(desired_torque_body_nm, current_B_body_t, future_B_body_t,
                               max_dipole_am2, *, torque_limit_nm=None,
                               regularization=0.35, max_compensation=2.0):
    """Conservative LTV magnetic authority scheduling.

    The achievable torque plane P=I-bb^T varies along the orbit.  Average future
    authority is used to precondition a *bounded* desired torque; the command is
    then allocated through the current m×B geometry.  This preserves the baseline
    PD torque bound while exploiting known future field geometry and avoids the
    excessive dipole saturation of an unconstrained finite-horizon regulator.
    """
    tau=np.asarray(desired_torque_body_nm,float)
    FB=np.atleast_2d(np.asarray(future_B_body_t,float))
    W=np.zeros((3,3)); count=0
    for B in FB:
        n=np.linalg.norm(B)
        if n<=1e-12: continue
        b=B/n; W += np.eye(3)-np.outer(b,b); count += 1
    if count==0:
        W=np.eye(3)
    else:
        W/=count
    reg=float(regularization)
    sched=np.linalg.solve(W+reg*np.eye(3),tau)
    # Prevent authority compensation from magnifying the original command without bound.
    tn=np.linalg.norm(tau); sn=np.linalg.norm(sched)
    if tn>0 and sn>float(max_compensation)*tn:
        sched*=float(max_compensation)*tn/sn
    if torque_limit_nm is not None:
        lim=float(torque_limit_nm); sn=np.linalg.norm(sched)
        if sn>lim and sn>0: sched*=lim/sn
    B=np.asarray(current_B_body_t,float); b2=float(B@B)
    if b2<1e-18:
        return np.zeros(3),sched,W
    m=np.cross(B,sched)/b2
    lim=np.asarray(max_dipole_am2,float)
    if lim.shape==(): lim=np.repeat(float(lim),3)
    return np.clip(m,-lim,lim),sched,W
