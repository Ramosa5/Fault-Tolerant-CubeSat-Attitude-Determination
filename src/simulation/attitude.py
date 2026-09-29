import numpy as np

def quat_normalize(q):
    q=np.asarray(q,dtype=float); return q/np.linalg.norm(q)

def quat_multiply(q1,q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2])

def quat_to_dcm(q):
    w,x,y,z=quat_normalize(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def small_angle_quat(dtheta):
    a=np.asarray(dtheta,float); n=np.linalg.norm(a)
    if n < 1e-12: return quat_normalize(np.r_[1.0,0.5*a])
    return np.r_[np.cos(n/2), np.sin(n/2)*a/n]

def quat_angle_error_deg(q_est,q_true):
    d=abs(float(np.dot(quat_normalize(q_est),quat_normalize(q_true))))
    return np.degrees(2*np.arccos(np.clip(d,-1,1)))

def _derivative(state, inertia, torque):
    q=state[:4]; omega=state[4:]
    qdot=0.5*quat_multiply(q, np.r_[0.0,omega])
    odot=np.linalg.solve(inertia, torque-np.cross(omega,inertia@omega))
    return np.r_[qdot,odot]

def propagate_attitude(times_s, q0=(1,0,0,0), omega0=(0.01,-0.015,0.02),
                       inertia_diag=(0.020,0.025,0.030), torque_body=(0,0,0)):
    t=np.asarray(times_s,float); I=np.diag(inertia_diag); tau=np.asarray(torque_body,float)
    out=np.zeros((len(t),7)); out[0,:4]=quat_normalize(q0); out[0,4:]=omega0
    for k in range(len(t)-1):
        h=t[k+1]-t[k]; s=out[k].copy()
        k1=_derivative(s,I,tau); k2=_derivative(s+h*k1/2,I,tau)
        k3=_derivative(s+h*k2/2,I,tau); k4=_derivative(s+h*k3,I,tau)
        sn=s+h*(k1+2*k2+2*k3+k4)/6; sn[:4]=quat_normalize(sn[:4]); out[k+1]=sn
    return out

def rotational_energy(omega,inertia_diag):
    I=np.diag(inertia_diag); return 0.5*np.einsum('...i,ij,...j->...',omega,I,omega)

def angular_momentum_norm(omega,inertia_diag):
    I=np.diag(inertia_diag); return np.linalg.norm(omega@I,axis=1)

def propagate_attitude_controlled(times_s, q0, omega0, inertia_diag, torque_callback):
    """RK4 rigid-body propagation with a state/time-dependent body torque."""
    t=np.asarray(times_s,float); I=np.diag(inertia_diag)
    out=np.zeros((len(t),7)); torques=np.zeros((len(t),3))
    out[0,:4]=quat_normalize(q0); out[0,4:]=omega0
    def deriv(tt,state):
        tau=np.asarray(torque_callback(float(tt),state[:4],state[4:]),float)
        return _derivative(state,I,tau)
    for k in range(len(t)-1):
        h=t[k+1]-t[k]; s=out[k].copy(); tk=t[k]
        k1=deriv(tk,s); k2=deriv(tk+h/2,s+h*k1/2)
        k3=deriv(tk+h/2,s+h*k2/2); k4=deriv(tk+h,s+h*k3)
        sn=s+h*(k1+2*k2+2*k3+k4)/6; sn[:4]=quat_normalize(sn[:4]); out[k+1]=sn
        torques[k]=np.asarray(torque_callback(float(t[k]),s[:4],s[4:]),float)
    torques[-1]=np.asarray(torque_callback(float(t[-1]),out[-1,:4],out[-1,4:]),float)
    return out, torques
