import numpy as np
from src.simulation.attitude import propagate_attitude, rotational_energy, angular_momentum_norm

def test_quaternion_norm_preserved():
    t=np.linspace(0,100,1001); s=propagate_attitude(t)
    assert np.max(np.abs(np.linalg.norm(s[:,:4],axis=1)-1)) < 1e-12

def test_torque_free_energy_conserved():
    t=np.linspace(0,100,1001); I=(.020,.025,.030); s=propagate_attitude(t,inertia_diag=I)
    E=rotational_energy(s[:,4:],I); assert (E.max()-E.min())/E.mean() < 1e-9

def test_torque_free_momentum_conserved():
    t=np.linspace(0,100,1001); I=(.020,.025,.030); s=propagate_attitude(t,inertia_diag=I)
    H=angular_momentum_norm(s[:,4:],I); assert (H.max()-H.min())/H.mean() < 1e-9
