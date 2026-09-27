import numpy as np
from src.simulation.orbit_model import *
from src.simulation.constants import *
def test_period_near_expected(): assert 5600 < circular_orbit_period(500e3) < 5700
def test_altitude_constant():
 t=np.linspace(0,circular_orbit_period(500e3),500); s=propagate_circular_two_body(t,500e3); r=np.linalg.norm(s[:,:3],axis=1); assert np.ptp(r)<1e-6
def test_speed_constant():
 t=np.linspace(0,circular_orbit_period(500e3),500); s=propagate_circular_two_body(t,500e3); v=np.linalg.norm(s[:,3:],axis=1); assert np.ptp(v)<1e-9
