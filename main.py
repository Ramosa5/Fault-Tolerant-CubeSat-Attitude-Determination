import numpy as np
from src.simulation.orbit_model import *
T=circular_orbit_period(500e3); t=np.linspace(0,T,1000); s=propagate_circular_two_body(t,500e3)
print(f'Ideal circular orbit period: {T/60:.2f} min')
print(f'Altitude mean: {np.mean(np.linalg.norm(s[:,:3],axis=1)-6378137)/1000:.3f} km')
print(f'Speed mean: {np.mean(np.linalg.norm(s[:,3:],axis=1))/1000:.3f} km/s')
