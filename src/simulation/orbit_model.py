import numpy as np
from .constants import MU_EARTH, R_EARTH

def circular_orbit_period(altitude_m, mu=MU_EARTH, radius=R_EARTH):
    r = radius + altitude_m
    return 2*np.pi*np.sqrt(r**3/mu)

def propagate_circular_two_body(times_s, altitude_m, inclination_deg=51.6, raan_deg=0.0, phase_deg=0.0, mu=MU_EARTH, radius=R_EARTH):
    """Analytic circular two-body propagation in an inertial frame (m, m/s)."""
    t = np.asarray(times_s, dtype=float)
    r = radius + altitude_m
    n = np.sqrt(mu/r**3)
    u = np.deg2rad(phase_deg) + n*t
    inc = np.deg2rad(inclination_deg); raan = np.deg2rad(raan_deg)
    cu, su = np.cos(u), np.sin(u); cO, sO = np.cos(raan), np.sin(raan); ci, si = np.cos(inc), np.sin(inc)
    x = r*(cO*cu - sO*su*ci); y = r*(sO*cu + cO*su*ci); z = r*(su*si)
    vx = r*n*(-cO*su - sO*cu*ci); vy = r*n*(-sO*su + cO*cu*ci); vz = r*n*(cu*si)
    return np.column_stack((x,y,z,vx,vy,vz))
