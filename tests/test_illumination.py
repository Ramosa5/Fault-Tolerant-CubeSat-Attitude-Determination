import numpy as np
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.orbit_model import propagate_circular_two_body, circular_orbit_period


def test_analytic_sun_vector_is_normalized_and_changes():
    t=np.array([0.0,86400.0])
    d,p=sun_direction_series(t,'analytic',epoch='2026-01-01T00:00:00+00:00')
    assert np.allclose(np.linalg.norm(d,axis=1),1.0,atol=1e-12)
    assert np.linalg.norm(d[1]-d[0])>1e-4
    assert np.all(np.linalg.norm(p,axis=1)>1e11)


def test_default_leo_contains_eclipse_for_reference_epoch():
    T=float(circular_orbit_period(500000.0))
    t=np.arange(0.0,T+2.0,2.0)
    orb=propagate_circular_two_body(t,500000.0,51.6,0.0,0.0)
    _,sp=sun_direction_series(t,'analytic',epoch='2026-01-01T00:00:00+00:00')
    state,frac=eclipse_geometry(orb[:,:3],sp)
    assert np.any(state==2)
    assert np.any(state==0)
    assert np.all((frac>=0.0)&(frac<=1.0))
