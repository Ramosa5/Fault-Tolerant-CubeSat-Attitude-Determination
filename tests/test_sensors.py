import numpy as np
from src.simulation.attitude import propagate_attitude
from src.simulation.sensors import simulate_sensors

def test_vector_sensors_are_unit_norm():
    t=np.arange(0,10,.1); truth=propagate_attitude(t); s=simulate_sensors(t,truth)
    assert np.allclose(np.linalg.norm(s['sun'],axis=1),1,atol=1e-12)
    assert np.allclose(np.linalg.norm(s['mag'],axis=1),1,atol=1e-12)

def test_sensor_simulation_reproducible():
    t=np.arange(0,2,.1); truth=propagate_attitude(t)
    a=simulate_sensors(t,truth,seed=7); b=simulate_sensors(t,truth,seed=7)
    assert np.allclose(a['gyro'],b['gyro'])
