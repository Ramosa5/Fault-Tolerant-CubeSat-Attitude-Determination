import numpy as np
from src.simulation.attitude import propagate_attitude
from src.simulation.sensors import simulate_sensors
from src.simulation.faults import apply_fault

def setup_data():
    t=np.arange(0,10.1,.1); truth=propagate_attitude(t); return t,simulate_sensors(t,truth)

def test_healthy_unchanged():
    t,s=setup_data(); f,y=apply_fault(s,t,'healthy',start_s=2,end_s=8)
    assert np.allclose(f['gyro'],s['gyro']) and np.all(y==0)

def test_bias_only_in_interval():
    t,s=setup_data(); f,y=apply_fault(s,t,'gyro_bias','medium',2,8)
    before=t<2; active=(t>=2)&(t<=8); after=t>8
    assert np.allclose(f['gyro'][before],s['gyro'][before]); assert np.any(np.abs(f['gyro'][active]-s['gyro'][active])>0); assert np.allclose(f['gyro'][after],s['gyro'][after]); assert np.all(y[active]==1)

def test_sun_loss_is_nan_only_in_interval():
    t,s=setup_data(); f,y=apply_fault(s,t,'sun_loss','medium',2,8)
    active=(t>=2)&(t<=8); assert np.all(np.isnan(f['sun'][active])); assert np.all(np.isfinite(f['sun'][~active])); assert np.all(y[active]==6)
