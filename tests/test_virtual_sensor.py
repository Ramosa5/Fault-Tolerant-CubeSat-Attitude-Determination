import numpy as np
from src.ml.virtual_sensor import VirtualSunMLP, build_virtual_sun_dataset, propagate_body_vector

def test_virtual_sun_model_output_shape_and_unit_norm():
    import torch
    m=VirtualSunMLP(hidden=(8,4)); x=torch.randn(3,9); y=m(x)
    assert y.shape==(3,3)
    assert np.allclose(torch.linalg.norm(y,dim=1).detach().numpy(),1.0,atol=1e-5)

def test_virtual_sun_dataset_shape():
    X,Y=build_virtual_sun_dataset(runs=2,duration_s=1.0,dt=.1,seed=7)
    assert X.ndim==2 and X.shape[1]==9
    assert Y.ndim==2 and Y.shape[1]==3
    assert len(X)==len(Y) and len(X)>0

def test_zero_rate_vector_propagation_is_constant():
    s=np.array([1.,2.,3.]); s/=np.linalg.norm(s)
    assert np.allclose(propagate_body_vector(s,[0,0,0],.1),s)
