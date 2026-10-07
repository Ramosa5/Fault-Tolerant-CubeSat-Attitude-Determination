import numpy as np
import torch

from src.ml.multisensor_virtual import build_model, MultiSensorPredictor
from src.scenarios.registry import SCENARIOS


def test_three_multisensor_models_output_nine_values():
    x=torch.zeros((2,8,12),dtype=torch.float32)
    for name in ('mlp','cnn','gru'):
        y=build_model(name,window=8)(x)
        assert tuple(y.shape)==(2,9)


def test_multisensor_predictor_normalizes_vector_outputs():
    model=build_model('mlp',window=4)
    xmu=np.zeros((1,1,9)); xsd=np.ones((1,1,9)); ymu=np.zeros((1,9)); ysd=np.ones((1,9))
    p=MultiSensorPredictor(model,4,xmu,xsd,ymu,ysd)
    g,s,m=p.predict_measurements(np.zeros(3),np.array([1.,0,0]),np.array([0,1.,0]),faulty=(True,False,True))
    assert g.shape==(3,) and s.shape==(3,) and m.shape==(3,)
    assert np.isfinite(g).all() and np.isfinite(s).all() and np.isfinite(m).all()
    assert np.isclose(np.linalg.norm(s),1.0,atol=1e-6)
    assert np.isclose(np.linalg.norm(m),1.0,atol=1e-6)


def test_model_comparison_scenario_registered():
    assert 'reaction_wheel_ml_model_campaign' in SCENARIOS
