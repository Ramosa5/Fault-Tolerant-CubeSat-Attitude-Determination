import numpy as np, pandas as pd, torch
from src.ml.dataset import split_runs,make_windows
from src.ml.models import build_model

def test_run_split_has_no_leakage():
    rows=[]
    for c,f in enumerate(['healthy','gyro_bias']):
        for r in range(6): rows.append({'run_id':c*6+r,'fault':f})
    s=split_runs(pd.DataFrame(rows),seed=1)
    groups={k:{r for r,v in s.items() if v==k} for k in ('train','val','test')}
    assert groups['train'].isdisjoint(groups['val']) and groups['train'].isdisjoint(groups['test']) and groups['val'].isdisjoint(groups['test'])

def test_windows_never_cross_runs():
    X=np.arange(40,dtype=np.float32).reshape(20,2); y=np.r_[np.zeros(10),np.ones(10)].astype(int); rid=np.r_[np.zeros(10),np.ones(10)].astype(int)
    W,Y,R,E=make_windows(X,y,rid,5,[0,1],stride=1)
    assert len(W)==12 and set(R)=={0,1} and np.all(W[:,0,0]//20==W[:,-1,0]//20)

def test_all_models_output_class_logits():
    for name in ('mlp','cnn','gru'):
        m=build_model(name,7,20,7); out=m(torch.zeros(4,20,7)); assert out.shape==(4,7)
