from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.simulation.attitude import propagate_attitude
from src.simulation.sensors import simulate_sensors

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n else v

def propagate_body_vector(vector_body, omega_body_rad_s, dt):
    """First-order propagation of an inertially fixed vector expressed in body coordinates.

    For a body rotating at omega, the body-frame coordinates obey s_dot = -omega x s.
    """
    s=np.asarray(vector_body,float); w=np.asarray(omega_body_rad_s,float)
    return _unit(s - np.cross(w,s)*float(dt))

class VirtualSunMLP(nn.Module):
    """Tiny physics-informed residual-correction network.

    The first three inputs are the gyro-propagated Sun vector. The network may only
    apply a bounded residual correction around that physical prediction.
    """
    def __init__(self, hidden=(48,24), residual_scale=0.05):
        super().__init__(); self.residual_scale=float(residual_scale)
        self.net=nn.Sequential(nn.Linear(9,int(hidden[0])),nn.ReLU(),
                               nn.Linear(int(hidden[0]),int(hidden[1])),nn.ReLU(),
                               nn.Linear(int(hidden[1]),3))
    def forward(self,x):
        prop=x[:,:3]
        y=prop+self.residual_scale*torch.tanh(self.net(x))
        return y/(torch.linalg.norm(y,dim=1,keepdim=True)+1e-8)

def _unit_q(rng):
    q=rng.normal(size=4); return q/np.linalg.norm(q)

def build_virtual_sun_dataset(runs=24,duration_s=40.0,dt=0.1,seed=8128,
                              gyro_noise_std=5e-5,vector_noise_std=2e-3):
    """Healthy examples for the physics-informed virtual Sun sensor.

    Teacher forcing uses the previous healthy Sun measurement as the propagation state.
    The network learns only the correction to gyro propagation rather than the full attitude map.
    """
    rng=np.random.default_rng(seed); X=[]; Y=[]
    for _ in range(int(runs)):
        local=int(rng.integers(1,2_000_000_000)); rr=np.random.default_rng(local)
        t=np.arange(0,float(duration_s)+dt/2,dt); q0=_unit_q(rr); w0=rr.uniform(-.035,.035,3)
        truth=propagate_attitude(t,q0=q0,omega0=w0)
        bias=rr.normal(0,2e-4,3)
        sens=simulate_sensors(t,truth,seed=local,gyro_bias=bias,
                              gyro_noise_std=gyro_noise_std,vector_noise_std=vector_noise_std)
        prev=sens['sun'][0].copy()
        for k in range(1,len(t)):
            w=sens['gyro'][k-1]-bias
            prop=propagate_body_vector(prev,w,dt)
            X.append(np.r_[prop,sens['mag'][k],w]); Y.append(sens['sun'][k])
            prev=sens['sun'][k]
    return np.asarray(X,np.float32),np.asarray(Y,np.float32)

def train_virtual_sun_model(out_dir, runs=24,duration_s=40.0,dt=0.1,hidden=(48,24),
                            epochs=18,batch_size=256,seed=8128,gyro_noise_std=5e-5,vector_noise_std=2e-3):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    X,Y=build_virtual_sun_dataset(runs,duration_s,dt,seed,gyro_noise_std,vector_noise_std)
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X)); nval=max(1,int(.2*len(idx))); va,tr=idx[:nval],idx[nval:]
    Xtr=X[tr]; Xv=X[va]
    model=VirtualSunMLP(hidden); opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss()
    loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Y[tr])),batch_size=batch_size,shuffle=True)
    hist=[]; best=None; best_val=float('inf'); xv=torch.from_numpy(Xv); yv=torch.from_numpy(Y[va])
    for ep in range(1,int(epochs)+1):
        model.train(); total=0.; n=0
        for xb,yb in loader:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb); n+=len(xb)
        model.eval();
        with torch.no_grad(): vl=crit(model(xv),yv).item()
        hist.append({'epoch':ep,'train_loss':total/max(n,1),'val_loss':vl})
        if vl<best_val: best_val=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pv=model(xv).numpy()
    dots=np.sum(pv*Y[va],axis=1); ang=np.degrees(np.arccos(np.clip(dots,-1,1)))
    payload={'state_dict':model.state_dict(),'hidden':tuple(hidden),'method':'bounded_residual_mlp_on_gyro_propagation'}
    torch.save(payload,out/'virtual_sun_model.pt'); pd.DataFrame(hist).to_csv(out/'learning_curve.csv',index=False)
    metrics={'samples':int(len(X)),'validation_samples':int(len(va)),'validation_angular_rmse_deg':float(np.sqrt(np.mean(ang**2))),
             'validation_angular_mean_deg':float(np.mean(ang)),'validation_angular_max_deg':float(np.max(ang)),
             'parameters':int(sum(p.numel() for p in model.parameters())),'method':'bounded_residual_mlp_on_gyro_propagation'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return VirtualSunPredictor(model),metrics

class VirtualSunPredictor:
    def __init__(self,model): self.model=model.eval()
    def predict(self, previous_sun_body, omega_body_rad_s, mag_body, dt):
        prop=propagate_body_vector(previous_sun_body,omega_body_rad_s,dt)
        x=np.r_[prop,np.asarray(mag_body,float),np.asarray(omega_body_rad_s,float)].astype(np.float32)[None,:]
        with torch.no_grad(): y=self.model(torch.from_numpy(x)).numpy()[0]
        return _unit(y)
