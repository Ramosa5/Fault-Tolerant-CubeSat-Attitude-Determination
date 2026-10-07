from __future__ import annotations
from collections import deque
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.simulation.attitude import propagate_attitude
from src.simulation.sensors import simulate_sensors


SENSOR_SLICES = {'gyro': slice(0,3), 'sun': slice(3,6), 'mag': slice(6,9)}


def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if n>1e-12 else v


class MultiSensorMLP(nn.Module):
    def __init__(self, n_features=12, window=20, hidden=(128,64), out_dim=9):
        super().__init__()
        self.net=nn.Sequential(
            nn.Flatten(), nn.Linear(n_features*window,int(hidden[0])), nn.ReLU(), nn.Dropout(.05),
            nn.Linear(int(hidden[0]),int(hidden[1])), nn.ReLU(), nn.Linear(int(hidden[1]),out_dim)
        )
    def forward(self,x): return self.net(x)


class MultiSensorCNN(nn.Module):
    def __init__(self, n_features=12, channels=(32,48), out_dim=9):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv1d(n_features,int(channels[0]),3,padding=1),nn.ReLU(),
            nn.Conv1d(int(channels[0]),int(channels[1]),3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head=nn.Linear(int(channels[1]),out_dim)
    def forward(self,x):
        z=self.features(x.transpose(1,2)).squeeze(-1)
        return self.head(z)


class MultiSensorGRU(nn.Module):
    def __init__(self, n_features=12, hidden=48, out_dim=9):
        super().__init__(); self.gru=nn.GRU(n_features,int(hidden),batch_first=True); self.head=nn.Linear(int(hidden),out_dim)
    def forward(self,x):
        _,h=self.gru(x); return self.head(h[-1])


def build_model(name, window, n_features=12, out_dim=9):
    name=str(name).lower()
    if name=='mlp': return MultiSensorMLP(n_features,window,out_dim=out_dim)
    if name=='cnn': return MultiSensorCNN(n_features,out_dim=out_dim)
    if name=='gru': return MultiSensorGRU(n_features,out_dim=out_dim)
    raise ValueError(f'Unknown virtual sensor model {name}')


def _random_mask_pattern(rng, n, p_mask=.55):
    """Create sensor availability masks with outages, including combined outages.

    At least one sensor is normally left available. A small all-missing fraction is
    retained to test temporal continuation rather than making the training problem
    artificially easy.
    """
    masks=np.ones((n,3),np.float32)
    k=0
    while k<n:
        if rng.random()<p_mask:
            dur=int(rng.integers(2, max(3,min(80,n-k)+1)))
            # Mostly single-sensor, sometimes two-sensor, rarely all-sensor masking.
            u=rng.random()
            count=1 if u<.68 else (2 if u<.95 else 3)
            ids=rng.choice(3,size=count,replace=False)
            masks[k:min(n,k+dur),ids]=0.0
            k+=dur
        else:
            k+=1
    return masks


def build_dataset(runs=24,duration_s=40.0,dt=0.1,window=20,seed=91031,
                  gyro_noise_std=5e-5,vector_noise_std=2e-3):
    rng=np.random.default_rng(seed); X=[]; Y=[]; T=[]
    for _ in range(int(runs)):
        local=int(rng.integers(1,2_000_000_000)); rr=np.random.default_rng(local)
        t=np.arange(0,float(duration_s)+dt/2,dt)
        q0=rr.normal(size=4); q0=q0/np.linalg.norm(q0)
        # Match the operational CubeSat regime more closely: most training runs are
        # low-rate pointing/recovery motion, with a minority of faster acquisition
        # trajectories for robustness.
        wlim=0.004 if rr.random()<0.75 else 0.02
        w0=rr.uniform(-wlim,wlim,3)
        truth=propagate_attitude(t,q0=q0,omega0=w0)
        bias=rr.normal(0,3e-4,3)
        sens=simulate_sensors(t,truth,seed=local,gyro_bias=bias,
                              gyro_noise_std=gyro_noise_std,vector_noise_std=vector_noise_std)
        healthy=np.column_stack([sens['gyro'],sens['sun'],sens['mag']]).astype(np.float32)
        # The virtual gyro target is the bias-free physical body rate.  At runtime
        # the current MEKF bias estimate is added back before feeding the synthetic
        # gyro measurement to the filter. This avoids asking ML to infer an
        # unobservable sensor bias from vector sensors.
        targets=healthy.copy(); targets[:,:3]=truth[:,4:].astype(np.float32)
        masks=_random_mask_pattern(rr,len(t))
        corrupted=healthy.copy()
        for si,sl in enumerate((slice(0,3),slice(3,6),slice(6,9))):
            corrupted[masks[:,si]<.5,sl]=0.0
        feat=np.column_stack([corrupted,masks]).astype(np.float32)
        for k in range(int(window)-1,len(t)):
            X.append(feat[k-window+1:k+1]); Y.append(targets[k]); T.append(masks[k])
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(T,np.float32)


def _fit_scalers(X,Y):
    # Do not normalize the three availability-mask channels.
    data=X[:,:,:9]
    xmu=data.mean(axis=(0,1),keepdims=True); xsd=data.std(axis=(0,1),keepdims=True); xsd[xsd<1e-7]=1.0
    ymu=Y.mean(axis=0,keepdims=True); ysd=Y.std(axis=0,keepdims=True); ysd[ysd<1e-7]=1.0
    return xmu,xsd,ymu,ysd


def _scale_x(X,xmu,xsd):
    out=X.copy(); out[:,:,:9]=(out[:,:,:9]-xmu)/xsd; return out


def _angles(pred,true,sl):
    p=pred[:,sl]; y=true[:,sl]
    p=p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12); y=y/(np.linalg.norm(y,axis=1,keepdims=True)+1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(p*y,axis=1),-1,1)))


def train_multisensor_model(model_name,out_dir,runs=24,duration_s=40.0,dt=0.1,window=20,
                            epochs=18,batch_size=256,seed=91031,
                            gyro_noise_std=5e-5,vector_noise_std=2e-3):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    X,Y,target_masks=build_dataset(runs,duration_s,dt,window,seed,gyro_noise_std,vector_noise_std)
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X)); nv=max(1,int(.2*len(idx))); va=idx[:nv]; tr=idx[nv:]
    xmu,xsd,ymu,ysd=_fit_scalers(X[tr],Y[tr])
    Xs=_scale_x(X,xmu,xsd); Ys=(Y-ymu)/ysd
    model=build_model(model_name,window)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),torch.from_numpy(Ys[tr])),batch_size=int(batch_size),shuffle=True)
    xv=torch.from_numpy(Xs[va]); yv=torch.from_numpy(Ys[va])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss(); hist=[]; best=None; bestv=float('inf')
    for ep in range(1,int(epochs)+1):
        model.train(); total=0.; nn_=0
        for xb,yb in loader:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb); nn_+=len(xb)
        model.eval()
        with torch.no_grad(): vl=crit(model(xv),yv).item()
        hist.append({'epoch':ep,'train_loss':total/max(1,nn_),'val_loss':vl})
        if vl<bestv: bestv=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pred=model(xv).numpy()*ysd+ymu
    sunang=_angles(pred,Y[va],slice(3,6)); magang=_angles(pred,Y[va],slice(6,9)); gyroerr=np.linalg.norm(pred[:,:3]-Y[va,:3],axis=1)
    metrics={
        'model':str(model_name),'window':int(window),'samples':int(len(X)),'validation_samples':int(len(va)),
        'parameters':int(sum(p.numel() for p in model.parameters())),
        'gyro_rmse_rad_s':float(np.sqrt(np.mean(gyroerr**2))),
        'sun_angular_rmse_deg':float(np.sqrt(np.mean(sunang**2))),
        'mag_angular_rmse_deg':float(np.sqrt(np.mean(magang**2))),
        'best_val_loss':float(bestv),
        'method':'temporal_multi_sensor_substitution_with_availability_masks'
    }
    payload={'state_dict':model.state_dict(),'model_name':str(model_name),'window':int(window),
             'xmu':xmu,'xsd':xsd,'ymu':ymu,'ysd':ysd}
    torch.save(payload,out/'model.pt'); pd.DataFrame(hist).to_csv(out/'learning_curve.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return MultiSensorPredictor(model,window,xmu,xsd,ymu,ysd),metrics


class MultiSensorPredictor:
    def __init__(self,model,window,xmu,xsd,ymu,ysd):
        self.model=model.eval(); self.window=int(window); self.xmu=np.asarray(xmu); self.xsd=np.asarray(xsd); self.ymu=np.asarray(ymu); self.ysd=np.asarray(ysd)
        self.history=deque(maxlen=self.window)
    def clone(self):
        # Predictor state must be independent for every scenario branch.
        return MultiSensorPredictor(copy.deepcopy(self.model),self.window,self.xmu.copy(),self.xsd.copy(),self.ymu.copy(),self.ysd.copy())
    def reset(self): self.history.clear()
    def predict_measurements(self,gyro,sun,mag,faulty=(False,False,False)):
        vals=[np.asarray(gyro,float).copy(),np.asarray(sun,float).copy(),np.asarray(mag,float).copy()]
        masks=np.ones(3,np.float32)
        for i,bad in enumerate(faulty):
            if bad or not np.all(np.isfinite(vals[i])):
                vals[i]=np.zeros(3,float); masks[i]=0.0
        row=np.r_[vals[0],vals[1],vals[2],masks].astype(np.float32)
        self.history.append(row)
        while len(self.history)<self.window:
            self.history.appendleft(row.copy())
        x=np.asarray(self.history,np.float32)[None,:,:]
        x[:,:,:9]=(x[:,:,:9]-self.xmu)/self.xsd
        with torch.no_grad(): pred=self.model(torch.from_numpy(x)).numpy()*self.ysd+self.ymu
        y=pred[0]
        out=[y[:3],_unit(y[3:6]),_unit(y[6:9])]
        # Feed recovered values back into temporal history while preserving masks.
        latest=self.history[-1].copy()
        for i,bad in enumerate(faulty):
            if bad or masks[i]<.5:
                latest[3*i:3*i+3]=out[i]
        self.history[-1]=latest
        return out[0],out[1],out[2]

TARGET_SLICE={'gyro':slice(0,3),'sun':slice(3,6),'mag':slice(6,9)}

class SingleSensorPredictor:
    def __init__(self,model,window,xmu,xsd,ymu,ysd,target):
        self.model=model.eval(); self.window=int(window); self.xmu=np.asarray(xmu); self.xsd=np.asarray(xsd)
        self.ymu=np.asarray(ymu); self.ysd=np.asarray(ysd); self.target=str(target); self.history=deque(maxlen=self.window)
    def clone(self):
        return SingleSensorPredictor(copy.deepcopy(self.model),self.window,self.xmu.copy(),self.xsd.copy(),self.ymu.copy(),self.ysd.copy(),self.target)
    def reset(self): self.history.clear()
    def predict(self,gyro,sun,mag,faulty=(False,False,False)):
        vals=[np.asarray(gyro,float).copy(),np.asarray(sun,float).copy(),np.asarray(mag,float).copy()]
        masks=np.ones(3,np.float32)
        for i,bad in enumerate(faulty):
            if bad or not np.all(np.isfinite(vals[i])):
                vals[i]=np.zeros(3,float); masks[i]=0.0
        row=np.r_[vals[0],vals[1],vals[2],masks].astype(np.float32)
        self.history.append(row)
        while len(self.history)<self.window: self.history.appendleft(row.copy())
        x=np.asarray(self.history,np.float32)[None,:,:]
        x[:,:,:9]=(x[:,:,:9]-self.xmu)/self.xsd
        with torch.no_grad(): y=self.model(torch.from_numpy(x)).numpy()*self.ysd+self.ymu
        v=np.asarray(y[0],float)
        if self.target in ('sun','mag'): v=_unit(v)
        return v


def train_sensor_model(model_name,target,out_dir,runs=24,duration_s=40.0,dt=0.1,window=20,
                       epochs=18,batch_size=256,seed=91031,gyro_noise_std=5e-5,vector_noise_std=2e-3):
    target=str(target).lower(); sl=TARGET_SLICE[target]
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    X,Y,_=build_dataset(runs,duration_s,dt,window,seed,gyro_noise_std,vector_noise_std)
    Y=Y[:,sl]
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X)); nv=max(1,int(.2*len(idx))); va=idx[:nv]; tr=idx[nv:]
    data=X[tr,:,:9]; xmu=data.mean(axis=(0,1),keepdims=True); xsd=data.std(axis=(0,1),keepdims=True); xsd[xsd<1e-7]=1.0
    ymu=Y[tr].mean(axis=0,keepdims=True); ysd=Y[tr].std(axis=0,keepdims=True); ysd[ysd<1e-7]=1.0
    Xs=_scale_x(X,xmu,xsd); Ys=(Y-ymu)/ysd
    model=build_model(model_name,window,out_dim=3)
    loader=DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),torch.from_numpy(Ys[tr])),batch_size=int(batch_size),shuffle=True)
    xv=torch.from_numpy(Xs[va]); yv=torch.from_numpy(Ys[va]); opt=torch.optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss()
    bestv=float('inf'); best=None; hist=[]
    for ep in range(1,int(epochs)+1):
        model.train(); total=0.; nn_=0
        for xb,yb in loader:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb); nn_+=len(xb)
        model.eval()
        with torch.no_grad(): vl=crit(model(xv),yv).item()
        hist.append({'epoch':ep,'train_loss':total/max(1,nn_),'val_loss':vl})
        if vl<bestv: bestv=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): pred=model(xv).numpy()*ysd+ymu
    if target=='gyro': metric=float(np.sqrt(np.mean(np.sum((pred-Y[va])**2,axis=1)))); metric_name='validation_rmse_rad_s'
    else:
        ang=_angles(pred,Y[va],slice(0,3)); metric=float(np.sqrt(np.mean(ang**2))); metric_name='validation_angular_rmse_deg'
    metrics={'model':str(model_name),'target_sensor':target,'window':int(window),'samples':int(len(X)),'validation_samples':int(len(va)),
             'parameters':int(sum(p.numel() for p in model.parameters())),metric_name:metric,'best_val_loss':float(bestv),
             'method':'sensor_specific_temporal_virtual_sensor'}
    torch.save({'state_dict':model.state_dict(),'model_name':str(model_name),'target':target,'window':int(window),'xmu':xmu,'xsd':xsd,'ymu':ymu,'ysd':ysd},out/'model.pt')
    pd.DataFrame(hist).to_csv(out/'learning_curve.csv',index=False); (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return SingleSensorPredictor(model,window,xmu,xsd,ymu,ysd,target),metrics


class SensorSpecificPredictorBundle:
    def __init__(self,predictors,metrics=None): self.predictors=dict(predictors); self.metrics=metrics or {}
    def clone(self): return SensorSpecificPredictorBundle({k:v.clone() for k,v in self.predictors.items()},copy.deepcopy(self.metrics))
    def reset(self):
        for p in self.predictors.values(): p.reset()
    def predict_measurements(self,gyro,sun,mag,faulty=(False,False,False)):
        vals=[]
        for target in ('gyro','sun','mag'):
            vals.append(self.predictors[target].predict(gyro,sun,mag,faulty=faulty))
        return tuple(vals)


def train_sensor_specific_bundle(model_name,out_dir,**kwargs):
    preds={}; mets={}
    for j,target in enumerate(('gyro','sun','mag')):
        kw=dict(kwargs); kw['seed']=int(kwargs.get('seed',91031))+j*101
        p,m=train_sensor_model(model_name,target,Path(out_dir)/target,**kw); preds[target]=p; mets[target]=m
    return SensorSpecificPredictorBundle(preds,mets),mets
