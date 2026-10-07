from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import copy, hashlib, json, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.simulation.attitude import propagate_attitude, quat_to_dcm, quat_normalize, quat_multiply
from src.simulation.control import desired_sun_orbit_dcm, axis_angle_quat
from src.simulation.orbit_model import propagate_circular_two_body
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.magnetic_field import magnetic_field_eci

ARCHITECTURES=("mlp","cnn","gru")
VARIANTS=("sensor_only","orbital_context","physics_residual")
TARGETS=("gyro","sun","mag")
TARGET_SLICE={"gyro":slice(0,3),"sun":slice(3,6),"mag":slice(6,9)}


def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if n>1e-12 else np.zeros_like(v)


def _quat_from_dcm(C):
    """Robust body-to-inertial DCM -> scalar-first quaternion."""
    C=np.asarray(C,float); tr=float(np.trace(C))
    if tr>0:
        s=math.sqrt(tr+1.0)*2; w=.25*s
        x=(C[2,1]-C[1,2])/s; y=(C[0,2]-C[2,0])/s; z=(C[1,0]-C[0,1])/s
    elif C[0,0]>C[1,1] and C[0,0]>C[2,2]:
        s=math.sqrt(1+C[0,0]-C[1,1]-C[2,2])*2
        w=(C[2,1]-C[1,2])/s; x=.25*s; y=(C[0,1]+C[1,0])/s; z=(C[0,2]+C[2,0])/s
    elif C[1,1]>C[2,2]:
        s=math.sqrt(1+C[1,1]-C[0,0]-C[2,2])*2
        w=(C[0,2]-C[2,0])/s; x=(C[0,1]+C[1,0])/s; y=.25*s; z=(C[1,2]+C[2,1])/s
    else:
        s=math.sqrt(1+C[2,2]-C[0,0]-C[1,1])*2
        w=(C[1,0]-C[0,1])/s; x=(C[0,2]+C[2,0])/s; y=(C[1,2]+C[2,1])/s; z=.25*s
    return quat_normalize(np.array([w,x,y,z],float))


def _rotation_vector_from_dcm(R):
    R=np.asarray(R,float); c=np.clip((np.trace(R)-1.0)/2.0,-1.0,1.0); ang=float(np.arccos(c))
    if ang<1e-9:
        return .5*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    s=2.0*np.sin(ang)
    axis=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/s
    return axis*ang


def build_orbital_context(states,sun_i,B_i,illumination):
    """25-feature context foundation excluding measured sensors/masks.

    The final per-step feature vector is:
      measured gyro(3), Sun(3), magnetometer(3), availability masks(3),
      r_ECI unit(3), v_ECI unit(3), Sun_ECI unit(3), B_ECI unit(3), illumination(1).
    """
    r=np.asarray(states[:,:3],float); v=np.asarray(states[:,3:],float)
    rhat=r/(np.linalg.norm(r,axis=1,keepdims=True)+1e-12)
    vhat=v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12)
    shat=np.asarray(sun_i,float); shat=shat/(np.linalg.norm(shat,axis=1,keepdims=True)+1e-12)
    bhat=np.asarray(B_i,float); bhat=bhat/(np.linalg.norm(bhat,axis=1,keepdims=True)+1e-12)
    return np.column_stack([rhat,vhat,shat,bhat,np.asarray(illumination,float)]).astype(np.float32)


def build_physics_baseline(states,sun_i,B_i,dt,pointing_axis='x'):
    """Planned-attitude baseline [omega_des, Sun_body_des, B_body_des]."""
    n=len(states); C=np.empty((n,3,3)); out=np.zeros((n,9),np.float32)
    for k in range(n):
        C[k]=desired_sun_orbit_dcm(sun_i[k],states[k,:3],states[k,3:],pointing_axis)
        out[k,3:6]=_unit(C[k].T@sun_i[k])
        out[k,6:9]=_unit(C[k].T@_unit(B_i[k]))
    if n>1:
        for k in range(1,n):
            # Approximate desired angular velocity in the previous body frame.
            R=C[k-1].T@C[k]
            out[k,:3]=_rotation_vector_from_dcm(R)/float(dt)
        out[0,:3]=out[1,:3]
    return out


def _random_mask_pattern(rng,n,p_mask=.55,max_outage_samples=80):
    masks=np.ones((n,3),np.float32); k=0
    while k<n:
        if rng.random()<p_mask:
            dur=int(rng.integers(2,max(3,min(int(max_outage_samples),n-k)+1)))
            u=rng.random(); count=1 if u<.68 else (2 if u<.95 else 3)
            ids=rng.choice(3,size=count,replace=False); masks[k:min(n,k+dur),ids]=0.; k+=dur
        else: k+=1
    return masks


def _apply_training_degradations(rng,healthy,masks,cfg):
    """Apply configurable training-time sensor faults without changing target truth.

    This allows the training distribution to cover loss/dropout plus bias, drift,
    increased noise and vector misalignment. The exact realization is shared by all
    architectures and variants because it is generated once in the common dataset.
    """
    x=np.asarray(healthy,float).copy(); n=len(x)
    # Missing sensors are represented by zero vectors + mask=0.
    for si,sl in enumerate((slice(0,3),slice(3,6),slice(6,9))): x[masks[:,si]<.5,sl]=0.
    if not bool(cfg.get('enable_nonmissing_faults',True)): return x.astype(np.float32)
    # Gyro bias/drift/noise on still-available samples.
    gb=float(cfg.get('gyro_bias_max_rad_s',0.006)); gd=float(cfg.get('gyro_drift_max_rad_s2',3e-5)); gn=float(cfg.get('gyro_extra_noise_max_rad_s',0.002))
    if rng.random()<float(cfg.get('gyro_degradation_probability',.5)):
        bias=rng.uniform(-gb,gb,3); drift=rng.uniform(-gd,gd,3); extra=rng.uniform(0,gn)
        tt=np.arange(n,dtype=float)
        gmask=masks[:,0]>.5; x[gmask,:3]+=bias+tt[gmask,None]*drift+rng.normal(0,extra,(gmask.sum(),3))
    # Sun vector extra noise.
    sn=float(cfg.get('sun_extra_noise_max',.03))
    if rng.random()<float(cfg.get('sun_degradation_probability',.35)):
        std=rng.uniform(0,sn); sm=masks[:,1]>.5
        for k in np.where(sm)[0]: x[k,3:6]=_unit(x[k,3:6]+rng.normal(0,std,3))
    # Magnetometer bias/noise/misalignment approximation by fixed rotation.
    mn=float(cfg.get('mag_extra_noise_max',.04)); ma=float(cfg.get('mag_misalignment_max_deg',20.0))
    if rng.random()<float(cfg.get('mag_degradation_probability',.5)):
        std=rng.uniform(0,mn); ang=np.deg2rad(rng.uniform(-ma,ma)); axis=_unit(rng.normal(size=3)); q=axis_angle_quat(axis,ang)
        # Rodrigues matrix from quaternion.
        w,a,b,c=q
        R=np.array([[1-2*(b*b+c*c),2*(a*b-c*w),2*(a*c+b*w)],
                    [2*(a*b+c*w),1-2*(a*a+c*c),2*(b*c-a*w)],
                    [2*(a*c-b*w),2*(b*c+a*w),1-2*(a*a+b*b)]])
        mm=masks[:,2]>.5
        for k in np.where(mm)[0]: x[k,6:9]=_unit(R@x[k,6:9]+rng.normal(0,std,3))
    return x.astype(np.float32)


def _feature_slice(variant):
    return slice(0,12) if str(variant)=='sensor_only' else slice(0,25)


def _hash_int_array(a):
    return hashlib.sha256(np.asarray(a,np.int64).tobytes()).hexdigest()


def generate_shared_dataset(cfg,gyro_noise_std=5e-5,vector_noise_std=2e-3):
    """Generate one common orbital dataset used by every architecture/variant.

    Splits are performed by complete simulation run IDs, never by overlapping
    windows, preventing temporal leakage and ensuring fair model comparisons.
    """
    runs=int(cfg.get('runs',36)); duration=float(cfg.get('duration_s',60.0)); dt=float(cfg.get('dt_s',0.1)); window=int(cfg.get('window',20)); seed=int(cfg.get('seed',97001))
    rng=np.random.default_rng(seed); X25=[]; Y=[]; BASE=[]; RUN=[]
    alt_rng=cfg.get('altitude_m_range',[400e3,650e3]); inc_rng=cfg.get('inclination_deg_range',[30.,98.]); epoch0=datetime.fromisoformat(str(cfg.get('epoch_utc','2026-01-01T00:00:00+00:00')).replace('Z','+00:00'))
    max_epoch_days=float(cfg.get('epoch_span_days',365.0)); mag_model=str(cfg.get('training_magnetic_field_model','dipole'))
    mask_cfg=cfg.get('masking',{}); degradation_cfg=cfg.get('degradations',{})
    run_meta=[]
    for rid in range(runs):
        rr=np.random.default_rng(int(rng.integers(1,2_000_000_000)))
        t=np.arange(0,duration+dt/2,dt); n=len(t)
        alt=float(rr.uniform(*alt_rng)); inc=float(rr.uniform(*inc_rng)); raan=float(rr.uniform(0,360)); phase=float(rr.uniform(0,360)); epoch=epoch0+timedelta(days=float(rr.uniform(0,max_epoch_days)))
        states=propagate_circular_two_body(t,alt,inc,raan,phase)
        sun_i,sun_pos=sun_direction_series(t,'analytic',epoch=epoch.isoformat())
        eclipse_state,illum=eclipse_geometry(states[:,:3],sun_pos)
        B_i=magnetic_field_eci(states[:,:3],t,epoch_utc=epoch,model=mag_model)
        ctx=build_orbital_context(states,sun_i,B_i,illum); base=build_physics_baseline(states,sun_i,B_i,dt)
        # Start near planned Sun-pointing attitude but vary attitude/rates broadly.
        C0=desired_sun_orbit_dcm(sun_i[0],states[0,:3],states[0,3:],'x'); qdes=_quat_from_dcm(C0)
        axis=_unit(rr.normal(size=3)); err=np.deg2rad(rr.uniform(0,float(cfg.get('initial_attitude_error_max_deg',35.0))))
        q0=quat_normalize(quat_multiply(axis_angle_quat(axis,err),qdes))
        wlim=float(cfg.get('body_rate_max_rad_s',.02)); w0=rr.uniform(-wlim,wlim,3)
        truth=propagate_attitude(t,q0=q0,omega0=w0,inertia_diag=tuple(cfg.get('training_inertia_kg_m2',[.001667,.001667,.001667])))
        bias=rr.normal(0,float(cfg.get('gyro_bias_training_std_rad_s',3e-4)),3)
        gyro=truth[:,4:]+bias+rng.normal(0,gyro_noise_std,(n,3))
        sun_b=np.zeros((n,3)); mag_b=np.zeros((n,3))
        for k,q in enumerate(truth[:,:4]):
            C=quat_to_dcm(q); sun_b[k]=_unit(C.T@sun_i[k]+rr.normal(0,vector_noise_std,3)); mag_b[k]=_unit(C.T@_unit(B_i[k])+rr.normal(0,vector_noise_std,3))
        healthy=np.column_stack([gyro,sun_b,mag_b]).astype(np.float32)
        targets=healthy.copy(); targets[:,:3]=truth[:,4:].astype(np.float32)
        masks=_random_mask_pattern(rr,n,p_mask=float(mask_cfg.get('probability',.55)),max_outage_samples=int(mask_cfg.get('max_outage_samples',80)))
        # Physical eclipse always removes Sun availability in training.
        masks[eclipse_state==2,1]=0.
        corrupted=_apply_training_degradations(rr,healthy,masks,degradation_cfg)
        feat=np.column_stack([corrupted,masks,ctx]).astype(np.float32) # 9+3+13=25
        for k in range(window-1,n):
            X25.append(feat[k-window+1:k+1]); Y.append(targets[k]); BASE.append(base[k]); RUN.append(rid)
        run_meta.append({'run_id':rid,'altitude_m':alt,'inclination_deg':inc,'raan_deg':raan,'phase_deg':phase,'epoch_utc':epoch.isoformat(),'beta_orbit_training':True})
    X25=np.asarray(X25,np.float32); Y=np.asarray(Y,np.float32); BASE=np.asarray(BASE,np.float32); RUN=np.asarray(RUN,np.int32)
    # Split by complete run IDs.
    ids=np.arange(runs); prng=np.random.default_rng(int(cfg.get('split_seed',seed+17))); ids=prng.permutation(ids)
    ftrain=float(cfg.get('train_fraction',.70)); fval=float(cfg.get('validation_fraction',.15)); ntr=max(1,int(round(runs*ftrain))); nva=max(1,int(round(runs*fval)))
    if ntr+nva>=runs: nva=max(1,runs-ntr-1)
    train_ids=np.sort(ids[:ntr]); val_ids=np.sort(ids[ntr:ntr+nva]); test_ids=np.sort(ids[ntr+nva:])
    split={'train_run_ids':train_ids.tolist(),'validation_run_ids':val_ids.tolist(),'test_run_ids':test_ids.tolist()}
    manifest={'runs':runs,'duration_s':duration,'dt_s':dt,'window':window,'samples':int(len(X25)),'features_context':25,'features_sensor_only':12,'seed':seed,'split_seed':int(cfg.get('split_seed',seed+17)),**split,
              'train_hash':_hash_int_array(train_ids),'validation_hash':_hash_int_array(val_ids),'test_hash':_hash_int_array(test_ids),'run_metadata':run_meta}
    return {'X25':X25,'Y':Y,'BASE':BASE,'RUN':RUN,'manifest':manifest}


def save_shared_dataset(dataset,out_dir,save_npz=True):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'shared_dataset_manifest.json').write_text(json.dumps(dataset['manifest'],indent=2),encoding='utf-8')
    if save_npz:
        np.savez_compressed(out/'shared_dataset.npz',X25=dataset['X25'],Y=dataset['Y'],BASE=dataset['BASE'],RUN=dataset['RUN'])


def _indices_for_split(run_ids,split_ids): return np.where(np.isin(run_ids,np.asarray(split_ids,int)))[0]


def _fit_x_scaler(X):
    mu=X.mean(axis=(0,1),keepdims=True); sd=X.std(axis=(0,1),keepdims=True); sd[sd<1e-7]=1.; return mu.astype(np.float32),sd.astype(np.float32)


def _fit_y_scaler(Y):
    mu=Y.mean(axis=0,keepdims=True); sd=Y.std(axis=0,keepdims=True); sd[sd<1e-7]=1.; return mu.astype(np.float32),sd.astype(np.float32)


class FairMLP(nn.Module):
    def __init__(self,n_features,window,hidden=(128,64),out_dim=3,dropout=.05):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(n_features*window,int(hidden[0])),nn.ReLU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden[0]),int(hidden[1])),nn.ReLU(),nn.Linear(int(hidden[1]),out_dim))
    def forward(self,x): return self.net(x)

class FairCNN(nn.Module):
    def __init__(self,n_features,channels=(32,48),out_dim=3,kernel_size=3):
        super().__init__(); p=int(kernel_size)//2; self.features=nn.Sequential(nn.Conv1d(n_features,int(channels[0]),int(kernel_size),padding=p),nn.ReLU(),nn.Conv1d(int(channels[0]),int(channels[1]),int(kernel_size),padding=p),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); self.head=nn.Linear(int(channels[1]),out_dim)
    def forward(self,x): return self.head(self.features(x.transpose(1,2)).squeeze(-1))

class FairGRU(nn.Module):
    def __init__(self,n_features,hidden=48,layers=1,out_dim=3):
        super().__init__(); self.gru=nn.GRU(n_features,int(hidden),num_layers=int(layers),batch_first=True); self.head=nn.Linear(int(hidden),out_dim)
    def forward(self,x): _,h=self.gru(x); return self.head(h[-1])


def _build_model(arch,n_features,window,hp):
    if arch=='mlp': return FairMLP(n_features,window,hidden=tuple(hp.get('hidden',[128,64])),dropout=float(hp.get('dropout',.05)))
    if arch=='cnn': return FairCNN(n_features,channels=tuple(hp.get('channels',[32,48])),kernel_size=int(hp.get('kernel_size',3)))
    if arch=='gru': return FairGRU(n_features,hidden=int(hp.get('hidden',48)),layers=int(hp.get('layers',1)))
    raise ValueError(arch)


def _angular_rmse(pred,true):
    p=pred/(np.linalg.norm(pred,axis=1,keepdims=True)+1e-12); y=true/(np.linalg.norm(true,axis=1,keepdims=True)+1e-12)
    a=np.degrees(np.arccos(np.clip(np.sum(p*y,axis=1),-1,1))); return float(np.sqrt(np.mean(a*a)))


@dataclass
class SharedScaler:
    xmu: np.ndarray
    xsd: np.ndarray
    ymu: np.ndarray
    ysd: np.ndarray


def prepare_variant_target(dataset,variant,target):
    sl=TARGET_SLICE[target]; fs=_feature_slice(variant); X=dataset['X25'][:,:,fs]; Y=dataset['Y'][:,sl].copy()
    if variant=='physics_residual': Y=Y-dataset['BASE'][:,sl]
    return X.astype(np.float32),Y.astype(np.float32)


def train_one_fair_model(arch,variant,target,dataset,out_dir,training_cfg,arch_cfg,shared_scaler=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); window=int(dataset['manifest']['window']); X,Y=prepare_variant_target(dataset,variant,target); run_ids=dataset['RUN']; man=dataset['manifest']
    tr=_indices_for_split(run_ids,man['train_run_ids']); va=_indices_for_split(run_ids,man['validation_run_ids']); te=_indices_for_split(run_ids,man['test_run_ids'])
    if shared_scaler is None:
        xmu,xsd=_fit_x_scaler(X[tr]); ymu,ysd=_fit_y_scaler(Y[tr]); scaler=SharedScaler(xmu,xsd,ymu,ysd)
    else: scaler=shared_scaler
    Xs=(X-scaler.xmu)/scaler.xsd; Ys=(Y-scaler.ymu)/scaler.ysd
    seed=int(training_cfg.get('weight_seed',98101)); arch_offset={'mlp':0,'cnn':100,'gru':200}[arch]; torch.manual_seed(seed+arch_offset); np.random.seed(seed+arch_offset)
    model=_build_model(arch,X.shape[-1],window,arch_cfg); batch=int(training_cfg.get('batch_size',256)); epochs=int(training_cfg.get('epochs',20)); lr=float(training_cfg.get('learning_rate',1e-3)); wd=float(training_cfg.get('weight_decay',0.0))
    # Same ordered training samples for every architecture; DataLoader shuffle uses the same generator seed.
    gen=torch.Generator().manual_seed(int(training_cfg.get('loader_seed',98201)))
    loader=DataLoader(TensorDataset(torch.from_numpy(Xs[tr]),torch.from_numpy(Ys[tr])),batch_size=batch,shuffle=True,generator=gen)
    xv=torch.from_numpy(Xs[va]); yv=torch.from_numpy(Ys[va]); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd); crit=nn.MSELoss(); bestv=float('inf'); best=None; hist=[]
    patience=int(training_cfg.get('early_stopping_patience',0)); stale=0
    for ep in range(1,epochs+1):
        model.train(); total=0.; count=0
        for xb,yb in loader:
            opt.zero_grad(); loss=crit(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb); count+=len(xb)
        model.eval();
        with torch.no_grad(): vl=float(crit(model(xv),yv).item())
        hist.append({'epoch':ep,'train_loss':total/max(1,count),'val_loss':vl})
        if vl<bestv-1e-12: bestv=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if patience>0 and stale>=patience: break
    model.load_state_dict(best); model.eval()
    def predict_idx(ix):
        with torch.no_grad(): p=model(torch.from_numpy(Xs[ix])).numpy()*scaler.ysd+scaler.ymu
        if variant=='physics_residual': p=p+dataset['BASE'][ix,TARGET_SLICE[target]]
        return p
    pv=predict_idx(va); pt=predict_idx(te); yv_true=dataset['Y'][va,TARGET_SLICE[target]]; yt_true=dataset['Y'][te,TARGET_SLICE[target]]
    if target=='gyro':
        val_metric=float(np.sqrt(np.mean(np.sum((pv-yv_true)**2,axis=1)))); test_metric=float(np.sqrt(np.mean(np.sum((pt-yt_true)**2,axis=1)))); metric='rmse_rad_s'
    else:
        val_metric=_angular_rmse(pv,yv_true); test_metric=_angular_rmse(pt,yt_true); metric='angular_rmse_deg'
    metrics={'architecture':arch,'variant':variant,'target_sensor':target,'n_features':int(X.shape[-1]),'window':window,'parameters':int(sum(p.numel() for p in model.parameters())),'train_samples':int(len(tr)),'validation_samples':int(len(va)),'test_samples':int(len(te)),'best_val_loss':bestv,f'validation_{metric}':val_metric,f'test_{metric}':test_metric,
             'train_split_hash':man['train_hash'],'validation_split_hash':man['validation_hash'],'test_split_hash':man['test_hash'],'fairness_note':'same shared trajectories, run-level split, fault realizations and sample indices for all architectures within each variant'}
    torch.save({'state_dict':model.state_dict(),'architecture':arch,'variant':variant,'target':target,'window':window,'n_features':int(X.shape[-1]),'xmu':scaler.xmu,'xsd':scaler.xsd,'ymu':scaler.ymu,'ysd':scaler.ysd},out/'model.pt')
    pd.DataFrame(hist).to_csv(out/'learning_curve.csv',index=False); (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return FairSensorPredictor(model,variant,target,window,scaler),metrics,scaler


class FairSensorPredictor:
    def __init__(self,model,variant,target,window,scaler):
        self.model=model.eval(); self.variant=str(variant); self.target=str(target); self.window=int(window); self.scaler=SharedScaler(np.asarray(scaler.xmu),np.asarray(scaler.xsd),np.asarray(scaler.ymu),np.asarray(scaler.ysd)); self.history=deque(maxlen=self.window); self.context=None; self.baseline=None; self.index=1
    def clone(self):
        q=FairSensorPredictor(copy.deepcopy(self.model),self.variant,self.target,self.window,copy.deepcopy(self.scaler));
        if self.context is not None: q.context=self.context.copy()
        if self.baseline is not None: q.baseline=self.baseline.copy()
        q.index=self.index; return q
    def reset(self): self.history.clear(); self.index=1
    def set_runtime_context(self,context,baseline): self.context=np.asarray(context,np.float32); self.baseline=np.asarray(baseline,np.float32); self.index=1
    def predict(self,gyro,sun,mag,faulty):
        vals=[np.asarray(gyro,float).copy(),np.asarray(sun,float).copy(),np.asarray(mag,float).copy()]; masks=np.ones(3,np.float32)
        for i,bad in enumerate(faulty):
            if bad or not np.all(np.isfinite(vals[i])): vals[i]=np.zeros(3); masks[i]=0.
        sensor=np.r_[vals[0],vals[1],vals[2],masks].astype(np.float32)
        if self.variant=='sensor_only': row=sensor
        else:
            if self.context is None: raise RuntimeError('orbital-context predictor requires set_runtime_context()')
            j=min(self.index,len(self.context)-1); row=np.r_[sensor,self.context[j]].astype(np.float32)
        self.history.append(row)
        while len(self.history)<self.window: self.history.appendleft(row.copy())
        x=np.asarray(self.history,np.float32)[None,:,:]; x=(x-self.scaler.xmu)/self.scaler.xsd
        with torch.no_grad(): y=self.model(torch.from_numpy(x)).numpy()*self.scaler.ysd+self.scaler.ymu
        v=np.asarray(y[0],float)
        if self.variant=='physics_residual':
            j=min(self.index,len(self.baseline)-1); v=v+self.baseline[j,TARGET_SLICE[self.target]]
        if self.target in ('sun','mag'): v=_unit(v)
        self.index+=1; return v


class FairPredictorBundle:
    def __init__(self,architecture,variant,predictors,metrics): self.architecture=architecture; self.variant=variant; self.predictors=dict(predictors); self.metrics=metrics
    def clone(self): return FairPredictorBundle(self.architecture,self.variant,{k:v.clone() for k,v in self.predictors.items()},copy.deepcopy(self.metrics))
    def reset(self):
        for p in self.predictors.values(): p.reset()
    def set_runtime_context(self,context,baseline):
        for p in self.predictors.values(): p.set_runtime_context(context,baseline)
    def predict_measurements(self,gyro,sun,mag,faulty=(False,False,False)):
        # Each target predictor sees the exact same current measurement row and masks.
        return tuple(self.predictors[t].predict(gyro,sun,mag,faulty) for t in TARGETS)


def train_fair_model_matrix(dataset,out_dir,cfg):
    """Train 3 architectures x 3 variants x 3 target sensors = 27 models.

    Scalers are fitted once per (variant,target) and reused across MLP/CNN/GRU,
    guaranteeing identical normalization in architecture comparisons.
    """
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    architectures=[str(x).lower() for x in cfg.get('architectures',ARCHITECTURES)]; variants=[str(x).lower() for x in cfg.get('variants',VARIANTS)]; targets=[str(x).lower() for x in cfg.get('targets',TARGETS)]
    training=cfg.get('training',{}); arch_hps=cfg.get('hyperparameters',{})
    bundles={}; rows=[]; scalers={}
    # Pre-fit and freeze scalers per variant/target from the same train split.
    for variant in variants:
        for target in targets:
            X,Y=prepare_variant_target(dataset,variant,target); tr=_indices_for_split(dataset['RUN'],dataset['manifest']['train_run_ids']); scalers[(variant,target)]=SharedScaler(*_fit_x_scaler(X[tr]),*_fit_y_scaler(Y[tr]))
    for arch in architectures:
        for variant in variants:
            preds={}; mets={}
            for target in targets:
                p,m,_=train_one_fair_model(arch,variant,target,dataset,out/arch/variant/target,training,arch_hps.get(arch,{}),shared_scaler=scalers[(variant,target)])
                preds[target]=p; mets[target]=m; rows.append(m)
            bundles[(arch,variant)]=FairPredictorBundle(arch,variant,preds,mets)
    metrics=pd.DataFrame(rows); metrics.to_csv(out/'model_validation_test_resources.csv',index=False)
    return bundles,metrics
