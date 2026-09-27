from pathlib import Path
import json, numpy as np, pandas as pd
from src.simulation.attitude import propagate_attitude
from src.simulation.sensors import simulate_sensors
from src.simulation.faults import apply_fault, FAULT_NAMES
from src.estimation.mekf import run_mekf

FAULTS=['healthy','gyro_bias','gyro_drift','gyro_noise','sun_noise','sun_dropout','sun_loss']
FEATURES=['gyro_res_x','gyro_res_y','gyro_res_z','sun_innovation_norm','mag_innovation_norm','sun_available','mag_available']

def _unit_quaternion(rng):
    q=rng.normal(size=4); return q/np.linalg.norm(q)

def generate_monte_carlo_dataset(out_dir='data/results/phase6/dataset', runs_per_class=12, duration_s=300., dt=.1, seed=20260927, force=False):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); cache=out/'runs.npz'; manifest_path=out/'run_manifest.csv'
    if cache.exists() and manifest_path.exists() and not force:
        z=np.load(cache); return z['X'],z['y'],z['run_id'],pd.read_csv(manifest_path)
    rng=np.random.default_rng(seed); frames=[]; manifest=[]; run_id=0
    for class_id,fault in enumerate(FAULTS):
        for rep in range(runs_per_class):
            local_seed=int(rng.integers(1,2_000_000_000)); rr=np.random.default_rng(local_seed)
            times=np.arange(0,duration_s+dt/2,dt)
            q0=_unit_quaternion(rr); omega0=rr.uniform(-.025,.025,3)
            truth=propagate_attitude(times,q0=q0,omega0=omega0)
            gyro_bias=rr.normal(0,2e-4,3)
            sensors=simulate_sensors(times,truth,seed=local_seed,gyro_bias=gyro_bias,
                                     gyro_noise_std=float(rr.uniform(4e-5,7e-5)),vector_noise_std=float(rr.uniform(1.5e-3,3e-3)))
            if fault=='healthy': start,end,severity=duration_s+1,duration_s+2,'medium'
            else:
                start=float(rr.uniform(.25*duration_s,.40*duration_s)); end=float(rr.uniform(.72*duration_s,.90*duration_s)); severity=str(rr.choice(['low','medium','high']))
            faulty,labels=apply_fault(sensors,times,fault,severity,start,end,seed=local_seed+1)
            qhat,bhat,d=run_mekf(times,faulty['gyro'],faulty['sun'],faulty['mag'],faulty['sun_i'],faulty['mag_i'],return_diagnostics=True)
            gyro_res=faulty['gyro']-truth[:,4:]-bhat
            sun_res=np.linalg.norm(d['sun_innovation'],axis=1); mag_res=np.linalg.norm(d['mag_innovation'],axis=1)
            feat=np.column_stack([gyro_res,sun_res,mag_res,d['sun_available'].astype(float),d['mag_available'].astype(float)])
            feat[~np.isfinite(feat)]=0.0
            frames.append((feat.astype(np.float32),labels.astype(np.int16),np.full(len(times),run_id,np.int32)))
            manifest.append(dict(run_id=run_id,class_id=class_id,fault=fault,replicate=rep,seed=local_seed,severity=severity,
                                 fault_start_s=start if fault!='healthy' else np.nan,fault_end_s=end if fault!='healthy' else np.nan,
                                 q0_w=q0[0],q0_x=q0[1],q0_y=q0[2],q0_z=q0[3],omega0_x=omega0[0],omega0_y=omega0[1],omega0_z=omega0[2]))
            run_id+=1
    X=np.concatenate([a for a,_,_ in frames]); y=np.concatenate([b for _,b,_ in frames]); rid=np.concatenate([c for _,_,c in frames])
    np.savez_compressed(cache,X=X,y=y,run_id=rid); m=pd.DataFrame(manifest); m.to_csv(manifest_path,index=False)
    (out/'metadata.json').write_text(json.dumps({'features':FEATURES,'fault_names':FAULT_NAMES,'runs_per_class':runs_per_class,'duration_s':duration_s,'dt_s':dt,'seed':seed},indent=2))
    return X,y,rid,m

def split_runs(manifest,seed=20260927):
    rng=np.random.default_rng(seed); split={}
    for fault,g in manifest.groupby('fault'):
        ids=g.run_id.to_numpy().copy(); rng.shuffle(ids); n=len(ids); nt=max(1,int(round(.20*n))); nv=max(1,int(round(.20*n)))
        if n<5: raise ValueError('Use at least 5 runs per class for run-level train/validation/test splitting.')
        for r in ids[:nt]: split[int(r)]='test'
        for r in ids[nt:nt+nv]: split[int(r)]='val'
        for r in ids[nt+nv:]: split[int(r)]='train'
    return split

def make_windows(X,y,run_id,window,selected_runs,stride=5):
    XX=[]; yy=[]; rr=[]; ee=[]
    for r in selected_runs:
        idx=np.flatnonzero(run_id==r); xr=X[idx]; yr=y[idx]
        for k in range(window-1,len(idx),stride):
            XX.append(xr[k-window+1:k+1]); yy.append(yr[k]); rr.append(r); ee.append(k)
    return np.asarray(XX,np.float32),np.asarray(yy,np.int64),np.asarray(rr,np.int32),np.asarray(ee,np.int32)
