from __future__ import annotations
from pathlib import Path
from datetime import datetime
import csv, json, shutil, subprocess, sys, time, traceback, tomllib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.simulation.constants import MU_EARTH,R_EARTH
from src.simulation.orbit_model import circular_orbit_period,propagate_circular_two_body
from src.verification.orekit_reference import propagate_orekit
from src.verification.metrics import comparison_metrics
from src.simulation.attitude import propagate_attitude,rotational_energy,angular_momentum_norm,quat_angle_error_deg
from src.simulation.sensors import simulate_sensors
from src.simulation.faults import apply_fault,FAULT_NAMES
from src.estimation.mekf import run_mekf
from src.ml.dataset import generate_monte_carlo_dataset,split_runs,make_windows,FAULTS
from src.ml.training import train_one
from src.experiments.run_phase6 import detection_delays

CONFIG=ROOT/'experiment_config.toml'

def mkdir(p): p.mkdir(parents=True,exist_ok=True); return p

def parse_scenarios(items):
    out=[]
    for s in items:
        a=s.split(':',1); out.append((a[0],a[1] if len(a)>1 else 'medium'))
    return out

def run_tests(out):
    p=subprocess.run([sys.executable,'-m','pytest','-v'],cwd=ROOT,text=True,capture_output=True)
    (out/'pytest_output.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    if p.returncode: raise RuntimeError('Unit tests failed; see pytest_output.txt')

def orbit_sanity(cfg,out):
    c=cfg['orbit']; period=circular_orbit_period(c['altitude_m']); n=int(c['sanity_orbits']*c['samples_per_orbit'])+1
    t=np.linspace(0,c['sanity_orbits']*period,n); x=propagate_circular_two_body(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg'])
    r=np.linalg.norm(x[:,:3],axis=1); v=np.linalg.norm(x[:,3:],axis=1)
    df=pd.DataFrame({'time_s':t,'x_m':x[:,0],'y_m':x[:,1],'z_m':x[:,2],'vx_mps':x[:,3],'vy_mps':x[:,4],'vz_mps':x[:,5],'altitude_m':r-R_EARTH,'speed_mps':v})
    df.to_csv(out/'orbit_sanity_timeseries.csv',index=False)
    s=pd.DataFrame([{'period_s':period,'altitude_mean_m':np.mean(r-R_EARTH),'altitude_range_m':np.ptp(r-R_EARTH),'speed_mean_mps':np.mean(v),'speed_range_mps':np.ptp(v)}]); s.to_csv(out/'orbit_sanity_summary.csv',index=False)
    return s.iloc[0].to_dict()

def orekit_verify(cfg,out):
    rows=[]
    for c in cfg['orekit']['cases']:
        period=circular_orbit_period(c['altitude_m']); t=np.linspace(0,c['orbits']*period,int(c['orbits']*cfg['orbit']['samples_per_orbit'])+1)
        ours=propagate_circular_two_body(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg']); ref=propagate_orekit(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg'],MU_EARTH,R_EARTH)
        m,pe,ve=comparison_metrics(ours,ref); row={'test':c['name'],'altitude_km':c['altitude_m']/1000,'inclination_deg':c['inclination_deg'],'orbits':c['orbits'],'duration_s':t[-1],**m}; rows.append(row)
        pd.DataFrame({'time_s':t,'position_error_m':pe,'velocity_error_mps':ve}).to_csv(out/f"{c['name']}_timeseries.csv",index=False)
    df=pd.DataFrame(rows); df.to_csv(out/'verification_summary.csv',index=False); (out/'verification_summary.json').write_text(df.to_json(orient='records',indent=2))
    return {'cases':len(df),'max_position_error_m':float(df.position_max_m.max())}

def attitude_run(cfg,out,plots):
    a=cfg['attitude']; s=cfg['sensors']; dt=a['dt_s']; t=np.arange(0,a['duration_s']+dt/2,dt); inertia=tuple(a['inertia_kg_m2'])
    truth=propagate_attitude(t,q0=np.array(a['q0']),omega0=np.array(a['omega0_rad_s']),inertia_diag=inertia)
    E=rotational_energy(truth[:,4:],inertia); H=angular_momentum_norm(truth[:,4:],inertia)
    sensors=simulate_sensors(t,truth,seed=s['seed'],gyro_bias=np.array(s['gyro_bias_rad_s']),gyro_noise_std=s['gyro_noise_std_rad_s'],vector_noise_std=s['vector_noise_std'])
    q,b=run_mekf(t,sensors['gyro'],sensors['sun'],sensors['mag'],sensors['sun_i'],sensors['mag_i']); err=np.array([quat_angle_error_deg(q[k],truth[k,:4]) for k in range(len(t))])
    pd.DataFrame({'time_s':t,'attitude_error_deg':err,'qw_true':truth[:,0],'qx_true':truth[:,1],'qy_true':truth[:,2],'qz_true':truth[:,3],'qw_est':q[:,0],'qx_est':q[:,1],'qy_est':q[:,2],'qz_est':q[:,3]}).to_csv(out/'phase2_4_timeseries.csv',index=False)
    row={'duration_s':t[-1],'dt_s':dt,'energy_rel_drift':np.ptp(E)/np.mean(E),'momentum_rel_drift':np.ptp(H)/np.mean(H),'attitude_rmse_deg':np.sqrt(np.mean(err**2)),'attitude_max_deg':err.max(),'attitude_final_deg':err[-1],'bias_error_final_rad_s':np.linalg.norm(b[-1]-sensors['gyro_bias_true'])}
    pd.DataFrame([row]).to_csv(out/'phase2_4_summary.csv',index=False)
    plt.figure(figsize=(8,4)); plt.plot(t,err); plt.xlabel('Time [s]'); plt.ylabel('Attitude error [deg]'); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(plots/'phase4_attitude_error.png',dpi=180); plt.close()
    return row

def fault_run(cfg,out,plots):
    f=cfg['fault_campaign']; a=cfg['attitude']; s=cfg['sensors']; dt=f['dt_s']; t=np.arange(0,f['duration_s']+dt/2,dt)
    truth=propagate_attitude(t,q0=np.array(a['q0']),omega0=np.array(a['omega0_rad_s']),inertia_diag=tuple(a['inertia_kg_m2']))
    base=simulate_sensors(t,truth,seed=s['seed'],gyro_bias=np.array(s['gyro_bias_rad_s']),gyro_noise_std=s['gyro_noise_std_rad_s'],vector_noise_std=s['vector_noise_std'])
    rows=[]; frames=[]
    for rid,(fault,severity) in enumerate(parse_scenarios(f['scenarios'])):
        faulty,labels=apply_fault(base,t,fault,severity,f['fault_start_s'],f['fault_end_s'],seed=f['seed_base']+rid); q,b,d=run_mekf(t,faulty['gyro'],faulty['sun'],faulty['mag'],faulty['sun_i'],faulty['mag_i'],return_diagnostics=True)
        err=np.array([quat_angle_error_deg(q[k],truth[k,:4]) for k in range(len(t))]); gr=faulty['gyro']-truth[:,4:]-b; sr=np.linalg.norm(d['sun_innovation'],axis=1); mr=np.linalg.norm(d['mag_innovation'],axis=1); feat=np.column_stack([gr,sr,mr,d['sun_available'].astype(float),d['mag_available'].astype(float)]); feat[~np.isfinite(feat)]=0
        active=labels>0; rows.append({'run_id':rid,'fault':fault,'severity':severity,'fault_start_s':f['fault_start_s'],'fault_end_s':f['fault_end_s'],'attitude_rmse_all_deg':np.sqrt(np.mean(err**2)),'attitude_rmse_fault_deg':np.sqrt(np.mean(err[active]**2)) if active.any() else np.nan,'attitude_max_deg':err.max(),'sun_availability_fault_pct':100*np.mean(d['sun_available'][active]) if active.any() else 100.})
        frames.append(pd.DataFrame({'run_id':rid,'time_s':t,'fault_id':labels,'gyro_res_x':feat[:,0],'gyro_res_y':feat[:,1],'gyro_res_z':feat[:,2],'sun_innovation_norm':feat[:,3],'mag_innovation_norm':feat[:,4],'sun_available':feat[:,5],'mag_available':feat[:,6],'attitude_error_deg':err}))
    df=pd.DataFrame(rows); df.to_csv(out/'phase5_scenario_summary.csv',index=False); pd.concat(frames).to_csv(out/'phase5_sample_dataset.csv',index=False); return {'scenarios':len(df),'max_fault_rmse_deg':float(df.attitude_rmse_fault_deg.max())}

def ml_run(cfg,out,plots):
    m=cfg['ml']; data=mkdir(out/'dataset'); X,y,rid,manifest=generate_monte_carlo_dataset(data,m['runs_per_class'],duration_s=m['duration_s'],dt=m['dt_s'],seed=m['dataset_seed'],force=m['force_regenerate_dataset'])
    split=split_runs(manifest,seed=m['split_seed']); manifest['split']=manifest.run_id.map(split); manifest.to_csv(out/'split_manifest.csv',index=False); results=[]; delays=[]
    for w in m['window_lengths']:
        sets={}
        for sp,stride in [('train',m['train_stride']),('val',m['val_stride']),('test',m['test_stride'])]:
            ids=manifest.loc[manifest.split==sp,'run_id'].to_numpy(); sets[sp]=make_windows(X,y,rid,w,ids,stride=stride)
        Xtr,ytr,_,_=sets['train']; Xv,yv,_,_=sets['val']; Xte,yte,rte,ete=sets['test']
        for model in m['models']:
            modelout=mkdir(out/f'{model}_N{w}'); met,pred,cm=train_one(model,w,Xtr,ytr,Xv,yv,Xte,yte,len(FAULTS),epochs=m['epochs'],out_dir=modelout)
            d=detection_delays(yte,pred,rte,ete,manifest,m['dt_s']); d['model']=model; d['window']=w; delays.append(d); met.update({'model':model,'window':w,'train_windows':len(ytr),'val_windows':len(yv),'test_windows':len(yte),'mean_detection_delay_s':float(d.detection_delay_s.mean()) if len(d) else np.nan}); results.append(met)
    res=pd.DataFrame(results); res.to_csv(out/'phase6_model_comparison.csv',index=False); pd.concat(delays,ignore_index=True).to_csv(out/'phase6_detection_delays.csv',index=False)
    for metric,label in [('f1_macro','Macro F1'),('mean_detection_delay_s','Detection delay [s]')]:
        plt.figure(figsize=(8,5));
        for model,g in res.groupby('model'): plt.plot(g.window,g[metric],marker='o',label=model.upper())
        plt.xscale('log'); plt.xlabel('Window length N [samples]'); plt.ylabel(label); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(plots/f'{metric}_vs_window.png',dpi=180); plt.close()
    return {'experiments':len(res),'dataset_runs':len(manifest)}

def main():
    with open(CONFIG,'rb') as f: cfg=tomllib.load(f)
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'); run_dir=mkdir(ROOT/cfg['run']['results_root']/f'results_{stamp}'); shutil.copy2(CONFIG,run_dir/'config_used.toml')
    manifest={'run_name':cfg['run']['name'],'started_at':datetime.now().isoformat(),'python':sys.version,'config':str(CONFIG),'steps':{}}; (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[('unit_tests',run_tests),('orbit_sanity',orbit_sanity),('orekit_verification',orekit_verify),('attitude_sensors_ekf',attitude_run),('fault_campaign',fault_run),('ml_training',ml_run)]
    print(f'\nResults directory: {run_dir}\n')
    for name,fn in jobs:
        if not cfg['steps'].get(name,False): print(f'[SKIP] {name}'); manifest['steps'][name]={'status':'skipped'}; continue
        print(f'[RUN ] {name}'); t0=time.time(); phase=mkdir(run_dir/name); plots=mkdir(phase/'plots')
        try:
            if name=='unit_tests': result=fn(phase)
            elif name in ('orbit_sanity','orekit_verification'): result=fn(cfg,phase)
            else: result=fn(cfg,phase,plots)
            manifest['steps'][name]={'status':'completed','elapsed_s':time.time()-t0,'summary':result}; print(f'[ OK ] {name} ({time.time()-t0:.1f}s)')
        except Exception as e:
            (phase/'ERROR.txt').write_text(traceback.format_exc()); manifest['steps'][name]={'status':'failed','elapsed_s':time.time()-t0,'error':str(e)}; print(f'[FAIL] {name}: {e}')
            (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
            if cfg['run']['stop_on_error']: raise
    manifest['finished_at']=datetime.now().isoformat(); (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)); print(f'\nFinished. All outputs are in:\n{run_dir}')
if __name__=='__main__': main()
