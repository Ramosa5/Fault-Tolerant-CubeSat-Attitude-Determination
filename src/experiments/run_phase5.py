from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.simulation.attitude import propagate_attitude, quat_angle_error_deg
from src.simulation.sensors import simulate_sensors
from src.simulation.faults import apply_fault, FAULT_NAMES
from src.estimation.mekf import run_mekf

SCENARIOS=[
 ('healthy','medium'),
 ('gyro_bias','low'),('gyro_bias','medium'),('gyro_bias','high'),
 ('gyro_drift','medium'),('gyro_noise','medium'),('sun_noise','medium'),
 ('sun_dropout','low'),('sun_dropout','high'),('sun_loss','medium')]
WINDOWS=(1,5,10,20,50,100)

def _fill_nan_features(x):
    x=np.asarray(x,float).copy(); x[~np.isfinite(x)]=0.0; return x

def make_windows(features, labels, run_id, lengths=WINDOWS):
    """Save end-labelled windows without crossing run boundaries."""
    result={}
    for n in lengths:
        X=[]; y=[]; end=[]
        for k in range(n-1,len(features)):
            X.append(features[k-n+1:k+1]); y.append(labels[k]); end.append(k)
        result[n]=(np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int16),np.asarray(end,dtype=np.int32),run_id)
    return result

def main():
    out=Path('data/results/phase5'); plots=Path('plots/phase5'); out.mkdir(parents=True,exist_ok=True); plots.mkdir(parents=True,exist_ok=True)
    dt=0.1; times=np.arange(0,600+dt,dt); truth=propagate_attitude(times,inertia_diag=(.020,.025,.030)); base=simulate_sensors(times,truth)
    summaries=[]; all_frames=[]; window_store={n:{'X':[],'y':[],'run':[],'end':[]} for n in WINDOWS}
    for run_id,(fault,severity) in enumerate(SCENARIOS):
        sensors,labels=apply_fault(base,times,fault,severity,start_s=200,end_s=500,seed=100+run_id)
        qhat,bhat,d=run_mekf(times,sensors['gyro'],sensors['sun'],sensors['mag'],sensors['sun_i'],sensors['mag_i'],return_diagnostics=True)
        err=np.array([quat_angle_error_deg(qhat[k],truth[k,:4]) for k in range(len(times))])
        sun_res=np.linalg.norm(d['sun_innovation'],axis=1); mag_res=np.linalg.norm(d['mag_innovation'],axis=1)
        gyro_res=sensors['gyro']-truth[:,4:]-bhat
        features=np.column_stack([gyro_res,sun_res,mag_res,d['sun_available'].astype(float),d['mag_available'].astype(float)])
        features=_fill_nan_features(features)
        frame=pd.DataFrame({'run_id':run_id,'time_s':times,'fault_id':labels,'fault_name':[FAULT_NAMES[int(v)] for v in labels],
          'scenario_fault':fault,'severity':severity,'gyro_res_x':features[:,0],'gyro_res_y':features[:,1],'gyro_res_z':features[:,2],
          'sun_innovation_norm':features[:,3],'mag_innovation_norm':features[:,4],'sun_available':features[:,5],
          'mag_available':features[:,6],'attitude_error_deg':err})
        all_frames.append(frame)
        active=labels>0
        summaries.append({'run_id':run_id,'fault':fault,'severity':severity,'fault_start_s':200,'fault_end_s':500,
          'attitude_rmse_all_deg':float(np.sqrt(np.mean(err**2))),
          'attitude_rmse_fault_deg':float(np.sqrt(np.mean(err[active]**2))) if active.any() else np.nan,
          'attitude_max_deg':float(err.max()),'sun_availability_fault_pct':float(100*np.mean(d['sun_available'][active])) if active.any() else 100.0})
        for n,(X,y,end,_) in make_windows(features,labels,run_id).items():
            window_store[n]['X'].append(X); window_store[n]['y'].append(y); window_store[n]['end'].append(end); window_store[n]['run'].append(np.full(len(y),run_id,dtype=np.int16))
        if fault in ('healthy','gyro_bias','sun_dropout','sun_loss'):
            fig=plt.figure(figsize=(9,5)); plt.plot(times,err,label='attitude error'); plt.axvspan(200,500,alpha=.12,label='fault interval'); plt.xlabel('Time [s]'); plt.ylabel('Attitude error [deg]'); plt.legend(); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(plots/f'{run_id:02d}_{fault}_{severity}_attitude_error.png',dpi=180); plt.close(fig)
    summary=pd.DataFrame(summaries); summary.to_csv(out/'phase5_scenario_summary.csv',index=False)
    samples=pd.concat(all_frames,ignore_index=True); samples.to_csv(out/'phase5_sample_dataset.csv',index=False)
    for n,s in window_store.items():
        X=np.concatenate(s['X']); y=np.concatenate(s['y']); runs=np.concatenate(s['run']); ends=np.concatenate(s['end'])
        np.savez_compressed(out/f'windows_N{n}.npz',X=X,y=y,run_id=runs,end_index=ends)
    metadata={'feature_order':['gyro_res_x','gyro_res_y','gyro_res_z','sun_innovation_norm','mag_innovation_norm','sun_available','mag_available'],
              'window_lengths':list(WINDOWS),'fault_classes':FAULT_NAMES,'sampling_hz':1/dt,
              'split_rule':'Split by run_id/scenario; never randomly split overlapping windows.'}
    (out/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('\nPHASE 5 FAULT-INJECTION RESULTS'); print(summary.to_string(index=False)); print('\nSaved:',out/'phase5_scenario_summary.csv'); print('ML windows:',', '.join(f'N={n}' for n in WINDOWS))
if __name__=='__main__': main()
