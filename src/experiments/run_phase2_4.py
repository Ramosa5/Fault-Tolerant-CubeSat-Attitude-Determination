from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.simulation.attitude import propagate_attitude, rotational_energy, angular_momentum_norm, quat_angle_error_deg
from src.simulation.sensors import simulate_sensors
from src.estimation.mekf import run_mekf

def main():
    out=Path('data/results'); plots=Path('plots'); out.mkdir(parents=True,exist_ok=True); plots.mkdir(exist_ok=True)
    dt=0.1; times=np.arange(0,600+dt,dt); inertia=(0.020,0.025,0.030)
    truth=propagate_attitude(times,inertia_diag=inertia)
    E=rotational_energy(truth[:,4:],inertia); H=angular_momentum_norm(truth[:,4:],inertia)
    sensors=simulate_sensors(times,truth)
    qhat,bhat=run_mekf(times,sensors['gyro'],sensors['sun'],sensors['mag'],sensors['sun_i'],sensors['mag_i'])
    err=np.array([quat_angle_error_deg(qhat[k],truth[k,:4]) for k in range(len(times))])
    df=pd.DataFrame({'time_s':times,'qw_true':truth[:,0],'qx_true':truth[:,1],'qy_true':truth[:,2],'qz_true':truth[:,3],
        'wx_true':truth[:,4],'wy_true':truth[:,5],'wz_true':truth[:,6],
        'gyro_x':sensors['gyro'][:,0],'gyro_y':sensors['gyro'][:,1],'gyro_z':sensors['gyro'][:,2],
        'qw_est':qhat[:,0],'qx_est':qhat[:,1],'qy_est':qhat[:,2],'qz_est':qhat[:,3],
        'bias_x_est':bhat[:,0],'bias_y_est':bhat[:,1],'bias_z_est':bhat[:,2],'attitude_error_deg':err})
    df.to_csv(out/'phase2_4_timeseries.csv',index=False)
    summary=pd.DataFrame([{'duration_s':times[-1],'dt_s':dt,'energy_rel_drift':(E.max()-E.min())/E.mean(),
      'momentum_rel_drift':(H.max()-H.min())/H.mean(),'attitude_rmse_deg':np.sqrt(np.mean(err**2)),
      'attitude_max_deg':err.max(),'attitude_final_deg':err[-1],
      'bias_error_final_rad_s':np.linalg.norm(bhat[-1]-sensors['gyro_bias_true'])}])
    summary.to_csv(out/'phase2_4_summary.csv',index=False)
    plt.figure(figsize=(8,4)); plt.plot(times,err); plt.xlabel('Time [s]'); plt.ylabel('Attitude error [deg]'); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(plots/'phase4_attitude_error.png',dpi=180); plt.close()
    plt.figure(figsize=(8,4)); plt.plot(times,truth[:,4:]); plt.xlabel('Time [s]'); plt.ylabel('Angular rate [rad/s]'); plt.legend(['wx','wy','wz']); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(plots/'phase2_angular_rates.png',dpi=180); plt.close()
    print('\nPHASE 2-4 RESULTS'); print(summary.to_string(index=False)); print('\nSaved:',out/'phase2_4_summary.csv')
if __name__=='__main__': main()
