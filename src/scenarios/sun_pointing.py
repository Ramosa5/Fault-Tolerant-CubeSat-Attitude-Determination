from pathlib import Path
import json
import numpy as np, pandas as pd
from src.simulation.attitude import propagate_attitude_controlled, quat_angle_error_deg
from src.simulation.control import sun_pointing_initial_quaternion, sun_pointing_torque, sun_pointing_error_deg
from src.simulation.sensors import simulate_sensors
from src.estimation.mekf import run_mekf
from src.visualization.publication import PALETTE, create_subplots, finalize_figure
from src.visualization.animation import create_orbit_attitude_animation

SUN_DEFAULT=np.array([1.0,0.2,0.1],float); SUN_DEFAULT/=np.linalg.norm(SUN_DEFAULT)

def _settling_time(t,err,threshold,hold_s):
    dt=float(np.median(np.diff(t))) if len(t)>1 else 1.0; n=max(1,int(np.ceil(hold_s/dt)))
    good=np.asarray(err)<=threshold
    for i in range(0,len(t)-n+1):
        if np.all(good[i:i+n]): return float(t[i])
    return np.nan

def run(cfg, global_cfg, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); plots=out/'plots'; plots.mkdir(exist_ok=True)
    duration=float(cfg.get('duration_s',600.0)); dt=float(cfg.get('dt_s',0.1)); t=np.arange(0,duration+0.5*dt,dt)
    axis=str(cfg.get('pointing_axis','x')); sun=np.asarray(cfg.get('sun_vector_eci',SUN_DEFAULT.tolist()),float); sun/=np.linalg.norm(sun)
    q0=sun_pointing_initial_quaternion(axis,sun,float(cfg.get('initial_error_deg',30.0)))
    omega0=np.asarray(cfg.get('omega0_rad_s',[0.0,0.0,0.0]),float); inertia=cfg.get('inertia_kg_m2',global_cfg['attitude']['inertia_kg_m2'])
    kp=float(cfg.get('kp',0.002)); kd=float(cfg.get('kd',0.01)); max_tau=cfg.get('max_torque_nm',0.002)
    cb=lambda tt,q,w: sun_pointing_torque(q,w,axis,sun,kp,kd,max_tau)
    truth,tau=propagate_attitude_controlled(t,q0,omega0,inertia,cb)
    sc=global_cfg['sensors']; sens=simulate_sensors(t,truth,seed=int(sc['seed']),gyro_bias=sc['gyro_bias_rad_s'],gyro_noise_std=sc['gyro_noise_std_rad_s'],vector_noise_std=sc['vector_noise_std'])
    # Override the sensor-model Sun reference only if scenario config differs from default.
    if np.linalg.norm(sun-sens['sun_i'])>1e-12:
        # Rebuild Sun body measurements consistently with selected scenario Sun vector.
        rng=np.random.default_rng(int(sc['seed'])+991); from src.simulation.attitude import quat_to_dcm
        sb=np.zeros((len(t),3))
        for k,q in enumerate(truth[:,:4]):
            v=quat_to_dcm(q).T@sun+rng.normal(0,float(sc['vector_noise_std']),3); sb[k]=v/np.linalg.norm(v)
        sens['sun']=sb; sens['sun_i']=sun
    qest,best=run_mekf(t,sens['gyro'],sens['sun'],sens['mag'],sens['sun_i'],sens['mag_i'],q0=q0,gyro_noise_std=float(sc['gyro_noise_std_rad_s']),vector_noise_std=float(sc['vector_noise_std']))
    true_err=np.array([sun_pointing_error_deg(q,axis,sun) for q in truth[:,:4]])
    est_err=np.array([sun_pointing_error_deg(q,axis,sun) for q in qest])
    attitude_err=np.array([quat_angle_error_deg(qe,qt) for qe,qt in zip(qest,truth[:,:4])])
    df=pd.DataFrame({'time_s':t,'true_sun_pointing_error_deg':true_err,'estimated_sun_pointing_error_deg':est_err,'attitude_estimation_error_deg':attitude_err,
                     'omega_x_rad_s':truth[:,4],'omega_y_rad_s':truth[:,5],'omega_z_rad_s':truth[:,6],
                     'torque_x_nm':tau[:,0],'torque_y_nm':tau[:,1],'torque_z_nm':tau[:,2]})
    df.to_csv(out/'sun_pointing_timeseries.csv',index=False)
    th=float(cfg.get('settling_threshold_deg',1.0)); hold=float(cfg.get('settling_duration_s',10.0))
    summary={'scenario':'sun_pointing','controller_feedback':'truth','pointing_axis':axis,'initial_error_deg':float(true_err[0]),'final_true_error_deg':float(true_err[-1]),
             'rmse_true_error_deg':float(np.sqrt(np.mean(true_err**2))),'max_true_error_deg':float(np.max(true_err)),'settling_threshold_deg':th,
             'settling_time_s':_settling_time(t,true_err,th,hold),'max_commanded_torque_nm':float(np.max(np.linalg.norm(tau,axis=1))),
             'attitude_estimation_rmse_deg':float(np.sqrt(np.mean(attitude_err**2)))}
    pd.DataFrame([summary]).to_csv(out/'sun_pointing_summary.csv',index=False); (out/'scenario_metadata.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # plots
    fig,axes=create_subplots(figsize=(8.8,5.0)); ax=axes[0]; ax.plot(t,true_err,label='True pointing error',color=PALETTE['blue_main']); ax.plot(t,est_err,label='Estimated pointing error',color=PALETTE['red_strong'],ls='--'); ax.axhline(th,color=PALETTE['neutral'],ls=':',label=f'{th:g}° threshold'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Sun-pointing error [deg]'); ax.set_title('Sun-pointing acquisition and tracking'); ax.legend(); finalize_figure(fig,plots/'sun_pointing_true_and_estimated_error_vs_time',formats=tuple(global_cfg['visualization']['formats']),dpi=int(global_cfg['visualization']['dpi']))
    fig,axes=create_subplots(2,1,figsize=(9.0,7.2),sharex=True); axes[0].plot(t,truth[:,4:]); axes[0].set_ylabel('Body rate [rad/s]'); axes[0].set_title('Sun-pointing body rates and control torque'); axes[1].plot(t,tau); axes[1].set_ylabel('Torque [N m]'); axes[1].set_xlabel('Time [s]'); finalize_figure(fig,plots/'sun_pointing_body_rates_and_control_torque',formats=tuple(global_cfg['visualization']['formats']),dpi=int(global_cfg['visualization']['dpi']))
    # animation
    ac=global_cfg.get('animation',{}); sac=cfg.get('animation',{})
    if bool(ac.get('enabled',False)) and bool(sac.get('enabled',True)):
        anim=out/'animations'; anim.mkdir(exist_ok=True); oc=global_cfg['orbit']
        kwargs={k:v for k,v in ac.items() if k not in ('enabled','phase4','phase5')}
        kwargs['body_pointing_axis']=axis; kwargs['sun_i']=sun; kwargs['mag_i']=sens['mag_i']; kwargs['show_sun_pointing_error']=True
        create_orbit_attitude_animation(anim/'sun_pointing_true_vs_estimated_orbit_attitude.html',t,truth[:,:4],qest,altitude_m=oc['altitude_m'],inclination_deg=oc['inclination_deg'],raan_deg=oc['raan_deg'],phase_deg=oc['phase_deg'],title='Sun-pointing scenario: true and estimated attitude',**kwargs)
    return summary
