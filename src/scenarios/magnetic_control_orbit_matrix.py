from __future__ import annotations
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd

from src.simulation.control import sun_pointing_initial_quaternion
from src.scenarios.magnetic_sun_pointing_comparison import _make_environment, _run_branch
from src.simulation.adcs_modes import SUN_POINTING
from src.visualization.publication import PALETTE, create_subplots, finalize_figure
from src.visualization.animation import create_orbit_attitude_animation


def _slug(text):
    return str(text).lower().replace(' ', '_').replace('|', '_').replace('/', '_').replace('\\', '_').replace('__','_').strip('_')

def _animation_case_enabled(profile, controller, anim_cfg):
    cases = anim_cfg.get('cases', ['all'])
    if isinstance(cases, str):
        cases = [cases]
    norm = {str(x).strip().lower() for x in cases}
    if 'all' in norm or '*' in norm:
        return True
    keys = {
        f'{profile}|{controller}'.lower(),
        f'{profile}:{controller}'.lower(),
        f'{profile}_{controller}'.lower(),
    }
    return bool(norm & keys)


def _settling_time_with_rate(t, point_deg, rate_deg_s, point_thr, rate_thr, hold_s):
    dt=float(t[1]-t[0]); need=max(1,int(np.ceil(float(hold_s)/dt))); count=0
    for k in range(len(t)):
        if point_deg[k] <= point_thr and rate_deg_s[k] <= rate_thr:
            count += 1
            if count >= need:
                return float(t[k-need+1])
        else:
            count=0
    return np.nan


def run(cfg, global_cfg, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); plots=out/'plots'; plots.mkdir(exist_ok=True)
    duration=float(cfg.get('duration_s',6000.0)); dt=float(cfg.get('dt_s',0.1)); t=np.arange(0,duration+dt/2,dt)
    controllers=list(cfg.get('controllers',['projected_pd','magnetic_ltv']))
    profiles=list(cfg.get('orbit_profiles',['iss_like','high_beta_sunlit']))
    sc=global_cfg['sensors']; seed=int(cfg.get('comparison_seed',9107)); rng=np.random.default_rng(seed); n=len(t)
    noise={'gyro':rng.normal(0,float(sc['gyro_noise_std_rad_s']),(n,3)),
           'sun':rng.normal(0,float(sc['vector_noise_std']),(n,3)),
           'mag':rng.normal(0,float(sc['vector_noise_std']),(n,3))}
    rows=[]; ts=[]; run_data={}
    for profile in profiles:
        for controller in controllers:
            cc=copy.deepcopy(cfg)
            cc['orbit_profile']=profile
            cc.setdefault('controller',{})['type']=controller
            # Matrix is a healthy-controller/orbit diagnostic, not a fault/ML experiment.
            cc['fault_start_s']=duration+1.0; cc['fault_end_s']=duration+2.0
            sun,sun_pos,states,eclipse_state,illum,B_i,orbit_meta=_make_environment(t,cc,global_cfg)
            axis=str(cc.get('pointing_axis','x'))
            q0=sun_pointing_initial_quaternion(axis,sun[0],float(cc.get('initial_error_deg',30.0)))
            omega0=np.asarray(cc.get('omega0_rad_s',[0,0,0]),float)
            r=_run_branch('healthy_mekf',cc,global_cfg,t,q0,omega0,sun,B_i,states,eclipse_state,noise,predictor=None)
            rate=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)); eclipse=eclipse_state==2; sunlight=~eclipse
            modes=cc.get('modes',{}); pthr=float(modes.get('pointing_threshold_deg',2.0)); rthr=float(modes.get('rate_threshold_deg_s',0.08)); hold=float(modes.get('pointing_hold_s',20.0))
            label=f'{profile} | {controller}'
            run_data[label]=(r,orbit_meta,eclipse_state,states,sun,B_i)
            rows.append({
                'orbit_profile':profile,'controller_type':controller,'beta_angle_deg':orbit_meta['beta_angle_deg'],
                'eclipse_fraction_pct':float(100*np.mean(eclipse)),'spacecraft_mass_kg':float(cc['spacecraft']['mass_kg']),
                'inertia_x_kg_m2':float(r['inertia'][0]),'inertia_y_kg_m2':float(r['inertia'][1]),'inertia_z_kg_m2':float(r['inertia'][2]),
                'minimum_pointing_error_deg':float(np.min(r['true_point'])),'final_pointing_error_deg':float(r['true_point'][-1]),
                'sunlight_pointing_rmse_deg':float(np.sqrt(np.mean(r['true_point'][sunlight]**2))) if np.any(sunlight) else np.nan,
                'eclipse_pointing_rmse_deg':float(np.sqrt(np.mean(r['true_point'][eclipse]**2))) if np.any(eclipse) else np.nan,
                'settled_acquisition_time_s':_settling_time_with_rate(t,r['true_point'],rate,pthr,rthr,hold),
                'sun_pointing_mode_fraction_pct':float(100*np.mean(r['mode']==SUN_POINTING)),
                'hold_duration_s':float(np.sum(r['mode']==SUN_POINTING)*dt),
                'hold_pointing_rmse_deg':float(np.sqrt(np.mean(r['true_point'][r['mode']==SUN_POINTING]**2))) if np.any(r['mode']==SUN_POINTING) else np.nan,
                'hold_max_pointing_error_deg':float(np.max(r['true_point'][r['mode']==SUN_POINTING])) if np.any(r['mode']==SUN_POINTING) else np.nan,
                'hold_mean_body_rate_deg_s':float(np.mean(rate[r['mode']==SUN_POINTING])) if np.any(r['mode']==SUN_POINTING) else np.nan,
                'final_body_rate_deg_s':float(rate[-1]),'max_body_rate_deg_s':float(np.max(rate)),
                'max_dipole_used_am2':float(np.max(np.linalg.norm(r['dip_actual'],axis=1))),
                'max_magnetic_torque_uNm':float(np.max(np.linalg.norm(r['tau_mtq'],axis=1))*1e6),
            })
            ts.append(pd.DataFrame({'time_s':t,'orbit_profile':profile,'controller_type':controller,
                'pointing_error_deg':r['true_point'],'attitude_estimation_error_deg':r['att_err'],'body_rate_deg_s':rate,
                'adcs_mode':r['mode'],'eclipse_state':eclipse_state,'illumination_fraction':illum,
                'dipole_norm_Am2':np.linalg.norm(r['dip_actual'],axis=1),'magnetic_torque_uNm':np.linalg.norm(r['tau_mtq'],axis=1)*1e6}))
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'magnetic_control_orbit_matrix_summary.csv',index=False)
    pd.concat(ts,ignore_index=True).to_csv(out/'magnetic_control_orbit_matrix_timeseries.csv',index=False)
    (out/'scenario_metadata.json').write_text(json.dumps({'scenario':'magnetic_control_orbit_matrix','controllers':controllers,'orbit_profiles':profiles,
        'purpose':'Controller/orbit isolation experiment using identical sensor noise and a healthy MEKF branch.'},indent=2),encoding='utf-8')

    formats=tuple(global_cfg['visualization']['formats']); dpi=int(global_cfg['visualization']['dpi'])
    colors=[PALETTE['blue_main'],PALETTE['green_3'],PALETTE['violet'],PALETTE['red_strong']]
    fig,axes=create_subplots(2,1,figsize=(10.5,8.0),sharex=True)
    for (label,(r,om,es,states,sun,B)),c in zip(run_data.items(),colors):
        axes[0].plot(t,r['true_point'],label=label,color=c)
        axes[1].plot(t,np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),label=label,color=c)
    axes[0].set_ylabel('Sun-pointing error [deg]'); axes[0].set_title('2×2 magnetic AOCS controller/orbit comparison'); axes[0].legend(fontsize=8,ncol=2)
    axes[1].set_ylabel('Body rate [deg/s]'); axes[1].set_xlabel('Time [s]')
    finalize_figure(fig,plots/'matrix_pointing_error_and_body_rate',formats=formats,dpi=dpi)

    fig,axes=create_subplots(1,2,figsize=(11,4.8)); ax0,ax1=axes
    labels=[f"{a}\n{b}" for a,b in zip(summary.orbit_profile,summary.controller_type)]
    x=np.arange(len(labels))
    ax0.bar(x,summary.sunlight_pointing_rmse_deg); ax0.set_xticks(x,labels,rotation=25,ha='right'); ax0.set_ylabel('Sunlight pointing RMSE [deg]'); ax0.set_title('Pointing performance')
    ax1.bar(x,summary.eclipse_fraction_pct); ax1.set_xticks(x,labels,rotation=25,ha='right'); ax1.set_ylabel('Eclipse fraction [%]'); ax1.set_title('Orbit illumination')
    finalize_figure(fig,plots/'matrix_pointing_rmse_and_eclipse_fraction',formats=formats,dpi=dpi)

    fig,axes=create_subplots(1,2,figsize=(11,4.8)); ax0,ax1=axes
    ax0.bar(x,summary.hold_duration_s); ax0.set_xticks(x,labels,rotation=25,ha='right'); ax0.set_ylabel('SUN_POINTING hold duration [s]'); ax0.set_title('Sustained hold duration')
    ax1.bar(x,summary.hold_pointing_rmse_deg); ax1.set_xticks(x,labels,rotation=25,ha='right'); ax1.set_ylabel('Hold pointing RMSE [deg]'); ax1.set_title('Pointing quality while in hold')
    finalize_figure(fig,plots/'matrix_sun_hold_duration_and_rmse',formats=formats,dpi=dpi)

    # Optional lightweight 3D animation for each controller/orbit combination.
    ac=global_cfg.get('animation',{})
    sac=cfg.get('animation',{})
    if bool(ac.get('enabled',False)) and bool(sac.get('enabled',True)):
        ad=out/'animations'; ad.mkdir(exist_ok=True)
        base_kwargs={k:v for k,v in ac.items() if k not in ('enabled','phase4','phase5')}
        axis=str(cfg.get('pointing_axis','x'))
        epoch=str(cfg.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00'))
        for label,(r,om,es,states,sun,B_i) in run_data.items():
            profile,controller=[x.strip() for x in label.split('|',1)]
            if not _animation_case_enabled(profile,controller,sac):
                continue
            body_rate=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1))
            title=f'{profile} + {controller} — magnetic AOCS controller/orbit matrix'
            filename=f'matrix_{_slug(profile)}_{_slug(controller)}_orbit_attitude.html'
            create_orbit_attitude_animation(
                ad/filename,t,r['truth'][:,:4],r['qhat'],
                altitude_m=float(om['altitude_m']),inclination_deg=float(om['inclination_deg']),
                raan_deg=float(om['raan_deg']),phase_deg=float(om.get('phase_deg',0.0)),
                body_pointing_axis=axis,sun_i=sun,mag_i=B_i,show_sun_pointing_error=True,
                eclipse_mask=np.asarray(es)==2,position_eci_m=states[:,:3],velocity_eci_mps=states[:,3:],
                earth_epoch_utc=epoch,mode_series=r['mode'],body_rate_deg_s=body_rate,
                fault_name='none',title=title,**base_kwargs
            )

    return {'scenario':'magnetic_control_orbit_matrix','cases':len(summary),'summary_file':str(out/'magnetic_control_orbit_matrix_summary.csv')}
