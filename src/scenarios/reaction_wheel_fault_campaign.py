from __future__ import annotations
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd

from src.simulation.control import sun_pointing_initial_quaternion
from src.scenarios.reaction_wheel_sun_pointing_comparison import _make_environment, _run_branch
from src.scenarios.sun_pointing_comparison import BRANCHES, LABELS
from src.ml.virtual_sensor import train_virtual_sun_model
from src.visualization.publication import PALETTE, create_subplots, finalize_figure
from src.visualization.animation import create_orbit_attitude_animation


def _slug(s):
    return ''.join(c if c.isalnum() else '_' for c in str(s).lower()).strip('_')


def _fault_case_active_mask(t, case):
    return (t >= float(case.get('start_s', 0.0))) & (t <= float(case.get('end_s', t[-1])))


def _case_description(case):
    parts=[]
    for key in ('sun','gyro','mag'):
        cfg=case.get(key,{}) or {}
        typ=str(cfg.get('type','none'))
        if typ!='none': parts.append(f'{key}:{typ}')
    return ', '.join(parts) if parts else 'healthy sensors'


def run(cfg, global_cfg, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    plots=out/'plots'; plots.mkdir(exist_ok=True)
    animations=out/'animations'; animations.mkdir(exist_ok=True)
    models=out/'models'; models.mkdir(exist_ok=True)

    base=copy.deepcopy(global_cfg['scenarios']['reaction_wheel_sun_pointing_comparison'])
    for k,v in cfg.get('overrides',{}).items():
        base[k]=v
    duration=float(cfg.get('duration_s',base.get('duration_s',6000.0)))
    dt=float(cfg.get('dt_s',base.get('dt_s',0.1)))
    base['duration_s']=duration; base['dt_s']=dt
    if 'orbit_profile' in cfg: base['orbit_profile']=cfg['orbit_profile']
    if 'magnetic_field_model' in cfg: base.setdefault('magnetic_field',{})['model']=cfg['magnetic_field_model']
    t=np.arange(0,duration+dt/2,dt)

    sun,sun_pos,states,eclipse_state,illum,B_i,orbit_meta=_make_environment(t,base,global_cfg)
    axis=str(base.get('pointing_axis','x'))
    q0=sun_pointing_initial_quaternion(axis,sun[0],float(base.get('initial_error_deg',30.0)))
    omega0=np.asarray(base.get('omega0_rad_s',[0,0,0]),float)
    sc=global_cfg['sensors']; n=len(t)

    seed=int(cfg.get('seed',9201)); rng=np.random.default_rng(seed)
    noise={'gyro':rng.normal(0,float(sc['gyro_noise_std_rad_s']),(n,3)),
           'sun':rng.normal(0,float(sc['vector_noise_std']),(n,3)),
           'mag':rng.normal(0,float(sc['vector_noise_std']),(n,3))}

    vc=base.get('virtual_sensor',{})
    predictor,vm=train_virtual_sun_model(models/'virtual_sun_sensor',runs=int(vc.get('train_runs',24)),duration_s=float(vc.get('train_duration_s',40.0)),dt=dt,
        hidden=tuple(vc.get('hidden',[48,24])),epochs=int(vc.get('epochs',18)),batch_size=int(vc.get('batch_size',256)),seed=int(vc.get('seed',8128)),
        gyro_noise_std=float(sc['gyro_noise_std_rad_s']),vector_noise_std=float(sc['vector_noise_std']))

    cases=list(cfg.get('cases',[]))
    if not cases:
        raise ValueError('reaction_wheel_fault_campaign requires at least one [[...cases]] entry')

    summary_rows=[]; timeseries=[]; all_results={}
    animation_cases={str(x).lower() for x in cfg.get('animation_cases',['all'])}
    for ci,case0 in enumerate(cases):
        case=copy.deepcopy(case0)
        name=str(case.get('name',f'F{ci}'))
        mask=_fault_case_active_mask(t,case)
        results={}
        for branch in BRANCHES:
            results[branch]=_run_branch(branch,base,global_cfg,t,q0,omega0,sun,B_i,states,eclipse_state,noise,
                                        predictor if branch=='ml_recovery' else None,fault_case=case)
            r=results[branch]
            point=np.asarray(r['true_point']); att=np.asarray(r['att_err']); rate=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1))
            def rmse(x,m): return float(np.sqrt(np.mean(np.asarray(x)[m]**2))) if np.any(m) else np.nan
            summary_rows.append({
                'case_id':ci,'case_name':name,'fault_description':_case_description(case),'branch':branch,'label':LABELS[branch],
                'fault_start_s':float(case.get('start_s',0.0)),'fault_end_s':float(case.get('end_s',duration)),
                'pointing_rmse_all_deg':rmse(point,np.ones(n,bool)),'pointing_rmse_fault_deg':rmse(point,mask),
                'attitude_rmse_all_deg':rmse(att,np.ones(n,bool)),'attitude_rmse_fault_deg':rmse(att,mask),
                'pointing_max_fault_deg':float(np.max(point[mask])) if np.any(mask) else np.nan,
                'attitude_max_fault_deg':float(np.max(att[mask])) if np.any(mask) else np.nan,
                'time_over_1deg_fault_pct':float(100*np.mean(point[mask]>1.0)) if np.any(mask) else 0.0,
                'time_over_2deg_fault_pct':float(100*np.mean(point[mask]>2.0)) if np.any(mask) else 0.0,
                'time_over_5deg_fault_pct':float(100*np.mean(point[mask]>5.0)) if np.any(mask) else 0.0,
                'final_pointing_error_deg':float(point[-1]),'max_body_rate_deg_s':float(np.max(rate)),
                'ml_recovery_usage_pct_fault':float(100*np.mean(r['ml_used'][mask])) if np.any(mask) else 0.0,
            })
            timeseries.append(pd.DataFrame({
                'case_id':ci,'case_name':name,'branch':branch,'time_s':t,'fault_active':mask.astype(int),
                'true_sun_pointing_error_deg':point,'attitude_estimation_error_deg':att,'body_rate_deg_s':rate,
                'adcs_mode':r['mode'],'eclipse_state':eclipse_state,'ml_recovery_active':r['ml_used'].astype(int),
                'sun_fault_active':r['sensor_faulted'].astype(int),'gyro_fault_active':r['gyro_faulted'].astype(int),'mag_fault_active':r['mag_faulted'].astype(int),
            }))
        all_results[name]=results

        # One comparison plot per fault case.
        fig,axes=create_subplots(2,1,figsize=(10.3,7.2),sharex=True)
        colors=[PALETTE['green_3'],PALETTE['red_strong'],PALETTE['blue_main']]
        for b,c in zip(BRANCHES,colors):
            axes[0].plot(t,results[b]['true_point'],label=LABELS[b],color=c)
            axes[1].plot(t,results[b]['att_err'],label=LABELS[b],color=c)
        for ax in axes: ax.axvspan(float(case.get('start_s',0)),float(case.get('end_s',duration)),color='0.35',alpha=.12)
        axes[0].set_ylabel('Sun-pointing error [deg]'); axes[0].set_title(f'{name}: {_case_description(case)}'); axes[0].legend(fontsize=9)
        axes[1].set_ylabel('MEKF attitude error [deg]'); axes[1].set_xlabel('Time [s]')
        finalize_figure(fig,plots/f'{ci:02d}_{_slug(name)}_pointing_and_estimation_error',formats=tuple(global_cfg['visualization']['formats']),dpi=int(global_cfg['visualization']['dpi']))

        # Lightweight animations for selected cases, all three branches.
        selected=('all' in animation_cases or str(ci) in animation_cases or name.lower() in animation_cases)
        if bool(global_cfg.get('animation',{}).get('enabled',False)) and bool(cfg.get('animation_enabled',True)) and selected:
            kwargs={k:v for k,v in global_cfg['animation'].items() if k not in ('enabled','phase4','phase5')}
            kwargs.update(body_pointing_axis=axis,sun_i=sun,mag_i=B_i,show_sun_pointing_error=True,eclipse_mask=np.asarray(eclipse_state)==2,
                          position_eci_m=states[:,:3],velocity_eci_mps=states[:,3:],earth_epoch_utc=str(base.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00')))
            for b in BRANCHES:
                r=results[b]
                create_orbit_attitude_animation(animations/f'{ci:02d}_{_slug(name)}_{b}.html',t,r['truth'][:,:4],r['qhat'],
                    altitude_m=float(orbit_meta['altitude_m']),inclination_deg=float(orbit_meta['inclination_deg']),raan_deg=float(orbit_meta['raan_deg']),phase_deg=float(orbit_meta.get('phase_deg',0.0)),
                    fault_name=_case_description(case) if b!='healthy_mekf' else 'none',severity='campaign',
                    fault_start_s=float(case.get('start_s',0.0)) if b!='healthy_mekf' else None,fault_end_s=float(case.get('end_s',duration)) if b!='healthy_mekf' else None,
                    title=f'{name} — {LABELS[b]}',mode_series=r['mode'],body_rate_deg_s=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),**kwargs)

    summary=pd.DataFrame(summary_rows)
    summary_path=out/'reaction_wheel_fault_campaign_summary.csv'
    timeseries_path=out/'reaction_wheel_fault_campaign_timeseries.csv'
    summary.to_csv(summary_path,index=False)
    pd.concat(timeseries,ignore_index=True).to_csv(timeseries_path,index=False)
    # Convenience copies in the timestamped run root so the user does not need to
    # search inside the scenario subdirectory.
    summary.to_csv(out.parent/'reaction_wheel_fault_campaign_summary.csv',index=False)
    pd.concat(timeseries,ignore_index=True).to_csv(out.parent/'reaction_wheel_fault_campaign_timeseries.csv',index=False)

    # Campaign-level plot: faulty MEKF vs ML by case.
    piv=summary.pivot(index='case_name',columns='branch',values='pointing_rmse_fault_deg')
    fig,axes=create_subplots(figsize=(11.0,5.5)); ax=axes[0]
    x=np.arange(len(piv.index)); w=.25
    for j,b in enumerate(BRANCHES):
        vals=piv[b].to_numpy() if b in piv else np.full(len(x),np.nan)
        ax.bar(x+(j-1)*w,vals,width=w,label=LABELS[b])
    ax.set_xticks(x,piv.index,rotation=30,ha='right'); ax.set_ylabel('Pointing RMSE during fault [deg]'); ax.set_title('Multi-sensor fault campaign: pointing degradation'); ax.legend(fontsize=9)
    finalize_figure(fig,plots/'campaign_pointing_rmse_by_fault_case_and_branch',formats=tuple(global_cfg['visualization']['formats']),dpi=int(global_cfg['visualization']['dpi']))

    (out/'scenario_metadata.json').write_text(json.dumps({'scenario':'reaction_wheel_fault_campaign','cases':cases,'orbit':orbit_meta,'virtual_sensor':vm},indent=2),encoding='utf-8')
    return {'scenario':'reaction_wheel_fault_campaign','cases':len(cases),'rows':len(summary)}
