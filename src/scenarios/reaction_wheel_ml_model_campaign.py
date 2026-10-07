from __future__ import annotations
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd

from src.simulation.control import sun_pointing_initial_quaternion
from src.scenarios.reaction_wheel_sun_pointing_comparison import _make_environment, _run_branch
from src.ml.multisensor_virtual import train_sensor_specific_bundle
from src.visualization.publication import PALETTE, create_subplots, finalize_figure
from src.visualization.animation import create_orbit_attitude_animation

BRANCHES=('healthy_mekf','faulty_mekf','fdir_only','mlp_recovery','cnn_recovery','gru_recovery')
LABELS={
    'healthy_mekf':'Healthy MEKF',
    'faulty_mekf':'Faulty sensors + MEKF',
    'fdir_only':'FDIR reject-only',
    'mlp_recovery':'Faulty + MEKF + MLP',
    'cnn_recovery':'Faulty + MEKF + 1D CNN',
    'gru_recovery':'Faulty + MEKF + GRU',
}
MODEL_FOR_BRANCH={'mlp_recovery':'mlp','cnn_recovery':'cnn','gru_recovery':'gru'}


def _slug(s): return ''.join(c if c.isalnum() else '_' for c in str(s).lower()).strip('_')

def _mask(t,case): return (t>=float(case.get('start_s',0.0)))&(t<=float(case.get('end_s',t[-1])))

def _desc(case):
    p=[]
    for key in ('sun','gyro','mag'):
        c=case.get(key,{}) or {}; typ=str(c.get('type','none'))
        if typ!='none': p.append(f'{key}:{typ}')
    return ', '.join(p) if p else 'healthy sensors'



def _family_severity(case):
    gf=case.get('gyro',{}) or {}; mf=case.get('mag',{}) or {}; sf=case.get('sun',{}) or {}
    gt=str(gf.get('type','none')).lower(); mt=str(mf.get('type','none')).lower(); st=str(sf.get('type','none')).lower()
    if gt=='bias_step' and mt=='none' and st=='none':
        return 'gyro_bias', float(np.max(np.abs(np.asarray(gf.get('bias_rad_s',[0,0,0]),float)))), 'rad/s'
    if mt=='misalignment' and gt=='none' and st=='none':
        return 'mag_misalignment', float(np.max(np.abs(np.asarray(mf.get('angles_deg',[0,0,0]),float)))), 'deg'
    if gt=='loss' and mt=='none' and st=='none': return 'gyro_loss', 1.0, 'binary'
    if mt=='loss' and gt=='none' and st=='none': return 'mag_loss', 1.0, 'binary'
    if st!='none' and gt=='none' and mt=='none': return 'sun_fault', 1.0, 'binary'
    if gt!='none' or mt!='none' or st!='none': return 'combined', np.nan, ''
    return 'baseline', 0.0, ''

def run(cfg,global_cfg,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    plots=out/'plots'; plots.mkdir(exist_ok=True)
    animations=out/'animations'; animations.mkdir(exist_ok=True)
    models_dir=out/'models'; models_dir.mkdir(exist_ok=True)

    base=copy.deepcopy(global_cfg['scenarios']['reaction_wheel_sun_pointing_comparison'])
    duration=float(cfg.get('duration_s',base.get('duration_s',6000.0))); dt=float(cfg.get('dt_s',base.get('dt_s',0.1)))
    base['duration_s']=duration; base['dt_s']=dt
    if 'orbit_profile' in cfg: base['orbit_profile']=cfg['orbit_profile']
    if 'magnetic_field_model' in cfg: base.setdefault('magnetic_field',{})['model']=cfg['magnetic_field_model']
    t=np.arange(0,duration+dt/2,dt)

    cases=list(cfg.get('cases',[]))
    if not cases:
        cases=copy.deepcopy(global_cfg.get('scenarios',{}).get('reaction_wheel_fault_campaign',{}).get('cases',[]))
    if not cases: raise ValueError('reaction_wheel_ml_model_campaign requires fault cases')

    sun,sun_pos,states,eclipse_state,illum,B_i,orbit_meta=_make_environment(t,base,global_cfg)
    axis=str(base.get('pointing_axis','x')); q0=sun_pointing_initial_quaternion(axis,sun[0],float(base.get('initial_error_deg',30.0)))
    omega0=np.asarray(base.get('omega0_rad_s',[0,0,0]),float)
    sc=global_cfg['sensors']; n=len(t); seed=int(cfg.get('seed',9311)); rng=np.random.default_rng(seed)
    noise={'gyro':rng.normal(0,float(sc['gyro_noise_std_rad_s']),(n,3)),
           'sun':rng.normal(0,float(sc['vector_noise_std']),(n,3)),
           'mag':rng.normal(0,float(sc['vector_noise_std']),(n,3))}

    mc=cfg.get('models',{})
    window=int(mc.get('window',20)); train_runs=int(mc.get('train_runs',24)); train_duration=float(mc.get('train_duration_s',40.0))
    epochs=int(mc.get('epochs',18)); batch_size=int(mc.get('batch_size',256)); model_seed=int(mc.get('seed',93001))
    predictors={}; model_metrics={}
    for j,name in enumerate(('mlp','cnn','gru')):
        pred,met=train_sensor_specific_bundle(name,models_dir/name,runs=train_runs,duration_s=train_duration,dt=dt,window=window,
                                               epochs=epochs,batch_size=batch_size,seed=model_seed+j*1000,
                                               gyro_noise_std=float(sc['gyro_noise_std_rad_s']),vector_noise_std=float(sc['vector_noise_std']))
        predictors[name]=pred; model_metrics[name]=met

    summary_rows=[]; timeseries=[]; animation_cases={str(x).lower() for x in cfg.get('animation_cases',['F3_sun_loss_large_gyro_bias','F5_sun_loss_mag_misaligned','F6_severe_estimator_corruption'])}
    formats=tuple(global_cfg['visualization']['formats']); dpi=int(global_cfg['visualization']['dpi'])
    colors=[PALETTE['green_3'],PALETTE['red_strong'],PALETTE['neutral_dark'],PALETTE['blue_main'],PALETTE['violet'],PALETTE['teal']]

    for ci,case0 in enumerate(cases):
        case=copy.deepcopy(case0); name=str(case.get('name',f'F{ci}')); mask=_mask(t,case); results={}
        fault_family,severity_value,severity_unit=_family_severity(case)
        for branch in BRANCHES:
            predictor=None
            if branch in MODEL_FOR_BRANCH:
                predictor=predictors[MODEL_FOR_BRANCH[branch]].clone(); predictor.reset()
            r=_run_branch(branch,base,global_cfg,t,q0,omega0,sun,B_i,states,eclipse_state,noise,predictor=predictor,fault_case=case, fdir_only=(branch=='fdir_only'))
            results[branch]=r
            point=np.asarray(r['true_point']); att=np.asarray(r['att_err']); rate=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1))
            def rmse(x,m): return float(np.sqrt(np.mean(np.asarray(x)[m]**2))) if np.any(m) else np.nan
            summary_rows.append({
                'case_id':ci,'case_name':name,'fault_description':_desc(case),'fault_family':fault_family,'severity_value':severity_value,'severity_unit':severity_unit,'branch':branch,'label':LABELS[branch],
                'model':MODEL_FOR_BRANCH.get(branch,'none'),'fault_start_s':float(case.get('start_s',0.0)),'fault_end_s':float(case.get('end_s',duration)),
                'pointing_rmse_all_deg':rmse(point,np.ones(n,bool)),'pointing_rmse_fault_deg':rmse(point,mask),
                'attitude_rmse_all_deg':rmse(att,np.ones(n,bool)),'attitude_rmse_fault_deg':rmse(att,mask),
                'pointing_max_fault_deg':float(np.max(point[mask])) if np.any(mask) else np.nan,
                'attitude_max_fault_deg':float(np.max(att[mask])) if np.any(mask) else np.nan,
                'time_over_1deg_fault_pct':float(100*np.mean(point[mask]>1.0)) if np.any(mask) else 0.0,
                'time_over_2deg_fault_pct':float(100*np.mean(point[mask]>2.0)) if np.any(mask) else 0.0,
                'time_over_5deg_fault_pct':float(100*np.mean(point[mask]>5.0)) if np.any(mask) else 0.0,
                'final_pointing_error_deg':float(point[-1]),'max_body_rate_deg_s':float(np.max(rate)),
                'ml_any_usage_pct_fault':float(100*np.mean(r['ml_used'][mask])) if np.any(mask) else 0.0,
                'ml_sun_usage_pct_fault':float(100*np.mean(r['ml_sun_used'][mask])) if np.any(mask) else 0.0,
                'ml_gyro_usage_pct_fault':float(100*np.mean(r['ml_gyro_used'][mask])) if np.any(mask) else 0.0,
                'ml_mag_usage_pct_fault':float(100*np.mean(r['ml_mag_used'][mask])) if np.any(mask) else 0.0,
                'ml_rejected_pct_fault':float(100*np.mean(r['ml_rejected'][mask])) if np.any(mask) else 0.0,
                'ml_sun_rejected_pct_fault':float(100*np.mean(r['ml_sun_rejected'][mask])) if np.any(mask) else 0.0,
                'ml_gyro_rejected_pct_fault':float(100*np.mean(r['ml_gyro_rejected'][mask])) if np.any(mask) else 0.0,
                'ml_mag_rejected_pct_fault':float(100*np.mean(r['ml_mag_rejected'][mask])) if np.any(mask) else 0.0,
            })
            timeseries.append(pd.DataFrame({
                'case_id':ci,'case_name':name,'branch':branch,'model':MODEL_FOR_BRANCH.get(branch,'none'),'time_s':t,'fault_active':mask.astype(int),
                'true_sun_pointing_error_deg':point,'attitude_estimation_error_deg':att,'body_rate_deg_s':rate,
                'adcs_mode':r['mode'],'eclipse_state':eclipse_state,'ml_recovery_active':r['ml_used'].astype(int),
                'ml_sun_used':r['ml_sun_used'].astype(int),'ml_gyro_used':r['ml_gyro_used'].astype(int),'ml_mag_used':r['ml_mag_used'].astype(int),
                'ml_sun_rejected':r['ml_sun_rejected'].astype(int),'ml_gyro_rejected':r['ml_gyro_rejected'].astype(int),'ml_mag_rejected':r['ml_mag_rejected'].astype(int),
                'ml_sun_gate_value_deg':r['ml_sun_gate_value'],'ml_gyro_gate_value_rad_s':r['ml_gyro_gate_value'],'ml_gyro_cross_sensor_gate_deg':r['ml_gyro_cross_sensor_gate_deg'],'ml_mag_gate_value_deg':r['ml_mag_gate_value'],
                'sun_fault_active':r['sensor_faulted'].astype(int),'gyro_fault_active':r['gyro_faulted'].astype(int),'mag_fault_active':r['mag_faulted'].astype(int),
            }))

        fig,axes=create_subplots(2,1,figsize=(10.7,7.4),sharex=True)
        for b,c in zip(BRANCHES,colors):
            axes[0].plot(t,results[b]['true_point'],label=LABELS[b],color=c)
            axes[1].plot(t,results[b]['att_err'],label=LABELS[b],color=c)
        for ax in axes: ax.axvspan(float(case.get('start_s',0)),float(case.get('end_s',duration)),color='0.35',alpha=.12)
        axes[0].set_ylabel('Sun-pointing error [deg]'); axes[0].set_title(f'{name}: {_desc(case)}'); axes[0].legend(fontsize=8,ncol=2)
        axes[1].set_ylabel('Attitude estimation error [deg]'); axes[1].set_xlabel('Time [s]')
        finalize_figure(fig,plots/f'{ci:02d}_{_slug(name)}_five_branch_model_comparison',formats=formats,dpi=dpi)

        selected=('all' in animation_cases or str(ci) in animation_cases or name.lower() in animation_cases)
        if bool(global_cfg.get('animation',{}).get('enabled',False)) and bool(cfg.get('animation_enabled',True)) and selected:
            kwargs={k:v for k,v in global_cfg['animation'].items() if k not in ('enabled','phase4','phase5')}
            kwargs.update(body_pointing_axis=axis,sun_i=sun,mag_i=B_i,show_sun_pointing_error=True,eclipse_mask=np.asarray(eclipse_state)==2,
                          position_eci_m=states[:,:3],velocity_eci_mps=states[:,3:],earth_epoch_utc=str(base.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00')))
            for b in BRANCHES:
                r=results[b]
                create_orbit_attitude_animation(animations/f'{ci:02d}_{_slug(name)}_{b}.html',t,r['truth'][:,:4],r['qhat'],
                    altitude_m=float(orbit_meta['altitude_m']),inclination_deg=float(orbit_meta['inclination_deg']),raan_deg=float(orbit_meta['raan_deg']),phase_deg=float(orbit_meta.get('phase_deg',0.0)),
                    fault_name=_desc(case) if b!='healthy_mekf' else 'none',severity='model_comparison',
                    fault_start_s=float(case.get('start_s',0.0)) if b!='healthy_mekf' else None,fault_end_s=float(case.get('end_s',duration)) if b!='healthy_mekf' else None,
                    title=f'{name} — {LABELS[b]}',mode_series=r['mode'],body_rate_deg_s=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),**kwargs)

    summary=pd.DataFrame(summary_rows); ts=pd.concat(timeseries,ignore_index=True)
    # Improvement is reported relative to the same-case faulty MEKF baseline. Positive
    # values mean lower error; negative values mean that recovery degraded performance.
    summary['pointing_improvement_vs_faulty_pct']=np.nan
    summary['attitude_improvement_vs_faulty_pct']=np.nan
    summary['pointing_improvement_vs_fdir_pct']=np.nan
    summary['attitude_improvement_vs_fdir_pct']=np.nan
    for cname,g in summary.groupby('case_name'):
        fb=g[g.branch=='faulty_mekf']; fd=g[g.branch=='fdir_only']
        ix=g.index
        if len(fb):
            p0=float(fb.iloc[0].pointing_rmse_fault_deg); a0=float(fb.iloc[0].attitude_rmse_fault_deg)
            if np.isfinite(p0) and abs(p0)>1e-12: summary.loc[ix,'pointing_improvement_vs_faulty_pct']=100.0*(p0-summary.loc[ix,'pointing_rmse_fault_deg'])/p0
            if np.isfinite(a0) and abs(a0)>1e-12: summary.loc[ix,'attitude_improvement_vs_faulty_pct']=100.0*(a0-summary.loc[ix,'attitude_rmse_fault_deg'])/a0
        if len(fd):
            pf=float(fd.iloc[0].pointing_rmse_fault_deg); af=float(fd.iloc[0].attitude_rmse_fault_deg)
            mlix=g[g.branch.isin(('mlp_recovery','cnn_recovery','gru_recovery'))].index
            if np.isfinite(pf) and abs(pf)>1e-12: summary.loc[mlix,'pointing_improvement_vs_fdir_pct']=100.0*(pf-summary.loc[mlix,'pointing_rmse_fault_deg'])/pf
            if np.isfinite(af) and abs(af)>1e-12: summary.loc[mlix,'attitude_improvement_vs_fdir_pct']=100.0*(af-summary.loc[mlix,'attitude_rmse_fault_deg'])/af
    summary.to_csv(out/'reaction_wheel_ml_model_campaign_summary.csv',index=False); ts.to_csv(out/'reaction_wheel_ml_model_campaign_timeseries.csv',index=False)
    summary.to_csv(out.parent/'reaction_wheel_ml_model_campaign_summary.csv',index=False); ts.to_csv(out.parent/'reaction_wheel_ml_model_campaign_timeseries.csv',index=False)

    # Campaign-level grouped model comparison.
    piv=summary.pivot(index='case_name',columns='branch',values='pointing_rmse_fault_deg')
    fig,axes=create_subplots(figsize=(12.2,5.8)); ax=axes[0]; x=np.arange(len(piv.index)); w=.13
    for j,(b,c) in enumerate(zip(BRANCHES,colors)):
        vals=piv[b].to_numpy() if b in piv else np.full(len(x),np.nan)
        ax.bar(x+(j-2.5)*w,vals,width=w,label=LABELS[b],color=c)
    ax.set_xticks(x,piv.index,rotation=30,ha='right'); ax.set_ylabel('Pointing RMSE during fault [deg]'); ax.set_title('ML architecture comparison under identical sensor faults'); ax.legend(fontsize=8,ncol=2)
    finalize_figure(fig,plots/'campaign_mlp_cnn_gru_pointing_rmse_comparison',formats=formats,dpi=dpi)

    # Severity-sweep figures separate FDIR value from actual ML recovery value.
    for family,xlabel,stem in (
        ('gyro_bias','Injected gyro bias severity [rad/s]','gyro_bias_severity_sweep'),
        ('mag_misalignment','Magnetometer misalignment severity [deg]','mag_misalignment_severity_sweep')):
        sub=summary[summary.fault_family==family].copy()
        if len(sub):
            fig,axes=create_subplots(figsize=(9.8,5.4)); ax=axes[0]
            for b,c in zip(BRANCHES,colors):
                bb=sub[sub.branch==b].sort_values('severity_value')
                if len(bb): ax.plot(bb.severity_value,bb.pointing_rmse_fault_deg,marker='o',label=LABELS[b],color=c)
            ax.set_xlabel(xlabel); ax.set_ylabel('Pointing RMSE during fault [deg]')
            ax.set_title(f'Recovery Architecture v2: {family.replace("_"," ")} sweep')
            ax.legend(fontsize=8,ncol=2)
            finalize_figure(fig,plots/stem,formats=formats,dpi=dpi)

    # Model resource / validation table.
    validation_rows=[]
    for model_name in ('mlp','cnn','gru'):
        for target,met in model_metrics[model_name].items(): validation_rows.append(met)
    pd.DataFrame(validation_rows).to_csv(out/'ml_model_validation_and_resources.csv',index=False)
    (out/'scenario_metadata.json').write_text(json.dumps({'scenario':'reaction_wheel_ml_model_campaign','cases':cases,'orbit':orbit_meta,'models':model_metrics,
        'fdir_note':'Recovery comparison uses oracle fault flags from injected faults to isolate reconstruction performance from fault-detection performance.'},indent=2),encoding='utf-8')
    return {'scenario':'reaction_wheel_ml_model_campaign','cases':len(cases),'branches':len(BRANCHES),'models':['mlp','cnn','gru']}
