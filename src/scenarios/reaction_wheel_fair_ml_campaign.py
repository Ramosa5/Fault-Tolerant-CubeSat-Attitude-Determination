from __future__ import annotations
from pathlib import Path
import copy, json
import numpy as np
import pandas as pd

from src.simulation.control import sun_pointing_initial_quaternion
from src.scenarios.reaction_wheel_sun_pointing_comparison import _make_environment, _run_branch
from src.ml.fair_orbital_models import (
    generate_shared_dataset, save_shared_dataset, train_fair_model_matrix,
    build_orbital_context, build_physics_baseline,
)
from src.visualization.publication import PALETTE, create_subplots, finalize_figure
from src.visualization.animation import create_orbit_attitude_animation

BASE_BRANCHES=('healthy_mekf','faulty_mekf','fdir_only')
BASE_LABELS={
    'healthy_mekf':'Healthy MEKF',
    'faulty_mekf':'Faulty sensors + MEKF',
    'fdir_only':'FDIR-only',
}


def _slug(s): return ''.join(c if c.isalnum() else '_' for c in str(s).lower()).strip('_')
def _fault_mask(t,case): return (t>=float(case.get('start_s',0.0)))&(t<=float(case.get('end_s',t[-1])))

def _fault_description(case):
    p=[]
    for key in ('sun','gyro','mag'):
        c=case.get(key,{}) or {}; typ=str(c.get('type','none'))
        if typ!='none': p.append(f'{key}:{typ}')
    return ', '.join(p) if p else 'healthy sensors'


def _family(case):
    explicit=str(case.get('group','')).strip()
    if explicit:
        return explicit
    s=str((case.get('sun',{}) or {}).get('type','none')).lower(); g=str((case.get('gyro',{}) or {}).get('type','none')).lower(); m=str((case.get('mag',{}) or {}).get('type','none')).lower()
    if s=='loss' or s=='sun_loss': return 'A_single_sensor_loss' if g=='none' and m=='none' else 'combined'
    if g=='loss' and s=='none' and m=='none': return 'A_single_sensor_loss'
    if m=='loss' and s=='none' and g=='none': return 'A_single_sensor_loss'
    if 'dropout' in (s,g,m): return 'B_intermittent_faults'
    if g in ('bias_step','drift') or m in ('misalignment','bias'): return 'C_bias_drift'
    return 'combined' if any(x!='none' for x in (s,g,m)) else 'baseline'


def _recovery_time(t,err,fault_end_s,sun_visible,threshold_deg=2.0,hold_s=10.0):
    t=np.asarray(t,float); e=np.asarray(err,float); vis=np.asarray(sun_visible,bool); dt=float(np.median(np.diff(t))); need=max(1,int(np.ceil(float(hold_s)/dt))); start=int(np.searchsorted(t,float(fault_end_s))); c=0
    for k in range(start,len(t)):
        if vis[k] and np.isfinite(e[k]) and e[k]<=float(threshold_deg):
            c+=1
            if c>=need: return float(t[k-need+1]-float(fault_end_s))
        else: c=0
    return np.nan


def _save_table(df,path,index=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path.with_suffix('.csv'),index=index)
    try: path.with_suffix('.md').write_text(df.to_markdown(index=index),encoding='utf-8')
    except Exception: pass
    try: path.with_suffix('.tex').write_text(df.to_latex(index=index,float_format=lambda x:f'{x:.4g}'),encoding='utf-8')
    except Exception: pass


def _branch_specs(cfg):
    mc=cfg.get('models',{}); arch=[str(x).lower() for x in mc.get('architectures',['mlp','cnn','gru'])]; var=[str(x).lower() for x in mc.get('variants',['sensor_only','orbital_context','physics_residual'])]
    specs=[(b,'none','none',BASE_LABELS[b]) for b in BASE_BRANCHES]
    for a in arch:
        for v in var:
            specs.append((f'{a}_{v}',a,v,f'{a.upper()} — {v.replace("_"," ")}'))
    return specs



def _generated_cases(cfg):
    fc=cfg.get('fault_scenarios',{})
    if not fc or not bool(fc.get('enabled',True)):
        return []
    start=float(fc.get('start_s',1000.0)); end=float(fc.get('end_s',3000.0)); groups={str(x).upper() for x in fc.get('groups',['A','B','C'])}; cases=[]
    if 'A' in groups:
        for sensor in fc.get('single_sensor_loss',['sun','gyro','mag']):
            sensor=str(sensor).lower(); c={'name':f'A_{sensor}_loss','group':'A_single_sensor_loss','start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'none'},'mag':{'type':'none'}}
            c[sensor]={'type':'sun_loss','severity':'high'} if sensor=='sun' else {'type':'loss'}
            cases.append(c)
    if 'B' in groups:
        for sensor in fc.get('dropout_sensors',['gyro','sun','mag']):
            sensor=str(sensor).lower()
            for p in fc.get('dropout_probabilities',[.10,.25,.50,.75]):
                prob=float(p); tag=int(round(prob*100)); c={'name':f'B_{sensor}_dropout_{tag:02d}pct','group':'B_intermittent_faults','start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'none'},'mag':{'type':'none'}}
                c[sensor]={'type':'sun_dropout','severity':'medium','probability':prob} if sensor=='sun' else {'type':'dropout','probability':prob}
                cases.append(c)
    if 'C' in groups:
        axis=np.asarray(fc.get('gyro_fault_axis',[1.0,-0.7,0.5]),float); axis=axis/(np.max(np.abs(axis))+1e-12)
        for level in fc.get('gyro_bias_levels_rad_s',[.001,.003,.006,.012]):
            lv=float(level); cases.append({'group':'C_bias_drift','name':f'C_gyro_bias_{str(lv).replace(".","p")}','start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'bias_step','bias_rad_s':(axis*lv).tolist()},'mag':{'type':'none'}})
        daxis=np.asarray(fc.get('gyro_drift_axis',[1.0,-0.75,0.5]),float); daxis=daxis/(np.max(np.abs(daxis))+1e-12)
        for level in fc.get('gyro_drift_levels_rad_s2',[1e-6,3e-6,8e-6]):
            lv=float(level); cases.append({'group':'C_bias_drift','name':f'C_gyro_drift_{lv:.0e}'.replace('+',''),'start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'drift','drift_rad_s2':(daxis*lv).tolist()},'mag':{'type':'none'}})
        for level in fc.get('mag_misalignment_levels_deg',[5.,10.,20.,30.]):
            lv=float(level); cases.append({'name':f'C_mag_misalignment_{int(lv)}deg','group':'C_bias_drift','start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'none'},'mag':{'type':'misalignment','angles_deg':[lv,-.7*lv,.5*lv]}})
    if 'D' in groups:
        selected={str(x).strip().lower() for x in fc.get('combined_cases',[
            'gyro_loss_sun_loss','gyro_loss_mag_loss','sun_loss_mag_loss',
            'gyro_bias_mag_misalignment','gyro_drift_sun_dropout',
            'gyro_dropout_mag_dropout','sun_dropout_mag_dropout','gyro_dropout_sun_dropout',
            'all_three_degraded','all_three_loss'])}
        dprob=float(fc.get('combined_dropout_probability',0.50))
        dbias=float(fc.get('combined_gyro_bias_rad_s',0.006))
        ddrift=float(fc.get('combined_gyro_drift_rad_s2',3.0e-6))
        dmis=float(fc.get('combined_mag_misalignment_deg',20.0))
        bax=np.asarray(fc.get('combined_gyro_bias_axis',[1.0,-0.7,0.5]),float); bax=bax/(np.max(np.abs(bax))+1e-12)
        dax=np.asarray(fc.get('combined_gyro_drift_axis',[1.0,-0.75,0.5]),float); dax=dax/(np.max(np.abs(dax))+1e-12)
        base_case=lambda name: {'name':name,'group':'D_multi_sensor_faults','start_s':start,'end_s':end,'sun':{'type':'none'},'gyro':{'type':'none'},'mag':{'type':'none'}}
        def add(key,c):
            if key in selected: cases.append(c)
        c=base_case('D1_gyro_loss_plus_sun_loss'); c['gyro']={'type':'loss'}; c['sun']={'type':'sun_loss','severity':'high'}; add('gyro_loss_sun_loss',c)
        c=base_case('D2_gyro_loss_plus_mag_loss'); c['gyro']={'type':'loss'}; c['mag']={'type':'loss'}; add('gyro_loss_mag_loss',c)
        c=base_case('D3_sun_loss_plus_mag_loss'); c['sun']={'type':'sun_loss','severity':'high'}; c['mag']={'type':'loss'}; add('sun_loss_mag_loss',c)
        c=base_case('D4_gyro_bias_plus_mag_misalignment'); c['gyro']={'type':'bias_step','bias_rad_s':(bax*dbias).tolist()}; c['mag']={'type':'misalignment','angles_deg':[dmis,-.7*dmis,.5*dmis]}; add('gyro_bias_mag_misalignment',c)
        c=base_case('D5_gyro_drift_plus_sun_dropout'); c['gyro']={'type':'drift','drift_rad_s2':(dax*ddrift).tolist()}; c['sun']={'type':'sun_dropout','severity':'high','probability':dprob}; add('gyro_drift_sun_dropout',c)
        c=base_case('D6_gyro_dropout_plus_mag_dropout'); c['gyro']={'type':'dropout','probability':dprob}; c['mag']={'type':'dropout','probability':dprob}; add('gyro_dropout_mag_dropout',c)
        c=base_case('D7_sun_dropout_plus_mag_dropout'); c['sun']={'type':'sun_dropout','severity':'high','probability':dprob}; c['mag']={'type':'dropout','probability':dprob}; add('sun_dropout_mag_dropout',c)
        c=base_case('D8_gyro_dropout_plus_sun_dropout'); c['gyro']={'type':'dropout','probability':dprob}; c['sun']={'type':'sun_dropout','severity':'high','probability':dprob}; add('gyro_dropout_sun_dropout',c)
        c=base_case('D9_all_three_degraded'); c['gyro']={'type':'bias_step','bias_rad_s':(bax*dbias).tolist()}; c['sun']={'type':'sun_dropout','severity':'high','probability':dprob}; c['mag']={'type':'misalignment','angles_deg':[dmis,-.7*dmis,.5*dmis]}; add('all_three_degraded',c)
        c=base_case('D10_all_three_loss'); c['gyro']={'type':'loss'}; c['sun']={'type':'sun_loss','severity':'high'}; c['mag']={'type':'loss'}; add('all_three_loss',c)
    return cases

def run(cfg,global_cfg,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); plots=out/'plots'; plots.mkdir(exist_ok=True); tables=out/'tables'; tables.mkdir(exist_ok=True); models_dir=out/'models'; models_dir.mkdir(exist_ok=True)
    animations=out/'animations'; animations.mkdir(exist_ok=True)

    base=copy.deepcopy(global_cfg['scenarios']['reaction_wheel_sun_pointing_comparison'])
    duration=float(cfg.get('duration_s',base.get('duration_s',6000.))); dt=float(cfg.get('dt_s',base.get('dt_s',.1))); base['duration_s']=duration; base['dt_s']=dt
    if 'orbit_profile' in cfg: base['orbit_profile']=cfg['orbit_profile']
    if 'magnetic_field_model' in cfg: base.setdefault('magnetic_field',{})['model']=cfg['magnetic_field_model']
    t=np.arange(0,duration+dt/2,dt); n=len(t)
    cases=[copy.deepcopy(c) for c in cfg.get('cases',[]) if bool(c.get('enabled',True))]
    if not cases: cases=_generated_cases(cfg)
    if not cases: raise ValueError('reaction_wheel_fair_ml_campaign requires explicit cases or [fault_scenarios] generation settings')

    # Runtime environment is identical for every branch and fault case.
    sun,sun_pos,states,eclipse_state,illum,B_i,orbit_meta=_make_environment(t,base,global_cfg)
    runtime_context=build_orbital_context(states,sun,B_i,illum); runtime_baseline=build_physics_baseline(states,sun,B_i,dt,str(base.get('pointing_axis','x')))
    axis=str(base.get('pointing_axis','x')); q0=sun_pointing_initial_quaternion(axis,sun[0],float(base.get('initial_error_deg',30.))); omega0=np.asarray(base.get('omega0_rad_s',[0,0,0]),float)
    sc=global_cfg['sensors']; seed=int(cfg.get('seed',99501)); rng=np.random.default_rng(seed)
    noise={'gyro':rng.normal(0,float(sc['gyro_noise_std_rad_s']),(n,3)),'sun':rng.normal(0,float(sc['vector_noise_std']),(n,3)),'mag':rng.normal(0,float(sc['vector_noise_std']),(n,3))}

    # One shared dataset for every architecture and variant.
    model_cfg=copy.deepcopy(cfg.get('models',{})); dataset_cfg=copy.deepcopy(model_cfg.get('dataset',{})); dataset_cfg.setdefault('dt_s',dt); dataset_cfg.setdefault('window',int(model_cfg.get('window',20)))
    dataset=generate_shared_dataset(dataset_cfg,gyro_noise_std=float(sc['gyro_noise_std_rad_s']),vector_noise_std=float(sc['vector_noise_std']))
    save_shared_dataset(dataset,models_dir/'shared_dataset',save_npz=bool(model_cfg.get('save_shared_dataset_npz',False)))
    # Flatten training config expected by model matrix helper.
    matrix_cfg={'architectures':model_cfg.get('architectures',['mlp','cnn','gru']),'variants':model_cfg.get('variants',['sensor_only','orbital_context','physics_residual']),'targets':model_cfg.get('targets',['gyro','sun','mag']),
                'training':model_cfg.get('training',{}),'hyperparameters':model_cfg.get('hyperparameters',{})}
    bundles,model_validation=train_fair_model_matrix(dataset,models_dir/'fair_matrix',matrix_cfg)
    _save_table(model_validation,tables/'model_validation_test_resources')

    specs=_branch_specs(cfg); summary_rows=[]; ts_rows=[]; sunlight_min=float(cfg.get('metrics',{}).get('sunlight_min_illumination',.05)); rec_thr=float(cfg.get('metrics',{}).get('recovery_threshold_deg',2.0)); rec_hold=float(cfg.get('metrics',{}).get('recovery_hold_s',10.0))
    fdir_cfg=copy.deepcopy(cfg.get('fdir',{})); fdir_mode=str(fdir_cfg.get('mode','classical_v2')).lower()
    if fdir_mode not in ('oracle','classical','classical_v2'):
        raise ValueError(f'Unsupported FDIR mode: {fdir_mode}. Use oracle, classical, or classical_v2.')
    formats=tuple(global_cfg['visualization']['formats']); dpi=int(global_cfg['visualization']['dpi']); palette=[PALETTE['green_3'],PALETTE['red_strong'],PALETTE['neutral_dark'],PALETTE['blue_main'],PALETTE['violet'],PALETTE['teal']]
    animation_cfg=cfg.get('animation',{}); animation_enabled=bool(animation_cfg.get('enabled',False)); animation_cases={str(x).lower() for x in animation_cfg.get('cases',[])}

    for ci,case0 in enumerate(cases):
        case=copy.deepcopy(case0); cname=str(case.get('name',f'case_{ci:02d}')); fmask=_fault_mask(t,case); sunlight=np.asarray(illum)>sunlight_min; primary=fmask&sunlight; results={}
        for bi,(branch,arch,variant,label) in enumerate(specs):
            predictor=None; run_branch=branch
            if arch!='none':
                predictor=bundles[(arch,variant)].clone(); predictor.reset(); predictor.set_runtime_context(runtime_context,runtime_baseline); run_branch='ml_recovery'
            elif branch=='healthy_mekf': run_branch='healthy_mekf'
            elif branch=='faulty_mekf': run_branch='faulty_mekf'
            elif branch=='fdir_only': run_branch='fdir_only'
            r=_run_branch(run_branch,base,global_cfg,t,q0,omega0,sun,B_i,states,eclipse_state,noise,predictor=predictor,fault_case=case,fdir_only=(branch=='fdir_only'),fdir_mode=fdir_mode,fdir_config=fdir_cfg)
            results[branch]=r; point=np.asarray(r['true_point']); att=np.asarray(r['att_err']); rate=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1))
            def rmse(x,m): return float(np.sqrt(np.mean(np.asarray(x)[m]**2))) if np.any(m) else np.nan
            row={'case_id':ci,'case_name':cname,'fault_group':_family(case),'fault_description':_fault_description(case),'branch':branch,'label':label,'architecture':arch,'variant':variant,
                 'fault_start_s':float(case.get('start_s',0.)),'fault_end_s':float(case.get('end_s',duration)),'sunlight_fault_samples':int(np.sum(primary)),
                 # Primary thesis metrics: only while fault is active AND Sun is physically visible.
                 'sunlight_fault_pointing_rmse_deg':rmse(point,primary),'sunlight_fault_attitude_rmse_deg':rmse(att,primary),
                 'sunlight_fault_pointing_max_deg':float(np.max(point[primary])) if np.any(primary) else np.nan,
                 'sunlight_fault_p_over_1deg_pct':float(100*np.mean(point[primary]>1.)) if np.any(primary) else np.nan,
                 'sunlight_fault_p_over_2deg_pct':float(100*np.mean(point[primary]>2.)) if np.any(primary) else np.nan,
                 'sunlight_fault_p_over_5deg_pct':float(100*np.mean(point[primary]>5.)) if np.any(primary) else np.nan,
                 'recovery_time_s':_recovery_time(t,point,float(case.get('end_s',duration)),sunlight,rec_thr,rec_hold),
                 # Secondary diagnostics retained for completeness.
                 'fault_pointing_rmse_all_illumination_deg':rmse(point,fmask),'fault_attitude_rmse_all_illumination_deg':rmse(att,fmask),'final_pointing_error_deg':float(point[-1]),'max_body_rate_deg_s':float(np.max(rate)),
                 'ml_any_usage_pct_sunlight_fault':float(100*np.mean(r['ml_used'][primary])) if np.any(primary) else 0.,
                 'ml_sun_usage_pct_sunlight_fault':float(100*np.mean(r['ml_sun_used'][primary])) if np.any(primary) else 0.,'ml_gyro_usage_pct_sunlight_fault':float(100*np.mean(r['ml_gyro_used'][primary])) if np.any(primary) else 0.,'ml_mag_usage_pct_sunlight_fault':float(100*np.mean(r['ml_mag_used'][primary])) if np.any(primary) else 0.,
                 'ml_sun_rejected_pct_sunlight_fault':float(100*np.mean(r['ml_sun_rejected'][primary])) if np.any(primary) else 0.,'ml_gyro_rejected_pct_sunlight_fault':float(100*np.mean(r['ml_gyro_rejected'][primary])) if np.any(primary) else 0.,'ml_mag_rejected_pct_sunlight_fault':float(100*np.mean(r['ml_mag_rejected'][primary])) if np.any(primary) else 0.,
                 'any_ml_rejection_pct_sunlight_fault':float(100*np.mean((r['ml_sun_rejected']|r['ml_gyro_rejected']|r['ml_mag_rejected'])[primary])) if np.any(primary) else 0.}

            # Score automatic FDIR separately from recovery. Ground-truth injected
            # fault flags are used only here, never by the classical detector.
            truth_masks={'gyro':np.asarray(r['gyro_faulted'],bool),'sun':np.asarray(r['sensor_faulted'],bool)&sunlight,'mag':np.asarray(r['mag_faulted'],bool)}
            det_masks={'gyro':np.asarray(r['fdir_detected_gyro'],bool),'sun':np.asarray(r['fdir_detected_sun'],bool),'mag':np.asarray(r['fdir_detected_mag'],bool)}
            confs={'gyro':np.asarray(r['fdir_conf_gyro'],float),'sun':np.asarray(r['fdir_conf_sun'],float),'mag':np.asarray(r['fdir_conf_mag'],float)}
            armed=np.asarray(r['fdir_armed'],bool)
            detector_active=bool(fdir_mode in ('classical','classical_v2') and branch not in ('healthy_mekf','faulty_mekf'))
            for sensor in ('gyro','sun','mag'):
                gt=truth_masks[sensor]; detm=det_masks[sensor]
                eval_mask=armed & (sunlight if sensor=='sun' else np.ones(n,bool))
                pos=gt&eval_mask; neg=(~gt)&eval_mask
                row[f'fdir_{sensor}_recall_pct']=float(100*np.mean(detm[pos])) if detector_active and np.any(pos) else np.nan
                row[f'fdir_{sensor}_false_alarm_pct']=float(100*np.mean(detm[neg])) if detector_active and np.any(neg) else np.nan
                row[f'fdir_{sensor}_mean_confidence_fault']=float(np.mean(confs[sensor][pos])) if detector_active and np.any(pos) else np.nan
                if detector_active:
                    tp=int(np.sum(detm & gt & eval_mask)); fp=int(np.sum(detm & (~gt) & eval_mask)); fn=int(np.sum((~detm) & gt & eval_mask))
                    precision=(tp/(tp+fp)) if (tp+fp)>0 else np.nan
                    recall=(tp/(tp+fn)) if (tp+fn)>0 else np.nan
                    row[f'fdir_{sensor}_precision_pct']=100*precision if np.isfinite(precision) else np.nan
                    row[f'fdir_{sensor}_f1_pct']=100*(2*precision*recall/(precision+recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision+recall)>0 else np.nan
                    starts=np.flatnonzero(gt & eval_mask & np.r_[True, ~(gt[:-1] & eval_mask[:-1])])
                    delays=[]
                    for st in starts:
                        idx=np.flatnonzero(detm[st:] & eval_mask[st:])
                        if len(idx): delays.append(float(t[st+idx[0]]-t[st]))
                    row[f'fdir_{sensor}_mean_detection_delay_s']=float(np.mean(delays)) if delays else np.nan
                else:
                    row[f'fdir_{sensor}_precision_pct']=np.nan; row[f'fdir_{sensor}_f1_pct']=np.nan; row[f'fdir_{sensor}_mean_detection_delay_s']=np.nan
            summary_rows.append(row)
            ts_rows.append(pd.DataFrame({'case_id':ci,'case_name':cname,'branch':branch,'architecture':arch,'variant':variant,'time_s':t,'fault_active':fmask.astype(int),'sun_visible':sunlight.astype(int),'primary_metric_mask':primary.astype(int),'illumination_fraction':illum,'eclipse_state':eclipse_state,'true_sun_pointing_error_deg':point,'attitude_estimation_error_deg':att,'body_rate_deg_s':rate,'adcs_mode':r['mode'],'ml_recovery_active':r['ml_used'].astype(int),'ml_sun_used':r['ml_sun_used'].astype(int),'ml_gyro_used':r['ml_gyro_used'].astype(int),'ml_mag_used':r['ml_mag_used'].astype(int),'ml_sun_rejected':r['ml_sun_rejected'].astype(int),'ml_gyro_rejected':r['ml_gyro_rejected'].astype(int),'ml_mag_rejected':r['ml_mag_rejected'].astype(int),
                'fdir_mode':r['fdir_mode'],'fdir_armed':r['fdir_armed'].astype(int),'fdir_gyro_confidence':r['fdir_conf_gyro'],'fdir_sun_confidence':r['fdir_conf_sun'],'fdir_mag_confidence':r['fdir_conf_mag'],
                'fdir_gyro_state':r['fdir_state_gyro'],'fdir_sun_state':r['fdir_state_sun'],'fdir_mag_state':r['fdir_state_mag'],
                'fdir_gyro_detected':r['fdir_detected_gyro'].astype(int),'fdir_sun_detected':r['fdir_detected_sun'].astype(int),'fdir_mag_detected':r['fdir_detected_mag'].astype(int),
                'fdir_gyro_reconstruct_request':r['fdir_flag_gyro'].astype(int),'fdir_sun_reconstruct_request':r['fdir_flag_sun'].astype(int),'fdir_mag_reconstruct_request':r['fdir_flag_mag'].astype(int),
                'fdir_gyro_rate_residual_rad_s':r['fdir_gyro_rate_residual'],'fdir_gyro_vector_rate_residual_rad_s':r['fdir_gyro_vector_rate_residual'],'fdir_gyro_cross_sensor_deg':r['fdir_gyro_cross_sensor'],'fdir_sun_innovation_deg':r['fdir_sun_innovation'],'fdir_mag_innovation_deg':r['fdir_mag_innovation'],
                'fdir_sun_mag_geometry_error_deg':r['fdir_geometry_error'],'fdir_sun_gyro_track_error_deg':r['fdir_sun_track_error'],'fdir_mag_gyro_track_error_deg':r['fdir_mag_track_error'],
                'fdir_gyro_hard_unavailable':r['fdir_hard_gyro'].astype(int),'fdir_sun_hard_unavailable':r['fdir_hard_sun'].astype(int),'fdir_mag_hard_unavailable':r['fdir_hard_mag'].astype(int),'fdir_sun_expected_unavailable':r['fdir_sun_expected_unavailable'].astype(int),
                'truth_gyro_fault':r['gyro_faulted'].astype(int),'truth_sun_fault':r['sensor_faulted'].astype(int),'truth_mag_fault':r['mag_faulted'].astype(int)}))

        # Per-case plot of primary pointing trajectories. Keep publication plots even when animation is disabled.
        if bool(cfg.get('plots_enabled',True)):
            fig,axes=create_subplots(figsize=(11.5,5.8)); ax=axes[0]
            for j,(branch,arch,variant,label) in enumerate(specs):
                ax.plot(t,results[branch]['true_point'],label=label,color=palette[j%len(palette)],lw=1.1)
            ax.axvspan(float(case.get('start_s',0.)),float(case.get('end_s',duration)),color='0.3',alpha=.1); ax.set_xlabel('Time [s]'); ax.set_ylabel('Sun-pointing error [deg]'); ax.set_title(f'{cname}: fair 3×3 ML comparison'); ax.legend(fontsize=7,ncol=3)
            finalize_figure(fig,plots/f'{ci:02d}_{_slug(cname)}_fair_3x3_comparison',formats=formats,dpi=dpi)

        if animation_enabled and ('all' in animation_cases or str(ci) in animation_cases or cname.lower() in animation_cases):
            kwargs={k:v for k,v in global_cfg.get('animation',{}).items() if k not in ('enabled','phase4','phase5')}; kwargs.update(body_pointing_axis=axis,sun_i=sun,mag_i=B_i,show_sun_pointing_error=True,eclipse_mask=np.asarray(eclipse_state)==2,position_eci_m=states[:,:3],velocity_eci_mps=states[:,3:],earth_epoch_utc=str(base.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00')))
            for branch,arch,variant,label in specs:
                r=results[branch]; create_orbit_attitude_animation(animations/f'{ci:02d}_{_slug(cname)}_{branch}.html',t,r['truth'][:,:4],r['qhat'],altitude_m=float(orbit_meta['altitude_m']),inclination_deg=float(orbit_meta['inclination_deg']),raan_deg=float(orbit_meta['raan_deg']),phase_deg=float(orbit_meta.get('phase_deg',0.0)),fault_name=_fault_description(case) if branch!='healthy_mekf' else 'none',severity='fair_comparison',fault_start_s=float(case.get('start_s',0.)) if branch!='healthy_mekf' else None,fault_end_s=float(case.get('end_s',duration)) if branch!='healthy_mekf' else None,title=f'{cname} — {label}',mode_series=r['mode'],body_rate_deg_s=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),**kwargs)

    summary=pd.DataFrame(summary_rows); ts=pd.concat(ts_rows,ignore_index=True)
    # Improvements relative to Faulty and FDIR baselines use the primary sunlight-only metric.
    summary['pointing_improvement_vs_faulty_pct']=np.nan; summary['pointing_improvement_vs_fdir_pct']=np.nan
    for cname,g in summary.groupby('case_name'):
        fb=g[g.branch=='faulty_mekf']; fd=g[g.branch=='fdir_only']; ix=g.index
        if len(fb):
            p0=float(fb.iloc[0].sunlight_fault_pointing_rmse_deg)
            if np.isfinite(p0) and abs(p0)>1e-12: summary.loc[ix,'pointing_improvement_vs_faulty_pct']=100*(p0-summary.loc[ix,'sunlight_fault_pointing_rmse_deg'])/p0
        if len(fd):
            p0=float(fd.iloc[0].sunlight_fault_pointing_rmse_deg); mlx=g[~g.branch.isin(BASE_BRANCHES)].index
            if np.isfinite(p0) and abs(p0)>1e-12: summary.loc[mlx,'pointing_improvement_vs_fdir_pct']=100*(p0-summary.loc[mlx,'sunlight_fault_pointing_rmse_deg'])/p0

    summary.to_csv(out/'reaction_wheel_fair_ml_campaign_summary.csv',index=False); ts.to_csv(out/'reaction_wheel_fair_ml_campaign_timeseries.csv',index=False)
    summary.to_csv(out.parent/'reaction_wheel_fair_ml_campaign_summary.csv',index=False); ts.to_csv(out.parent/'reaction_wheel_fair_ml_campaign_timeseries.csv',index=False)

    # Thesis-ready tables.
    primary_cols=['case_name','fault_group','fault_description','branch','architecture','variant','sunlight_fault_pointing_rmse_deg','sunlight_fault_attitude_rmse_deg','sunlight_fault_p_over_2deg_pct','sunlight_fault_p_over_5deg_pct','recovery_time_s','pointing_improvement_vs_faulty_pct','pointing_improvement_vs_fdir_pct']
    _save_table(summary[primary_cols],tables/'all_primary_sunlight_results')
    fdir_cols=['case_name','fault_group','branch','architecture','variant','fdir_gyro_recall_pct','fdir_gyro_precision_pct','fdir_gyro_f1_pct','fdir_gyro_false_alarm_pct','fdir_gyro_mean_detection_delay_s','fdir_gyro_mean_confidence_fault','fdir_sun_recall_pct','fdir_sun_precision_pct','fdir_sun_f1_pct','fdir_sun_false_alarm_pct','fdir_sun_mean_detection_delay_s','fdir_sun_mean_confidence_fault','fdir_mag_recall_pct','fdir_mag_precision_pct','fdir_mag_f1_pct','fdir_mag_false_alarm_pct','fdir_mag_mean_detection_delay_s','fdir_mag_mean_confidence_fault']
    _save_table(summary[fdir_cols],tables/'fdir_detection_performance')
    for arch in [str(x).lower() for x in model_cfg.get('architectures',['mlp','cnn','gru'])]:
        q=summary[(summary.architecture==arch)|summary.branch.isin(BASE_BRANCHES)][primary_cols]
        _save_table(q,tables/f'{arch}_three_variant_comparison')
    for variant in [str(x).lower() for x in model_cfg.get('variants',['sensor_only','orbital_context','physics_residual'])]:
        q=summary[(summary.variant==variant)|summary.branch.isin(BASE_BRANCHES)][primary_cols]
        _save_table(q,tables/f'{variant}_architecture_comparison')
    for fam,g in summary.groupby('fault_group'):
        _save_table(g[primary_cols],tables/f'fault_group_{_slug(fam)}')

    # Compact winner table per case (ML only) and architecture/variant aggregates.
    ml=summary[~summary.branch.isin(BASE_BRANCHES)].copy(); winner=[]
    for cname,g in ml.groupby('case_name'):
        gg=g[np.isfinite(g.sunlight_fault_pointing_rmse_deg)]
        if len(gg): winner.append(gg.sort_values('sunlight_fault_pointing_rmse_deg').iloc[0][['case_name','architecture','variant','sunlight_fault_pointing_rmse_deg','sunlight_fault_attitude_rmse_deg','pointing_improvement_vs_fdir_pct']].to_dict())
    _save_table(pd.DataFrame(winner),tables/'best_ml_variant_by_fault_case')
    agg=ml.groupby(['architecture','variant'],as_index=False).agg(mean_sunlight_pointing_rmse_deg=('sunlight_fault_pointing_rmse_deg','mean'),median_sunlight_pointing_rmse_deg=('sunlight_fault_pointing_rmse_deg','median'),mean_attitude_rmse_deg=('sunlight_fault_attitude_rmse_deg','mean'),mean_improvement_vs_fdir_pct=('pointing_improvement_vs_fdir_pct','mean'))
    _save_table(agg,tables/'architecture_variant_aggregate')

    # Duplicate thesis-ready tables at run-root level for easy retrieval.
    root_tables=out.parent/'comparison_tables'; root_tables.mkdir(exist_ok=True)
    for src in tables.iterdir():
        if src.is_file():
            (root_tables/src.name).write_bytes(src.read_bytes())

    # Campaign comparison plot based on the primary sunlight metric.
    if bool(cfg.get('plots_enabled',True)):
        fig,axes=create_subplots(figsize=(13.0,6.2)); ax=axes[0]; piv=summary.pivot(index='case_name',columns='branch',values='sunlight_fault_pointing_rmse_deg'); x=np.arange(len(piv)); w=min(.07,.8/max(1,len(specs)))
        for j,(branch,arch,variant,label) in enumerate(specs):
            vals=piv[branch].to_numpy() if branch in piv else np.full(len(x),np.nan); ax.bar(x+(j-(len(specs)-1)/2)*w,vals,width=w,label=label)
        ax.set_xticks(x,piv.index,rotation=35,ha='right'); ax.set_ylabel('Sunlight fault pointing RMSE [deg]'); ax.set_title('Fair MLP/CNN/GRU × variant comparison (fault active + Sun visible)'); ax.legend(fontsize=6,ncol=3)
        finalize_figure(fig,plots/'fair_campaign_primary_sunlight_rmse',formats=formats,dpi=dpi)

    metadata={'scenario':'reaction_wheel_fair_ml_campaign','branch_count':len(specs),'branches':[x[0] for x in specs],'architectures':model_cfg.get('architectures',['mlp','cnn','gru']),'variants':model_cfg.get('variants',['sensor_only','orbital_context','physics_residual']),'primary_metric':'Sun-pointing RMSE during fault while illumination_fraction > configured sunlight_min_illumination','sunlight_min_illumination':sunlight_min,'dataset_manifest':dataset['manifest'],'orbit':orbit_meta,'animation_enabled':animation_enabled,'fdir_mode':fdir_mode,'fdir_config':fdir_cfg,'fairness':'All architectures within a variant use identical shared trajectories, complete-run train/validation/test split, sample windows, injected training faults and normalization.'}
    (out/'scenario_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return {'scenario':'reaction_wheel_fair_ml_campaign','cases':len(cases),'branches':len(specs),'architectures':model_cfg.get('architectures',['mlp','cnn','gru']),'variants':model_cfg.get('variants',['sensor_only','orbital_context','physics_residual'])}
