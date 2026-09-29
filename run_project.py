from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, shutil, subprocess, sys, time, traceback, tomllib
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.simulation.constants import MU_EARTH,R_EARTH
from src.simulation.orbit_model import circular_orbit_period,propagate_circular_two_body
from src.verification.orekit_reference import propagate_orekit
from src.verification.metrics import comparison_metrics
from src.simulation.attitude import propagate_attitude,rotational_energy,angular_momentum_norm,quat_angle_error_deg,quat_to_dcm
from src.simulation.sensors import simulate_sensors
from src.simulation.faults import apply_fault
from src.estimation.mekf import run_mekf
from src.ml.dataset import generate_monte_carlo_dataset,split_runs,make_windows,FAULTS
from src.ml.training import train_one
from src.experiments.run_phase6 import detection_delays
from src.visualization.publication import (
    PALETTE, DEFAULT_COLORS, FigureStyle, apply_publication_style, create_subplots,
    finalize_figure, make_trend, make_grouped_bar, make_heatmap, add_fault_span
)
from src.visualization.animation import create_orbit_attitude_animation
from src.scenarios.registry import run_scenario, SCENARIOS

CONFIG=ROOT/'experiment_config.toml'

def mkdir(p): p.mkdir(parents=True,exist_ok=True); return p

def parse_scenarios(items):
    out=[]
    for s in items:
        a=s.split(':',1); out.append((a[0],a[1] if len(a)>1 else 'medium'))
    return out

def viz_options(cfg):
    v=cfg.get('visualization',{})
    return {
        'enabled':bool(v.get('enabled',True)),
        'formats':tuple(v.get('formats',['png','pdf'])),
        'dpi':int(v.get('dpi',300)),
        'font_size':int(v.get('font_size',15)),
        'axes_linewidth':float(v.get('axes_linewidth',2.0)),
    }

def animation_options(cfg):
    a=cfg.get('animation',{})
    return {
        'enabled':bool(a.get('enabled',False)),
        'frame_stride':int(a.get('frame_stride',20)),
        'max_frames':int(a.get('max_frames',600)),
        'start_time_s':None if str(a.get('start_time_s','none')).lower()=='none' else float(a.get('start_time_s')),
        'end_time_s':None if str(a.get('end_time_s','none')).lower()=='none' else float(a.get('end_time_s')),
        'ground_track_max_points':int(a.get('ground_track_max_points',900)),
        'include_plotlyjs':a.get('include_plotlyjs','cdn'),
        'playback_frame_ms':int(a.get('playback_frame_ms',80)),
        'save_mp4':bool(a.get('save_mp4',False)),
        'mp4_fps':int(a.get('mp4_fps',20)),
        'mp4_max_frames':int(a.get('mp4_max_frames',900)),
        'mp4_dpi':int(a.get('mp4_dpi',110)),
        'vector_length_km':float(a.get('vector_length_km',900.0)),
        'satellite_visual_size_km':tuple(float(x) for x in a.get('satellite_visual_size_km',[140.0,90.0,90.0])),
        'sun_vector_length_scale':float(a.get('sun_vector_length_scale',1.35)),
        'sun_arrowhead_size_km':float(a.get('sun_arrowhead_size_km',140.0)),
        'body_pointing_axis':str(a.get('body_pointing_axis','x')),
        'show_true_axes':bool(a.get('show_true_axes',True)),
        'show_estimated_axes':bool(a.get('show_estimated_axes',True)),
        'show_sun_vector':bool(a.get('show_sun_vector',True)),
        'show_mag_vector':bool(a.get('show_mag_vector',True)),
        'show_velocity_vector':bool(a.get('show_velocity_vector',True)),
        'show_nadir_vector':bool(a.get('show_nadir_vector',True)),
        'show_full_orbit':bool(a.get('show_full_orbit',True)),
    }

def _animation_kwargs(cfg):
    a=animation_options(cfg)
    return {k:v for k,v in a.items() if k!='enabled'}

def savefig(fig, plots, name, cfg, pad=1.0):
    v=viz_options(cfg)
    if not v['enabled']:
        import matplotlib.pyplot as plt
        plt.close(fig); return []
    try: fig.tight_layout(pad=pad)
    except Exception: pass
    return finalize_figure(fig, plots/name, formats=v['formats'], dpi=v['dpi'])

def run_tests(cfg,out,plots):
    report=out/'unit_test_report.xml'
    p=subprocess.run([sys.executable,'-m','pytest','-v',f'--junitxml={report}'],cwd=ROOT,text=True,capture_output=True)
    (out/'pytest_output.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    if viz_options(cfg)['enabled'] and report.exists():
        root=ET.parse(report).getroot()
        # pytest may use <testsuites><testsuite> or a direct <testsuite>
        suite=root if root.tag=='testsuite' else root.find('testsuite')
        counts={k:int(suite.attrib.get(k,0)) for k in ('tests','failures','errors','skipped')}
        passed=max(0,counts['tests']-counts['failures']-counts['errors']-counts['skipped'])
        cats=['Passed','Failed','Errors','Skipped']; vals=[passed,counts['failures'],counts['errors'],counts['skipped']]
        fig,axes=create_subplots(figsize=(7.5,4.5)); ax=axes[0]
        bars=ax.bar(cats,vals,color=[PALETTE['green_3'],PALETTE['red_strong'],PALETTE['violet'],PALETTE['neutral']],edgecolor='black',linewidth=1.0)
        ax.set_ylabel('Number of tests'); ax.set_title('Automated regression test outcome')
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,str(v),ha='center',va='bottom')
        savefig(fig,plots,'unit_tests_pass_fail_summary',cfg)
    if p.returncode: raise RuntimeError('Unit tests failed; see pytest_output.txt')
    return {'return_code':p.returncode}

def orbit_sanity(cfg,out,plots):
    c=cfg['orbit']; period=circular_orbit_period(c['altitude_m']); n=int(c['sanity_orbits']*c['samples_per_orbit'])+1
    t=np.linspace(0,c['sanity_orbits']*period,n); x=propagate_circular_two_body(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg'])
    r=np.linalg.norm(x[:,:3],axis=1); v=np.linalg.norm(x[:,3:],axis=1)
    altitude=r-R_EARTH
    df=pd.DataFrame({'time_s':t,'x_m':x[:,0],'y_m':x[:,1],'z_m':x[:,2],'vx_mps':x[:,3],'vy_mps':x[:,4],'vz_mps':x[:,5],'altitude_m':altitude,'speed_mps':v})
    df.to_csv(out/'orbit_sanity_timeseries.csv',index=False)
    s=pd.DataFrame([{'period_s':period,'altitude_mean_m':np.mean(altitude),'altitude_range_m':np.ptp(altitude),'speed_mean_mps':np.mean(v),'speed_range_mps':np.ptp(v)}]); s.to_csv(out/'orbit_sanity_summary.csv',index=False)
    if viz_options(cfg)['enabled']:
        import matplotlib.pyplot as plt
        # 3D inertial trajectory with Earth sphere
        fig=plt.figure(figsize=(7.4,6.4)); ax=fig.add_subplot(111,projection='3d')
        ax.plot(x[:,0]/1e3,x[:,1]/1e3,x[:,2]/1e3,color=PALETTE['blue_main'],lw=2.2,label='CubeSat trajectory')
        u=np.linspace(0,2*np.pi,60); vv=np.linspace(0,np.pi,30); R=R_EARTH/1e3
        xs=R*np.outer(np.cos(u),np.sin(vv)); ys=R*np.outer(np.sin(u),np.sin(vv)); zs=R*np.outer(np.ones_like(u),np.cos(vv))
        ax.plot_surface(xs,ys,zs,color=PALETTE['blue_secondary'],alpha=.18,linewidth=0)
        ax.scatter([x[0,0]/1e3],[x[0,1]/1e3],[x[0,2]/1e3],s=45,color=PALETTE['red_strong'],label='Initial state')
        ax.set_xlabel('ECI x [km]'); ax.set_ylabel('ECI y [km]'); ax.set_zlabel('ECI z [km]'); ax.set_title('Orbit sanity check: inertial trajectory'); ax.legend(loc='upper left')
        lim=np.max(np.abs(x[:,:3]))/1e3*1.08; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        savefig(fig,plots,'orbit_sanity_eci_3d_trajectory',cfg)
        # altitude
        fig,axes=create_subplots(figsize=(8.2,4.6)); ax=axes[0]; ax.plot(t/60,altitude/1e3,color=PALETTE['blue_main']); ax.set_xlabel('Time [min]'); ax.set_ylabel('Altitude [km]'); ax.set_title('Orbit sanity check: altitude constancy');
        savefig(fig,plots,'orbit_sanity_altitude_vs_time',cfg)
        # speed
        fig,axes=create_subplots(figsize=(8.2,4.6)); ax=axes[0]; ax.plot(t/60,v/1e3,color=PALETTE['teal']); ax.set_xlabel('Time [min]'); ax.set_ylabel('Speed [km/s]'); ax.set_title('Orbit sanity check: orbital speed constancy')
        savefig(fig,plots,'orbit_sanity_speed_vs_time',cfg)
    return s.iloc[0].to_dict()

def orekit_verify(cfg,out,plots):
    rows=[]; curves=[]
    for c in cfg['orekit']['cases']:
        period=circular_orbit_period(c['altitude_m']); t=np.linspace(0,c['orbits']*period,int(c['orbits']*cfg['orbit']['samples_per_orbit'])+1)
        ours=propagate_circular_two_body(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg']); ref=propagate_orekit(t,c['altitude_m'],c['inclination_deg'],c['raan_deg'],c['phase_deg'],MU_EARTH,R_EARTH)
        m,pe,ve=comparison_metrics(ours,ref); row={'test':c['name'],'altitude_km':c['altitude_m']/1000,'inclination_deg':c['inclination_deg'],'orbits':c['orbits'],'duration_s':t[-1],**m}; rows.append(row)
        pd.DataFrame({'time_s':t,'position_error_m':pe,'velocity_error_mps':ve}).to_csv(out/f"{c['name']}_timeseries.csv",index=False)
        curves.append((c['name'],t,pe,ve))
    df=pd.DataFrame(rows); df.to_csv(out/'verification_summary.csv',index=False); (out/'verification_summary.json').write_text(df.to_json(orient='records',indent=2))
    if viz_options(cfg)['enabled']:
        # Error trends, normalized to orbit progress for cross-case readability
        fig,axes=create_subplots(figsize=(9.4,5.1)); ax=axes[0]
        for i,(name,t,pe,ve) in enumerate(curves):
            ax.plot(t/60,np.maximum(pe,1e-15),label=name,color=DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
        ax.set_yscale('log'); ax.set_xlabel('Time [min]'); ax.set_ylabel('Position difference [m]'); ax.set_title('Orekit cross-verification: position error'); ax.legend(fontsize=9,ncol=2)
        savefig(fig,plots,'orekit_position_error_vs_time_all_cases',cfg)
        fig,axes=create_subplots(figsize=(9.4,5.1)); ax=axes[0]
        for i,(name,t,pe,ve) in enumerate(curves):
            ax.plot(t/60,np.maximum(ve,1e-18),label=name,color=DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
        ax.set_yscale('log'); ax.set_xlabel('Time [min]'); ax.set_ylabel('Velocity difference [m/s]'); ax.set_title('Orekit cross-verification: velocity error'); ax.legend(fontsize=9,ncol=2)
        savefig(fig,plots,'orekit_velocity_error_vs_time_all_cases',cfg)
        # Summary bars
        labels=df['test'].str.replace('_',' ',regex=False).tolist()
        fig,axes=create_subplots(figsize=(10.5,5.4)); ax=axes[0]; bars=ax.bar(labels,df.position_rmse_m,color=PALETTE['blue_main'],edgecolor='black',linewidth=1); ax.set_yscale('log'); ax.set_ylabel('Position RMSE [m]'); ax.set_title('Orekit cross-verification: position RMSE by scenario'); ax.tick_params(axis='x',rotation=30)
        savefig(fig,plots,'orekit_position_rmse_by_verification_case',cfg)
        fig,axes=create_subplots(figsize=(10.5,5.4)); ax=axes[0]; ax.bar(labels,df.velocity_rmse_mps,color=PALETTE['teal'],edgecolor='black',linewidth=1); ax.set_yscale('log'); ax.set_ylabel('Velocity RMSE [m/s]'); ax.set_title('Orekit cross-verification: velocity RMSE by scenario'); ax.tick_params(axis='x',rotation=30)
        savefig(fig,plots,'orekit_velocity_rmse_by_verification_case',cfg)
    return {'cases':len(df),'max_position_error_m':float(df.position_max_m.max())}

def attitude_run(cfg,out,plots):
    a=cfg['attitude']; s=cfg['sensors']; dt=a['dt_s']; t=np.arange(0,a['duration_s']+dt/2,dt); inertia=tuple(a['inertia_kg_m2'])
    truth=propagate_attitude(t,q0=np.array(a['q0']),omega0=np.array(a['omega0_rad_s']),inertia_diag=inertia)
    E=rotational_energy(truth[:,4:],inertia); H=angular_momentum_norm(truth[:,4:],inertia)
    sensors=simulate_sensors(t,truth,seed=s['seed'],gyro_bias=np.array(s['gyro_bias_rad_s']),gyro_noise_std=s['gyro_noise_std_rad_s'],vector_noise_std=s['vector_noise_std'])
    q,b,d=run_mekf(t,sensors['gyro'],sensors['sun'],sensors['mag'],sensors['sun_i'],sensors['mag_i'],return_diagnostics=True); err=np.array([quat_angle_error_deg(q[k],truth[k,:4]) for k in range(len(t))])
    pd.DataFrame({'time_s':t,'attitude_error_deg':err,'qw_true':truth[:,0],'qx_true':truth[:,1],'qy_true':truth[:,2],'qz_true':truth[:,3],'qw_est':q[:,0],'qx_est':q[:,1],'qy_est':q[:,2],'qz_est':q[:,3], 'bias_x_est':b[:,0],'bias_y_est':b[:,1],'bias_z_est':b[:,2]}).to_csv(out/'phase2_4_timeseries.csv',index=False)
    row={'duration_s':t[-1],'dt_s':dt,'energy_rel_drift':np.ptp(E)/np.mean(E),'momentum_rel_drift':np.ptp(H)/np.mean(H),'attitude_rmse_deg':np.sqrt(np.mean(err**2)),'attitude_max_deg':err.max(),'attitude_final_deg':err[-1],'bias_error_final_rad_s':np.linalg.norm(b[-1]-sensors['gyro_bias_true'])}
    pd.DataFrame([row]).to_csv(out/'phase2_4_summary.csv',index=False)
    if viz_options(cfg)['enabled']:
        # Phase 2: dynamics
        fig,axes=create_subplots(figsize=(8.7,4.8)); ax=axes[0]; make_trend(ax,t,[truth[:,4],truth[:,5],truth[:,6]],['ωx','ωy','ωz'],xlabel='Time [s]',ylabel='Angular velocity [rad/s]'); ax.set_title('Phase 2: true body angular velocity'); ax.legend()
        savefig(fig,plots,'phase2_true_body_angular_velocity_vs_time',cfg)
        erel=(E-E[0])/E[0]; hrel=(H-H[0])/H[0]
        fig,axes=create_subplots(figsize=(8.7,4.8)); ax=axes[0]; ax.plot(t,erel,color=PALETTE['blue_main']); ax.set_xlabel('Time [s]'); ax.set_ylabel('Relative change'); ax.set_title('Phase 2: rotational energy conservation')
        savefig(fig,plots,'phase2_rotational_energy_relative_change_vs_time',cfg)
        fig,axes=create_subplots(figsize=(8.7,4.8)); ax=axes[0]; ax.plot(t,hrel,color=PALETTE['teal']); ax.set_xlabel('Time [s]'); ax.set_ylabel('Relative change'); ax.set_title('Phase 2: angular momentum conservation')
        savefig(fig,plots,'phase2_angular_momentum_relative_change_vs_time',cfg)
        # Phase 3: sensor behaviour
        fig,axes=create_subplots(3,1,figsize=(9.0,8.5),sharex=True)
        for i,(lab,ax) in enumerate(zip(['x','y','z'],axes)):
            ax.plot(t,truth[:,4+i],color=PALETTE['neutral_dark'],lw=1.6,label='True rate'); ax.plot(t,sensors['gyro'][:,i],color=PALETTE['blue_main'],lw=1.0,alpha=.75,label='Gyro measurement'); ax.set_ylabel(f'ω{lab} [rad/s]')
            if i==0: ax.legend(ncol=2)
        axes[-1].set_xlabel('Time [s]'); axes[0].set_title('Phase 3: gyroscope measurements versus true angular rate')
        savefig(fig,plots,'phase3_gyroscope_measurements_vs_true_angular_rate',cfg,pad=.8)
        fig,axes=create_subplots(2,1,figsize=(9.0,7.0),sharex=True)
        for j,(name,arr,ax) in enumerate([('Sun sensor',sensors['sun'],axes[0]),('Magnetometer',sensors['mag'],axes[1])]):
            make_trend(ax,t,[arr[:,0],arr[:,1],arr[:,2]],['x','y','z'],ylabel='Normalized body vector'); ax.set_title(f'Phase 3: {name} body-frame measurement'); ax.legend(ncol=3)
        axes[-1].set_xlabel('Time [s]')
        savefig(fig,plots,'phase3_vector_sensor_body_frame_measurements',cfg,pad=.8)
        # innovations
        fig,axes=create_subplots(2,1,figsize=(9.0,7.0),sharex=True)
        sun_norm=np.linalg.norm(np.nan_to_num(d['sun_innovation']),axis=1); mag_norm=np.linalg.norm(np.nan_to_num(d['mag_innovation']),axis=1)
        axes[0].plot(t,sun_norm,color=PALETTE['blue_secondary']); axes[0].set_ylabel('Sun innovation norm'); axes[0].set_title('Phase 4: MEKF vector-measurement innovations')
        axes[1].plot(t,mag_norm,color=PALETTE['teal']); axes[1].set_ylabel('Mag innovation norm'); axes[1].set_xlabel('Time [s]')
        savefig(fig,plots,'phase4_mekf_measurement_innovation_norms',cfg,pad=.8)
        # Phase 4: estimation quality
        fig,axes=create_subplots(figsize=(8.7,4.8)); ax=axes[0]; ax.plot(t,err,color=PALETTE['red_strong']); ax.set_xlabel('Time [s]'); ax.set_ylabel('Attitude error [deg]'); ax.set_title('Phase 4: MEKF attitude estimation error')
        savefig(fig,plots,'phase4_mekf_attitude_error_vs_time',cfg)
        fig,axes=create_subplots(2,2,figsize=(10.0,7.3),sharex=True); qnames=['qw','qx','qy','qz']
        for i,ax in enumerate(axes):
            ax.plot(t,truth[:,i],color=PALETTE['neutral_dark'],label='True'); ax.plot(t,q[:,i],color=PALETTE['blue_main'],ls='--',label='Estimated'); ax.set_ylabel(qnames[i]);
            if i>=2: ax.set_xlabel('Time [s]')
        axes[0].legend(ncol=2); axes[0].set_title('Phase 4: true and estimated quaternion')
        savefig(fig,plots,'phase4_mekf_true_vs_estimated_quaternion_components',cfg,pad=.8)
        fig,axes=create_subplots(figsize=(8.7,4.8)); ax=axes[0]
        for i,lab in enumerate(['x','y','z']): ax.plot(t,b[:,i],color=DEFAULT_COLORS[i],label=f'Estimated bias {lab}')
        for i in range(3): ax.axhline(sensors['gyro_bias_true'][i],color=DEFAULT_COLORS[i],ls=':',lw=1.5)
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Gyro bias [rad/s]'); ax.set_title('Phase 4: gyroscope bias estimation'); ax.legend(ncol=3)
        savefig(fig,plots,'phase4_mekf_gyroscope_bias_estimate_vs_true',cfg)
    anim_cfg=cfg.get('animation',{})
    if animation_options(cfg)['enabled'] and anim_cfg.get('phase4',{}).get('enabled',True):
        animations=mkdir(out/'animations')
        oc=cfg['orbit']
        create_orbit_attitude_animation(
            animations/'phase4_nominal_true_vs_estimated_orbit_attitude.html',
            t, truth[:,:4], q,
            altitude_m=oc['altitude_m'], inclination_deg=oc['inclination_deg'],
            raan_deg=oc['raan_deg'], phase_deg=oc['phase_deg'],
            sun_i=sensors['sun_i'], mag_i=sensors['mag_i'],
            title='Phase 4 nominal orbit and attitude: true vs MEKF estimate',
            **_animation_kwargs(cfg),
        )
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
        frame=pd.DataFrame({'run_id':rid,'time_s':t,'fault_id':labels,'gyro_res_x':feat[:,0],'gyro_res_y':feat[:,1],'gyro_res_z':feat[:,2],'sun_innovation_norm':feat[:,3],'mag_innovation_norm':feat[:,4],'sun_available':feat[:,5],'mag_available':feat[:,6],'attitude_error_deg':err}); frames.append(frame)
        if viz_options(cfg)['enabled']:
            # One diagnostic figure for every configured fault scenario.
            fig,axes=create_subplots(3,1,figsize=(10.0,9.2),sharex=True)
            axes[0].plot(t,err,color=PALETTE['red_strong']); axes[0].set_ylabel('Attitude error [deg]'); axes[0].set_title(f'Fault scenario {rid}: {fault} ({severity})')
            axes[1].plot(t,feat[:,3],color=PALETTE['blue_main'],label='Sun innovation norm'); axes[1].plot(t,feat[:,4],color=PALETTE['teal'],label='Mag innovation norm'); axes[1].set_ylabel('Innovation norm'); axes[1].legend(ncol=2)
            axes[2].plot(t,feat[:,5]*100,color=PALETTE['blue_secondary'],label='Sun available'); axes[2].plot(t,feat[:,6]*100,color=PALETTE['green_3'],label='Mag available'); axes[2].set_ylabel('Availability [%]'); axes[2].set_xlabel('Time [s]'); axes[2].set_ylim(-5,105); axes[2].legend(ncol=2)
            for ax in axes:
                if fault!='healthy': add_fault_span(ax,f['fault_start_s'],f['fault_end_s'])
            savefig(fig,plots,f'fault_scenario_{rid:02d}_{fault}_{severity}_attitude_innovations_availability',cfg,pad=.8)
        anim_cfg=cfg.get('animation',{})
        phase5_cfg=anim_cfg.get('phase5',{})
        selected=set(int(x) for x in phase5_cfg.get('scenario_ids',[]))
        if animation_options(cfg)['enabled'] and phase5_cfg.get('enabled',True) and rid in selected:
            animations=mkdir(out/'animations'); oc=cfg['orbit']
            create_orbit_attitude_animation(
                animations/f'phase5_scenario_{rid:02d}_{fault}_{severity}_orbit_attitude.html',
                t, truth[:,:4], q,
                altitude_m=oc['altitude_m'], inclination_deg=oc['inclination_deg'],
                raan_deg=oc['raan_deg'], phase_deg=oc['phase_deg'],
                sun_i=faulty['sun_i'], mag_i=faulty['mag_i'],
                fault_name=fault, severity=severity, fault_start_s=f['fault_start_s'], fault_end_s=f['fault_end_s'],
                title=f'Phase 5 scenario {rid}: {fault} ({severity}) orbit and attitude',
                **_animation_kwargs(cfg),
            )
    df=pd.DataFrame(rows); df.to_csv(out/'phase5_scenario_summary.csv',index=False); pd.concat(frames).to_csv(out/'phase5_sample_dataset.csv',index=False)
    if viz_options(cfg)['enabled']:
        labels=[f"{r.fault}\n{r.severity}" for _,r in df.iterrows()]
        vals=df.attitude_rmse_fault_deg.fillna(df.attitude_rmse_all_deg).to_numpy()
        fig,axes=create_subplots(figsize=(11.5,5.6)); ax=axes[0]; colors=[PALETTE['neutral'] if f=='healthy' else PALETTE['red_strong'] for f in df.fault]; ax.bar(labels,vals,color=colors,edgecolor='black',linewidth=1); ax.set_ylabel('Attitude RMSE [deg]'); ax.set_title('Phase 5: estimator degradation by fault scenario'); ax.tick_params(axis='x',rotation=30)
        savefig(fig,plots,'phase5_attitude_rmse_by_fault_scenario',cfg)
        fig,axes=create_subplots(figsize=(11.5,5.6)); ax=axes[0]; ax.bar(labels,df.sun_availability_fault_pct,color=PALETTE['blue_secondary'],edgecolor='black',linewidth=1); ax.set_ylabel('Sun-sensor availability during fault [%]'); ax.set_ylim(0,105); ax.set_title('Phase 5: Sun-sensor availability by fault scenario'); ax.tick_params(axis='x',rotation=30)
        savefig(fig,plots,'phase5_sun_sensor_availability_by_fault_scenario',cfg)
    return {'scenarios':len(df),'max_fault_rmse_deg':float(df.attitude_rmse_fault_deg.max())}

def ml_run(cfg,out,plots):
    m=cfg['ml']; data=mkdir(out/'dataset'); X,y,rid,manifest=generate_monte_carlo_dataset(data,m['runs_per_class'],duration_s=m['duration_s'],dt=m['dt_s'],seed=m['dataset_seed'],force=m['force_regenerate_dataset'])
    split=split_runs(manifest,seed=m['split_seed']); manifest['split']=manifest.run_id.map(split); manifest.to_csv(out/'split_manifest.csv',index=False); results=[]; delays=[]
    if viz_options(cfg)['enabled']:
        counts=manifest.groupby(['fault','split']).size().unstack(fill_value=0).reindex(index=FAULTS,fill_value=0)
        fig,axes=create_subplots(figsize=(10.0,5.4)); ax=axes[0]; make_grouped_bar(ax,counts.index,[counts.get(k,pd.Series(0,index=counts.index)).to_numpy() for k in ['train','val','test']],['Train','Validation','Test'],ylabel='Number of independent runs'); ax.set_title('Phase 6: run-level dataset split by fault class'); ax.tick_params(axis='x',rotation=25); ax.legend(ncol=3)
        savefig(fig,plots,'phase6_run_level_train_validation_test_split_by_class',cfg)
    for w in m['window_lengths']:
        sets={}
        for sp,stride in [('train',m['train_stride']),('val',m['val_stride']),('test',m['test_stride'])]:
            ids=manifest.loc[manifest.split==sp,'run_id'].to_numpy(); sets[sp]=make_windows(X,y,rid,w,ids,stride=stride)
        Xtr,ytr,_,_=sets['train']; Xv,yv,_,_=sets['val']; Xte,yte,rte,ete=sets['test']
        for model in m['models']:
            modelout=mkdir(out/f'{model}_N{w}'); met,pred,cm=train_one(model,w,Xtr,ytr,Xv,yv,Xte,yte,len(FAULTS),epochs=m['epochs'],out_dir=modelout)
            d=detection_delays(yte,pred,rte,ete,manifest,m['dt_s']); d['model']=model; d['window']=w; delays.append(d); met.update({'model':model,'window':w,'train_windows':len(ytr),'val_windows':len(yv),'test_windows':len(yte),'mean_detection_delay_s':float(d.detection_delay_s.mean()) if len(d) else np.nan}); results.append(met)
            if viz_options(cfg)['enabled']:
                # learning curve
                lc=pd.read_csv(modelout/'learning_curve.csv'); fig,axes=create_subplots(figsize=(7.8,4.8)); ax=axes[0]; make_trend(ax,lc.epoch,[lc.train_loss,lc.val_loss],['Training loss','Validation loss'],xlabel='Epoch',ylabel='Cross-entropy loss'); ax.set_title(f'Phase 6: {model.upper()} learning curve, N={w}'); ax.legend()
                savefig(fig,plots,f'phase6_{model}_N{w}_learning_curve_train_vs_validation_loss',cfg)
                # confusion matrix
                fig,axes=create_subplots(figsize=(8.0,6.8)); ax=axes[0]; make_heatmap(ax,cm,x_labels=FAULTS,y_labels=FAULTS,cbar_label='Test windows',annotate=True); ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title(f'Phase 6: {model.upper()} confusion matrix, N={w}')
                savefig(fig,plots,f'phase6_{model}_N{w}_test_confusion_matrix',cfg,pad=.7)
    res=pd.DataFrame(results); res.to_csv(out/'phase6_model_comparison.csv',index=False); dd=pd.concat(delays,ignore_index=True) if delays else pd.DataFrame(); dd.to_csv(out/'phase6_detection_delays.csv',index=False)
    if viz_options(cfg)['enabled'] and len(res):
        specs=[
            ('f1_macro','Macro F1','phase6_macro_f1_vs_window_length'),
            ('accuracy','Accuracy','phase6_accuracy_vs_window_length'),
            ('mean_detection_delay_s','Detection delay [s]','phase6_mean_detection_delay_vs_window_length'),
            ('inference_ms_cpu','CPU inference time [ms]','phase6_cpu_inference_time_vs_window_length'),
            ('model_size_mb','Model size [MB]','phase6_model_size_vs_window_length'),
            ('parameters','Trainable parameters','phase6_parameter_count_vs_window_length'),
        ]
        for metric,label,name in specs:
            fig,axes=create_subplots(figsize=(8.6,5.1)); ax=axes[0]
            for i,(model,g) in enumerate(res.groupby('model')):
                g=g.sort_values('window'); ax.plot(g.window,g[metric],marker='o',ms=6,label=model.upper(),color=DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
            ax.set_xscale('log'); ax.set_xlabel('Window length N [samples]'); ax.set_ylabel(label); ax.set_title(label+' across temporal windows'); ax.legend()
            savefig(fig,plots,name,cfg)
        # Resource/performance overview at all experiments
        fig,axes=create_subplots(figsize=(8.4,5.4)); ax=axes[0]
        markers={'mlp':'o','cnn':'s','gru':'^'}
        for i,(model,g) in enumerate(res.groupby('model')):
            ax.scatter(g.inference_ms_cpu,g.f1_macro,s=np.maximum(45,g.model_size_mb.to_numpy()*500),label=model.upper(),marker=markers.get(model,'o'),color=DEFAULT_COLORS[i%len(DEFAULT_COLORS)],alpha=.8,edgecolor='black',linewidth=.7)
        ax.set_xlabel('CPU inference time [ms]'); ax.set_ylabel('Macro F1'); ax.set_title('Phase 6: fault-detection performance versus inference cost'); ax.legend()
        savefig(fig,plots,'phase6_macro_f1_vs_cpu_inference_time_resource_tradeoff',cfg)
        if len(dd):
            delay_summary=dd.groupby(['model','fault'],as_index=False).detection_delay_s.mean()
            pivot=delay_summary.pivot(index='fault',columns='model',values='detection_delay_s').reindex(index=[f for f in FAULTS if f!='healthy'])
            fig,axes=create_subplots(figsize=(9.0,5.8)); ax=axes[0]; make_grouped_bar(ax,pivot.index,[pivot.get(m,pd.Series(np.nan,index=pivot.index)).to_numpy() for m in m['models']],[x.upper() for x in m['models']],ylabel='Mean detection delay [s]'); ax.set_title('Phase 6: mean detection delay by fault class'); ax.tick_params(axis='x',rotation=25); ax.legend(ncol=len(m['models']))
            savefig(fig,plots,'phase6_mean_detection_delay_by_fault_class_and_model',cfg)
    return {'experiments':len(res),'dataset_runs':len(manifest)}

def scenario_run(cfg,out,plots):
    sc=cfg.get('scenario',{})
    name=sc.get('active','sun_pointing')
    scenario_cfg=cfg.get('scenarios',{}).get(name,{})
    result=run_scenario(name,scenario_cfg,cfg,out)
    return {'active_scenario':name, **{k:v for k,v in result.items() if isinstance(v,(str,int,float,bool))}}

def main():
    with open(CONFIG,'rb') as f: cfg=tomllib.load(f)
    v=viz_options(cfg); apply_publication_style(FigureStyle(font_size=v['font_size'],axes_linewidth=v['axes_linewidth']))
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'); run_dir=mkdir(ROOT/cfg['run']['results_root']/f'results_{stamp}'); shutil.copy2(CONFIG,run_dir/'config_used.toml')
    manifest={'run_name':cfg['run']['name'],'started_at':datetime.now().isoformat(),'python':sys.version,'config':str(CONFIG),'visualization':v,'animation':animation_options(cfg),'steps':{}}; (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[('unit_tests',run_tests),('orbit_sanity',orbit_sanity),('orekit_verification',orekit_verify),('attitude_sensors_ekf',attitude_run),('fault_campaign',fault_run),('ml_training',ml_run),('scenario',scenario_run)]
    print(f'\nResults directory: {run_dir}\n')
    for name,fn in jobs:
        if not cfg['steps'].get(name,False): print(f'[SKIP] {name}'); manifest['steps'][name]={'status':'skipped'}; continue
        print(f'[RUN ] {name}'); t0=time.time(); phase=mkdir(run_dir/name); plots=mkdir(phase/'plots')
        try:
            result=fn(cfg,phase,plots)
            manifest['steps'][name]={'status':'completed','elapsed_s':time.time()-t0,'summary':result}; print(f'[ OK ] {name} ({time.time()-t0:.1f}s)')
        except Exception as e:
            (phase/'ERROR.txt').write_text(traceback.format_exc()); manifest['steps'][name]={'status':'failed','elapsed_s':time.time()-t0,'error':str(e)}; print(f'[FAIL] {name}: {e}')
            (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
            if cfg['run']['stop_on_error']: raise
    manifest['finished_at']=datetime.now().isoformat(); (run_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)); print(f'\nFinished. All outputs are in:\n{run_dir}')
if __name__=='__main__': main()
