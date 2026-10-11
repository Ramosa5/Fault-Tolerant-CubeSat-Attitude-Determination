"""Experiment 3: OFAT robustness campaign calling REAL Experiment 1 and 2.

Usage: python experiment3.py --dry-run
       python experiment3.py --limit 1 --experiment 1
       python experiment3.py --experiment both --resume
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import traceback
import time
import threading
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.simulation.orbit_model import circular_orbit_period
from src.simulation.spacecraft import box_inertia_diag
from src.scenarios import reaction_wheel_fair_ml_campaign as experiment1
from src.scenarios import reaction_wheel_ml_fdir_campaign as experiment2

ROOT = Path(__file__).resolve().parent
PALETTE = {'EKF':'#3E4451','MLP':'#0072B2','CNN':'#D55E00','GRU':'#009E73'}
# Candidate CubeSat envelopes (m). Approximate standard outer-envelope dimensions;
# deployables, rails, separation hardware and constraints are not modelled.
FORM_FACTORS = {
 '1U':([.10,.10,.10],2.0),
 '2U':([.10,.10,.20],4.0),
 '3U':([.10,.10,.30],6.0),
 '6U':([.20,.10,.30],12.0),
 '12U':([.20,.20,.30],24.0),
}
BASE = {'mass_kg':4.0, 'dimensions_m':[.20,.10,.30], 'form_factor':'6U', 'inclination_deg':45.0}

def cases():
    # Independent OFAT variations; one common reference case is shared.
    out=[]
    for inc in [5,25,45,65,85]:
        out.append(('inclination',str(inc),{**BASE,'inclination_deg':float(inc)}))
    for m in [1,2,4,6,12]:
        out.append(('mass',str(m),{**BASE,'mass_kg':float(m)}))
    for ff,(dims,max_mass) in FORM_FACTORS.items():
        # constant 1 kg reference mass ensures every geometry is within envelope mass limit
        out.append(('geometry',ff,{**BASE,'mass_kg':1.0,'dimensions_m':dims,'form_factor':ff}))
    return out

def build_manifest(altitude_m, dt):
    result=[]
    period=float(circular_orbit_period(altitude_m))
    # sample grid explicitly includes time at exactly two orbital periods
    steps=max(2, math.ceil(2*period/dt))
    actual_dt=2*period/steps
    for sweep, value, c in cases():
        inertia=box_inertia_diag(c['mass_kg'],c['dimensions_m'])
        result.append({'case_id':f'{sweep}_{value.replace(".","p")}', 'sweep':sweep,'value':value,
          'mass_kg':c['mass_kg'],'form_factor':c['form_factor'], 'dimensions_m':json.dumps(c['dimensions_m']),
          'inclination_deg':c['inclination_deg'],'altitude_m':altitude_m,'raan_deg':0.0,'phase_deg':0.0,
          'orbital_period_s':period,'duration_s':2*period,'dt_requested_s':dt,'dt_actual_s':actual_dt,
          'n_steps':steps, 'Jxx_kg_m2':inertia[0],'Jyy_kg_m2':inertia[1],'Jzz_kg_m2':inertia[2]})
    return pd.DataFrame(result)

def _set_config(base_cfg,row,exp,seed):
    cfg=copy.deepcopy(base_cfg)
    cfg['orbit'].update({k:float(row[k]) for k in ['altitude_m','inclination_deg','raan_deg','phase_deg']})
    satellite=cfg['scenarios']['reaction_wheel_sun_pointing_comparison']['spacecraft']
    satellite.update(mass_kg=float(row.mass_kg),dimensions_m=json.loads(row.dimensions_m))
    satellite.pop('inertia_kg_m2',None)
    # Orbit profile 'global' explicitly references the user-controlled orbit, rather than fixed high-beta.
    b=cfg['scenarios']['reaction_wheel_sun_pointing_comparison']; b['orbit_profile']='global'
    name='reaction_wheel_fair_ml_campaign' if exp==1 else 'reaction_wheel_ml_fdir_campaign'
    ec=copy.deepcopy(cfg['scenarios'][name]); ec['orbit_profile']='global'; ec['dt_s']=float(row.dt_actual_s)
    ec['plots_enabled']=False; ec['save_timeseries']=False
    if exp==1:
        ec['duration_s']=float(row.duration_s)
        ec['seed']=seed
        # One variant is chosen and held fixed across all configurations.
        ec['models']['architectures']=['mlp','cnn','gru']
        ec['models']['variants']=['sensor_only']
        ec['animation']={'enabled':False}
        ec['models']['orbit_uncertainty_sweep']={'enabled':False}
        ec['models']['dataset']['seed']=int(ec['models']['dataset'].get('seed',99501))
        # 2-orbit fault centred at 1 orbit; same timing and severity for every case.
        ec['cases']=[{'name':'sun_sensor_loss','group':'A_single_sensor_loss','sun':{'type':'sun_loss','severity':'high'},'gyro':{'type':'none'},'mag':{'type':'none'},
                      'start_s':float(row.duration_s*.45),'end_s':float(row.duration_s*.75)}]
    else:
        ec['evaluation']['duration_s']=float(row.duration_s)
        ec['evaluation']['fault_start_s']=float(row.duration_s*.45)
        ec['evaluation']['fault_end_s']=float(row.duration_s*.75)
        ec['models']['architectures']=['mlp','cnn','gru']
        # Frozen training and evaluation seeds for paired comparisons.
        ec['evaluation']['seed']=int(ec['evaluation'].get('seed',17011))
    return cfg,ec

def _collect_exp1(folder,row):
    df=pd.read_csv(folder/'reaction_wheel_fair_ml_campaign_summary.csv')
    # EKF baseline is faulty_mekf (not healthy reference). No hidden averaging of variants.
    df=df[df['branch'].isin(['faulty_mekf','mlp_sensor_only','cnn_sensor_only','gru_sensor_only'])].copy()
    df['method']=df['architecture'].str.upper(); df.loc[df.branch=='faulty_mekf','method']='EKF'
    df['experiment']=1; df['case_id']=row.case_id; df['sweep']=row.sweep; df['value']=row.value
    return df

def _collect_exp2(folder,row):
    df=pd.read_csv(folder/'reaction_wheel_ml_fdir_campaign_summary.csv')
    # Original experiment 2 measures FDIR detection, not synthetic sensor replacement.
    # Classical FDIR is the benchmark; hybrid_* mixes ML with classical detector.
    mapping={'classical_v31':'EKF','hybrid_mlp':'MLP','hybrid_cnn':'CNN','hybrid_gru':'GRU'}
    df=df[df.detector.isin(mapping)].copy(); df['method']=df.detector.map(mapping)
    df['experiment']=2; df['case_id']=row.case_id; df['sweep']=row.sweep; df['value']=row.value
    return df

def _fig(df,y,title,path):
    if y not in df: return
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for ax,(sweep,g) in zip(axes,df.groupby('sweep',sort=False)):
        order=[str(c[1]) for c in cases() if c[0]==sweep]
        for method in ['EKF','MLP','CNN','GRU']:
            u=g[g.method==method].copy()
            if u.empty: continue
            series=u.groupby('value')[y].mean().reindex(order)
            ax.plot(range(len(order)),series.to_numpy(),label=method,color=PALETTE[method],marker='o',linewidth=2)
        ax.set_xticks(range(len(order)),order); ax.set_xlabel(sweep);ax.set_title(sweep.capitalize());ax.grid(alpha=.2)
        ax.set_ylabel(y.replace('_',' '))
    axes[0].legend(ncol=4,loc='best');fig.suptitle(title)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.png'),dpi=240,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def plots(result_dir):
    files={1:result_dir/'experiment1.csv',2:result_dir/'experiment2.csv'}
    for exp,p in files.items():
        if not p.exists(): continue
        df=pd.read_csv(p)
        if df.empty:continue
        if exp==1:
            for y in ['sunlight_fault_pointing_rmse_deg','sunlight_fault_attitude_rmse_deg','recovery_time_s','pointing_improvement_vs_fdir_pct']:
                _fig(df,y,f'Experiment 1 · {y}',result_dir/'figures'/f'exp1_{y}')
        else:
            for y in ['macro_f1_pct','mean_false_alarm_rate_pct','mean_detection_delay_s','macro_recall_pct']:
                _fig(df,y,f'Experiment 2 · {y}',result_dir/'figures'/f'exp2_{y}')

class Tee:
    """Mirror experiment stdout/stderr to console and a persistent log file."""
    def __init__(self, terminal, logfile):
        self.terminal=terminal
        self.logfile=logfile
    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    def isatty(self):
        return self.terminal.isatty()


def fmt_seconds(seconds):
    seconds=int(max(0, seconds))
    hours, remainder=divmod(seconds,3600)
    minutes, seconds=divmod(remainder,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def progress_line(completed,total,started,case,experiment,elapsed_case,successful,failed):
    pct=100*completed/total if total else 100.0
    width=24
    filled=round(width*completed/total) if total else width
    bar='█'*filled+'░'*(width-filled)
    elapsed=time.monotonic()-started
    eta=elapsed/completed*(total-completed) if completed else None
    eta_text=fmt_seconds(eta) if eta is not None else 'calculating'
    return (f'[CAMPAIGN] [{bar}] {completed}/{total} ({pct:.1f}%) | '
            f'elapsed {fmt_seconds(elapsed)} | ETA {eta_text} | '
            f'OK {successful}, failed {failed} | current exp{experiment}/{case} '
            f'({fmt_seconds(elapsed_case)})')


def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--limit',type=int)
    ap.add_argument('--experiment',choices=['1','2','both'],default='both')
    ap.add_argument('--results',default='results/experiment3');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--dt',type=float,default=.1);ap.add_argument('--plots-only',action='store_true')
    ap.add_argument('--heartbeat',type=float,default=30.0,help='Seconds between progress updates (default: 30; 0 disables)')
    a=ap.parse_args(); out=ROOT/a.results;out.mkdir(parents=True,exist_ok=True)
    if a.plots_only: plots(out);return
    if not (0<a.dt<=10):raise ValueError('--dt must be >0 and <=10 seconds')
    with (ROOT/'experiment_config.toml').open('rb') as f: base_cfg=tomllib.load(f)
    man=build_manifest(float(base_cfg['orbit']['altitude_m']),a.dt)
    man.to_csv(out/'manifest.csv',index=False)
    (out/'color_palette.json').write_text(json.dumps(PALETTE,indent=2))
    if a.dry_run:
        print(f'Manifest created: {len(man)} configurations, two full orbits each; {out}');return
    exps=[1,2] if a.experiment=='both' else [int(a.experiment)]
    selected=man.iloc[:a.limit] if a.limit is not None else man
    if a.heartbeat < 0:
        raise ValueError('--heartbeat must be >= 0')
    tasks=[(index,row,exp) for index,row in selected.iterrows() for exp in exps]
    total=len(tasks)
    started=time.monotonic()
    log=[]
    with (out/'campaign.log').open('a',encoding='utf-8',buffering=1) as lf:
        with redirect_stdout(Tee(sys.stdout,lf)),redirect_stderr(Tee(sys.stderr,lf)):
            print(f'\n=== Experiment 3 started {datetime.now().astimezone().isoformat(timespec="seconds")} ===',flush=True)
            print(f'Cases: {len(selected)} | Experiments: {exps} | Runs: {total} | dt: {a.dt}s | heartbeat: {a.heartbeat}s',flush=True)
            print(f'Logging to {out / "campaign.log"}',flush=True)
            for task_number,(index,row,exp) in enumerate(tasks,1):
                folder=out/'runs'/f'exp{exp}'/row.case_id
                expected=folder/('reaction_wheel_fair_ml_campaign_summary.csv' if exp==1 else 'reaction_wheel_ml_fdir_campaign_summary.csv')
                case_started=time.monotonic()
                stop=threading.Event()
                completed=task_number-1
                ok=sum(x['status'] in ('ok','skipped') for x in log)
                failed=sum(x['status']=='failed' for x in log)
                print('\n'+'='*72,flush=True)
                print(f'[START {task_number}/{total}] exp{exp} | {row.case_id} | '
                      f'inclination={row.inclination_deg:g}° mass={row.mass_kg:g}kg form={row.form_factor} | '
                      f'2 orbits={row.duration_s/60:.1f} simulated minutes | '
                      f'wall time {datetime.now().astimezone().isoformat(timespec="seconds")}',flush=True)
                def heartbeat():
                    while not stop.wait(a.heartbeat):
                        print(progress_line(completed,total,started,row.case_id,exp,
                                            time.monotonic()-case_started,ok,failed),flush=True)
                watcher=None
                if a.heartbeat>0:
                    watcher=threading.Thread(target=heartbeat,name='experiment3-progress',daemon=True)
                    watcher.start()
                status='ok'
                error=''
                try:
                    if a.resume and expected.exists():
                        print(f'[RESUME] Existing output detected: {expected}; validating summary.',flush=True)
                        status='skipped'
                    else:
                        folder.mkdir(parents=True,exist_ok=True)
                        cfg,ec=_set_config(base_cfg,row,exp,99501)
                        (folder/'case_config.json').write_text(json.dumps({'orbit':cfg['orbit'],'spacecraft':cfg['scenarios']['reaction_wheel_sun_pointing_comparison']['spacecraft'],'experiment_config':ec},indent=2,default=str))
                        print('[RUNNING] Calling original experiment code. Its own messages appear below.',flush=True)
                        if exp==1: experiment1.run(ec,cfg,folder)
                        else: experiment2.run(ec,cfg,folder)
                    summary=_collect_exp1(folder,row) if exp==1 else _collect_exp2(folder,row)
                    summary.to_csv(folder/'exp3_normalized.csv',index=False)
                except Exception as err:
                    status='failed'
                    error=repr(err)
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    print(f'[ERROR] {row.case_id} exp{exp}: {err}',flush=True)
                    traceback.print_exc()
                finally:
                    stop.set()
                    if watcher: watcher.join(timeout=1.0)
                    elapsed_case=time.monotonic()-case_started
                    log.append({'experiment':exp,'case_id':row.case_id,'status':status,
                                'wall_time_s':round(elapsed_case,3),'error':error})
                    print(f'[END {task_number}/{total}] status={status.upper()} | duration={fmt_seconds(elapsed_case)}',flush=True)
                    pd.DataFrame(log).to_csv(out/'status.csv',index=False)
                    for e in exps:
                        parts=list((out/'runs'/f'exp{e}').glob('*/exp3_normalized.csv'))
                        if parts:
                            pd.concat([pd.read_csv(q) for q in parts],ignore_index=True).to_csv(out/f'experiment{e}.csv',index=False)
                    try:
                        plots(out)
                    except Exception as plot_error:
                        print(f'[PLOT WARNING] {plot_error}',flush=True)
                    print(progress_line(task_number,total,started,row.case_id,exp,elapsed_case,
                                        sum(x['status'] in ('ok','skipped') for x in log),
                                        sum(x['status']=='failed' for x in log)),flush=True)
            print(f'Campaign wall time: {fmt_seconds(time.monotonic()-started)}',flush=True)
    print(f'Finished. Results: {out}. Failed runs: {sum(x["status"]=="failed" for x in log)}')

if __name__=='__main__':run()
