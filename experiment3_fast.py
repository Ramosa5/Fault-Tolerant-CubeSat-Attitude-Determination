"""Fast Experiment 3: reduced 8-case OFAT campaign with 3 independent workers.

Run from multifault_v2: python -u experiment3_fast.py --resume --workers 3
Only the parent process writes aggregate CSVs/plots. Each worker owns a case folder.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
import pandas as pd
import experiment3 as core

ROOT = Path(__file__).resolve().parent
BASE_CASES = [
    ('inclination','5'),('inclination','45'),('inclination','85'),
    ('mass','2'),('mass','12'),
    ('geometry','1U'),('geometry','3U'),('geometry','6U'),
]

def selected_manifest(dt):
    import tomllib
    with (ROOT/'experiment_config.toml').open('rb') as f: cfg=tomllib.load(f)
    df=core.build_manifest(float(cfg['orbit']['altitude_m']),dt)
    keys={(s,v) for s,v in BASE_CASES}
    df=df[df.apply(lambda r:(r['sweep'],str(r['value'])) in keys,axis=1)].copy()
    order={k:i for i,k in enumerate(BASE_CASES)}
    df['sort_key']=df.apply(lambda r:order[(r['sweep'],str(r['value']))],axis=1)
    return df.sort_values('sort_key').drop(columns='sort_key').reset_index(drop=True)

def result_path(out,exp,case_id):
    return out/'runs'/f'exp{exp}'/case_id

def summary_name(exp):
    return 'reaction_wheel_fair_ml_campaign_summary.csv' if exp==1 else 'reaction_wheel_ml_fdir_campaign_summary.csv'

def normalized_valid(folder,exp):
    p=folder/'exp3_normalized.csv'
    if not p.exists() or p.stat().st_size==0:return False
    try:
        df=pd.read_csv(p)
        return not df.empty and {'method','case_id','experiment'}.issubset(df.columns)
    except Exception:return False

def restore_old(out, exp, row, old):
    """Copy only matching completed old experiment3 runs after verifying physical setup."""
    source=result_path(old,exp,row.case_id)
    target=result_path(out,exp,row.case_id)
    if normalized_valid(target,exp):return False
    conf=source/'case_config.json'
    sm=source/summary_name(exp)
    if not conf.exists() or not sm.exists():return False
    try:
        old_cfg=json.loads(conf.read_text(encoding='utf-8'))
        orbit=old_cfg['orbit']
        sc=old_cfg['spacecraft']
        if abs(float(orbit['inclination_deg'])-float(row.inclination_deg))>1e-8:return False
        if abs(float(sc['mass_kg'])-float(row.mass_kg))>1e-8:return False
        if list(sc['dimensions_m'])!=json.loads(row.dimensions_m):return False
        ec=old_cfg['experiment_config']
        duration=float(ec['duration_s'] if exp==1 else ec['evaluation']['duration_s'])
        if abs(duration-float(row.duration_s))>1e-5:return False
        if abs(float(ec['dt_s'])-float(row.dt_actual_s))>1e-8:return False
        # Reuse only the exact same experiment input, not merely a matching filename.
        if exp==1:
            old_cases=ec['cases']
            if len(old_cases)!=1 or old_cases[0].get('name')!='sun_sensor_loss':return False
        tmp=target.with_name(target.name+'.importing')
        if tmp.exists():shutil.rmtree(tmp)
        shutil.copytree(source,tmp)
        if target.exists():shutil.rmtree(target)
        tmp.rename(target)
        return True
    except Exception:return False

def worker(exp, row, folder, cfg_path):
    """Invoked as a separate Python process: no shared global state with other workers."""
    import tomllib
    with cfg_path.open('rb') as f:base_cfg=tomllib.load(f)
    folder.mkdir(parents=True,exist_ok=True)
    config,ec=core._set_config(base_cfg,row,exp,99501)
    (folder/'case_config.json').write_text(json.dumps({
      'orbit':config['orbit'],
      'spacecraft':config['scenarios']['reaction_wheel_sun_pointing_comparison']['spacecraft'],
      'experiment_config':ec},indent=2,default=str),encoding='utf-8')
    print(f'[WORKER] START exp{exp}/{row.case_id}',flush=True)
    if exp==1:core.experiment1.run(ec,config,folder)
    else:core.experiment2.run(ec,config,folder)
    normalized=core._collect_exp1(folder,row) if exp==1 else core._collect_exp2(folder,row)
    if normalized.empty:raise RuntimeError('Experiment returned an empty normalized summary')
    normalized.to_csv(folder/'exp3_normalized.csv',index=False)
    print(f'[WORKER] DONE exp{exp}/{row.case_id}',flush=True)

def execute_task(task, root, script, env):
    exp,row=task
    folder=result_path(root,exp,row.case_id)
    folder.mkdir(parents=True,exist_ok=True)
    log=folder/'worker.log'
    args=[sys.executable,'-u',str(script),'--worker',str(exp),str(row.case_id),'--results',str(root)]
    start=time.monotonic()
    with log.open('a',encoding='utf-8',buffering=1) as f:
        f.write(f'\n===== START {datetime.now().isoformat(timespec="seconds")} =====\n')
        proc=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,check=False)
        f.write(f'===== END exit={proc.returncode} wall={time.monotonic()-start:.1f}s =====\n')
    return proc.returncode, time.monotonic()-start, str(log)

def rebuild(out,exps):
    for exp in exps:
        parts=sorted((out/'runs'/f'exp{exp}').glob('*/exp3_normalized.csv'))
        dfs=[]
        for p in parts:
            try:
                df=pd.read_csv(p)
                if not df.empty:dfs.append(df)
            except Exception:pass
        if dfs:pd.concat(dfs,ignore_index=True).to_csv(out/f'experiment{exp}.csv',index=False)
    try:core.plots(out)
    except Exception as exc:print(f'[PLOT WARNING] {exc}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--experiment',choices=['both','1','2'],default='both')
    p.add_argument('--results',default='results/experiment3_fast')
    p.add_argument('--old-results',default='results/experiment3',help='Previously completed campaign to import from')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dt',type=float,default=0.1)
    p.add_argument('--heartbeat',type=int,default=30)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-import',action='store_true',help='Do not copy completed runs from older campaign')
    p.add_argument('--worker',nargs=2,metavar=('EXPERIMENT','CASE'))
    a=p.parse_args()
    if not 1<=a.workers<=6: p.error('--workers must be between 1 and 6')
    if a.dt<=0 or a.dt>10:p.error('--dt must be in (0,10]')
    if a.heartbeat<1:p.error('--heartbeat must be >= 1')
    out=(ROOT/a.results).resolve();out.mkdir(parents=True,exist_ok=True)
    man=selected_manifest(a.dt)
    if len(man)!=8:raise RuntimeError(f'Expected 8 cases, got {len(man)}')
    exps=[1,2] if a.experiment=='both' else [int(a.experiment)]
    if a.worker:
        exp=int(a.worker[0]);case_id=a.worker[1]
        rows=man[man.case_id==case_id]
        if len(rows)!=1 or exp not in [1,2]:raise ValueError('Invalid worker case')
        try:worker(exp,rows.iloc[0],result_path(out,exp,case_id),ROOT/'experiment_config.toml')
        except BaseException:
            folder=result_path(out,exp,case_id)
            (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            traceback.print_exc();sys.exit(1)
        return
    man.to_csv(out/'manifest.csv',index=False)
    (out/'color_palette.json').write_text(json.dumps(core.PALETTE,indent=2),encoding='utf-8')
    tasks=[(exp,row) for _,row in man.iterrows() for exp in exps]
    print(f'[FAST] {len(man)} OFAT cases, {len(tasks)} runs, 2 orbits each, {a.workers} parallel workers',flush=True)
    for _,r in man.iterrows(): print(f'  {r.case_id}: i={r.inclination_deg}deg m={r.mass_kg}kg {r.form_factor}',flush=True)
    if a.dry_run: print(f'[FAST] Manifest saved: {out/"manifest.csv"}',flush=True);return
    old=(ROOT/a.old_results).resolve()
    statuses={}
    pending=[]
    for exp,row in tasks:
        key=f'exp{exp}/{row.case_id}'
        folder=result_path(out,exp,row.case_id)
        if a.resume and normalized_valid(folder,exp):
            statuses[key]='skipped';continue
        if a.resume and not a.no_import and old!=out and restore_old(out,exp,row,old):
            statuses[key]='imported';print(f'[IMPORT] {key}',flush=True);continue
        pending.append((exp,row))
    print(f'[FAST] reuse={len(statuses)} pending={len(pending)}',flush=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='2'
    start=time.monotonic()
    with (out/'campaign_fast.log').open('a',encoding='utf-8',buffering=1) as log:
        def emit(message):
            print(message,flush=True);log.write(message+'\n')
        def status_csv():
            pd.DataFrame([{'run':k,'status':v} for k,v in sorted(statuses.items())]).to_csv(out/'status_fast.csv',index=False)
        status_csv()
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            # Each thread supervises an independent OS process, not CPU-bound Python work.
            futures={pool.submit(execute_task,t,out,Path(__file__).resolve(),env):t for t in pending}
            active=set(futures)
            last=time.monotonic()
            while active:
                completed,active=cf.wait(active,timeout=1,return_when=cf.FIRST_COMPLETED)
                for fut in completed:
                    exp,row=futures[fut];key=f'exp{exp}/{row.case_id}'
                    try:
                        code,wall,log_path=fut.result()
                        ok=code==0 and normalized_valid(result_path(out,exp,row.case_id),exp)
                        statuses[key]='ok' if ok else 'failed'
                        emit(f'[END] {key}: {statuses[key]} time={core.fmt_seconds(wall)} log={log_path}')
                    except Exception as exc:
                        statuses[key]='failed';emit(f'[ERROR] {key}: {exc!r}')
                    status_csv()
                    rebuild(out,exps)
                now=time.monotonic()
                if now-last>=a.heartbeat:
                    done=len(statuses);total=len(tasks)
                    current=[f'exp{futures[f][0]}/{futures[f][1].case_id}' for f in active]
                    emit(f'[PROGRESS] {done}/{total} ({100*done/total:.1f}%) | elapsed {core.fmt_seconds(now-start)} | active: {", ".join(current)}')
                    last=now
        rebuild(out,exps)
        emit(f'[FINISH] total={len(tasks)} ok={sum(v=="ok" for v in statuses.values())} reused={sum(v in ("skipped","imported") for v in statuses.values())} failed={sum(v=="failed" for v in statuses.values())} elapsed={core.fmt_seconds(time.monotonic()-start)}')

if __name__=='__main__':main()
