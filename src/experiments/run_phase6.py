from pathlib import Path
import argparse, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.ml.dataset import generate_monte_carlo_dataset,split_runs,make_windows,FAULTS
from src.ml.training import train_one

WINDOWS=(1,5,10,20,50,100); MODELS=('mlp','cnn','gru')

def detection_delays(y_true,y_pred,runs,end_idx,manifest,dt):
    rows=[]
    for r in np.unique(runs):
        m=manifest.loc[manifest.run_id==r].iloc[0]
        if m.fault=='healthy' or not np.isfinite(m.fault_start_s): continue
        mask=runs==r; e=end_idx[mask]; yp=y_pred[mask]; times=e*dt; after=times>=m.fault_start_s
        # first correct nonhealthy class prediction after onset
        target=int(m.class_id); hits=np.flatnonzero(after & (yp==target))
        delay=float(times[hits[0]]-m.fault_start_s) if len(hits) else np.nan
        rows.append({'run_id':int(r),'fault':m.fault,'detection_delay_s':delay})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs-per-class',type=int,default=12); ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--duration',type=float,default=300.0); ap.add_argument('--windows',default='1,5,10,20,50,100'); ap.add_argument('--models',default='mlp,cnn,gru'); ap.add_argument('--force-data',action='store_true')
    args=ap.parse_args(); windows=tuple(int(x) for x in args.windows.split(',')); models=tuple(x.strip().lower() for x in args.models.split(','))
    root=Path('data/results/phase6'); root.mkdir(parents=True,exist_ok=True); plots=Path('plots/phase6'); plots.mkdir(parents=True,exist_ok=True)
    X,y,rid,manifest=generate_monte_carlo_dataset(root/'dataset',args.runs_per_class,duration_s=args.duration,force=args.force_data); split=split_runs(manifest)
    manifest['split']=manifest.run_id.map(split); manifest.to_csv(root/'split_manifest.csv',index=False)
    results=[]; delays=[]; dt=.1
    print(f'Phase 6 dataset: {len(manifest)} independent runs ({args.runs_per_class}/class)')
    print(manifest.groupby(['fault','split']).size().unstack(fill_value=0).to_string())
    for w in windows:
        sets={}
        for s in ('train','val','test'):
            ids=manifest.loc[manifest.split==s,'run_id'].to_numpy(); sets[s]=make_windows(X,y,rid,w,ids,stride=5 if s!='test' else 1)
        Xtr,ytr,_,_=sets['train']; Xv,yv,_,_=sets['val']; Xte,yte,rte,ete=sets['test']
        for model in models:
            print(f'\nTraining {model.upper()} N={w}: train={len(ytr)}, val={len(yv)}, test={len(yte)}')
            out=root/f'{model}_N{w}'; met,pred,cm=train_one(model,w,Xtr,ytr,Xv,yv,Xte,yte,len(FAULTS),epochs=args.epochs,out_dir=out)
            d=detection_delays(yte,pred,rte,ete,manifest,dt); d['model']=model; d['window']=w; delays.append(d)
            met.update({'model':model,'window':w,'train_windows':len(ytr),'val_windows':len(yv),'test_windows':len(yte),'mean_detection_delay_s':float(d.detection_delay_s.mean()) if len(d) else np.nan})
            results.append(met); print(f"  F1={met['f1_macro']:.4f}, accuracy={met['accuracy']:.4f}, delay={met['mean_detection_delay_s']:.3f}s, params={met['parameters']}")
    res=pd.DataFrame(results); res.to_csv(root/'phase6_model_comparison.csv',index=False)
    dd=pd.concat(delays,ignore_index=True) if delays else pd.DataFrame(); dd.to_csv(root/'phase6_detection_delays.csv',index=False)
    for metric,ylabel in [('f1_macro','Macro F1'),('mean_detection_delay_s','Detection delay [s]')]:
        fig=plt.figure(figsize=(8,5))
        for m,g in res.groupby('model'): plt.plot(g.window,g[metric],marker='o',label=m.upper())
        plt.xscale('log'); plt.xlabel('Window length N [samples]'); plt.ylabel(ylabel); plt.grid(True,alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(plots/f'{metric}_vs_window.png',dpi=180); plt.close(fig)
    print('\nPHASE 6 MODEL COMPARISON'); print(res.sort_values(['model','window']).to_string(index=False)); print('\nSaved:',root/'phase6_model_comparison.csv')
if __name__=='__main__': main()
