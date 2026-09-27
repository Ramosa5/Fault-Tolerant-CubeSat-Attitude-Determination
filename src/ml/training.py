from pathlib import Path
import time, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from .models import build_model,count_parameters

def standardize(train,val,test):
    mu=train.mean(axis=(0,1),keepdims=True); sd=train.std(axis=(0,1),keepdims=True); sd[sd<1e-8]=1
    return (train-mu)/sd,(val-mu)/sd,(test-mu)/sd,mu,sd

def train_one(model_name,window,Xtr,ytr,Xv,yv,Xte,yte,n_classes,epochs=12,batch_size=256,seed=42,out_dir=None):
    torch.manual_seed(seed); np.random.seed(seed); device=torch.device('cpu')
    Xtr,Xv,Xte,mu,sd=standardize(Xtr,Xv,Xte)
    model=build_model(model_name,Xtr.shape[2],window,n_classes).to(device)
    # balanced class weights from training data
    counts=np.bincount(ytr,minlength=n_classes); weights=len(ytr)/(n_classes*np.maximum(counts,1)); criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    tr=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=batch_size,shuffle=True)
    xv=torch.from_numpy(Xv); yv_t=torch.from_numpy(yv); best=None; best_loss=float('inf'); hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0.; n=0
        for xb,yb in tr:
            opt.zero_grad(); loss=criterion(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(xb); n+=len(xb)
        model.eval()
        with torch.no_grad(): vl=criterion(model(xv),yv_t).item()
        hist.append({'epoch':ep,'train_loss':total/n,'val_loss':vl})
        if vl<best_loss: best_loss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval(); xt=torch.from_numpy(Xte)
    with torch.no_grad(): pred=model(xt).argmax(1).numpy()
    p,r,f,_=precision_recall_fscore_support(yte,pred,average='macro',zero_division=0); acc=accuracy_score(yte,pred); cm=confusion_matrix(yte,pred,labels=np.arange(n_classes))
    # CPU inference timing, batch=1
    one=xt[:1]; reps=min(1000,max(100,len(xt))); t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(reps): model(one)
    infer_ms=(time.perf_counter()-t0)*1000/reps
    params=count_parameters(model); size_mb=sum(p.numel()*p.element_size() for p in model.parameters())/1024**2
    if out_dir:
        out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'mean':mu,'std':sd},out/'model.pt'); pd.DataFrame(hist).to_csv(out/'learning_curve.csv',index=False); np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
    return {'accuracy':acc,'precision_macro':p,'recall_macro':r,'f1_macro':f,'parameters':params,'model_size_mb':size_mb,'inference_ms_cpu':infer_ms,'best_val_loss':best_loss},pred,cm
