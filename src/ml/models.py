import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, n_features, window, n_classes, hidden=(64,32)):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(n_features*window,hidden[0]),nn.ReLU(),nn.Dropout(.1),
                               nn.Linear(hidden[0],hidden[1]),nn.ReLU(),nn.Linear(hidden[1],n_classes))
    def forward(self,x): return self.net(x)

class CNN1D(nn.Module):
    def __init__(self,n_features,window,n_classes):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(n_features,24,3,padding=1),nn.ReLU(),
                                    nn.Conv1d(24,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Linear(32,n_classes)
    def forward(self,x):
        x=x.transpose(1,2); return self.head(self.features(x).squeeze(-1))

class GRUClassifier(nn.Module):
    def __init__(self,n_features,window,n_classes,hidden=32):
        super().__init__(); self.gru=nn.GRU(n_features,hidden,batch_first=True); self.head=nn.Linear(hidden,n_classes)
    def forward(self,x):
        _,h=self.gru(x); return self.head(h[-1])

def build_model(name,n_features,window,n_classes):
    if name=='mlp': return MLP(n_features,window,n_classes)
    if name=='cnn': return CNN1D(n_features,window,n_classes)
    if name=='gru': return GRUClassifier(n_features,window,n_classes)
    raise ValueError(name)

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
