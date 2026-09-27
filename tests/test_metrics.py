import numpy as np
from src.verification.metrics import comparison_metrics
def test_zero_error_metrics():
 a=np.array([[1.,2,3,4,5,6],[2,3,4,5,6,7]]); m,_,_=comparison_metrics(a,a); assert m['position_rmse_m']==0 and m['velocity_rmse_mps']==0
def test_known_position_error():
 a=np.zeros((2,6)); b=a.copy(); b[:,0]=3.; b[:,1]=4.; m,_,_=comparison_metrics(a,b); assert m['position_rmse_m']==5 and m['position_max_m']==5
