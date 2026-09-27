import numpy as np

def comparison_metrics(ours, ref):
    dp = ours[:,:3]-ref[:,:3]; dv=ours[:,3:]-ref[:,3:]
    pe=np.linalg.norm(dp,axis=1); ve=np.linalg.norm(dv,axis=1)
    return {
      'position_rmse_m': float(np.sqrt(np.mean(pe**2))), 'position_max_m': float(pe.max()),
      'velocity_rmse_mps': float(np.sqrt(np.mean(ve**2))), 'velocity_max_mps': float(ve.max()),
      'position_final_m': float(pe[-1]), 'velocity_final_mps': float(ve[-1])
    }, pe, ve
