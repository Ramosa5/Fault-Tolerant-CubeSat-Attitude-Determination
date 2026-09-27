from pathlib import Path
import csv, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.simulation.constants import *
from src.simulation.orbit_model import circular_orbit_period, propagate_circular_two_body
from src.verification.orekit_reference import propagate_orekit
from src.verification.metrics import comparison_metrics

CASES=[
 ('V1_nominal_1orbit',500e3,51.6,0.,0.,1.0),
 ('V2_low_inclination',500e3,10.,40.,15.,1.0),
 ('V3_high_inclination',500e3,97.,120.,90.,1.0),
 ('V4_altitude_400km',400e3,51.6,0.,0.,1.0),
 ('V5_altitude_700km',700e3,51.6,0.,0.,1.0),
 ('V6_long_10orbits',500e3,51.6,25.,35.,10.0),
]

def main():
    out=ROOT/'data'/'results'; out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name,alt,inc,raan,phase,norb in CASES:
        period=circular_orbit_period(alt); times=np.linspace(0,norb*period,int(norb*360)+1)
        ours=propagate_circular_two_body(times,alt,inc,raan,phase)
        ref=propagate_orekit(times,alt,inc,raan,phase,MU_EARTH,R_EARTH)
        m,pe,ve=comparison_metrics(ours,ref)
        row={'test':name,'altitude_km':alt/1000,'inclination_deg':inc,'orbits':norb,'duration_s':times[-1],**m}; summaries.append(row)
        with open(out/f'{name}_timeseries.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['time_s','position_error_m','velocity_error_mps']); w.writerows(zip(times,pe,ve))
    fields=list(summaries[0]);
    with open(out/'verification_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summaries)
    with open(out/'verification_summary.json','w') as f: json.dump(summaries,f,indent=2)
    print('\nOREKIT VERIFICATION RESULTS')
    for r in summaries: print(f"{r['test']}: pos RMSE={r['position_rmse_m']:.6g} m, pos max={r['position_max_m']:.6g} m, vel RMSE={r['velocity_rmse_mps']:.6g} m/s")
    print(f"\nSaved: {out/'verification_summary.csv'}")
if __name__=='__main__': main()
