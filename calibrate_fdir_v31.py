from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DIAGNOSTICS = {
    "gyro_vector_rate_residual_rad_s": "fdir_gyro_vector_rate_residual_rad_s",
    "gyro_cross_sensor_deg": "fdir_gyro_cross_sensor_deg",
    "sun_mag_geometry_error_deg": "fdir_sun_mag_geometry_error_deg",
}


def _summary(x):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    if not len(x):
        return None
    qs=[50,90,95,99,99.5,99.9,100]
    return {f"p{q:g}":float(np.percentile(x,q)) for q in qs}


def main():
    ap=argparse.ArgumentParser(description="Summarize healthy FDIR residuals for v3.1 threshold calibration.")
    ap.add_argument("timeseries_csv",type=Path)
    ap.add_argument("--branch",default="fdir_only")
    ap.add_argument("--start-s",type=float,default=200.0)
    ap.add_argument("--end-s",type=float,default=700.0)
    ap.add_argument("--chunksize",type=int,default=300_000)
    args=ap.parse_args()

    usecols=["branch","time_s","fault_active","fdir_armed",*DIAGNOSTICS.values()]
    acc={k:[] for k in DIAGNOSTICS}

    for ch in pd.read_csv(args.timeseries_csv,usecols=usecols,chunksize=args.chunksize):
        m=(ch["branch"]==args.branch)&(ch["fdir_armed"]==1)&(ch["fault_active"]==0)
        m&=(ch["time_s"]>=args.start_s)&(ch["time_s"]<=args.end_s)
        x=ch.loc[m]
        for name,col in DIAGNOSTICS.items():
            a=x[col].to_numpy(float)
            a=a[np.isfinite(a)]
            if len(a):
                acc[name].append(a)

    print(f"Healthy calibration window: {args.start_s:g}--{args.end_s:g} s, branch={args.branch}")
    out={}
    for name in DIAGNOSTICS:
        a=np.concatenate(acc[name]) if acc[name] else np.array([])
        out[name]=_summary(a)
        print(f"\n{name}")
        if out[name] is None:
            print("  no finite samples")
        else:
            for k,v in out[name].items():
                print(f"  {k:>6}: {v:.9g}")

    # Conservative warning suggestions: 10% above the p99.9 healthy tail.
    print("\nConservative warning suggestions (1.10 x healthy p99.9):")
    for name,s in out.items():
        if s is not None:
            print(f"  {name}: {1.10*s['p99.9']:.9g}")


if __name__=="__main__":
    main()
