import numpy as np

def propagate_orekit(times_s, altitude_m, inclination_deg, raan_deg, phase_deg, mu, earth_radius):
    """Reference Keplerian propagation with Orekit. Requires conda package 'orekit'."""
    try:
        import orekit; orekit.initVM()
        from org.orekit.frames import FramesFactory
        from org.orekit.orbits import KeplerianOrbit, PositionAngleType
        from org.orekit.propagation.analytical import KeplerianPropagator
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
    except Exception as exc:
        raise RuntimeError("Orekit Python wrapper is not available. Install with conda-forge.") from exc
    frame=FramesFactory.getEME2000()
    date0=AbsoluteDate(2026,1,1,0,0,0.0,TimeScalesFactory.getTAI())
    a=float(earth_radius+altitude_m)
    orbit=KeplerianOrbit(a,0.0,float(np.deg2rad(inclination_deg)),0.0,float(np.deg2rad(raan_deg)),
        float(np.deg2rad(phase_deg)),PositionAngleType.TRUE,frame,date0,float(mu))
    prop=KeplerianPropagator(orbit); rows=[]
    for sec in np.asarray(times_s,dtype=float):
        state=prop.propagate(date0.shiftedBy(float(sec))); pv=state.getPVCoordinates(frame); pos,vel=pv.getPosition(),pv.getVelocity()
        rows.append([pos.getX(),pos.getY(),pos.getZ(),vel.getX(),vel.getY(),vel.getZ()])
    return np.asarray(rows)
