from __future__ import annotations
import numpy as np
from .illumination import sun_direction_series


def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v)
    return v/n if n>0 else v


def orbit_normal_from_elements(inclination_deg: float, raan_deg: float) -> np.ndarray:
    i=np.deg2rad(float(inclination_deg)); O=np.deg2rad(float(raan_deg))
    return np.array([np.sin(i)*np.sin(O), -np.sin(i)*np.cos(O), np.cos(i)])


def beta_angle_deg(sun_eci, inclination_deg: float, raan_deg: float) -> float:
    h=orbit_normal_from_elements(inclination_deg,raan_deg)
    s=_unit(sun_eci)
    return float(np.degrees(np.arcsin(np.clip(np.dot(h,s),-1.0,1.0))))


def high_beta_raan_deg(sun_eci, inclination_deg: float) -> float:
    """Choose the RAAN that maximizes |beta| for a fixed inclination.

    This creates a terminator-like/high-beta geometry suitable for sunlight-rich
    controller testing.  Both antipodal plane normals are considered and the one
    with the larger absolute Sun projection is returned.
    """
    s=_unit(sun_eci); alpha=np.arctan2(s[1],s[0])
    candidates=[np.degrees(alpha+np.pi/2)%360.0, np.degrees(alpha+3*np.pi/2)%360.0]
    return float(max(candidates,key=lambda O: abs(beta_angle_deg(s,inclination_deg,O))))


def resolve_orbit_profile(profile: str, global_orbit: dict, *, epoch_utc: str,
                          high_beta_cfg: dict | None=None) -> dict:
    """Resolve a named scenario orbit to explicit circular-orbit elements."""
    name=str(profile or 'global').lower()
    if name in ('global','configured','default'):
        out={k:float(global_orbit[k]) for k in ('altitude_m','inclination_deg','raan_deg','phase_deg')}
        out['profile']='global'
        sun,_=sun_direction_series([0.0],'analytic',epoch=epoch_utc)
        out['beta_angle_deg']=beta_angle_deg(sun[0],out['inclination_deg'],out['raan_deg'])
        return out
    if name in ('iss_like','iss-like'):
        out=dict(profile='iss_like',altitude_m=500000.0,inclination_deg=51.6,raan_deg=0.0,phase_deg=0.0)
        sun,_=sun_direction_series([0.0],'analytic',epoch=epoch_utc)
        out['beta_angle_deg']=beta_angle_deg(sun[0],out['inclination_deg'],out['raan_deg'])
        return out
    if name in ('high_beta_sunlit','high-beta-sunlit','high_beta'):
        hc=high_beta_cfg or {}
        alt=float(hc.get('altitude_m',550000.0)); inc=float(hc.get('inclination_deg',97.6)); phase=float(hc.get('phase_deg',0.0))
        sun,_=sun_direction_series([0.0],'analytic',epoch=epoch_utc)
        if str(hc.get('raan_mode','auto')).lower()=='auto':
            raan=high_beta_raan_deg(sun[0],inc)
        else:
            raan=float(hc.get('raan_deg',0.0))
        return dict(profile='high_beta_sunlit',altitude_m=alt,inclination_deg=inc,raan_deg=raan,phase_deg=phase,
                    beta_angle_deg=beta_angle_deg(sun[0],inc,raan))
    raise ValueError(f"Unknown orbit profile '{profile}'. Use global, iss_like, or high_beta_sunlit.")
