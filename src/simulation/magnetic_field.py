from __future__ import annotations
from datetime import datetime
import numpy as np

EARTH_ROTATION_RAD_S = 7.2921150e-5


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def _julian_date(dt: datetime) -> float:
    """UTC datetime to Julian Date (sufficient for simulation frame rotation)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(__import__('datetime').timezone.utc).replace(tzinfo=None)
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond/1e6)/60)/60)/24
    if m <= 2:
        y -= 1; m += 12
    A = y // 100
    B = 2 - A + A // 4
    return (int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + B - 1524.5)


def gmst_angle_rad(epoch_utc: datetime) -> float:
    jd = _julian_date(epoch_utc)
    T = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629*(jd-2451545.0) + 0.000387933*T*T - T*T*T/38710000.0
    return np.deg2rad(gmst_deg % 360.0)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]], float)  # ECI -> ECEF


def _ecef_positions(position_eci_m, times_s, epoch_utc):
    p = np.asarray(position_eci_m, float)
    th0 = gmst_angle_rad(epoch_utc)
    out = np.empty_like(p)
    angles = th0 + EARTH_ROTATION_RAD_S*np.asarray(times_s, float)
    for i, a in enumerate(angles):
        out[i] = _rz(a) @ p[i]
    return out, angles


def _spherical_basis(theta, phi):
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    er = np.stack([st*cp, st*sp, ct], axis=-1)
    et = np.stack([ct*cp, ct*sp, -st], axis=-1)  # increasing colatitude (south)
    ep = np.stack([-sp, cp, np.zeros_like(phi)], axis=-1)
    return er, et, ep


def _dipole_field_ecef(position_ecef_m):
    """Centered tilted-dipole fallback; output Tesla in ECEF.

    This is intentionally only a fallback/sanity model. Use IGRF-14 for thesis runs.
    """
    p = np.asarray(position_ecef_m, float)
    r = np.linalg.norm(p, axis=1)
    rhat = p / r[:,None]
    # Approximate north geomagnetic dipole axis in geographic coordinates.
    lat = np.deg2rad(80.65); lon = np.deg2rad(-72.68)
    mhat = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    # Equatorial surface field ~30 microtesla scaled as r^-3.
    re = 6371.2e3; Beq = 30e-6
    scale = Beq*(re/r)**3
    return scale[:,None] * (3*rhat*(rhat@mhat)[:,None] - mhat)


def magnetic_field_eci(position_eci_m, times_s, *, epoch_utc, model='igrf14'):
    """Return Earth's magnetic field in ECI coordinates [Tesla].

    ``igrf14`` uses the IAGA IGRF-14 model through the pure-Python ``ppigrf`` package.
    A ``dipole`` model is available for fast tests and as an explicit fallback.
    """
    p_i = np.asarray(position_eci_m, float)
    t = np.asarray(times_s, float)
    if len(p_i) != len(t):
        raise ValueError('position_eci_m and times_s must have matching lengths')
    p_e, angles = _ecef_positions(p_i, t, epoch_utc)
    key = str(model).lower()
    if key == 'dipole':
        B_e = _dipole_field_ecef(p_e)
    elif key == 'igrf14':
        try:
            import ppigrf
        except ImportError as exc:
            raise RuntimeError('IGRF-14 magnetic field requires ppigrf. Install with: pip install ppigrf') from exc
        r_km = np.linalg.norm(p_e, axis=1)/1000.0
        theta = np.arccos(np.clip(p_e[:,2]/np.linalg.norm(p_e,axis=1), -1, 1))
        phi = np.arctan2(p_e[:,1], p_e[:,0])
        # IGRF coefficients barely change over one orbit; evaluate all orbital positions at the epoch date.
        Br, Bt, Bp = ppigrf.igrf_gc(r_km, np.rad2deg(theta), np.rad2deg(phi), epoch_utc.replace(tzinfo=None))
        Br = np.asarray(Br, float).reshape(-1); Bt=np.asarray(Bt,float).reshape(-1); Bp=np.asarray(Bp,float).reshape(-1)
        er, et, ep = _spherical_basis(theta, phi)
        B_e = (Br[:,None]*er + Bt[:,None]*et + Bp[:,None]*ep) * 1e-9  # nT -> T
    else:
        raise ValueError("magnetic field model must be 'igrf14' or 'dipole'")
    B_i = np.empty_like(B_e)
    for i, a in enumerate(angles):
        B_i[i] = _rz(a).T @ B_e[i]
    return B_i
