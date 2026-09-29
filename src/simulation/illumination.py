from __future__ import annotations
from datetime import datetime, timezone, timedelta
import numpy as np
from .constants import R_EARTH

AU_M = 149_597_870_700.0
R_SUN_M = 695_700_000.0


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _parse_epoch(epoch: str | datetime) -> datetime:
    if isinstance(epoch, datetime):
        dt = epoch
    else:
        text = str(epoch).replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _days_since_j2000(dt: datetime) -> float:
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (dt - j2000).total_seconds() / 86400.0


def analytic_sun_position_eci(epoch: str | datetime, times_s) -> np.ndarray:
    """Approximate geocentric Sun position in an equatorial inertial frame.

    Uses a compact low-precision solar ephemeris suitable for simulation and
    visualization. It captures the annual change in Sun direction and Earth-Sun
    distance without requiring external ephemeris files.
    """
    epoch_dt = _parse_epoch(epoch)
    t = np.asarray(times_s, dtype=float)
    out = np.zeros((len(t), 3), dtype=float)
    for i, sec in enumerate(t):
        dt = epoch_dt + timedelta(seconds=float(sec))
        n = _days_since_j2000(dt)
        L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
        g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)
        lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2.0 * g)
        eps = np.deg2rad(23.439 - 0.0000004 * n)
        # Approximate Sun-Earth distance in AU.
        r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
        d = r_au * AU_M
        out[i] = d * np.array([
            np.cos(lam),
            np.cos(eps) * np.sin(lam),
            np.sin(eps) * np.sin(lam),
        ])
    return out


def sun_direction_series(times_s, model='analytic', *, epoch='2026-01-01T00:00:00+00:00', constant_vector=(1.0, 0.2, 0.1)):
    t = np.asarray(times_s, dtype=float)
    model = str(model).lower()
    if model == 'constant':
        v = _unit(constant_vector)
        return np.repeat(v[None, :], len(t), axis=0), np.repeat((v * AU_M)[None, :], len(t), axis=0)
    if model == 'analytic':
        pos = analytic_sun_position_eci(epoch, t)
        return np.array([_unit(v) for v in pos]), pos
    raise ValueError("sun model must be 'constant' or 'analytic'")


def eclipse_geometry(spacecraft_position_eci_m, sun_position_eci_m, *, earth_radius_m=R_EARTH, sun_radius_m=R_SUN_M):
    """Return eclipse state for each spacecraft position.

    State values: 0=sunlight, 1=penumbra, 2=umbra/full eclipse.
    Also returns a simple illumination fraction: 1.0, 0.5, or 0.0.
    The angular-disk test accounts for the finite apparent radii of Earth and Sun.
    """
    r = np.atleast_2d(np.asarray(spacecraft_position_eci_m, dtype=float))
    s = np.atleast_2d(np.asarray(sun_position_eci_m, dtype=float))
    if len(s) == 1 and len(r) > 1:
        s = np.repeat(s, len(r), axis=0)
    if len(r) != len(s):
        raise ValueError('spacecraft and Sun position arrays must have matching lengths')
    state = np.zeros(len(r), dtype=int)
    frac = np.ones(len(r), dtype=float)
    for i, (ri, si) in enumerate(zip(r, s)):
        to_earth = -ri
        to_sun = si - ri
        de = np.linalg.norm(to_earth)
        ds = np.linalg.norm(to_sun)
        if de <= earth_radius_m or ds <= sun_radius_m:
            continue
        alpha_e = np.arcsin(np.clip(earth_radius_m / de, -1.0, 1.0))
        alpha_s = np.arcsin(np.clip(sun_radius_m / ds, -1.0, 1.0))
        sep = np.arccos(np.clip(np.dot(to_earth / de, to_sun / ds), -1.0, 1.0))
        # Earth disk and Sun disk do not overlap.
        if sep >= alpha_e + alpha_s:
            continue
        # Sun disk is completely covered by Earth.
        if alpha_e >= alpha_s and sep <= alpha_e - alpha_s:
            state[i] = 2
            frac[i] = 0.0
        else:
            state[i] = 1
            frac[i] = 0.5
    return state, frac
