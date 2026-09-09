"""Reference-frame transformations used by the environment simulation."""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from .constants import R_EARTH_KM, SECONDS_PER_DAY


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def julian_date(dt: datetime) -> float:
    """UTC Julian Date; adequate for the low-order Phase-1 environment models."""
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


def gmst_rad_from_jd(jd: float) -> float:
    """Greenwich mean sidereal time using a standard low-order expression."""
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t**2
        - t**3 / 38710000.0
    )
    return np.deg2rad(gmst_deg % 360.0)


def rot3(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci_km: np.ndarray, jd: float) -> np.ndarray:
    return rot3(gmst_rad_from_jd(jd)) @ r_eci_km


def ecef_to_eci(v_ecef: np.ndarray, jd: float) -> np.ndarray:
    return rot3(gmst_rad_from_jd(jd)).T @ v_ecef


def ecef_to_geodetic_spherical(r_ecef_km: np.ndarray) -> tuple[float, float, float]:
    """Spherical-Earth lat/lon/alt used only for diagnostic ground-track plotting."""
    x, y, z = r_ecef_km
    rho = np.linalg.norm(r_ecef_km)
    lat = np.arcsin(z / rho)
    lon = np.arctan2(y, x)
    alt_km = rho - R_EARTH_KM
    return lat, lon, alt_km


def lvlh_dcm_eci_to_lvlh(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    """ECI -> LVLH DCM.

    LVLH axes used here:
      +x: radial outward
      +z: orbit angular-momentum direction
      +y: completes right-handed triad (approximately along-track)
    """
    x_hat = r_eci_km / np.linalg.norm(r_eci_km)
    z_hat = np.cross(r_eci_km, v_eci_km_s)
    z_hat = z_hat / np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)
    return np.vstack((x_hat, y_hat, z_hat))


def jd_series(start_dt: datetime, t_s: np.ndarray) -> np.ndarray:
    jd0 = julian_date(start_dt)
    return jd0 + t_s / SECONDS_PER_DAY
