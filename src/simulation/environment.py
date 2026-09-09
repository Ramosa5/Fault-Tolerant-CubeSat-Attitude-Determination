"""Low-order environmental reference-vector models for CubeSat ADCS research.

These models are deliberately lightweight and deterministic.  They are suitable for
algorithm development and controlled fault-tolerance experiments, not mission-grade
orbit determination or magnetic-field prediction.
"""
from __future__ import annotations

import numpy as np

from .constants import AU_KM, R_EARTH_KM
from .frames import ecef_to_eci, eci_to_ecef, lvlh_dcm_eci_to_lvlh


def sun_vector_eci(jd: float) -> tuple[np.ndarray, float]:
    """Approximate Earth-to-Sun vector in ECI and distance [km].

    Accuracy is sufficient for Phase-1 sensor/reference-vector generation.
    """
    n = jd - 2451545.0
    mean_long_deg = (280.460 + 0.9856474 * n) % 360.0
    mean_anom_deg = (357.528 + 0.9856003 * n) % 360.0
    g = np.deg2rad(mean_anom_deg)
    lam = np.deg2rad((mean_long_deg + 1.915 * np.sin(g) + 0.020 * np.sin(2.0 * g)) % 360.0)
    eps = np.deg2rad(23.439 - 0.0000004 * n)
    distance_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2.0 * g)
    unit = np.array([
        np.cos(lam),
        np.cos(eps) * np.sin(lam),
        np.sin(eps) * np.sin(lam),
    ])
    unit = unit / np.linalg.norm(unit)
    return unit, distance_au * AU_KM


def in_cylindrical_eclipse(r_sat_eci_km: np.ndarray, sun_hat_eci: np.ndarray) -> bool:
    """Return True when satellite is inside a cylindrical Earth umbra approximation."""
    behind_earth = float(np.dot(r_sat_eci_km, sun_hat_eci)) < 0.0
    perpendicular = r_sat_eci_km - np.dot(r_sat_eci_km, sun_hat_eci) * sun_hat_eci
    return bool(behind_earth and np.linalg.norm(perpendicular) < R_EARTH_KM)


def dipole_axis_ecef(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    m = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return m / np.linalg.norm(m)


def magnetic_field_ecef_tesla(
    r_ecef_km: np.ndarray,
    equatorial_surface_uT: float,
    dipole_lat_deg: float,
    dipole_lon_deg: float,
) -> np.ndarray:
    """Centered tilted dipole magnetic field in ECEF [T]."""
    r = np.linalg.norm(r_ecef_km)
    r_hat = r_ecef_km / r
    m_hat = dipole_axis_ecef(dipole_lat_deg, dipole_lon_deg)
    b0_t = equatorial_surface_uT * 1e-6
    scale = b0_t * (R_EARTH_KM / r) ** 3
    return scale * (3.0 * np.dot(m_hat, r_hat) * r_hat - m_hat)


def compute_environment(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    jd: np.ndarray,
    magnetic_equatorial_surface_uT: float,
    magnetic_dipole_lat_deg: float,
    magnetic_dipole_lon_deg: float,
) -> dict[str, np.ndarray]:
    n = len(jd)
    sun_eci = np.empty((n, 3))
    sun_lvlh = np.empty((n, 3))
    mag_eci_t = np.empty((n, 3))
    mag_lvlh_t = np.empty((n, 3))
    eclipse = np.empty(n, dtype=bool)
    sun_mag_angle_deg = np.empty(n)

    for i in range(n):
        s_hat, _sun_dist = sun_vector_eci(jd[i])
        r_ecef = eci_to_ecef(r_eci_km[i], jd[i])
        b_ecef = magnetic_field_ecef_tesla(
            r_ecef,
            magnetic_equatorial_surface_uT,
            magnetic_dipole_lat_deg,
            magnetic_dipole_lon_deg,
        )
        b_eci = ecef_to_eci(b_ecef, jd[i])
        c_eci_to_lvlh = lvlh_dcm_eci_to_lvlh(r_eci_km[i], v_eci_km_s[i])

        sun_eci[i] = s_hat
        sun_lvlh[i] = c_eci_to_lvlh @ s_hat
        mag_eci_t[i] = b_eci
        mag_lvlh_t[i] = c_eci_to_lvlh @ b_eci
        eclipse[i] = in_cylindrical_eclipse(r_eci_km[i], s_hat)

        b_hat = b_eci / np.linalg.norm(b_eci)
        cosang = np.clip(float(np.dot(s_hat, b_hat)), -1.0, 1.0)
        sun_mag_angle_deg[i] = np.rad2deg(np.arccos(cosang))

    return {
        "sun_eci": sun_eci,
        "sun_lvlh": sun_lvlh,
        "mag_eci_t": mag_eci_t,
        "mag_lvlh_t": mag_lvlh_t,
        "eclipse": eclipse,
        "sun_mag_angle_deg": sun_mag_angle_deg,
    }
