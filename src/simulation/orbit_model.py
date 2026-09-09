"""Orbit initialization and propagation.

The implementation intentionally uses a transparent numerical model rather than a
black-box astrodynamics package.  Phase 1 is intended to be auditable and easy to
extend with attitude/sensor models later.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM


@dataclass(frozen=True)
class OrbitalElements:
    semi_major_axis_km: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float


def rotation_1(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def elements_to_state(elements: OrbitalElements) -> tuple[np.ndarray, np.ndarray]:
    """Convert classical elements to ECI position/velocity in km and km/s."""
    a = elements.semi_major_axis_km
    e = elements.eccentricity
    nu = elements.true_anomaly_rad
    p = a * (1.0 - e**2)

    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(MU_EARTH_KM3_S2 / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    q_pqw_to_eci = (
        rotation_3(elements.raan_rad)
        @ rotation_1(elements.inclination_rad)
        @ rotation_3(elements.arg_perigee_rad)
    )
    return q_pqw_to_eci @ r_pf, q_pqw_to_eci @ v_pf


def two_body_acceleration(r_km: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(r_km)
    return -MU_EARTH_KM3_S2 * r_km / r**3


def j2_acceleration(r_km: np.ndarray) -> np.ndarray:
    """First-order J2 acceleration in ECI [km/s^2]."""
    x, y, z = r_km
    r2 = float(np.dot(r_km, r_km))
    r = np.sqrt(r2)
    z2_r2 = z * z / r2
    factor = 1.5 * J2_EARTH * MU_EARTH_KM3_S2 * R_EARTH_KM**2 / r**5
    return factor * np.array([
        x * (5.0 * z2_r2 - 1.0),
        y * (5.0 * z2_r2 - 1.0),
        z * (5.0 * z2_r2 - 3.0),
    ])


def dynamics(_t: float, state: np.ndarray, use_j2: bool) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    a = two_body_acceleration(r)
    if use_j2:
        a = a + j2_acceleration(r)
    return np.concatenate((v, a))


def propagate_orbit(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    duration_s: float,
    sample_time_s: float,
    use_j2: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate an orbit and return t, r_ECI, v_ECI."""
    t_eval = np.arange(0.0, duration_s, sample_time_s)
    if len(t_eval) == 0 or not np.isclose(t_eval[-1], duration_s):
        t_eval = np.append(t_eval, duration_s)
    y0 = np.concatenate((r0_km, v0_km_s))
    sol = solve_ivp(
        lambda t, y: dynamics(t, y, use_j2),
        (0.0, duration_s),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Orbit integration failed: {sol.message}")
    return sol.t, sol.y[:3].T, sol.y[3:].T


def kepler_period_s(semi_major_axis_km: float) -> float:
    return 2.0 * np.pi * np.sqrt(semi_major_axis_km**3 / MU_EARTH_KM3_S2)


def specific_orbital_energy(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(r_km, axis=1)
    v2 = np.sum(v_km_s * v_km_s, axis=1)
    return 0.5 * v2 - MU_EARTH_KM3_S2 / r


def angular_momentum_norm(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.cross(r_km, v_km_s), axis=1)


def specific_total_energy(
    r_km: np.ndarray, v_km_s: np.ndarray, use_j2: bool = True
) -> np.ndarray:
    """Conserved specific mechanical energy for the selected static gravity model."""
    r = np.linalg.norm(r_km, axis=1)
    kinetic = 0.5 * np.sum(v_km_s * v_km_s, axis=1)
    potential = -MU_EARTH_KM3_S2 / r
    if use_j2:
        z_over_r = r_km[:, 2] / r
        p2 = 0.5 * (3.0 * z_over_r**2 - 1.0)
        potential = potential * (1.0 - J2_EARTH * (R_EARTH_KM / r) ** 2 * p2)
    return kinetic + potential


def angular_momentum_z(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """z component of specific angular momentum; conserved by the axisymmetric J2 model."""
    return np.cross(r_km, v_km_s)[:, 2]
