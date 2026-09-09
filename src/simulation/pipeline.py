"""Phase-1 orchestration and dataframe generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd
import yaml

from .constants import R_EARTH_KM
from .environment import compute_environment
from .frames import ecef_to_geodetic_spherical, eci_to_ecef, jd_series, parse_utc
from .orbit_model import (
    OrbitalElements,
    angular_momentum_norm,
    angular_momentum_z,
    elements_to_state,
    kepler_period_s,
    propagate_orbit,
    specific_orbital_energy,
    specific_total_energy,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    sim = config["simulation"]
    orb = config["orbit"]
    env_cfg = config["environment"]

    a_km = R_EARTH_KM + float(orb["altitude_km"])
    elements = OrbitalElements(
        semi_major_axis_km=a_km,
        eccentricity=float(orb["eccentricity"]),
        inclination_rad=np.deg2rad(float(orb["inclination_deg"])),
        raan_rad=np.deg2rad(float(orb["raan_deg"])),
        arg_perigee_rad=np.deg2rad(float(orb["arg_perigee_deg"])),
        true_anomaly_rad=np.deg2rad(float(orb["true_anomaly_deg"])),
    )
    r0, v0 = elements_to_state(elements)
    period_s = kepler_period_s(a_km)
    duration_s = float(sim["duration_orbits"]) * period_s

    t_s, r_eci, v_eci = propagate_orbit(
        r0,
        v0,
        duration_s=duration_s,
        sample_time_s=float(sim["sample_time_s"]),
        use_j2=bool(orb["use_j2"]),
        rtol=float(sim["integrator_rtol"]),
        atol=float(sim["integrator_atol"]),
    )

    start_dt = parse_utc(sim["start_utc"])
    jd = jd_series(start_dt, t_s)
    env = compute_environment(
        r_eci,
        v_eci,
        jd,
        magnetic_equatorial_surface_uT=float(env_cfg["magnetic_equatorial_surface_uT"]),
        magnetic_dipole_lat_deg=float(env_cfg["magnetic_dipole_lat_deg"]),
        magnetic_dipole_lon_deg=float(env_cfg["magnetic_dipole_lon_deg"]),
    )

    r_ecef = np.vstack([eci_to_ecef(r_eci[i], jd[i]) for i in range(len(t_s))])
    geod = np.array([ecef_to_geodetic_spherical(r) for r in r_ecef])
    lat_deg = np.rad2deg(geod[:, 0])
    lon_deg = np.rad2deg(geod[:, 1])
    altitude_km = np.linalg.norm(r_eci, axis=1) - R_EARTH_KM
    speed_km_s = np.linalg.norm(v_eci, axis=1)
    energy = specific_orbital_energy(r_eci, v_eci)
    hnorm = angular_momentum_norm(r_eci, v_eci)
    total_energy = specific_total_energy(r_eci, v_eci, use_j2=bool(orb["use_j2"]))
    hz = angular_momentum_z(r_eci, v_eci)
    b_uT = env["mag_eci_t"] * 1e6
    bmag_uT = np.linalg.norm(b_uT, axis=1)

    data: dict[str, np.ndarray] = {
        "time_s": t_s,
        "time_orbits": t_s / period_s,
        "jd_utc": jd,
        "r_eci_x_km": r_eci[:, 0], "r_eci_y_km": r_eci[:, 1], "r_eci_z_km": r_eci[:, 2],
        "v_eci_x_km_s": v_eci[:, 0], "v_eci_y_km_s": v_eci[:, 1], "v_eci_z_km_s": v_eci[:, 2],
        "r_ecef_x_km": r_ecef[:, 0], "r_ecef_y_km": r_ecef[:, 1], "r_ecef_z_km": r_ecef[:, 2],
        "latitude_deg": lat_deg,
        "longitude_deg": lon_deg,
        "altitude_km": altitude_km,
        "speed_km_s": speed_km_s,
        "specific_energy_km2_s2": energy,
        "angular_momentum_km2_s": hnorm,
        "specific_total_energy_km2_s2": total_energy,
        "angular_momentum_z_km2_s": hz,
        "sun_eci_x": env["sun_eci"][:, 0], "sun_eci_y": env["sun_eci"][:, 1], "sun_eci_z": env["sun_eci"][:, 2],
        "sun_lvlh_x": env["sun_lvlh"][:, 0], "sun_lvlh_y": env["sun_lvlh"][:, 1], "sun_lvlh_z": env["sun_lvlh"][:, 2],
        "mag_eci_x_uT": b_uT[:, 0], "mag_eci_y_uT": b_uT[:, 1], "mag_eci_z_uT": b_uT[:, 2],
        "mag_lvlh_x_uT": env["mag_lvlh_t"][:, 0] * 1e6,
        "mag_lvlh_y_uT": env["mag_lvlh_t"][:, 1] * 1e6,
        "mag_lvlh_z_uT": env["mag_lvlh_t"][:, 2] * 1e6,
        "mag_magnitude_uT": bmag_uT,
        "sun_mag_angle_deg": env["sun_mag_angle_deg"],
        "eclipse": env["eclipse"].astype(int),
    }
    df = pd.DataFrame(data)

    summary: dict[str, float | int | bool] = {
        "samples": int(len(df)),
        "period_s": float(period_s),
        "period_min": float(period_s / 60.0),
        "duration_s": float(t_s[-1]),
        "use_j2": bool(orb["use_j2"]),
        "altitude_mean_km": float(df.altitude_km.mean()),
        "altitude_min_km": float(df.altitude_km.min()),
        "altitude_max_km": float(df.altitude_km.max()),
        "speed_mean_km_s": float(df.speed_km_s.mean()),
        "eclipse_fraction": float(df.eclipse.mean()),
        "mag_min_uT": float(df.mag_magnitude_uT.min()),
        "mag_max_uT": float(df.mag_magnitude_uT.max()),
        "sun_mag_angle_min_deg": float(df.sun_mag_angle_deg.min()),
        "sun_mag_angle_max_deg": float(df.sun_mag_angle_deg.max()),
        "conserved_energy_relative_span": float((total_energy.max() - total_energy.min()) / abs(total_energy.mean())),
        "hz_relative_span": float((hz.max() - hz.min()) / max(abs(hz.mean()), 1e-15)),
    }
    return df, summary


def save_outputs(df: pd.DataFrame, summary: dict[str, Any], config: dict[str, Any], root: Path) -> None:
    csv_path = root / config["output"]["csv"]
    summary_path = root / config["output"]["summary_json"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
