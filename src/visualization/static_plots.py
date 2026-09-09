"""Static Phase-1 diagnostic and thesis-ready figures."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulation.constants import R_EARTH_KM


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_orbit_3d(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.25, linewidth=0)
    ax.plot(df.r_eci_x_km, df.r_eci_y_km, df.r_eci_z_km, linewidth=1.4, label="Orbit")
    ax.scatter(df.r_eci_x_km.iloc[0], df.r_eci_y_km.iloc[0], df.r_eci_z_km.iloc[0], s=35, label="Start")
    limit = 1.1 * np.max(np.linalg.norm(df[["r_eci_x_km","r_eci_y_km","r_eci_z_km"]].to_numpy(), axis=1))
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit), xlabel="ECI x [km]", ylabel="ECI y [km]", zlabel="ECI z [km]")
    ax.set_title("Phase 1 — propagated CubeSat orbit in ECI")
    ax.legend()
    _save(fig, out, dpi)


def plot_altitude_speed(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df.time_orbits, df.altitude_km)
    axes[0].set_ylabel("Altitude [km]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df.time_orbits, df.speed_km_s)
    axes[1].set_ylabel("Speed [km/s]")
    axes[1].set_xlabel("Elapsed orbital periods")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Orbital-state sanity check")
    _save(fig, out, dpi)


def plot_position_components(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df.time_orbits, df.r_eci_x_km, label="x")
    ax.plot(df.time_orbits, df.r_eci_y_km, label="y")
    ax.plot(df.time_orbits, df.r_eci_z_km, label="z")
    ax.set(xlabel="Elapsed orbital periods", ylabel="ECI position [km]", title="ECI position components")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    _save(fig, out, dpi)


def plot_ground_track(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # break wrap-around lines at +/-180 deg
    lon = df.longitude_deg.to_numpy().copy()
    lat = df.latitude_deg.to_numpy()
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0]
    start = 0
    for j in np.append(jumps, len(lon) - 1):
        ax.plot(lon[start:j+1], lat[start:j+1], linewidth=1.2)
        start = j + 1
    ax.scatter(lon[0], lat[0], s=25, label="Start")
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude [deg]", ylabel="Latitude [deg]", title="Diagnostic ground track")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out, dpi)


def plot_reference_vectors(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for key, label in [("sun_lvlh_x", "x"), ("sun_lvlh_y", "y"), ("sun_lvlh_z", "z")]:
        axes[0].plot(df.time_orbits, df[key], label=label)
    axes[0].set_ylabel("Sun unit vector")
    axes[0].set_title("Sun reference vector in LVLH")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3)
    for key, label in [("mag_lvlh_x_uT", "x"), ("mag_lvlh_y_uT", "y"), ("mag_lvlh_z_uT", "z")]:
        axes[1].plot(df.time_orbits, df[key], label=label)
    axes[1].set_ylabel("Magnetic field [µT]")
    axes[1].set_xlabel("Elapsed orbital periods")
    axes[1].set_title("Magnetic reference vector in LVLH")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=3)
    _save(fig, out, dpi)


def plot_eclipse_geometry(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].step(df.time_orbits, df.eclipse, where="post")
    axes[0].set_ylabel("Eclipse flag")
    axes[0].set_yticks([0, 1], labels=["Sunlit", "Eclipse"])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Natural Sun-sensor availability driver")
    axes[1].plot(df.time_orbits, df.sun_mag_angle_deg)
    axes[1].set_ylabel("Sun–B angle [deg]")
    axes[1].set_xlabel("Elapsed orbital periods")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Two-vector geometry diagnostic")
    _save(fig, out, dpi)


def plot_invariants(df: pd.DataFrame, out: Path, dpi: int = 160) -> None:
    energy0 = float(df.specific_total_energy_km2_s2.iloc[0])
    hz0 = float(df.angular_momentum_z_km2_s.iloc[0])
    de_ppm = (df.specific_total_energy_km2_s2 - energy0) / abs(energy0) * 1e6
    dhz_ppm = (df.angular_momentum_z_km2_s - hz0) / max(abs(hz0), 1e-15) * 1e6
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df.time_orbits, de_ppm)
    axes[0].set_ylabel("Δ total energy [ppm]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df.time_orbits, dhz_ppm)
    axes[1].set_ylabel("Δ h_z [ppm]")
    axes[1].set_xlabel("Elapsed orbital periods")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Conserved-quantity numerical diagnostics")
    _save(fig, out, dpi)


def generate_static_plots(df: pd.DataFrame, results_dir: Path, dpi: int = 160) -> list[Path]:
    jobs = [
        (plot_orbit_3d, "01_orbit_3d.png"),
        (plot_altitude_speed, "02_altitude_speed.png"),
        (plot_position_components, "03_eci_position.png"),
        (plot_ground_track, "04_ground_track.png"),
        (plot_reference_vectors, "05_reference_vectors_lvlh.png"),
        (plot_eclipse_geometry, "06_eclipse_and_geometry.png"),
        (plot_invariants, "07_orbit_invariants.png"),
    ]
    outputs: list[Path] = []
    for fn, name in jobs:
        path = results_dir / name
        fn(df, path, dpi)
        outputs.append(path)
    return outputs
