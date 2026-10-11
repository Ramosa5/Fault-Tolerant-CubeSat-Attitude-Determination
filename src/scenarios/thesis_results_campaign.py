from __future__ import annotations

"""Curated end-to-end thesis validation and results campaign.

This scenario produces one publication-ready package that follows the development
sequence of the thesis:

1. orbit propagation and Orekit cross-verification,
2. rigid-body dynamics and sensor model verification,
3. nominal MEKF attitude determination,
4. reaction-wheel AOCS closed-loop validation,
5. Experiment 1: ML virtual-sensor/recovery evaluation,
6. Experiment 2: ML-assisted FDIR evaluation,
7. lightweight-model resource comparison.

The visual style adapts the documented conventions of
https://github.com/ChenLiu-1996/figures4papers: minimalist axes, sans-serif
publication typography, semantic colours, black-edged comparison bars, wide
multi-panel layouts and simultaneous PNG/PDF export.
"""

from pathlib import Path
import copy
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.simulation.constants import MU_EARTH, R_EARTH
from src.simulation.orbit_model import circular_orbit_period, propagate_circular_two_body
from src.verification.orekit_reference import propagate_orekit
from src.verification.metrics import comparison_metrics
from src.simulation.attitude import (
    propagate_attitude,
    rotational_energy,
    angular_momentum_norm,
    quat_angle_error_deg,
    quat_to_dcm,
)
from src.simulation.sensors import simulate_sensors
from src.estimation.mekf import run_mekf
from src.visualization.publication import (
    PALETTE,
    DEFAULT_COLORS,
    create_subplots,
    finalize_figure,
)
from src.scenarios import reaction_wheel_sun_pointing_comparison as rw_aocs
from src.scenarios import reaction_wheel_fair_ml_campaign as experiment1
from src.scenarios import reaction_wheel_ml_fdir_campaign as experiment2


SENSORS = ("gyro", "sun", "mag")


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _save_table(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=index)
    try:
        path.with_suffix(".md").write_text(df.to_markdown(index=index), encoding="utf-8")
    except Exception:
        pass
    try:
        path.with_suffix(".tex").write_text(
            df.to_latex(index=index, float_format=lambda x: f"{x:.5g}"),
            encoding="utf-8",
        )
    except Exception:
        pass


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.10, 1.06, label, transform=ax.transAxes,
        fontsize=17, fontweight="bold", va="bottom", ha="left",
    )


def _annotate_bars(ax, bars, fmt: str = "{:.2f}", fontsize: int = 8) -> None:
    for bar in bars:
        v = float(bar.get_height())
        if np.isfinite(v):
            ax.annotate(
                fmt.format(v),
                (bar.get_x() + bar.get_width() / 2.0, v),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize,
            )


def _finish(fig, path: Path, global_cfg: dict, *, dpi: int | None = None) -> list[Path]:
    vis = global_cfg.get("visualization", {})
    formats = tuple(vis.get("formats", ["png", "pdf"]))
    dpi = int(dpi or vis.get("dpi", 300))
    try:
        fig.tight_layout(pad=1.8)
    except Exception:
        pass
    return finalize_figure(fig, path, formats=formats, dpi=dpi)


def _shade_eclipses(ax, t: np.ndarray, eclipse_state: np.ndarray) -> None:
    eclipse = np.asarray(eclipse_state) == 2
    if not np.any(eclipse):
        return
    starts = np.flatnonzero(eclipse & np.r_[True, ~eclipse[:-1]])
    ends = np.flatnonzero(eclipse & np.r_[~eclipse[1:], True])
    for st, en in zip(starts, ends):
        ax.axvspan(t[st], t[en], color="0.2", alpha=0.10, lw=0)


def _angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(np.sum(an * bn, axis=1), -1.0, 1.0)))


def _master_row(stage: str, metric: str, value, unit: str, subsection: str, source: str) -> dict:
    try:
        value = float(value)
    except Exception:
        pass
    return {
        "stage": stage,
        "metric": metric,
        "value": value,
        "unit": unit,
        "thesis_subsection": subsection,
        "source": source,
    }


def _orbit_validation(cfg: dict, global_cfg: dict, out: Path, figures: Path,
                      master: list[dict], figure_manifest: list[dict]) -> dict:
    stage = out / "01_orbit_propagator"
    stage.mkdir(parents=True, exist_ok=True)
    orbit_cfg = global_cfg["orbit"]

    # Analytical sanity check.
    period = circular_orbit_period(float(orbit_cfg["altitude_m"]))
    t_nom = np.linspace(
        0.0,
        float(orbit_cfg.get("sanity_orbits", 1.0)) * period,
        int(float(orbit_cfg.get("sanity_orbits", 1.0)) * int(orbit_cfg.get("samples_per_orbit", 360))) + 1,
    )
    nominal = propagate_circular_two_body(
        t_nom,
        float(orbit_cfg["altitude_m"]),
        float(orbit_cfg["inclination_deg"]),
        float(orbit_cfg["raan_deg"]),
        float(orbit_cfg["phase_deg"]),
    )
    alt = np.linalg.norm(nominal[:, :3], axis=1) - R_EARTH
    speed = np.linalg.norm(nominal[:, 3:], axis=1)
    sanity = {
        "orbital_period_s": period,
        "altitude_range_m": float(np.ptp(alt)),
        "speed_range_mps": float(np.ptp(speed)),
        "mean_altitude_m": float(np.mean(alt)),
        "mean_speed_mps": float(np.mean(speed)),
    }
    _save_table(pd.DataFrame([sanity]), stage / "orbit_sanity_metrics")

    master.extend([
        _master_row("Orbit propagator", "Orbital period", period, "s", "4.2 Propagator verification", "orbit_sanity_metrics.csv"),
        _master_row("Orbit propagator", "Altitude range", sanity["altitude_range_m"], "m", "4.2 Propagator verification", "orbit_sanity_metrics.csv"),
        _master_row("Orbit propagator", "Speed range", sanity["speed_range_mps"], "m/s", "4.2 Propagator verification", "orbit_sanity_metrics.csv"),
    ])

    # Orekit cross-verification.
    rows = []
    curves = []
    requested = {str(x) for x in cfg.get("orekit_case_names", [])}
    cases = global_cfg.get("orekit", {}).get("cases", [])
    if requested:
        cases = [c for c in cases if str(c.get("name")) in requested]
    orekit_enabled = bool(cfg.get("run_orekit_cross_verification", True))
    orekit_error = None
    if orekit_enabled:
        for case in cases:
            try:
                p = circular_orbit_period(float(case["altitude_m"]))
                tt = np.linspace(
                    0.0,
                    float(case.get("orbits", 1.0)) * p,
                    int(float(case.get("orbits", 1.0)) * int(orbit_cfg.get("samples_per_orbit", 360))) + 1,
                )
                ours = propagate_circular_two_body(
                    tt, float(case["altitude_m"]), float(case["inclination_deg"]),
                    float(case["raan_deg"]), float(case["phase_deg"]),
                )
                ref = propagate_orekit(
                    tt, float(case["altitude_m"]), float(case["inclination_deg"]),
                    float(case["raan_deg"]), float(case["phase_deg"]), MU_EARTH, R_EARTH,
                )
                metrics, pe, ve = comparison_metrics(ours, ref)
                row = {"case": case["name"], **metrics}
                rows.append(row)
                curves.append((str(case["name"]), tt, pe, ve))
            except Exception as exc:
                orekit_error = str(exc)
                if bool(cfg.get("require_orekit", False)):
                    raise
                break

    verification = pd.DataFrame(rows)
    if len(verification):
        _save_table(verification, stage / "orekit_cross_verification_metrics")
        master.extend([
            _master_row("Orbit propagator", "Worst position RMSE vs Orekit", verification.position_rmse_m.max(), "m", "4.2 Propagator verification", "orekit_cross_verification_metrics.csv"),
            _master_row("Orbit propagator", "Worst velocity RMSE vs Orekit", verification.velocity_rmse_mps.max(), "m/s", "4.2 Propagator verification", "orekit_cross_verification_metrics.csv"),
        ])

        fig, axes = create_subplots(1, 3, figsize=(17.0, 4.8))
        nominal_curve = next((c for c in curves if "nominal" in c[0].lower()), curves[0])
        axes[0].plot(nominal_curve[1] / 60.0, np.maximum(nominal_curve[2], 1e-15), color=PALETTE["blue_main"])
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Time [min]")
        axes[0].set_ylabel("Position difference [m]")
        axes[0].set_title("Nominal analytical–Orekit difference")
        _panel_label(axes[0], "A")

        x = np.arange(len(verification))
        b1 = axes[1].bar(x, verification.position_rmse_m, color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.2)
        axes[1].set_yscale("log")
        axes[1].set_xticks(x, verification.case, rotation=35, ha="right")
        axes[1].set_ylabel("Position RMSE [m]")
        axes[1].set_title("Cross-verification across orbit cases")
        _panel_label(axes[1], "B")

        b2 = axes[2].bar(x, verification.velocity_rmse_mps, color=PALETTE["teal"], edgecolor="black", linewidth=1.2)
        axes[2].set_yscale("log")
        axes[2].set_xticks(x, verification.case, rotation=35, ha="right")
        axes[2].set_ylabel("Velocity RMSE [m/s]")
        axes[2].set_title("Velocity-state agreement")
        _panel_label(axes[2], "C")
        _finish(fig, figures / "01_orbit_propagator_verification", global_cfg)
        figure_manifest.append({
            "figure": "01_orbit_propagator_verification",
            "thesis_subsection": "4.2 Propagator verification",
            "purpose": "Analytical circular-orbit propagator cross-verification against Orekit across nominal and off-nominal orbit cases.",
        })
    else:
        (stage / "orekit_status.json").write_text(json.dumps({"enabled": orekit_enabled, "error": orekit_error}, indent=2), encoding="utf-8")

    return {"sanity": sanity, "verification_cases": len(verification), "orekit_error": orekit_error}


def _nominal_estimation(cfg: dict, global_cfg: dict, out: Path, figures: Path,
                        master: list[dict], figure_manifest: list[dict]) -> dict:
    stage = out / "02_dynamics_sensors_mekf"
    stage.mkdir(parents=True, exist_ok=True)
    ac = copy.deepcopy(global_cfg["attitude"])
    sc = copy.deepcopy(global_cfg["sensors"])
    duration = float(cfg.get("nominal_duration_s", ac["duration_s"]))
    dt = float(cfg.get("nominal_dt_s", ac["dt_s"]))
    t = np.arange(0.0, duration + dt / 2.0, dt)
    inertia = tuple(float(x) for x in ac["inertia_kg_m2"])

    truth = propagate_attitude(
        t, q0=np.asarray(ac["q0"], float), omega0=np.asarray(ac["omega0_rad_s"], float), inertia_diag=inertia,
    )
    energy = rotational_energy(truth[:, 4:], inertia)
    momentum = angular_momentum_norm(truth[:, 4:], inertia)
    energy_drift = float(np.ptp(energy) / max(abs(float(np.mean(energy))), 1e-30))
    momentum_drift = float(np.ptp(momentum) / max(abs(float(np.mean(momentum))), 1e-30))

    sensors = simulate_sensors(
        t, truth, seed=int(sc["seed"]), gyro_bias=np.asarray(sc["gyro_bias_rad_s"], float),
        gyro_noise_std=float(sc["gyro_noise_std_rad_s"]), vector_noise_std=float(sc["vector_noise_std"]),
    )

    # Sensor-model error against the exact simulated state.
    gyro_noise = sensors["gyro"] - truth[:, 4:] - sensors["gyro_bias_true"]
    sun_true = np.empty_like(sensors["sun"])
    mag_true = np.empty_like(sensors["mag"])
    for k, q in enumerate(truth[:, :4]):
        C = quat_to_dcm(q)
        sun_true[k] = C.T @ sensors["sun_i"]
        mag_true[k] = C.T @ sensors["mag_i"]
    sun_err = _angular_error_deg(sensors["sun"], sun_true)
    mag_err = _angular_error_deg(sensors["mag"], mag_true)

    qhat, bhat, diag = run_mekf(
        t, sensors["gyro"], sensors["sun"], sensors["mag"], sensors["sun_i"], sensors["mag_i"], return_diagnostics=True,
    )
    attitude_error = np.array([quat_angle_error_deg(qhat[k], truth[k, :4]) for k in range(len(t))])
    bias_error = bhat - sensors["gyro_bias_true"]
    bias_norm = np.linalg.norm(bias_error, axis=1)

    metrics = {
        "energy_relative_drift": energy_drift,
        "angular_momentum_relative_drift": momentum_drift,
        "gyro_noise_std_x_rad_s": float(np.std(gyro_noise[:, 0])),
        "gyro_noise_std_y_rad_s": float(np.std(gyro_noise[:, 1])),
        "gyro_noise_std_z_rad_s": float(np.std(gyro_noise[:, 2])),
        "sun_sensor_angular_rmse_deg": float(np.sqrt(np.mean(sun_err ** 2))),
        "sun_sensor_angular_p95_deg": float(np.percentile(sun_err, 95)),
        "mag_sensor_angular_rmse_deg": float(np.sqrt(np.mean(mag_err ** 2))),
        "mag_sensor_angular_p95_deg": float(np.percentile(mag_err, 95)),
        "mekf_attitude_rmse_deg": float(np.sqrt(np.mean(attitude_error ** 2))),
        "mekf_attitude_p95_deg": float(np.percentile(attitude_error, 95)),
        "mekf_attitude_max_deg": float(np.max(attitude_error)),
        "mekf_attitude_final_deg": float(attitude_error[-1]),
        "mekf_final_bias_error_rad_s": float(bias_norm[-1]),
    }
    _save_table(pd.DataFrame([metrics]), stage / "nominal_dynamics_sensor_mekf_metrics")

    master.extend([
        _master_row("Attitude dynamics", "Relative rotational-energy drift", energy_drift, "-", "4.3 Dynamics verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("Attitude dynamics", "Relative angular-momentum drift", momentum_drift, "-", "4.3 Dynamics verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("Sensor models", "Sun-vector angular RMSE", metrics["sun_sensor_angular_rmse_deg"], "deg", "4.4 Sensor-model verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("Sensor models", "Magnetometer angular RMSE", metrics["mag_sensor_angular_rmse_deg"], "deg", "4.4 Sensor-model verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("MEKF", "Nominal attitude RMSE", metrics["mekf_attitude_rmse_deg"], "deg", "4.5 Nominal MEKF verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("MEKF", "Maximum attitude error", metrics["mekf_attitude_max_deg"], "deg", "4.5 Nominal MEKF verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
        _master_row("MEKF", "Final gyro-bias error", metrics["mekf_final_bias_error_rad_s"], "rad/s", "4.5 Nominal MEKF verification", "nominal_dynamics_sensor_mekf_metrics.csv"),
    ])

    # Dynamics figure.
    fig, axes = create_subplots(1, 2, figsize=(13.0, 4.8))
    for i, lab in enumerate((r"$\omega_x$", r"$\omega_y$", r"$\omega_z$")):
        axes[0].plot(t, truth[:, 4 + i], label=lab, color=DEFAULT_COLORS[i])
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angular rate [rad/s]")
    axes[0].set_title("Torque-free rigid-body dynamics")
    axes[0].legend(ncol=3)
    _panel_label(axes[0], "A")

    erel = np.abs((energy - energy[0]) / max(abs(float(energy[0])), 1e-30))
    hrel = np.abs((momentum - momentum[0]) / max(abs(float(momentum[0])), 1e-30))
    axes[1].plot(t, np.maximum(erel, 1e-18), color=PALETTE["blue_main"], label="Rotational energy")
    axes[1].plot(t, np.maximum(hrel, 1e-18), color=PALETTE["teal"], label="Angular momentum")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Absolute relative deviation")
    axes[1].set_title("Conservation-law numerical error")
    axes[1].legend()
    _panel_label(axes[1], "B")
    _finish(fig, figures / "02_attitude_dynamics_verification", global_cfg)
    figure_manifest.append({
        "figure": "02_attitude_dynamics_verification",
        "thesis_subsection": "4.3 Dynamics verification",
        "purpose": "Torque-free angular-rate evolution and numerical conservation of rotational energy and angular momentum.",
    })

    # Sensor figure.
    fig, axes = create_subplots(1, 2, figsize=(13.0, 4.8))
    bp = axes[0].boxplot(
        [gyro_noise[:, 0] * 1e6, gyro_noise[:, 1] * 1e6, gyro_noise[:, 2] * 1e6],
        tick_labels=["x", "y", "z"], patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
        boxprops={"facecolor": PALETTE["blue_secondary"], "edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black"}, capprops={"color": "black"},
    )
    axes[0].axhline(float(sc["gyro_noise_std_rad_s"]) * 1e6, color=PALETTE["red_strong"], ls="--", lw=1.5, label="Configured 1σ")
    axes[0].axhline(-float(sc["gyro_noise_std_rad_s"]) * 1e6, color=PALETTE["red_strong"], ls="--", lw=1.5)
    axes[0].set_ylabel("Gyroscope noise [µrad/s]")
    axes[0].set_title("Gyroscope stochastic error")
    axes[0].legend()
    _panel_label(axes[0], "A")

    cats = ["Sun sensor", "Magnetometer"]
    rmse = [metrics["sun_sensor_angular_rmse_deg"], metrics["mag_sensor_angular_rmse_deg"]]
    p95 = [metrics["sun_sensor_angular_p95_deg"], metrics["mag_sensor_angular_p95_deg"]]
    x = np.arange(2)
    w = 0.32
    b1 = axes[1].bar(x - w / 2, rmse, width=w, color=PALETTE["blue_main"], edgecolor="black", linewidth=1.2, label="RMSE")
    b2 = axes[1].bar(x + w / 2, p95, width=w, color=PALETTE["green_3"], edgecolor="black", linewidth=1.2, label="95th percentile")
    axes[1].set_xticks(x, cats)
    axes[1].set_ylabel("Angular measurement error [deg]")
    axes[1].set_title("Vector-sensor measurement error")
    axes[1].legend()
    _annotate_bars(axes[1], b1, "{:.3f}")
    _annotate_bars(axes[1], b2, "{:.3f}")
    _panel_label(axes[1], "B")
    _finish(fig, figures / "03_sensor_model_verification", global_cfg)
    figure_manifest.append({
        "figure": "03_sensor_model_verification",
        "thesis_subsection": "4.4 Sensor-model verification",
        "purpose": "Empirical gyroscope noise and angular errors of the simulated Sun sensor and magnetometer.",
    })

    # MEKF figure.
    fig, axes = create_subplots(1, 2, figsize=(13.0, 4.8))
    axes[0].plot(t, attitude_error, color=PALETTE["red_strong"])
    axes[0].axhline(metrics["mekf_attitude_rmse_deg"], color=PALETTE["neutral_dark"], ls="--", lw=1.4, label=f"RMSE = {metrics['mekf_attitude_rmse_deg']:.4f}°")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Attitude error [deg]")
    axes[0].set_title("Nominal MEKF attitude estimation")
    axes[0].legend()
    _panel_label(axes[0], "A")

    for i, lab in enumerate(("x", "y", "z")):
        axes[1].plot(t, bhat[:, i] * 1e3, color=DEFAULT_COLORS[i], label=f"Estimated {lab}")
        axes[1].axhline(sensors["gyro_bias_true"][i] * 1e3, color=DEFAULT_COLORS[i], ls=":", lw=1.4)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Gyroscope bias [mrad/s]")
    axes[1].set_title("MEKF gyroscope-bias convergence")
    axes[1].legend(ncol=3, fontsize=9)
    _panel_label(axes[1], "B")
    _finish(fig, figures / "04_mekf_nominal_performance", global_cfg)
    figure_manifest.append({
        "figure": "04_mekf_nominal_performance",
        "thesis_subsection": "4.5 Nominal MEKF verification",
        "purpose": "Attitude-estimation error and gyroscope-bias convergence in the nominal sensor case.",
    })

    if bool(cfg.get("save_nominal_timeseries", False)):
        pd.DataFrame({
            "time_s": t,
            "attitude_error_deg": attitude_error,
            "bias_error_norm_rad_s": bias_norm,
            "sun_sensor_error_deg": sun_err,
            "mag_sensor_error_deg": mag_err,
        }).to_csv(stage / "nominal_validation_timeseries.csv", index=False)

    return metrics


def _aocs_validation(cfg: dict, global_cfg: dict, out: Path, figures: Path,
                     master: list[dict], figure_manifest: list[dict]) -> dict:
    stage = out / "03_aocs_closed_loop"
    stage.mkdir(parents=True, exist_ok=True)
    acfg = copy.deepcopy(global_cfg["scenarios"]["reaction_wheel_sun_pointing_comparison"])
    duration = float(cfg.get("aocs_duration_s", acfg.get("duration_s", 6000.0)))
    dt = float(cfg.get("aocs_dt_s", acfg.get("dt_s", 0.1)))
    acfg["duration_s"] = duration
    acfg["dt_s"] = dt
    if "aocs_orbit_profile" in cfg:
        acfg["orbit_profile"] = str(cfg["aocs_orbit_profile"])
    if "aocs_magnetic_field_model" in cfg:
        acfg.setdefault("magnetic_field", {})["model"] = str(cfg["aocs_magnetic_field_model"])

    t = np.arange(0.0, duration + dt / 2.0, dt)
    sun, sun_pos, states, eclipse_state, illumination, B_i, orbit_meta = rw_aocs._make_environment(t, acfg, global_cfg)
    axis = str(acfg.get("pointing_axis", "x"))
    q0 = rw_aocs.sun_pointing_initial_quaternion(axis, sun[0], float(acfg.get("initial_error_deg", 30.0)))
    omega0 = np.asarray(acfg.get("omega0_rad_s", [0.0, 0.0, 0.0]), float)
    sc = global_cfg["sensors"]
    rng = np.random.default_rng(int(cfg.get("aocs_seed", acfg.get("comparison_seed", 7301))))
    noise = {
        "gyro": rng.normal(0.0, float(sc["gyro_noise_std_rad_s"]), (len(t), 3)),
        "sun": rng.normal(0.0, float(sc["vector_noise_std"]), (len(t), 3)),
        "mag": rng.normal(0.0, float(sc["vector_noise_std"]), (len(t), 3)),
    }
    result = rw_aocs._run_branch(
        "healthy_mekf", acfg, global_cfg, t, q0, omega0, sun, B_i, states,
        eclipse_state, noise, predictor=None, fault_case=None, fdir_only=False, fdir_mode="oracle", fdir_config=None,
    )

    sunlight = np.asarray(eclipse_state) != 2
    pointing = np.asarray(result["true_point"], float)
    att = np.asarray(result["att_err"], float)
    mode_cfg = acfg.get("modes", {})
    acquisition = rw_aocs._settling_time(
        t, pointing, float(mode_cfg.get("pointing_threshold_deg", 2.0)), float(mode_cfg.get("pointing_hold_s", 15.0)),
    )
    metrics = {
        "sunlight_pointing_rmse_deg": float(np.sqrt(np.mean(pointing[sunlight] ** 2))) if np.any(sunlight) else np.nan,
        "pointing_max_deg": float(np.max(pointing)),
        "attitude_rmse_deg": float(np.sqrt(np.mean(att ** 2))),
        "initial_acquisition_time_s": acquisition,
        "max_reaction_wheel_torque_uNm": float(np.max(np.linalg.norm(result["rw_tau"], axis=1)) * 1e6),
        "max_wheel_momentum_mNms": float(np.max(np.linalg.norm(result["wheel_h"], axis=1)) * 1e3),
        "max_wheel_speed_rpm": float(np.max(np.abs(result["wheel_speed_rpm"]))),
        "momentum_dump_active_pct": float(100.0 * np.mean(result["dump_active"])),
        "orbit_beta_deg": float(orbit_meta.get("beta_angle_deg", np.nan)),
    }
    _save_table(pd.DataFrame([metrics]), stage / "aocs_nominal_metrics")
    master.extend([
        _master_row("AOCS", "Sunlight Sun-pointing RMSE", metrics["sunlight_pointing_rmse_deg"], "deg", "4.6 Closed-loop AOCS validation", "aocs_nominal_metrics.csv"),
        _master_row("AOCS", "Initial acquisition time", metrics["initial_acquisition_time_s"], "s", "4.6 Closed-loop AOCS validation", "aocs_nominal_metrics.csv"),
        _master_row("AOCS", "Maximum wheel torque", metrics["max_reaction_wheel_torque_uNm"], "µN m", "4.6 Closed-loop AOCS validation", "aocs_nominal_metrics.csv"),
        _master_row("AOCS", "Maximum wheel speed", metrics["max_wheel_speed_rpm"], "rpm", "4.6 Closed-loop AOCS validation", "aocs_nominal_metrics.csv"),
    ])

    fig, axes = create_subplots(1, 3, figsize=(17.0, 4.8))
    axes[0].plot(t / 60.0, pointing, color=PALETTE["blue_main"])
    _shade_eclipses(axes[0], t / 60.0, eclipse_state)
    axes[0].axhline(float(mode_cfg.get("pointing_threshold_deg", 2.0)), color=PALETTE["red_strong"], ls="--", lw=1.3, label="Pointing threshold")
    axes[0].set_xlabel("Time [min]")
    axes[0].set_ylabel("Sun-pointing error [deg]")
    axes[0].set_title("Closed-loop Sun pointing")
    axes[0].legend(fontsize=9)
    _panel_label(axes[0], "A")

    for i, lab in enumerate(("x", "y", "z")):
        axes[1].plot(t / 60.0, result["wheel_speed_rpm"][:, i], label=lab, color=DEFAULT_COLORS[i])
    axes[1].set_xlabel("Time [min]")
    axes[1].set_ylabel("Wheel speed [rpm]")
    axes[1].set_title("Reaction-wheel utilization")
    axes[1].legend(ncol=3)
    _panel_label(axes[1], "B")

    axes[2].step(t / 60.0, result["mode_code"], where="post", color=PALETTE["neutral_dark"])
    axes[2].set_yticks(list(rw_aocs.MODE_CODE.values()), list(rw_aocs.MODE_CODE.keys()))
    axes[2].set_xlabel("Time [min]")
    axes[2].set_ylabel("AOCS mode")
    axes[2].set_title("Operational state-machine evolution")
    _shade_eclipses(axes[2], t / 60.0, eclipse_state)
    _panel_label(axes[2], "C")
    _finish(fig, figures / "05_reaction_wheel_aocs_closed_loop", global_cfg)
    figure_manifest.append({
        "figure": "05_reaction_wheel_aocs_closed_loop",
        "thesis_subsection": "4.6 Closed-loop AOCS validation",
        "purpose": "Nominal Sun-pointing error, reaction-wheel utilization and AOCS operational mode transitions including eclipse intervals.",
    })

    return metrics


def _experiment1_results(cfg: dict, global_cfg: dict, out: Path, figures: Path,
                         master: list[dict], figure_manifest: list[dict]) -> dict:
    stage = out / "04_experiment1_ml_recovery"
    stage.mkdir(parents=True, exist_ok=True)
    ecfg = copy.deepcopy(global_cfg["scenarios"]["reaction_wheel_fair_ml_campaign"])
    ecfg["plots_enabled"] = bool(cfg.get("component_plots_enabled", False))
    ecfg["save_timeseries"] = bool(cfg.get("save_component_timeseries", False))
    if "experiment1_groups" in cfg:
        ecfg.setdefault("fault_scenarios", {})["groups"] = list(cfg["experiment1_groups"])
    experiment1.run(ecfg, global_cfg, stage)

    summary = pd.read_csv(stage / "reaction_wheel_fair_ml_campaign_summary.csv")
    ml = summary[~summary.branch.isin(["healthy_mekf", "faulty_mekf", "fdir_only"])].copy()
    if len(ml):
        med_improvement = float(np.nanmedian(ml.pointing_improvement_vs_fdir_pct))
        positive_fraction = float(100.0 * np.mean(ml.pointing_improvement_vs_fdir_pct > 0.0))
    else:
        med_improvement = np.nan
        positive_fraction = np.nan
    metrics = {
        "evaluated_fault_cases": int(summary.case_name.nunique()),
        "ml_branches": int(ml.branch.nunique()),
        "median_ml_pointing_improvement_vs_fdir_pct": med_improvement,
        "ml_branch_case_fraction_improving_fdir_pct": positive_fraction,
    }
    _save_table(pd.DataFrame([metrics]), stage / "experiment1_thesis_summary")
    master.extend([
        _master_row("Experiment 1 — ML recovery", "Median ML improvement vs FDIR-only", med_improvement, "%", "4.8 Experiment 1: virtual-sensor recovery", "experiment1_thesis_summary.csv"),
        _master_row("Experiment 1 — ML recovery", "ML branch/case combinations improving FDIR", positive_fraction, "%", "4.8 Experiment 1: virtual-sensor recovery", "experiment1_thesis_summary.csv"),
    ])

    # Heatmap: improvement of every fixed ML architecture/variant branch over FDIR-only.
    if len(ml):
        ml["model_variant"] = ml["architecture"].str.upper() + "\n" + ml["variant"].str.replace("_", " ")
        pivot = ml.pivot_table(index="case_name", columns="model_variant", values="pointing_improvement_vs_fdir_pct", aggfunc="mean")
        preferred = []
        for arch in ("MLP", "CNN", "GRU"):
            for variant in ("sensor only", "orbital context", "physics residual"):
                key = f"{arch}\n{variant}"
                if key in pivot.columns:
                    preferred.append(key)
        pivot = pivot.reindex(columns=preferred or list(pivot.columns))
        arr = pivot.to_numpy(float)
        finite = arr[np.isfinite(arr)]
        lim = max(10.0, float(np.nanpercentile(np.abs(finite), 95))) if len(finite) else 10.0
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        fig, axes = create_subplots(figsize=(13.5, max(5.2, 0.48 * len(pivot) + 2.2)))
        ax = axes[0]
        im = ax.imshow(arr, aspect="auto", cmap="RdBu", norm=norm)
        ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
        ax.set_title("Experiment 1: observed ML change relative to FDIR-only")
        ax.set_xlabel("ML architecture / input variant")
        ax.set_ylabel("Fault case")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i,j]:.0f}%", ha="center", va="center", fontsize=7,
                            color="white" if abs(arr[i, j]) > 0.55 * lim else "black")
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Pointing improvement vs FDIR-only [%]\n(positive = better)")
        _finish(fig, figures / "07_experiment1_virtual_sensor_recovery", global_cfg)
        figure_manifest.append({
            "figure": "07_experiment1_virtual_sensor_recovery",
            "thesis_subsection": "4.8 Experiment 1: virtual-sensor recovery",
            "purpose": "Per-case performance change of the nine fixed MLP/CNN/GRU recovery variants relative to the classical FDIR-only branch.",
        })

    return metrics


def _detector_category_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df.groupby(["fault_category", "detector"], as_index=False).agg(
        precision_pct=("precision_pct", "mean"),
        recall_pct=("recall_pct", "mean"),
        f1_pct=("f1_pct", "mean"),
        far_pct=("false_alarm_rate_pct", "mean"),
        detection_delay_s=("mean_detection_delay_s", "mean"),
    )


def _experiment2_results(cfg: dict, global_cfg: dict, out: Path, figures: Path,
                         master: list[dict], figure_manifest: list[dict]) -> dict:
    stage = out / "05_experiment2_ml_fdir"
    stage.mkdir(parents=True, exist_ok=True)
    ecfg = copy.deepcopy(global_cfg["scenarios"]["reaction_wheel_ml_fdir_campaign"])
    ecfg["plots_enabled"] = bool(cfg.get("component_plots_enabled", False))
    ecfg["save_timeseries"] = bool(cfg.get("save_component_timeseries", False))
    experiment2.run(ecfg, global_cfg, stage)

    metrics_df = pd.read_csv(stage / "tables" / "detector_metrics_by_case_sensor.csv")
    selection = pd.read_csv(stage / "tables" / "heterogeneous_hybrid_sensor_selection.csv")
    resources = pd.read_csv(stage / "tables" / "model_resource_metrics.csv")
    stability = pd.read_csv(stage / "tables" / "heterogeneous_hybrid_validation_run_stability.csv")

    category = _detector_category_table(metrics_df)
    _save_table(category, stage / "experiment2_category_summary")

    # Classical FDIR baseline is reported separately before the ML experiments.
    classical_cat = category[category.detector == "classical_v31"].copy()
    _save_table(classical_cat, stage / "classical_fdir_category_summary")
    for cat, label in (("soft_or_gradual", "soft/gradual"), ("hard_or_intermittent", "hard/intermittent")):
        q = classical_cat[classical_cat.fault_category == cat]
        if len(q):
            r = q.iloc[0]
            master.extend([
                _master_row("Classical FDIR", f"{label} recall", r.recall_pct, "%", "4.7 Classical FDIR baseline", "classical_fdir_category_summary.csv"),
                _master_row("Classical FDIR", f"{label} F1", r.f1_pct, "%", "4.7 Classical FDIR baseline", "classical_fdir_category_summary.csv"),
                _master_row("Classical FDIR", f"{label} false-alarm rate", r.far_pct, "%", "4.7 Classical FDIR baseline", "classical_fdir_category_summary.csv"),
            ])

    if len(classical_cat):
        fig, axes = create_subplots(1, 3, figsize=(15.5, 4.7))
        cat_order = [("soft_or_gradual", "Soft / gradual"), ("hard_or_intermittent", "Hard / intermittent")]
        x = np.arange(len(cat_order))
        for ax, metric_name, ylabel, panel, color in [
            (axes[0], "recall_pct", "Recall [%]", "A", PALETTE["blue_main"]),
            (axes[1], "f1_pct", "F1 score [%]", "B", PALETTE["green_3"]),
            (axes[2], "far_pct", "False-alarm rate [%]", "C", PALETTE["red_strong"]),
        ]:
            vals=[]
            for cat,_ in cat_order:
                q=classical_cat[classical_cat.fault_category==cat]
                vals.append(float(q.iloc[0][metric_name]) if len(q) else np.nan)
            bars=ax.bar(x,vals,color=color,edgecolor="black",linewidth=1.2,width=.62)
            ax.set_xticks(x,[name for _,name in cat_order])
            ax.set_ylabel(ylabel)
            _annotate_bars(ax,bars,"{:.2f}")
            _panel_label(ax,panel)
        axes[0].set_title("Classical FDIR sensitivity")
        axes[1].set_title("Classical FDIR detection quality")
        axes[2].set_title("Classical FDIR false alarms")
        _finish(fig, figures / "06_classical_fdir_baseline", global_cfg)
        figure_manifest.append({
            "figure": "06_classical_fdir_baseline",
            "thesis_subsection": "4.7 Classical FDIR baseline",
            "purpose": "Recall, F1 and false-alarm rate of the final classical FDIR separated into soft/gradual and hard/intermittent fault categories.",
        })

    overall = metrics_df.groupby("detector", as_index=False).agg(
        precision_pct=("precision_pct", "mean"), recall_pct=("recall_pct", "mean"),
        f1_pct=("f1_pct", "mean"), far_pct=("false_alarm_rate_pct", "mean"),
        detection_delay_s=("mean_detection_delay_s", "mean"),
    )
    _save_table(overall, stage / "experiment2_overall_summary")

    robust = overall[overall.detector == "hybrid_validation_selected"]
    classical = overall[overall.detector == "classical_v31"]
    robust_f1 = float(robust.iloc[0].f1_pct) if len(robust) else np.nan
    robust_far = float(robust.iloc[0].far_pct) if len(robust) else np.nan
    classical_f1 = float(classical.iloc[0].f1_pct) if len(classical) else np.nan
    classical_far = float(classical.iloc[0].far_pct) if len(classical) else np.nan
    selected_map = ", ".join(f"{r.sensor}:{r.selected_architecture}" for _, r in selection.iterrows())
    metrics = {
        "classical_macro_f1_pct": classical_f1,
        "classical_macro_far_pct": classical_far,
        "robust_hybrid_macro_f1_pct": robust_f1,
        "robust_hybrid_macro_far_pct": robust_far,
        "robust_hybrid_f1_change_pp": robust_f1 - classical_f1 if np.isfinite(robust_f1) and np.isfinite(classical_f1) else np.nan,
        "selected_sensor_models": selected_map,
    }
    _save_table(pd.DataFrame([metrics]), stage / "experiment2_thesis_summary")
    master.extend([
        _master_row("Experiment 2 — ML-assisted FDIR", "Classical macro F1", classical_f1, "%", "4.9 Experiment 2: ML-assisted FDIR", "experiment2_overall_summary.csv"),
        _master_row("Experiment 2 — ML-assisted FDIR", "Robust hybrid macro F1", robust_f1, "%", "4.9 Experiment 2: ML-assisted FDIR", "experiment2_overall_summary.csv"),
        _master_row("Experiment 2 — ML-assisted FDIR", "Classical mean FAR", classical_far, "%", "4.9 Experiment 2: ML-assisted FDIR", "experiment2_overall_summary.csv"),
        _master_row("Experiment 2 — ML-assisted FDIR", "Robust hybrid mean FAR", robust_far, "%", "4.9 Experiment 2: ML-assisted FDIR", "experiment2_overall_summary.csv"),
    ])

    detector_order = ["classical_v31", "hybrid_mlp", "hybrid_cnn", "hybrid_gru", "hybrid_validation_selected"]
    labels = ["Classical", "Hybrid MLP", "Hybrid CNN", "Hybrid GRU", "Robust selected"]
    colors = [PALETTE["neutral_dark"], PALETTE["blue_secondary"], PALETTE["teal"], PALETTE["violet"], PALETTE["blue_main"]]
    cats = [("soft_or_gradual", "Soft / gradual faults"), ("hard_or_intermittent", "Hard / intermittent faults")]

    fig, axes = create_subplots(1, 2, figsize=(14.5, 5.0))
    x = np.arange(len(cats))
    width = 0.15
    for j, (det, label, color) in enumerate(zip(detector_order, labels, colors)):
        f1vals = []
        farvals = []
        for cat, _ in cats:
            q = category[(category.fault_category == cat) & (category.detector == det)]
            f1vals.append(float(q.iloc[0].f1_pct) if len(q) else np.nan)
            farvals.append(float(q.iloc[0].far_pct) if len(q) else np.nan)
        axes[0].bar(x + (j - 2) * width, f1vals, width=width, label=label, color=color, edgecolor="black", linewidth=1.0)
        axes[1].bar(x + (j - 2) * width, farvals, width=width, label=label, color=color, edgecolor="black", linewidth=1.0)
    for ax in axes:
        ax.set_xticks(x, [b for _, b in cats])
    axes[0].set_ylabel("Macro F1 [%]")
    axes[0].set_title("Fault-detection performance")
    _panel_label(axes[0], "A")
    axes[1].set_ylabel("False-alarm rate [%]")
    axes[1].set_title("False-alarm cost")
    _panel_label(axes[1], "B")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=5, fontsize=8, frameon=False)
    _finish(fig, figures / "08_experiment2_ml_assisted_fdir", global_cfg)
    figure_manifest.append({
        "figure": "08_experiment2_ml_assisted_fdir",
        "thesis_subsection": "4.9 Experiment 2: ML-assisted FDIR",
        "purpose": "Classical and ML-assisted FDIR performance separated into soft/gradual and hard/intermittent fault categories.",
    })

    # Fault-family detail: F1 and FAR of classical vs the final robust branch.
    family = metrics_df.groupby(["fault_family", "detector"], as_index=False).agg(
        f1_pct=("f1_pct", "mean"), far_pct=("false_alarm_rate_pct", "mean")
    )
    soft_families = [x for x in ("gyro_bias", "gyro_drift", "gyro_noise", "mag_misalignment", "mag_bias", "sun_noise", "combined_soft") if x in set(family.fault_family)]
    fig, axes = create_subplots(1, 2, figsize=(15.5, 5.0))
    x = np.arange(len(soft_families)); w = 0.36
    for ax, metric_name, ylabel, panel in [
        (axes[0], "f1_pct", "Macro F1 [%]", "A"),
        (axes[1], "far_pct", "False-alarm rate [%]", "B"),
    ]:
        cvals=[]; rvals=[]
        for fam in soft_families:
            qc=family[(family.fault_family==fam)&(family.detector=="classical_v31")]
            qr=family[(family.fault_family==fam)&(family.detector=="hybrid_validation_selected")]
            cvals.append(float(qc.iloc[0][metric_name]) if len(qc) else np.nan)
            rvals.append(float(qr.iloc[0][metric_name]) if len(qr) else np.nan)
        ax.bar(x-w/2,cvals,width=w,color=PALETTE["neutral"],edgecolor="black",linewidth=1.0,label="Classical")
        ax.bar(x+w/2,rvals,width=w,color=PALETTE["blue_main"],edgecolor="black",linewidth=1.0,label="Robust selected")
        ax.set_xticks(x,[f.replace("_"," ") for f in soft_families],rotation=30,ha="right")
        ax.set_ylabel(ylabel)
        ax.legend()
        _panel_label(ax,panel)
    axes[0].set_title("Robust detector by soft-fault family")
    axes[1].set_title("Run-level robustness versus false alarms")
    _finish(fig, figures / "09_final_robust_fdir_by_fault_family", global_cfg)
    figure_manifest.append({
        "figure": "09_final_robust_fdir_by_fault_family",
        "thesis_subsection": "4.9 Experiment 2: ML-assisted FDIR",
        "purpose": "Final robust validation-selected detector compared with classical FDIR for each soft-fault family.",
    })

    # Lightweight resource comparison.
    resources = resources.copy()
    resources["architecture"] = resources.architecture.str.upper()
    for _, rr in resources.iterrows():
        master.extend([
            _master_row(f"{rr.architecture} resources", "Trainable parameters", rr.parameters, "parameters", "4.10 Computational-resource assessment", "model_resource_metrics.csv"),
            _master_row(f"{rr.architecture} resources", "Serialized model size", rr.model_size_mb, "MB", "4.10 Computational-resource assessment", "model_resource_metrics.csv"),
            _master_row(f"{rr.architecture} resources", "CPU inference time", rr.inference_ms_cpu_batch1, "ms", "4.10 Computational-resource assessment", "model_resource_metrics.csv"),
        ])
    fig, axes = create_subplots(1, 3, figsize=(16.0, 4.7))
    rc = {"MLP": PALETTE["blue_main"], "CNN": PALETTE["teal"], "GRU": PALETTE["violet"]}
    bars = axes[0].bar(resources.architecture, resources.parameters, color=[rc.get(x, PALETTE["neutral"]) for x in resources.architecture], edgecolor="black", linewidth=1.2)
    axes[0].set_ylabel("Trainable parameters")
    axes[0].set_title("Model complexity")
    _annotate_bars(axes[0], bars, "{:.0f}")
    _panel_label(axes[0], "A")
    bars = axes[1].bar(resources.architecture, resources.model_size_mb, color=[rc.get(x, PALETTE["neutral"]) for x in resources.architecture], edgecolor="black", linewidth=1.2)
    axes[1].set_ylabel("Serialized model size [MB]")
    axes[1].set_title("Memory footprint")
    _annotate_bars(axes[1], bars, "{:.3f}")
    _panel_label(axes[1], "B")
    bars = axes[2].bar(resources.architecture, resources.inference_ms_cpu_batch1, color=[rc.get(x, PALETTE["neutral"]) for x in resources.architecture], edgecolor="black", linewidth=1.2)
    axes[2].set_ylabel("CPU inference time [ms]")
    axes[2].set_title("Inference cost")
    _annotate_bars(axes[2], bars, "{:.3f}")
    _panel_label(axes[2], "C")
    _finish(fig, figures / "10_lightweight_model_resource_comparison", global_cfg)
    figure_manifest.append({
        "figure": "10_lightweight_model_resource_comparison",
        "thesis_subsection": "4.10 Computational-resource assessment",
        "purpose": "Trainable parameters, serialized model size and CPU inference cost of MLP, CNN and GRU fault detectors.",
    })

    # Copy the key validation tables to a convenient thesis-level directory.
    for src in [
        stage / "tables" / "heterogeneous_hybrid_sensor_selection.csv",
        stage / "tables" / "heterogeneous_hybrid_validation_run_stability.csv",
        stage / "tables" / "model_resource_metrics.csv",
    ]:
        if src.exists():
            shutil.copy2(src, out / "tables" / src.name)

    return metrics


def _write_thesis_guide(out: Path, figure_manifest: list[dict]) -> None:
    lines = [
        "# Thesis results package\n",
        "The figures in `figures/` are ordered to match the recommended Testing and Results chapter.\n",
        "## Recommended figure order\n",
    ]
    for i, item in enumerate(sorted(figure_manifest, key=lambda x: x["figure"]), 1):
        lines.append(f"{i}. **{item['figure']}** — {item['thesis_subsection']} — {item['purpose']}\n")
    lines.extend([
        "\n## Interpretation rule\n",
        "Use only metrics produced by the corresponding table/figure. Do not claim that ML is globally superior when the robust validation-selected branch retains classical FDIR. Homogeneous ML hybrids are exploratory comparators; the robust selected branch is the deployment-oriented result.\n",
    ])
    (out / "THESIS_RESULTS_README.md").write_text("".join(lines), encoding="utf-8")


def run(cfg: dict, global_cfg: dict, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    master: list[dict] = []
    figure_manifest: list[dict] = []
    stage_status: dict[str, dict] = {}

    if bool(cfg.get("run_orbit", True)):
        stage_status["orbit"] = _orbit_validation(cfg, global_cfg, out, figures, master, figure_manifest)
    if bool(cfg.get("run_nominal_estimation", True)):
        stage_status["nominal_estimation"] = _nominal_estimation(cfg, global_cfg, out, figures, master, figure_manifest)
    if bool(cfg.get("run_aocs", True)):
        stage_status["aocs"] = _aocs_validation(cfg, global_cfg, out, figures, master, figure_manifest)
    if bool(cfg.get("run_experiment1", True)):
        stage_status["experiment1"] = _experiment1_results(cfg, global_cfg, out, figures, master, figure_manifest)
    if bool(cfg.get("run_experiment2", True)):
        stage_status["experiment2"] = _experiment2_results(cfg, global_cfg, out, figures, master, figure_manifest)

    master_df = pd.DataFrame(master)
    _save_table(master_df, tables / "thesis_master_metrics")
    manifest_df = pd.DataFrame(sorted(figure_manifest, key=lambda x: x["figure"]))
    _save_table(manifest_df, tables / "thesis_figure_manifest")
    _write_thesis_guide(out, figure_manifest)
    (out / "thesis_results_metadata.json").write_text(
        json.dumps({
            "scenario": "thesis_results_campaign",
            "stage_status": stage_status,
            "figure_count": len(figure_manifest),
            "style_reference": "ChenLiu-1996/figures4papers (documented publication-style conventions)",
            "component_timeseries_saved": bool(cfg.get("save_component_timeseries", False)),
        }, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "scenario": "thesis_results_campaign",
        "figures": len(figure_manifest),
        "metrics": len(master_df),
        "experiment1_run": bool(cfg.get("run_experiment1", True)),
        "experiment2_run": bool(cfg.get("run_experiment2", True)),
    }
