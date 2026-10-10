from __future__ import annotations

from pathlib import Path
import copy
import json
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.simulation.control import sun_pointing_initial_quaternion
from src.scenarios.reaction_wheel_sun_pointing_comparison import _make_environment, _run_branch
from src.scenarios.reaction_wheel_fair_ml_campaign import _runtime_mission_context, _save_table
from src.ml.fdir_classifier import (
    SENSORS,
    train_multilabel_fdir_model,
    apply_probability_filter,
    choose_runtime_multilabel_thresholds,
    evaluate_runtime_multilabel_thresholds,
    save_ml_fdir_bundle,
)
from src.visualization.publication import PALETTE, create_subplots, finalize_figure


FEATURE_NAMES_BASE = [
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "sun_body_x", "sun_body_y", "sun_body_z",
    "mag_body_x", "mag_body_y", "mag_body_z",
    "gyro_available", "sun_available", "mag_available",
    "gyro_norm_rad_s", "sun_norm", "mag_norm",
    "gyro_rate_residual_rad_s",
    "gyro_vector_rate_residual_rad_s",
    "gyro_cross_sensor_deg",
    "sun_innovation_deg",
    "mag_innovation_deg",
    "sun_mag_geometry_error_deg",
    "sun_gyro_track_error_deg",
    "mag_gyro_track_error_deg",
    "gyro_rate_residual_valid",
    "gyro_vector_rate_residual_valid",
    "gyro_cross_sensor_valid",
    "sun_innovation_valid",
    "mag_innovation_valid",
    "sun_mag_geometry_valid",
    "sun_gyro_track_valid",
    "mag_gyro_track_valid",
]

CONTEXT_NAMES = [
    "r_eci_unit_x", "r_eci_unit_y", "r_eci_unit_z",
    "v_eci_unit_x", "v_eci_unit_y", "v_eci_unit_z",
    "sun_eci_unit_x", "sun_eci_unit_y", "sun_eci_unit_z",
    "B_eci_unit_x", "B_eci_unit_y", "B_eci_unit_z",
    "illumination_fraction", "sin_orbital_phase", "cos_orbital_phase",
    "sin_beta", "cos_beta",
]


SOFT_FAMILIES = {
    "gyro_bias", "gyro_drift", "gyro_noise", "mag_misalignment",
    "mag_bias", "sun_noise", "combined_soft",
}
HARD_FAMILIES = {"hard_loss", "dropout"}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")


def _unit_axis(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _base_case(name: str, family: str, start_s: float, end_s: float) -> dict:
    return {
        "name": name,
        "family": family,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "sun": {"type": "none"},
        "gyro": {"type": "none"},
        "mag": {"type": "none"},
    }


def _sample_training_case(family: str, rng: np.random.Generator, start_s: float, end_s: float, variant_index: int | None = None) -> dict:
    fam = str(family).lower()
    c = _base_case(f"train_{fam}", fam, start_s, end_s)
    if fam == "healthy":
        return c
    if fam == "gyro_bias":
        mag = float(rng.uniform(5e-4, 1.2e-2))
        c["gyro"] = {"type": "bias_step", "bias_rad_s": (_unit_axis(rng) * mag).tolist()}
    elif fam == "gyro_drift":
        mag = float(10 ** rng.uniform(np.log10(5e-7), np.log10(1.5e-5)))
        c["gyro"] = {"type": "drift", "drift_rad_s2": (_unit_axis(rng) * mag).tolist()}
    elif fam == "gyro_noise":
        c["gyro"] = {"type": "noise", "noise_std_rad_s": float(rng.uniform(4e-4, 3e-3))}
    elif fam == "mag_misalignment":
        a = float(rng.uniform(3.0, 30.0))
        c["mag"] = {"type": "misalignment", "angles_deg": [a, -.7 * a, .5 * a]}
    elif fam == "mag_bias":
        a = _unit_axis(rng) * float(rng.uniform(.01, .08))
        c["mag"] = {"type": "bias", "bias_body": a.tolist()}
    elif fam == "sun_noise":
        c["sun"] = {"type": "sun_noise", "severity": str(rng.choice(["low", "medium", "high"]))}
    elif fam == "hard_loss":
        sensor = SENSORS[int(variant_index) % 3] if variant_index is not None else str(rng.choice(SENSORS))
        if sensor == "sun":
            c["sun"] = {"type": "sun_loss", "severity": "high"}
        else:
            c[sensor] = {"type": "loss"}
    elif fam == "dropout":
        sensor = SENSORS[int(variant_index) % 3] if variant_index is not None else str(rng.choice(SENSORS))
        p = float(rng.uniform(.2, .7))
        if sensor == "sun":
            c["sun"] = {"type": "sun_dropout", "severity": "high", "probability": p}
        else:
            c[sensor] = {"type": "dropout", "probability": p}
    elif fam == "combined_soft":
        variant = int(variant_index) % 3 if variant_index is not None else int(rng.integers(0, 3))
        if variant == 0:
            bg = float(rng.uniform(.001, .010))
            a = float(rng.uniform(5.0, 25.0))
            c["gyro"] = {"type": "bias_step", "bias_rad_s": (_unit_axis(rng) * bg).tolist()}
            c["mag"] = {"type": "misalignment", "angles_deg": [a, -.7 * a, .5 * a]}
        elif variant == 1:
            dg = float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-5)))
            c["gyro"] = {"type": "drift", "drift_rad_s2": (_unit_axis(rng) * dg).tolist()}
            c["sun"] = {"type": "sun_dropout", "severity": "high", "probability": float(rng.uniform(.2, .6))}
        else:
            bg = float(rng.uniform(.001, .010))
            a = float(rng.uniform(5.0, 25.0))
            c["gyro"] = {"type": "bias_step", "bias_rad_s": (_unit_axis(rng) * bg).tolist()}
            c["sun"] = {"type": "sun_dropout", "severity": "high", "probability": float(rng.uniform(.2, .6))}
            c["mag"] = {"type": "misalignment", "angles_deg": [a, -.7 * a, .5 * a]}
    else:
        raise ValueError(f"Unknown ML-FDIR training family: {family}")
    return c


def _evaluation_cases(cfg: dict) -> list[dict]:
    ec = cfg.get("evaluation", {})
    start = float(ec.get("fault_start_s", 300.0))
    end = float(ec.get("fault_end_s", 1400.0))
    out: list[dict] = [_base_case("E00_healthy_control", "healthy", start, end)]

    axis = np.asarray(ec.get("gyro_fault_axis", [1.0, -.7, .5]), float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    for x in ec.get("gyro_bias_levels_rad_s", [.001, .003, .006, .012]):
        x = float(x)
        c = _base_case(f"E_gyro_bias_{x:g}", "gyro_bias", start, end)
        c["gyro"] = {"type": "bias_step", "bias_rad_s": (axis * x).tolist()}
        out.append(c)
    da = np.asarray(ec.get("gyro_drift_axis", [1.0, -.75, .5]), float)
    da = da / (np.linalg.norm(da) + 1e-12)
    for x in ec.get("gyro_drift_levels_rad_s2", [1e-6, 3e-6, 8e-6]):
        x = float(x)
        c = _base_case(f"E_gyro_drift_{x:.0e}", "gyro_drift", start, end)
        c["gyro"] = {"type": "drift", "drift_rad_s2": (da * x).tolist()}
        out.append(c)
    for x in ec.get("gyro_noise_levels_rad_s", [8e-4, 2e-3]):
        x = float(x)
        c = _base_case(f"E_gyro_noise_{x:g}", "gyro_noise", start, end)
        c["gyro"] = {"type": "noise", "noise_std_rad_s": x}
        out.append(c)
    for a in ec.get("mag_misalignment_levels_deg", [5., 10., 20., 30.]):
        a = float(a)
        c = _base_case(f"E_mag_misalignment_{a:g}deg", "mag_misalignment", start, end)
        c["mag"] = {"type": "misalignment", "angles_deg": [a, -.7 * a, .5 * a]}
        out.append(c)
    for b in ec.get("mag_bias_levels", [.02, .06]):
        b = float(b)
        c = _base_case(f"E_mag_bias_{b:g}", "mag_bias", start, end)
        c["mag"] = {"type": "bias", "bias_body": [b, -.6 * b, .4 * b]}
        out.append(c)
    for sev in ec.get("sun_noise_severities", ["low", "medium", "high"]):
        c = _base_case(f"E_sun_noise_{sev}", "sun_noise", start, end)
        c["sun"] = {"type": "sun_noise", "severity": str(sev)}
        out.append(c)

    # Hard-fault controls: classical validity logic should remain the reference.
    for sensor in ("gyro", "sun", "mag"):
        c = _base_case(f"E_{sensor}_loss", "hard_loss", start, end)
        c[sensor] = {"type": "sun_loss", "severity": "high"} if sensor == "sun" else {"type": "loss"}
        out.append(c)
        c = _base_case(f"E_{sensor}_dropout_50pct", "dropout", start, end)
        c[sensor] = {"type": "sun_dropout", "severity": "high", "probability": .5} if sensor == "sun" else {"type": "dropout", "probability": .5}
        out.append(c)

    # Representative multi-sensor cases carried over from Experiment 1.
    c = _base_case("E_D4_gyro_bias_plus_mag_misalignment", "combined_soft", start, end)
    c["gyro"] = {"type": "bias_step", "bias_rad_s": (axis * .006).tolist()}
    c["mag"] = {"type": "misalignment", "angles_deg": [20., -14., 10.]}
    out.append(c)
    c = _base_case("E_D5_gyro_drift_plus_sun_dropout", "combined_soft", start, end)
    c["gyro"] = {"type": "drift", "drift_rad_s2": (da * 3e-6).tolist()}
    c["sun"] = {"type": "sun_dropout", "severity": "high", "probability": .5}
    out.append(c)
    c = _base_case("E_D9_all_three_degraded", "combined_soft", start, end)
    c["gyro"] = {"type": "bias_step", "bias_rad_s": (axis * .006).tolist()}
    c["sun"] = {"type": "sun_dropout", "severity": "high", "probability": .5}
    c["mag"] = {"type": "misalignment", "angles_deg": [20., -14., 10.]}
    out.append(c)

    # Matched-severity design: cases in the same sweep share initial attitude,
    # orbit, sensor-noise realization and fault RNG seed for a given repeat. Only
    # the injected fault severity differs. Post-fault closed-loop trajectories may
    # naturally diverge because the fault changes the estimator/controller state.
    matched_families = {"gyro_bias", "gyro_drift", "gyro_noise", "mag_misalignment", "mag_bias", "sun_noise"}
    for case in out:
        fam = str(case["family"])
        if fam in matched_families:
            case["match_group"] = fam
        else:
            case["match_group"] = str(case["name"])
    return out


def _fault_category(family: str) -> str:
    fam = str(family).lower()
    if fam == "healthy":
        return "healthy_control"
    if fam in SOFT_FAMILIES:
        return "soft_or_gradual"
    if fam in HARD_FAMILIES:
        return "hard_or_intermittent"
    return "other"


def _sanitize_residual(a: np.ndarray, clip: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, float)
    valid = np.isfinite(a)
    x = np.where(valid, a, 0.0)
    x = np.clip(x, -float(clip), float(clip))
    return x.astype(np.float32), valid.astype(np.float32)


def _build_features(result: dict, context: np.ndarray) -> tuple[np.ndarray, list[str]]:
    gyro = np.asarray(result["gyro"], float)
    sun = np.asarray(result["sun_raw"], float)
    mag = np.asarray(result["mag"], float)
    n = len(gyro)

    # Interface-level availability only; no classical confidence/state is exposed
    # to the ML classifier. In the simulator a gyro loss/dropout is encoded as an
    # exact zero vector. Real hardware should substitute the device validity bit.
    gyro_finite = np.all(np.isfinite(gyro), axis=1)
    gyro_available = (gyro_finite & (np.linalg.norm(np.nan_to_num(gyro), axis=1) > 1e-10)).astype(np.float32)
    sun_available = np.all(np.isfinite(sun), axis=1).astype(np.float32)
    mag_available = np.all(np.isfinite(mag), axis=1).astype(np.float32)

    g = np.nan_to_num(gyro, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    s = np.nan_to_num(sun, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    m = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    g = np.clip(g, -0.5, 0.5)
    norms = np.column_stack([
        np.linalg.norm(g, axis=1), np.linalg.norm(s, axis=1), np.linalg.norm(m, axis=1)
    ]).astype(np.float32)

    residual_specs = [
        ("fdir_gyro_rate_residual", .5),
        ("fdir_gyro_vector_rate_residual", .5),
        ("fdir_gyro_cross_sensor", 180.),
        ("fdir_sun_innovation", 180.),
        ("fdir_mag_innovation", 180.),
        ("fdir_geometry_error", 180.),
        ("fdir_sun_track_error", 180.),
        ("fdir_mag_track_error", 180.),
    ]
    residual_values = []
    residual_valid = []
    for key, lim in residual_specs:
        x, v = _sanitize_residual(result[key], lim)
        residual_values.append(x)
        residual_valid.append(v)
    rv = np.column_stack(residual_values).astype(np.float32)
    rm = np.column_stack(residual_valid).astype(np.float32)

    base = np.column_stack([
        g, s, m,
        gyro_available, sun_available, mag_available,
        norms, rv, rm,
    ]).astype(np.float32)
    ctx = np.nan_to_num(np.asarray(context, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if ctx.shape != (n, 17):
        raise ValueError(f"Expected 17 orbital-context features, got {ctx.shape}")
    return np.column_stack([base, ctx]).astype(np.float32), FEATURE_NAMES_BASE + CONTEXT_NAMES


def _randomize_known_orbit(mission_cfg: dict, rng: np.random.Generator) -> dict:
    m = copy.deepcopy(mission_cfg or {})
    m["enabled"] = bool(m.get("enabled", True))
    m["runtime_knowledge_altitude_error_m"] = float(rng.uniform(-float(m.get("knowledge_altitude_uncertainty_m", 1000.0)), float(m.get("knowledge_altitude_uncertainty_m", 1000.0))))
    m["runtime_knowledge_inclination_error_deg"] = float(rng.uniform(-float(m.get("knowledge_inclination_uncertainty_deg", .5)), float(m.get("knowledge_inclination_uncertainty_deg", .5))))
    m["runtime_knowledge_raan_error_deg"] = float(rng.uniform(-float(m.get("knowledge_raan_uncertainty_deg", .25)), float(m.get("knowledge_raan_uncertainty_deg", .25))))
    m["runtime_knowledge_phase_error_deg"] = float(rng.uniform(-float(m.get("knowledge_phase_uncertainty_deg", 1.0)), float(m.get("knowledge_phase_uncertainty_deg", 1.0))))
    return m


def _simulate_sequence(case: dict, *, base_template: dict, global_cfg: dict, duration_s: float,
                       dt_s: float, seed: int, fdir_cfg: dict, mission_cfg: dict,
                       initial_error_max_deg: float, body_rate_max_rad_s: float,
                       orbit_dispersion: dict | None = None) -> dict:
    rng = np.random.default_rng(int(seed))
    base = copy.deepcopy(base_template)
    gcfg = copy.deepcopy(global_cfg)
    base["duration_s"] = float(duration_s)
    base["dt_s"] = float(dt_s)
    base["orbit_profile"] = "global"
    # Couple stochastic fault realization to the complete-run seed. Matched
    # severity cases reuse the same seed, while independent repeats get new
    # dropout/noise realizations.
    base["fault_seed"] = int(seed) + 7919

    # Small physical-orbit dispersion makes complete simulation runs independent.
    od = orbit_dispersion or {}
    gcfg["orbit"] = copy.deepcopy(global_cfg["orbit"])
    gcfg["orbit"]["altitude_m"] = float(gcfg["orbit"].get("altitude_m", 500000.0)) + float(rng.uniform(-float(od.get("altitude_half_range_m", 5000.0)), float(od.get("altitude_half_range_m", 5000.0))))
    gcfg["orbit"]["inclination_deg"] = float(gcfg["orbit"].get("inclination_deg", 51.6)) + float(rng.uniform(-float(od.get("inclination_half_range_deg", .10)), float(od.get("inclination_half_range_deg", .10))))
    gcfg["orbit"]["raan_deg"] = float(gcfg["orbit"].get("raan_deg", 0.0)) + float(rng.uniform(-float(od.get("raan_half_range_deg", .10)), float(od.get("raan_half_range_deg", .10))))
    gcfg["orbit"]["phase_deg"] = float(gcfg["orbit"].get("phase_deg", 0.0)) + float(rng.uniform(-float(od.get("phase_half_range_deg", 180.0)), float(od.get("phase_half_range_deg", 180.0))))

    t = np.arange(0.0, float(duration_s) + float(dt_s) / 2.0, float(dt_s))
    sun, sun_pos, states, eclipse_state, illumination, B_i, orbit_meta = _make_environment(t, base, gcfg)
    axis = str(base.get("pointing_axis", "x"))
    q0 = sun_pointing_initial_quaternion(axis, sun[0], float(rng.uniform(0.0, float(initial_error_max_deg))))
    omega0 = rng.uniform(-float(body_rate_max_rad_s), float(body_rate_max_rad_s), 3)
    sc = global_cfg["sensors"]
    n = len(t)
    noise = {
        "gyro": rng.normal(0.0, float(sc["gyro_noise_std_rad_s"]), (n, 3)),
        "sun": rng.normal(0.0, float(sc["vector_noise_std"]), (n, 3)),
        "mag": rng.normal(0.0, float(sc["vector_noise_std"]), (n, 3)),
    }
    result = _run_branch(
        "ml_fdir_dataset", base, gcfg, t, q0, omega0, sun, B_i, states,
        eclipse_state, noise, predictor=None, fault_case=case, fdir_only=False,
        fdir_mode="classical_v31", fdir_config=fdir_cfg,
    )
    mission_run = _randomize_known_orbit(mission_cfg, rng)
    context, _, orbit_knowledge = _runtime_mission_context(
        t, base, gcfg, {"mission_orbit": mission_run}, sun, sun_pos, states, B_i, illumination
    )
    features, feature_names = _build_features(result, context)
    sun_visible = np.asarray(eclipse_state) != 2
    labels = np.column_stack([
        np.asarray(result["gyro_faulted"], bool),
        np.asarray(result["sensor_faulted"], bool) & sun_visible,
        np.asarray(result["mag_faulted"], bool),
    ]).astype(np.int64)
    return {
        "time": t,
        "features": features,
        "feature_names": feature_names,
        "labels": labels,
        "armed": np.asarray(result["fdir_armed"], bool),
        "sun_expected_unavailable": np.asarray(result["fdir_sun_expected_unavailable"], bool),
        "classical_detect": np.column_stack([
            result["fdir_detected_gyro"], result["fdir_detected_sun"], result["fdir_detected_mag"]
        ]).astype(bool),
        "hard_unavailable": np.column_stack([
            result["fdir_hard_gyro"], result["fdir_hard_sun"], result["fdir_hard_mag"]
        ]).astype(bool),
        "pointing_error_deg": np.asarray(result["true_point"], float),
        "attitude_error_deg": np.asarray(result["att_err"], float),
        "illumination": np.asarray(illumination, float),
        "eclipse_state": np.asarray(eclipse_state, int),
        "case": copy.deepcopy(case),
        "orbit": orbit_meta,
        "orbit_knowledge": orbit_knowledge,
    }


def _stratified_run_split(run_records: list[dict], seed: int,
                          validation_fraction: float = 0.20,
                          test_fraction: float = 0.20) -> dict[str, list[int]]:
    rng = np.random.default_rng(int(seed))
    by_family: dict[str, list[int]] = {}
    for r in run_records:
        by_family.setdefault(str(r["family"]), []).append(int(r["run_id"]))
    split = {"train": [], "validation": [], "test": []}
    vf = float(np.clip(validation_fraction, 0.0, .45))
    tf = float(np.clip(test_fraction, 0.0, .45))
    for fam, ids0 in sorted(by_family.items()):
        ids = np.asarray(ids0, int)
        rng.shuffle(ids)
        n = len(ids)
        if n >= 5:
            nv = max(1, int(round(vf * n)))
            nt = max(1, int(round(tf * n)))
            while n - nv - nt < max(2, int(round(.5 * n))):
                if nv >= nt and nv > 1:
                    nv -= 1
                elif nt > 1:
                    nt -= 1
                else:
                    break
            split["validation"].extend(ids[:nv].tolist())
            split["test"].extend(ids[nv:nv + nt].tolist())
            split["train"].extend(ids[nv + nt:].tolist())
        elif n == 4:
            split["validation"].append(int(ids[0])); split["test"].append(int(ids[1])); split["train"].extend(ids[2:].tolist())
        elif n == 3:
            split["validation"].append(int(ids[0])); split["test"].append(int(ids[1])); split["train"].append(int(ids[2]))
        elif n == 2:
            split["test"].append(int(ids[0])); split["train"].append(int(ids[1]))
        else:
            split["train"].extend(ids.tolist())
    return split


def _windows_for_runs(sequences: dict[int, dict], run_ids: list[int], window: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for rid in run_ids:
        seq = sequences[int(rid)]
        f = np.asarray(seq["features"], np.float32)
        y = np.asarray(seq["labels"], np.int64)
        armed = np.asarray(seq["armed"], bool)
        w = int(window)
        for end in range(w - 1, len(f), max(1, int(stride))):
            start = end - w + 1
            if not np.all(armed[start:end + 1]):
                continue
            xs.append(f[start:end + 1])
            ys.append(y[end])
    if not xs:
        raise RuntimeError("No ML-FDIR windows generated for requested run split")
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)


def _threshold_records_for_bundle(bundle, sequences: dict[int, dict], run_ids: list[int], batch_size: int) -> list[dict]:
    """Predict complete runs once for runtime-faithful threshold calibration/scoring."""
    records = []
    for rid in run_ids:
        seq = sequences[int(rid)]
        p = bundle.predict_sequence(seq["features"], batch_size=int(batch_size))
        armed = np.asarray(seq["armed"], bool)
        expected_sun = np.asarray(seq["sun_expected_unavailable"], bool)
        eval_mask = np.column_stack([armed, armed & ~expected_sun, armed]).astype(bool)
        classical = np.asarray(seq["classical_detect"], bool).copy()
        classical[:, 1] &= ~expected_sun
        classical &= eval_mask
        records.append({
            "run_id": int(rid),
            "probabilities": p,
            "truth": np.asarray(seq["labels"], bool),
            "eval_mask": eval_mask,
            "classical": classical,
        })
    return records



def _select_validation_heterogeneous_hybrid(calibration_df: pd.DataFrame,
                                            min_f1_gain_pct: float = 0.25) -> pd.DataFrame:
    """Select one validation-approved ML architecture per sensor.

    Selection uses ONLY hybrid threshold-calibration results from complete validation
    sequences. ``classical_only`` is an explicit candidate and is retained whenever
    no ML architecture improves validation F1 by at least ``min_f1_gain_pct``.
    This prevents test-set model selection and allows different sensors to use
    different lightweight architectures.
    """
    df = calibration_df.copy()
    df = df[df["mode"] == "hybrid_augmentation"].copy()
    rows = []
    for sensor in SENSORS:
        q = df[(df["sensor"] == sensor) & (df["feasible"].astype(bool))].copy()
        if q.empty:
            rows.append({
                "sensor": sensor, "selected_architecture": "classical_only",
                "selected_threshold": 1.01, "validation_precision_pct": np.nan,
                "validation_recall_pct": np.nan, "validation_f1_pct": np.nan,
                "validation_far_pct": np.nan, "classical_baseline_f1_pct": np.nan,
                "classical_baseline_far_pct": np.nan, "validation_f1_gain_pct": 0.0,
                "validation_far_increment_pct": 0.0, "selection_reason": "no_feasible_ml_candidate",
            })
            continue

        base_f1 = float(q["classical_baseline_f1_pct"].dropna().iloc[0])
        base_far = float(q["classical_baseline_far_pct"].dropna().iloc[0])
        # Prefer validation F1, then recall, precision, lower FAR and finally fewer
        # parameters indirectly via deterministic architecture name ordering.
        q = q.sort_values(
            ["f1_pct", "recall_pct", "precision_pct", "false_alarm_rate_pct", "architecture"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        best = q.iloc[0]
        gain = float(best["f1_pct"] - base_f1)
        far_inc = float(best["false_alarm_rate_pct"] - base_far)
        selected = str(best["architecture"]) if gain >= float(min_f1_gain_pct) else "classical_only"
        rows.append({
            "sensor": sensor,
            "selected_architecture": selected,
            "selected_threshold": float(best["threshold"]) if selected != "classical_only" else 1.01,
            "validation_precision_pct": float(best["precision_pct"]) if selected != "classical_only" else float(best["classical_baseline_precision_pct"]),
            "validation_recall_pct": float(best["recall_pct"]) if selected != "classical_only" else float(best["classical_baseline_recall_pct"]),
            "validation_f1_pct": float(best["f1_pct"]) if selected != "classical_only" else base_f1,
            "validation_far_pct": float(best["false_alarm_rate_pct"]) if selected != "classical_only" else base_far,
            "classical_baseline_f1_pct": base_f1,
            "classical_baseline_far_pct": base_far,
            "validation_f1_gain_pct": gain if selected != "classical_only" else 0.0,
            "validation_far_increment_pct": far_inc if selected != "classical_only" else 0.0,
            "selection_reason": "validation_f1_gain" if selected != "classical_only" else "classical_retained_no_minimum_gain",
        })
    return pd.DataFrame(rows)


def _score_detector(time: np.ndarray, truth: np.ndarray, detected: np.ndarray,
                    eval_mask: np.ndarray) -> dict:
    t = np.asarray(time, float)
    y = np.asarray(truth, bool) & np.asarray(eval_mask, bool)
    d = np.asarray(detected, bool) & np.asarray(eval_mask, bool)
    m = np.asarray(eval_mask, bool)
    yp = y[m].astype(int)
    dp = d[m].astype(int)
    if len(yp) == 0:
        return {k: np.nan for k in ("precision_pct", "recall_pct", "f1_pct", "false_alarm_rate_pct", "episode_detection_probability_pct", "mean_detection_delay_s")}
    if np.any(yp):
        pr, rc, f1, _ = precision_recall_fscore_support(yp, dp, average="binary", zero_division=0)
        pr_pct = float(100.0 * pr); rc_pct = float(100.0 * rc); f1_pct = float(100.0 * f1)
    else:
        # No positive ground-truth samples: detection precision/recall/F1 are not
        # defined for this sensor/case. False-alarm rate is the relevant metric.
        pr_pct = np.nan; rc_pct = np.nan; f1_pct = np.nan
    neg = (~y) & m
    far = float(100.0 * np.mean(d[neg])) if np.any(neg) else np.nan

    # Contiguous truth episodes. Detection must occur before that episode ends.
    active = y & m
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    found = 0
    delays = []
    for st, en in zip(starts, ends):
        hit = np.flatnonzero(d[st:en + 1])
        if len(hit):
            found += 1
            delays.append(float(t[st + int(hit[0])] - t[st]))
    ep = float(100.0 * found / len(starts)) if len(starts) else np.nan
    return {
        "precision_pct": pr_pct,
        "recall_pct": rc_pct,
        "f1_pct": f1_pct,
        "false_alarm_rate_pct": far,
        "episode_detection_probability_pct": ep,
        "mean_detection_delay_s": float(np.mean(delays)) if delays else np.nan,
    }


def _plot_probability_timeline(out_path: Path, seq: dict, probabilities: dict[str, np.ndarray],
                               smooth: dict[str, np.ndarray], global_cfg: dict, title: str):
    formats = tuple(global_cfg["visualization"]["formats"])
    dpi = int(global_cfg["visualization"]["dpi"])
    t = seq["time"]
    truth = seq["labels"]
    classical = seq["classical_detect"]
    colors = {"mlp": PALETTE["blue_main"], "cnn": PALETTE["teal"], "gru": PALETTE["violet"]}
    fig, axes = create_subplots(3, 1, figsize=(11.0, 8.5), sharex=True)
    for j, sensor in enumerate(SENSORS):
        ax = axes[j]
        ax.fill_between(t, 0, truth[:, j], step="post", alpha=.10, color=PALETTE["red_strong"], label="Injected fault")
        ax.step(t, classical[:, j].astype(float), where="post", color=PALETTE["neutral_dark"], lw=1.0, label="Classical v3.1")
        for arch in ("mlp", "cnn", "gru"):
            if arch in smooth:
                ax.plot(t, smooth[arch][:, j], lw=1.0, color=colors[arch], label=f"{arch.upper()} probability")
        ax.set_ylim(-.05, 1.05)
        ax.set_ylabel(sensor)
        if j == 0:
            ax.set_title(title)
            ax.legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time [s]")
    finalize_figure(fig, out_path, formats=formats, dpi=dpi)


def run(cfg, global_cfg, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out / "models"; models_dir.mkdir(exist_ok=True)
    tables = out / "tables"; tables.mkdir(exist_ok=True)
    plots = out / "plots"; plots.mkdir(exist_ok=True)

    base = copy.deepcopy(global_cfg["scenarios"]["reaction_wheel_sun_pointing_comparison"])
    if "orbit_profile" in cfg:
        base["orbit_profile"] = cfg["orbit_profile"]
    if "magnetic_field_model" in cfg:
        base.setdefault("magnetic_field", {})["model"] = str(cfg["magnetic_field_model"])

    fdir_cfg = copy.deepcopy(global_cfg["scenarios"]["reaction_wheel_fair_ml_campaign"].get("fdir", {}))
    fdir_cfg.update(copy.deepcopy(cfg.get("classical_fdir", {})))
    fdir_cfg["mode"] = "classical_v31"

    model_cfg = cfg.get("models", {})
    training_cfg = model_cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    mission_cfg = cfg.get("mission_orbit", global_cfg["scenarios"]["reaction_wheel_fair_ml_campaign"].get("models", {}).get("dataset", {}).get("mission_orbit", {}))
    families = [str(x).lower() for x in dataset_cfg.get("families", [
        "healthy", "gyro_bias", "gyro_drift", "gyro_noise", "mag_misalignment",
        "mag_bias", "sun_noise", "hard_loss", "dropout", "combined_soft",
    ])]
    runs_per_family = int(dataset_cfg.get("runs_per_family", 4))
    duration_train = float(dataset_cfg.get("duration_s", 400.0))
    dt = float(cfg.get("dt_s", .1))
    fault_start_range = [float(x) for x in dataset_cfg.get("fault_start_range_s", [80.0, 130.0])]
    fault_duration_range = [float(x) for x in dataset_cfg.get("fault_duration_range_s", [180.0, 260.0])]
    initial_error_max = float(dataset_cfg.get("initial_attitude_error_max_deg", 25.0))
    body_rate_max = float(dataset_cfg.get("body_rate_max_rad_s", .005))
    orbit_dispersion = copy.deepcopy(dataset_cfg.get("orbit_dispersion", {}))
    master_seed = int(dataset_cfg.get("seed", 12001))
    rng = np.random.default_rng(master_seed)

    # Training/validation/test dataset. Split is by COMPLETE simulation run and
    # stratified by fault family; no overlapping window can cross run boundaries.
    sequences: dict[int, dict] = {}
    run_records = []
    run_id = 0
    for fam in families:
        for rep in range(runs_per_family):
            start = float(rng.uniform(fault_start_range[0], fault_start_range[1]))
            dur = float(rng.uniform(fault_duration_range[0], fault_duration_range[1]))
            end = min(duration_train - 5.0, start + dur)
            case = _sample_training_case(fam, rng, start, end, variant_index=rep)
            case["name"] = f"train_{fam}_{rep:02d}"
            seq_seed = int(rng.integers(1, 2**31 - 1))
            seq = _simulate_sequence(
                case, base_template=base, global_cfg=global_cfg, duration_s=duration_train,
                dt_s=dt, seed=seq_seed, fdir_cfg=fdir_cfg, mission_cfg=mission_cfg,
                initial_error_max_deg=initial_error_max, body_rate_max_rad_s=body_rate_max,
                orbit_dispersion=orbit_dispersion,
            )
            sequences[run_id] = seq
            run_records.append({
                "run_id": run_id, "family": fam, "case_name": case["name"], "seed": seq_seed,
                "fault_start_s": start, "fault_end_s": end,
                "orbit": seq["orbit"], "orbit_knowledge": seq["orbit_knowledge"],
                "case": case,
            })
            run_id += 1

    split = _stratified_run_split(
        run_records, int(dataset_cfg.get("split_seed", 12018)),
        validation_fraction=float(dataset_cfg.get("validation_fraction", .20)),
        test_fraction=float(dataset_cfg.get("test_fraction", .20)),
    )
    window = int(model_cfg.get("window", 50))
    stride = int(dataset_cfg.get("window_stride_samples", 2))
    Xtr, ytr = _windows_for_runs(sequences, split["train"], window, stride)
    Xva, yva = _windows_for_runs(sequences, split["validation"], window, stride)
    Xte, yte = _windows_for_runs(sequences, split["test"], window, stride)
    feature_names = sequences[0]["feature_names"]
    run_family_lookup = {int(r["run_id"]): str(r["family"]) for r in run_records}
    threshold_calibration_run_ids = [
        int(rid) for rid in split["validation"]
        if run_family_lookup[int(rid)] == "healthy" or run_family_lookup[int(rid)] in SOFT_FAMILIES
    ]
    if not threshold_calibration_run_ids:
        threshold_calibration_run_ids = list(split["validation"])

    manifest = {
        "scenario": "reaction_wheel_ml_fdir_campaign",
        "purpose": "Experiment 2: lightweight ML for fault detection/isolation rather than virtual-sensor reconstruction",
        "feature_count": len(feature_names), "feature_names": feature_names,
        "window": window, "dt_s": dt, "window_duration_s": window * dt,
        "window_stride_samples": stride,
        "families": families, "runs_per_family": runs_per_family,
        "validation_fraction": float(dataset_cfg.get("validation_fraction", .20)),
        "test_fraction": float(dataset_cfg.get("test_fraction", .20)),
        "train_run_ids": split["train"], "validation_run_ids": split["validation"], "test_run_ids": split["test"],
        "threshold_calibration_run_ids": threshold_calibration_run_ids,
        "threshold_calibration_scope": "healthy + soft/gradual validation runs; hard-loss/dropout validation runs are excluded because deterministic classical FDIR owns hard availability",
        "train_windows": int(len(Xtr)), "validation_windows": int(len(Xva)), "test_windows": int(len(Xte)),
        "run_metadata": run_records,
        "fairness": "MLP/CNN/GRU use the same complete-run split, windows, labels, feature normalization protocol and fault realizations.",
        "truth_usage": "Injected fault labels are used only as supervised targets and for scoring; they are not input features.",
        "classical_role": "FDIR v3.1 is the deterministic baseline and supplies physical residual calculations; its confidence/state outputs are not ML input features.",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundles = {}
    model_metric_frames = []
    resource_rows = []
    threshold_calibration_frames = []
    sequence_test_frames = []
    pr_alpha = float(model_cfg.get("probability_ewma_alpha", .30))
    persist = int(model_cfg.get("persistence_samples", 3))
    reset_samples = int(model_cfg.get("reset_samples", 2))
    max_ml_far = float(model_cfg.get("threshold_max_false_alarm_rate_pct", 1.0))
    max_hybrid_incremental_far = float(model_cfg.get("hybrid_max_incremental_far_pct", 1.0))
    heterogeneous_enabled = bool(model_cfg.get("heterogeneous_hybrid_enabled", True))
    heterogeneous_min_gain = float(model_cfg.get("heterogeneous_min_validation_f1_gain_pct", 0.25))
    inference_batch = int(model_cfg.get("inference_batch_size", 1024))

    for arch in [str(x).lower() for x in model_cfg.get("architectures", ["mlp", "cnn", "gru"])]:
        bundle, sensor_metrics = train_multilabel_fdir_model(
            arch, Xtr, ytr, Xva, yva, Xte, yte,
            feature_names=feature_names, window=window, training_cfg=training_cfg,
            out_dir=models_dir / arch,
        )

        # IMPORTANT: operational thresholds are calibrated on COMPLETE validation
        # sequences AFTER the exact runtime EWMA + persistence + arming/eclipse
        # reset logic. This removes the v1 raw-window/runtime mismatch.
        val_records = _threshold_records_for_bundle(bundle, sequences, threshold_calibration_run_ids, inference_batch)
        ml_th, ml_cal = choose_runtime_multilabel_thresholds(
            val_records, alpha=pr_alpha, persistence_samples=persist,
            reset_samples=reset_samples, max_false_alarm_rate_pct=max_ml_far,
            hybrid=False,
        )
        hy_th, hy_cal = choose_runtime_multilabel_thresholds(
            val_records, alpha=pr_alpha, persistence_samples=persist,
            reset_samples=reset_samples, hybrid=True,
            max_incremental_far_pct=max_hybrid_incremental_far,
        )
        ml_cal.insert(0, "architecture", arch)
        hy_cal.insert(0, "architecture", arch)
        threshold_calibration_frames.extend([ml_cal, hy_cal])
        bundle.thresholds = np.asarray(ml_th, float)
        bundle.hybrid_thresholds = np.asarray(hy_th, float)
        bundle.metrics["runtime_thresholds"] = {s: float(ml_th[i]) for i, s in enumerate(SENSORS)}
        bundle.metrics["hybrid_augmentation_thresholds"] = {s: float(hy_th[i]) for i, s in enumerate(SENSORS)}
        bundle.metrics["threshold_calibration"] = {
            "applied_after": "causal EWMA + persistence + FDIR arming/eclipse reset",
            "max_pure_ml_false_alarm_rate_pct": max_ml_far,
            "max_hybrid_incremental_false_alarm_rate_pct": max_hybrid_incremental_far,
        }
        save_ml_fdir_bundle(bundle, models_dir / arch)

        # Final model test metrics use held-out COMPLETE runs and the same runtime
        # decision path; the old window-level table remains only a diagnostic.
        test_records = _threshold_records_for_bundle(bundle, sequences, split["test"], inference_batch)
        ml_test = evaluate_runtime_multilabel_thresholds(
            test_records, bundle.thresholds, alpha=pr_alpha,
            persistence_samples=persist, reset_samples=reset_samples, hybrid=False,
        )
        ml_test.insert(0, "architecture", arch)
        hy_test = evaluate_runtime_multilabel_thresholds(
            test_records, bundle.hybrid_thresholds, alpha=pr_alpha,
            persistence_samples=persist, reset_samples=reset_samples, hybrid=True,
        )
        hy_test.insert(0, "architecture", arch)
        sequence_test_frames.extend([ml_test, hy_test])

        bundles[arch] = bundle
        model_metric_frames.append(sensor_metrics)
        resource_rows.append(bundle.metrics)

    model_sensor_metrics = pd.concat(model_metric_frames, ignore_index=True)
    _save_table(model_sensor_metrics, tables / "model_window_level_test_metrics_preliminary")
    threshold_calibration_df = pd.concat(threshold_calibration_frames, ignore_index=True)
    _save_table(threshold_calibration_df, tables / "runtime_threshold_calibration")

    heterogeneous_selection = _select_validation_heterogeneous_hybrid(
        threshold_calibration_df, min_f1_gain_pct=heterogeneous_min_gain,
    )
    if not heterogeneous_enabled:
        heterogeneous_selection["selected_architecture"] = "classical_only"
        heterogeneous_selection["selected_threshold"] = 1.01
        heterogeneous_selection["selection_reason"] = "disabled_in_config"
        heterogeneous_selection["validation_f1_gain_pct"] = 0.0
        heterogeneous_selection["validation_far_increment_pct"] = 0.0
    _save_table(heterogeneous_selection, tables / "heterogeneous_hybrid_sensor_selection")
    heterogeneous_arch_by_sensor = dict(zip(
        heterogeneous_selection["sensor"], heterogeneous_selection["selected_architecture"]
    ))

    sequence_test_df = pd.concat(sequence_test_frames, ignore_index=True)
    _save_table(sequence_test_df, tables / "model_complete_run_test_metrics")
    resource_df = pd.DataFrame(resource_rows)
    _save_table(resource_df[[
        "architecture", "parameters", "model_size_mb", "inference_ms_cpu_batch1",
        "test_macro_f1", "n_features", "window", "train_samples", "validation_samples", "test_samples"
    ]], tables / "model_resource_metrics")

    # Held-out mission-like evaluation sweep.
    eval_cfg = cfg.get("evaluation", {})
    duration_eval = float(eval_cfg.get("duration_s", 1600.0))
    eval_seed = int(eval_cfg.get("seed", 13001))
    eval_repeats = max(1, int(eval_cfg.get("repeats", 4)))
    eval_cases = _evaluation_cases(cfg)
    match_groups = sorted({str(c.get("match_group", c["name"])) for c in eval_cases})
    match_group_index = {g: i for i, g in enumerate(match_groups)}
    metric_rows = []
    timeseries_rows = []
    representative = {str(x) for x in eval_cfg.get("probability_plot_cases", [
        "E_gyro_drift_3e-06", "E_mag_misalignment_20deg", "E_D9_all_three_degraded"
    ])}

    for repeat_id in range(eval_repeats):
        for ci, case in enumerate(eval_cases):
            match_group = str(case.get("match_group", case["name"]))
            # Same seed for all severity levels in one matched sweep/repeat.
            matched_seed = int(eval_seed + repeat_id * 1000003 + match_group_index[match_group] * 1009)
            seq = _simulate_sequence(
                case, base_template=base, global_cfg=global_cfg, duration_s=duration_eval,
                dt_s=dt, seed=matched_seed, fdir_cfg=fdir_cfg, mission_cfg=mission_cfg,
                initial_error_max_deg=float(eval_cfg.get("initial_attitude_error_max_deg", 20.0)),
                body_rate_max_rad_s=float(eval_cfg.get("body_rate_max_rad_s", .003)),
                orbit_dispersion=orbit_dispersion,
            )
            t = seq["time"]
            truth = seq["labels"].astype(bool)
            armed = seq["armed"].astype(bool)
            expected_sun = seq["sun_expected_unavailable"].astype(bool)
            active_mask = np.column_stack([armed, armed & ~expected_sun, armed]).astype(bool)
            classical = seq["classical_detect"].astype(bool)
            classical[:, 1] &= ~expected_sun
            classical &= active_mask
            hard = seq["hard_unavailable"].astype(bool)
            prob_by_arch = {}
            smooth_by_arch = {}
            ml_det_by_arch = {}
            hybrid_by_arch = {}
            hybrid_aug_by_arch = {}
            for arch, bundle in bundles.items():
                p = bundle.predict_sequence(seq["features"], batch_size=inference_batch)
                smooth, det = apply_probability_filter(
                    p, bundle.thresholds, alpha=pr_alpha,
                    persistence_samples=persist, reset_samples=reset_samples,
                    active_mask=active_mask,
                )
                _, ml_aug = apply_probability_filter(
                    p, bundle.hybrid_thresholds, alpha=pr_alpha,
                    persistence_samples=persist, reset_samples=reset_samples,
                    active_mask=active_mask,
                )
                # v2 hybrid preserves ALL classical v3.1 decisions (hard + soft)
                # and only adds ML detections accepted by a stricter validation-
                # calibrated threshold with <= configured incremental FAR.
                hybrid = classical | ml_aug
                hybrid &= active_mask
                prob_by_arch[arch] = p
                smooth_by_arch[arch] = smooth
                ml_det_by_arch[arch] = det
                hybrid_aug_by_arch[arch] = ml_aug & ~classical
                hybrid_by_arch[arch] = hybrid

            # Validation-selected heterogeneous hybrid: each sensor may use a
            # different architecture, or classical-only. The mapping is frozen
            # before these held-out mission-like evaluation cases are generated.
            heterogeneous_hybrid = classical.copy()
            heterogeneous_aug = np.zeros_like(classical, dtype=bool)
            for j, sensor in enumerate(SENSORS):
                selected_arch = heterogeneous_arch_by_sensor.get(sensor, "classical_only")
                if selected_arch != "classical_only":
                    heterogeneous_hybrid[:, j] = hybrid_by_arch[selected_arch][:, j]
                    heterogeneous_aug[:, j] = hybrid_aug_by_arch[selected_arch][:, j]

            detectors = {"classical_v31": classical}
            for arch in bundles:
                detectors[f"ml_{arch}"] = ml_det_by_arch[arch]
                detectors[f"hybrid_{arch}"] = hybrid_by_arch[arch]
            detectors["hybrid_validation_selected"] = heterogeneous_hybrid

            for det_name, det in detectors.items():
                for j, sensor in enumerate(SENSORS):
                    eval_mask = active_mask[:, j]
                    metrics = _score_detector(t, truth[:, j], det[:, j], eval_mask)
                    metric_rows.append({
                        "case_id": ci, "repeat_id": repeat_id, "matched_seed": matched_seed,
                        "match_group": match_group,
                        "case_name": case["name"], "fault_family": case["family"],
                        "fault_category": _fault_category(case["family"]),
                        "detector": det_name, "sensor": sensor,
                        **metrics,
                    })

            frame = pd.DataFrame({
                "case_id": ci, "repeat_id": repeat_id, "matched_seed": matched_seed,
                "match_group": match_group,
                "case_name": case["name"], "fault_family": case["family"],
                "time_s": t, "fdir_armed": armed.astype(int),
                "sun_expected_unavailable": expected_sun.astype(int),
                "illumination_fraction": seq["illumination"], "eclipse_state": seq["eclipse_state"],
                "truth_gyro_fault": truth[:, 0].astype(int), "truth_sun_fault": truth[:, 1].astype(int), "truth_mag_fault": truth[:, 2].astype(int),
                "classical_gyro_detected": classical[:, 0].astype(int), "classical_sun_detected": classical[:, 1].astype(int), "classical_mag_detected": classical[:, 2].astype(int),
                "hard_gyro_unavailable": hard[:, 0].astype(int), "hard_sun_unavailable": hard[:, 1].astype(int), "hard_mag_unavailable": hard[:, 2].astype(int),
                "sun_pointing_error_deg": seq["pointing_error_deg"], "attitude_error_deg": seq["attitude_error_deg"],
            })
            for arch in bundles:
                p = prob_by_arch[arch]; sm = smooth_by_arch[arch]; md = ml_det_by_arch[arch]
                hy = hybrid_by_arch[arch]; aug = hybrid_aug_by_arch[arch]
                for j, sensor in enumerate(SENSORS):
                    frame[f"{arch}_{sensor}_probability"] = p[:, j]
                    frame[f"{arch}_{sensor}_probability_ewma"] = sm[:, j]
                    frame[f"ml_{arch}_{sensor}_detected"] = md[:, j].astype(int)
                    frame[f"hybrid_{arch}_{sensor}_ml_augmentation"] = aug[:, j].astype(int)
                    frame[f"hybrid_{arch}_{sensor}_detected"] = hy[:, j].astype(int)
            for j, sensor in enumerate(SENSORS):
                frame[f"hybrid_validation_selected_{sensor}_ml_augmentation"] = heterogeneous_aug[:, j].astype(int)
                frame[f"hybrid_validation_selected_{sensor}_detected"] = heterogeneous_hybrid[:, j].astype(int)
            timeseries_rows.append(frame)

            if repeat_id == 0 and bool(cfg.get("plots_enabled", True)) and case["name"] in representative:
                _plot_probability_timeline(
                    plots / f"{ci:02d}_{_slug(case['name'])}_fault_probabilities",
                    seq, prob_by_arch, smooth_by_arch, global_cfg,
                    title=f"{case['name']}: ML-assisted FDIR probabilities",
                )

    metrics_df = pd.DataFrame(metric_rows)
    _save_table(metrics_df, tables / "detector_metrics_by_case_sensor")
    matched_case_sensor = metrics_df.groupby(
        ["case_name", "fault_family", "fault_category", "match_group", "detector", "sensor"], as_index=False
    ).agg(
        precision_mean_pct=("precision_pct", "mean"), precision_std_pct=("precision_pct", "std"),
        recall_mean_pct=("recall_pct", "mean"), recall_std_pct=("recall_pct", "std"),
        f1_mean_pct=("f1_pct", "mean"), f1_std_pct=("f1_pct", "std"),
        far_mean_pct=("false_alarm_rate_pct", "mean"), far_std_pct=("false_alarm_rate_pct", "std"),
        episode_detection_mean_pct=("episode_detection_probability_pct", "mean"),
        detection_delay_mean_s=("mean_detection_delay_s", "mean"), detection_delay_std_s=("mean_detection_delay_s", "std"),
        repeats=("repeat_id", "nunique"),
    )
    _save_table(matched_case_sensor, tables / "matched_severity_case_sensor_aggregate")
    case_agg = metrics_df.groupby(
        ["case_name", "fault_family", "fault_category", "detector"], as_index=False
    ).agg(
        macro_precision_pct=("precision_pct", "mean"),
        macro_recall_pct=("recall_pct", "mean"),
        macro_f1_pct=("f1_pct", "mean"),
        mean_false_alarm_rate_pct=("false_alarm_rate_pct", "mean"),
        mean_episode_detection_probability_pct=("episode_detection_probability_pct", "mean"),
        mean_detection_delay_s=("mean_detection_delay_s", "mean"),
    )
    _save_table(case_agg, tables / "detector_metrics_by_case")
    family_agg = metrics_df.groupby(
        ["fault_category", "fault_family", "detector"], as_index=False
    ).agg(
        macro_precision_pct=("precision_pct", "mean"),
        macro_recall_pct=("recall_pct", "mean"),
        macro_f1_pct=("f1_pct", "mean"),
        mean_false_alarm_rate_pct=("false_alarm_rate_pct", "mean"),
        mean_episode_detection_probability_pct=("episode_detection_probability_pct", "mean"),
        mean_detection_delay_s=("mean_detection_delay_s", "mean"),
    )
    _save_table(family_agg, tables / "detector_metrics_by_fault_family")
    overall = metrics_df.groupby("detector", as_index=False).agg(
        macro_precision_pct=("precision_pct", "mean"),
        macro_recall_pct=("recall_pct", "mean"),
        macro_f1_pct=("f1_pct", "mean"),
        mean_false_alarm_rate_pct=("false_alarm_rate_pct", "mean"),
        mean_episode_detection_probability_pct=("episode_detection_probability_pct", "mean"),
        mean_detection_delay_s=("mean_detection_delay_s", "mean"),
    )
    _save_table(overall, tables / "detector_overall_aggregate")

    # A focused thesis table for the exact soft faults motivating Experiment 2.
    focus = metrics_df[metrics_df["fault_family"].isin([
        "gyro_bias", "gyro_drift", "mag_misalignment", "combined_soft"
    ])].copy()
    _save_table(focus, tables / "soft_fault_focus_metrics")

    # Compact plots across detector families.
    if bool(cfg.get("plots_enabled", True)):
        plot_df = family_agg.groupby(["fault_category", "detector"], as_index=False)["macro_f1_pct"].mean()
        cats = [x for x in ["soft_or_gradual", "hard_or_intermittent"] if x in set(plot_df.fault_category)]
        detectors_order = ["classical_v31", "ml_mlp", "ml_cnn", "ml_gru", "hybrid_mlp", "hybrid_cnn", "hybrid_gru", "hybrid_validation_selected"]
        fig, axes = create_subplots(figsize=(11.0, 5.5)); ax = axes[0]
        x = np.arange(len(cats)); width = .095
        colors = [PALETTE["neutral_dark"], PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"],
                  PALETTE["green_3"], PALETTE["red_strong"], PALETTE["blue_main"], PALETTE["neutral_dark"]]
        for j, det in enumerate(detectors_order):
            vals = []
            for cat in cats:
                q = plot_df[(plot_df.fault_category == cat) & (plot_df.detector == det)]
                vals.append(float(q.iloc[0].macro_f1_pct) if len(q) else np.nan)
            ax.bar(x + (j - (len(detectors_order)-1)/2) * width, vals, width=width, label=det, color=colors[j])
        ax.set_xticks(x, [c.replace("_", " ") for c in cats]); ax.set_ylabel("Macro F1 [%]")
        ax.set_title("Experiment 2: classical vs ML-assisted FDIR")
        ax.legend(fontsize=7, ncol=4)
        finalize_figure(fig, plots / "ml_fdir_macro_f1_by_fault_category", formats=tuple(global_cfg["visualization"]["formats"]), dpi=int(global_cfg["visualization"]["dpi"]))

    # Timeseries can be large: write it once, zip it, and optionally remove CSV.
    ts = pd.concat(timeseries_rows, ignore_index=True)
    ts_csv = out / "reaction_wheel_ml_fdir_campaign_timeseries.csv"
    ts.to_csv(ts_csv, index=False)
    ts_zip = out / "reaction_wheel_ml_fdir_campaign_timeseries.zip"
    with zipfile.ZipFile(ts_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(ts_csv, arcname=ts_csv.name)
    if not bool(cfg.get("keep_uncompressed_timeseries", False)):
        ts_csv.unlink(missing_ok=True)

    metrics_df.to_csv(out / "reaction_wheel_ml_fdir_campaign_metrics.csv", index=False)
    case_agg.to_csv(out / "reaction_wheel_ml_fdir_campaign_summary.csv", index=False)
    # Duplicate the small outputs at scenario parent for easy retrieval.
    case_agg.to_csv(out.parent / "reaction_wheel_ml_fdir_campaign_summary.csv", index=False)
    metrics_df.to_csv(out.parent / "reaction_wheel_ml_fdir_campaign_metrics.csv", index=False)

    metadata = {
        "scenario": "reaction_wheel_ml_fdir_campaign",
        "experiment": "Experiment 2 — ML-assisted FDIR",
        "hypothesis": "Lightweight temporal ML may add more value in gradual fault detection/isolation than in direct virtual-sensor replacement.",
        "detectors": ["classical_v31", "ml_mlp", "ml_cnn", "ml_gru", "hybrid_mlp", "hybrid_cnn", "hybrid_gru", "hybrid_validation_selected"],
        "hybrid_definition": "complete classical v3.1 decision (hard + soft) OR validation-gated ML augmentation; ML augmentation is calibrated to add at most the configured incremental FAR",
        "heterogeneous_hybrid": {
            "selection_domain": "complete validation sequences only",
            "minimum_validation_f1_gain_pct": heterogeneous_min_gain,
            "selected_architecture_by_sensor": heterogeneous_arch_by_sensor,
            "test_set_used_for_selection": False,
        },
        "primary_metrics": ["precision", "recall", "F1", "false alarm rate", "episode detection probability", "detection delay"],
        "secondary_context": "Pointing/attitude errors are saved in the evaluation timeseries but recovery performance remains Experiment 1; Experiment 2 isolates the detection question.",
        "dataset_manifest": manifest,
        "model_metrics": {arch: bundle.metrics for arch, bundle in bundles.items()},
        "probability_filter": {"ewma_alpha": pr_alpha, "persistence_samples": persist, "reset_samples": reset_samples, "reset_before_arming_and_expected_sun_eclipse": True},
        "threshold_calibration": {"pure_ml_max_far_pct": max_ml_far, "hybrid_max_incremental_far_pct": max_hybrid_incremental_far, "domain": "complete validation sequences after runtime EWMA+persistence"},
        "sensor_model_selection": heterogeneous_selection.to_dict(orient="records"),
        "evaluation_design": {"repeats": eval_repeats, "matched_severity_sweeps": True, "same_seed_within_match_group_per_repeat": True},
    }
    (out / "scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "scenario": "reaction_wheel_ml_fdir_campaign",
        "training_runs": len(run_records),
        "evaluation_cases": len(eval_cases),
        "evaluation_repeats": eval_repeats,
        "evaluation_runs": len(eval_cases) * eval_repeats,
        "feature_count": len(feature_names),
        "window": window,
        "detectors": 8,
    }
