from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.simulation.attitude import (
    propagate_attitude_controlled, quat_angle_error_deg, quat_to_dcm,
    quat_multiply, small_angle_quat, quat_normalize,
)
from src.simulation.control import sun_pointing_initial_quaternion, sun_pointing_torque, sun_pointing_error_deg
from src.simulation.orbit_model import propagate_circular_two_body
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.adcs_modes import (
    SunPointingModeManager, SUN_ACQUISITION, SUN_POINTING, ECLIPSE_DRIFT, SUN_REACQUISITION,
    MODE_CODE, control_enabled,
)
from src.estimation.mekf import MEKF
from src.ml.virtual_sensor import train_virtual_sun_model
from src.visualization.publication import PALETTE, create_subplots, finalize_figure, add_fault_span
from src.visualization.animation import create_orbit_attitude_animation

BRANCHES = ("healthy_mekf", "faulty_mekf", "ml_recovery")
LABELS = {
    "healthy_mekf": "Healthy MEKF",
    "faulty_mekf": "Faulty sensor + MEKF",
    "ml_recovery": "Faulty sensor + MEKF + ML recovery",
}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _sensor_measurement(q, omega, gyro_bias, gyro_noise, sun_noise, mag_noise, sun_i, mag_i, sun_visible=True):
    C = quat_to_dcm(q)
    g = np.asarray(omega) + np.asarray(gyro_bias) + np.asarray(gyro_noise)
    s = _unit(C.T @ sun_i + sun_noise) if sun_visible else np.full(3, np.nan)
    m = _unit(C.T @ mag_i + mag_noise)
    return g, s, m


def _apply_sun_fault(z, fault_type, severity, active, rng):
    if not active:
        return z.copy(), False
    lev = {'low': 1.0, 'medium': 2.0, 'high': 4.0}[severity]
    if fault_type == 'sun_loss':
        return np.full(3, np.nan), True
    if fault_type == 'sun_dropout':
        if rng.random() < min(.15 * lev, .75):
            return np.full(3, np.nan), True
        return z.copy(), False
    if fault_type == 'sun_noise':
        if not np.all(np.isfinite(z)):
            return z.copy(), True
        return _unit(z + rng.normal(0, 8e-3 * lev, 3)), True
    raise ValueError('sun_pointing_comparison supports sun_loss, sun_dropout, or sun_noise')


def _one_step_truth(dt, state, inertia, tau):
    cb = lambda tt, q, w: tau
    tr, _ = propagate_attitude_controlled(np.array([0.0, float(dt)]), state[:4], state[4:], inertia, cb)
    return tr[-1]


def _settling_time(t, err, threshold, hold_s, start_s=0.0):
    t = np.asarray(t); e = np.asarray(err)
    dt = float(np.median(np.diff(t))); n = max(1, int(np.ceil(hold_s / dt)))
    start = int(np.searchsorted(t, start_s))
    for i in range(start, len(t) - n + 1):
        if np.all(e[i:i+n] <= threshold):
            return float(t[i])
    return np.nan


def _make_environment(t, cfg, global_cfg):
    illum = cfg.get('illumination', {})
    model = str(illum.get('sun_model', 'analytic'))
    epoch = str(illum.get('epoch_utc', '2026-01-01T00:00:00+00:00'))
    constant = cfg.get('sun_vector_eci', [1.0, .2, .1])
    sun_dir, sun_pos = sun_direction_series(t, model, epoch=epoch, constant_vector=constant)

    oc = global_cfg['orbit']
    states = propagate_circular_two_body(
        t, float(oc['altitude_m']), float(oc['inclination_deg']),
        float(oc['raan_deg']), float(oc['phase_deg'])
    )
    eclipse_enabled = bool(illum.get('eclipse_enabled', True))
    if eclipse_enabled:
        eclipse_state, illumination = eclipse_geometry(states[:, :3], sun_pos)
    else:
        eclipse_state = np.zeros(len(t), dtype=int)
        illumination = np.ones(len(t), dtype=float)
    return sun_dir, sun_pos, states, eclipse_state, illumination


def _run_branch(branch, cfg, global_cfg, t, q0, omega0, sun_series, mag_i, eclipse_state, noise, predictor=None):
    sc = global_cfg['sensors']; dt = float(t[1] - t[0]); n = len(t)
    inertia = cfg.get('inertia_kg_m2', global_cfg['attitude']['inertia_kg_m2'])
    kp = float(cfg.get('kp', .002)); kd = float(cfg.get('kd', .01)); maxt = cfg.get('max_torque_nm', .002)
    axis = str(cfg.get('pointing_axis', 'x'))
    fs = float(cfg.get('fault_start_s', 200.)); fe = float(cfg.get('fault_end_s', 400.))
    ft = str(cfg.get('fault_type', 'sun_loss')); sev = str(cfg.get('fault_severity', 'medium'))
    eclipse_as_loss = bool(cfg.get('illumination', {}).get('eclipse_sun_sensor_unavailable', True))
    mode_cfg = cfg.get('modes', {})
    mode_threshold = float(mode_cfg.get('pointing_threshold_deg', cfg.get('settling_threshold_deg', 1.0)))
    mode_hold = float(mode_cfg.get('pointing_hold_s', cfg.get('settling_duration_s', 10.0)))
    mgr = SunPointingModeManager(threshold_deg=mode_threshold, hold_s=mode_hold, dt_s=dt)

    filt = MEKF(q0=q0, gyro_noise_std=float(sc['gyro_noise_std_rad_s']), vector_noise_std=float(sc['vector_noise_std']))
    truth = np.zeros((n, 7)); qhat = np.zeros((n, 4)); bhat = np.zeros((n, 3)); tau = np.zeros((n, 3))
    gyro = np.zeros((n, 3)); sun_raw = np.full((n, 3), np.nan); sun_used = np.full((n, 3), np.nan); mag = np.zeros((n, 3))
    ml_used = np.zeros(n, bool); ml_candidate = np.zeros(n, bool); ml_rejected = np.zeros(n, bool); sensor_faulted = np.zeros(n, bool)
    mode = np.empty(n, dtype=object); mode_code = np.zeros(n, dtype=int); control_on = np.zeros(n, dtype=bool)
    truth[0, :4] = q0; truth[0, 4:] = omega0; qhat[0] = filt.q; bhat[0] = filt.b
    visible0 = not (eclipse_as_loss and eclipse_state[0] == 2)
    gyro[0], sun_raw[0], mag[0] = _sensor_measurement(q0, omega0, sc['gyro_bias_rad_s'], noise['gyro'][0], noise['sun'][0], noise['mag'][0], sun_series[0], mag_i, visible0)
    sun_used[0] = sun_raw[0]
    # If the run starts in eclipse, initialize the virtual sensor from the estimator-predicted Sun direction.
    anchor = sun_raw[0].copy() if np.all(np.isfinite(sun_raw[0])) else quat_to_dcm(qhat[0]).T @ sun_series[0]
    initial_est_point = sun_pointing_error_deg(qhat[0], axis, sun_series[0])
    mode[0] = mgr.update(in_full_eclipse=bool(eclipse_state[0] == 2), real_sun_available=bool(np.all(np.isfinite(sun_raw[0]))), estimated_pointing_error_deg=initial_est_point)
    mode_code[0] = MODE_CODE[mode[0]]; control_on[0] = control_enabled(mode[0])
    frng = np.random.default_rng(int(cfg.get('fault_seed', 4501)))

    for k in range(1, n):
        sun_i = sun_series[k]
        omega_est = gyro[k-1] - bhat[k-1]
        # The previous sample's ADCS mode determines the control action over this step.
        if control_enabled(mode[k-1]):
            tau[k-1] = sun_pointing_torque(qhat[k-1], omega_est, axis, sun_i, kp, kd, maxt)
        else:
            # ECLIPSE_DRIFT: no Sun-pointing torque. The estimator continues running.
            tau[k-1] = np.zeros(3)
        truth[k] = _one_step_truth(dt, truth[k-1], inertia, tau[k-1])
        sun_visible = not (eclipse_as_loss and eclipse_state[k] == 2)
        gyro[k], sun_raw[k], mag[k] = _sensor_measurement(
            truth[k, :4], truth[k, 4:], sc['gyro_bias_rad_s'], noise['gyro'][k], noise['sun'][k], noise['mag'][k], sun_i, mag_i, sun_visible
        )
        fault_active = (t[k] >= fs and t[k] <= fe)
        measured = sun_raw[k].copy()
        if branch != 'healthy_mekf':
            measured, faulted = _apply_sun_fault(measured, ft, sev, fault_active, frng)
            sensor_faulted[k] = faulted or (fault_active and ft == 'sun_noise')

        # ML recovery is only for a sensor fault while the Sun is physically visible.
        # During eclipse there are no photons, so no virtual measurement is injected.
        physical_sun_visible = not (eclipse_as_loss and eclipse_state[k] == 2)
        needs_recovery = physical_sun_visible and (not np.all(np.isfinite(measured)) or (branch == 'ml_recovery' and fault_active and ft == 'sun_noise'))
        if branch == 'ml_recovery' and needs_recovery:
            omega_virtual = gyro[k-1] - bhat[k-1]
            candidate = predictor.predict(anchor, omega_virtual, mag[k], dt); ml_candidate[k] = True
            qpred = quat_normalize(quat_multiply(qhat[k-1], small_angle_quat(omega_virtual * dt)))
            predicted_sun = quat_to_dcm(qpred).T @ sun_i
            gate = float(cfg.get('ml_gate_deg', 2.0))
            disagreement = np.degrees(np.arccos(np.clip(np.dot(_unit(candidate), _unit(predicted_sun)), -1, 1)))
            if disagreement <= gate:
                blend = float(cfg.get('ml_blend', .25))
                measured = _unit((1.0 - blend) * predicted_sun + blend * candidate)
                ml_used[k] = True; anchor = measured.copy()
            else:
                measured = np.full(3, np.nan); ml_rejected[k] = True; anchor = _unit(candidate)
        elif np.all(np.isfinite(measured)):
            anchor = measured.copy()

        sun_used[k] = measured
        sun_noise_override = float(cfg.get('ml_virtual_sensor_noise_std', .01)) if ml_used[k] else None
        q, b, _ = filt.step(gyro[k-1], sun_used[k], mag[k], sun_i, mag_i, dt, sun_noise_std=sun_noise_override)
        qhat[k] = q; bhat[k] = b

        # Mode logic uses the MEKF estimate and a REAL Sun-sensor availability flag.
        # An ML virtual vector does not count as physical Sun detection for eclipse exit.
        est_point_now = sun_pointing_error_deg(qhat[k], axis, sun_i)
        real_sun_available = bool(np.all(np.isfinite(measured))) and not bool(ml_used[k])
        mode[k] = mgr.update(in_full_eclipse=bool(eclipse_state[k] == 2), real_sun_available=real_sun_available, estimated_pointing_error_deg=est_point_now)
        mode_code[k] = MODE_CODE[mode[k]]; control_on[k] = control_enabled(mode[k])

    tau[-1] = tau[-2]
    true_point = np.array([sun_pointing_error_deg(q, axis, s) for q, s in zip(truth[:, :4], sun_series)])
    est_point = np.array([sun_pointing_error_deg(q, axis, s) for q, s in zip(qhat, sun_series)])
    att_err = np.array([quat_angle_error_deg(qe, qt) for qe, qt in zip(qhat, truth[:, :4])])
    sun_rec_err = np.full(n, np.nan)
    # Compare virtual/recovered measurement with the true noise-free body Sun direction, including eclipse.
    for k in range(n):
        if np.all(np.isfinite(sun_used[k])):
            target = _unit(quat_to_dcm(truth[k, :4]).T @ sun_series[k])
            sun_rec_err[k] = np.degrees(np.arccos(np.clip(np.dot(_unit(sun_used[k]), target), -1, 1)))
    return dict(truth=truth, qhat=qhat, bhat=bhat, tau=tau, gyro=gyro, sun_raw=sun_raw, sun_used=sun_used, mag=mag,
                ml_used=ml_used, ml_candidate=ml_candidate, ml_rejected=ml_rejected, sensor_faulted=sensor_faulted,
                mode=mode, mode_code=mode_code, control_on=control_on,
                true_point=true_point, est_point=est_point, att_err=att_err, sun_rec_err=sun_rec_err)


def run(cfg, global_cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    plots = out / 'plots'; plots.mkdir(exist_ok=True); models = out / 'models'; models.mkdir(exist_ok=True)
    duration = float(cfg.get('duration_s', 3000.)); dt = float(cfg.get('dt_s', .1)); t = np.arange(0, duration + dt/2, dt)
    axis = str(cfg.get('pointing_axis', 'x')); mag_i = _unit(cfg.get('mag_vector_eci', [.25, -.10, .96]))
    sun_series, sun_pos, orbit_states, eclipse_state, illumination = _make_environment(t, cfg, global_cfg)
    q0 = sun_pointing_initial_quaternion(axis, sun_series[0], float(cfg.get('initial_error_deg', 30.)))
    omega0 = np.asarray(cfg.get('omega0_rad_s', [0, 0, 0]), float)

    sc = global_cfg['sensors']; seed = int(cfg.get('comparison_seed', 7301)); rng = np.random.default_rng(seed); n = len(t)
    noise = {
        'gyro': rng.normal(0, float(sc['gyro_noise_std_rad_s']), (n, 3)),
        'sun': rng.normal(0, float(sc['vector_noise_std']), (n, 3)),
        'mag': rng.normal(0, float(sc['vector_noise_std']), (n, 3)),
    }
    vc = cfg.get('virtual_sensor', {})
    predictor, vm = train_virtual_sun_model(
        models/'virtual_sun_sensor', runs=int(vc.get('train_runs', 24)), duration_s=float(vc.get('train_duration_s', 40.)), dt=dt,
        hidden=tuple(vc.get('hidden', [48, 24])), epochs=int(vc.get('epochs', 18)), batch_size=int(vc.get('batch_size', 256)),
        seed=int(vc.get('seed', 8128)), gyro_noise_std=float(sc['gyro_noise_std_rad_s']), vector_noise_std=float(sc['vector_noise_std'])
    )
    results = {b: _run_branch(b, cfg, global_cfg, t, q0, omega0, sun_series, mag_i, eclipse_state, noise, predictor if b == 'ml_recovery' else None) for b in BRANCHES}

    fs = float(cfg.get('fault_start_s', 200.)); fe = float(cfg.get('fault_end_s', 400.))
    active = (t >= fs) & (t <= fe); eclipse = eclipse_state == 2; sunlight = ~eclipse
    post = t > fe; pre = t < fs; th = float(cfg.get('settling_threshold_deg', 1.)); hold = float(cfg.get('settling_duration_s', 10.))
    rows = []; frames = []
    for b, r in results.items():
        def rmse(mask, x): return float(np.sqrt(np.mean(np.asarray(x)[mask]**2))) if np.any(mask) else np.nan
        rows.append({
            'branch': b, 'label': LABELS[b], 'fault_type': 'none' if b == 'healthy_mekf' else cfg.get('fault_type', 'sun_loss'),
            'pointing_rmse_all_deg': rmse(np.ones(n,bool), r['true_point']), 'pointing_rmse_fault_deg': rmse(active, r['true_point']),
            'pointing_rmse_eclipse_deg': rmse(eclipse, r['true_point']), 'pointing_rmse_sunlight_deg': rmse(sunlight, r['true_point']),
            'attitude_rmse_fault_deg': rmse(active, r['att_err']), 'attitude_rmse_eclipse_deg': rmse(eclipse, r['att_err']),
            'pointing_max_eclipse_deg': float(np.max(r['true_point'][eclipse])) if np.any(eclipse) else np.nan,
            'final_pointing_error_deg': float(r['true_point'][-1]), 'settling_time_s': _settling_time(t, r['true_point'], th, hold),
            'ml_recovery_usage_pct_fault': float(100*np.mean(r['ml_used'][active])) if np.any(active) else 0.,
            'ml_recovery_usage_pct_eclipse': float(100*np.mean(r['ml_used'][eclipse])) if np.any(eclipse) else 0.,
            'virtual_sun_rmse_eclipse_deg': rmse(eclipse & np.isfinite(r['sun_rec_err']), np.nan_to_num(r['sun_rec_err'])),
            'eclipse_drift_duration_s': float(np.sum(r['mode'] == ECLIPSE_DRIFT) * dt),
            'reacquisition_duration_s': float(np.sum(r['mode'] == SUN_REACQUISITION) * dt),
            'sun_pointing_mode_fraction_pct': float(100.0 * np.mean(r['mode'] == SUN_POINTING)),
        })
        frames.append(pd.DataFrame({
            'time_s': t, 'branch': b, 'sun_eci_x': sun_series[:,0], 'sun_eci_y': sun_series[:,1], 'sun_eci_z': sun_series[:,2],
            'eclipse_state': eclipse_state, 'illumination_fraction': illumination,
            'true_sun_pointing_error_deg': r['true_point'], 'estimated_sun_pointing_error_deg': r['est_point'],
            'attitude_estimation_error_deg': r['att_err'], 'ml_recovery_active': r['ml_used'].astype(int),
            'sensor_fault_active': r['sensor_faulted'].astype(int), 'virtual_sun_error_deg': r['sun_rec_err'],
            'adcs_mode': r['mode'], 'adcs_mode_code': r['mode_code'], 'sun_pointing_control_enabled': r['control_on'].astype(int),
            'torque_x_nm': r['tau'][:,0], 'torque_y_nm': r['tau'][:,1], 'torque_z_nm': r['tau'][:,2],
        }))
    summary = pd.DataFrame(rows); summary.to_csv(out/'sun_pointing_comparison_summary.csv', index=False)
    pd.concat(frames).to_csv(out/'sun_pointing_comparison_timeseries.csv', index=False)

    eclipse_idx = np.where(eclipse)[0]
    meta = {
        'scenario': 'sun_pointing_comparison', 'branches': list(BRANCHES),
        'illumination': cfg.get('illumination', {}), 'eclipse_samples': int(eclipse.sum()),
        'first_eclipse_s': float(t[eclipse_idx[0]]) if len(eclipse_idx) else None,
        'last_eclipse_s': float(t[eclipse_idx[-1]]) if len(eclipse_idx) else None,
        'virtual_sensor': vm,
        'adcs_modes': list(MODE_CODE.keys()),
        'mode_configuration': cfg.get('modes', {}),
    }
    (out/'scenario_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    formats = tuple(global_cfg['visualization']['formats']); dpi = int(global_cfg['visualization']['dpi'])
    # Sun ephemeris + eclipse timeline.
    fig, axes = create_subplots(2, 1, figsize=(9.6, 7.0), sharex=True)
    for i, lab in enumerate(['x','y','z']): axes[0].plot(t, sun_series[:,i], label=f'Sun ECI {lab}')
    axes[0].set_ylabel('Normalized Sun vector'); axes[0].set_title('Time-varying Sun direction and orbital eclipse'); axes[0].legend(ncol=3)
    axes[1].plot(t, illumination, color=PALETTE['blue_main']); axes[1].set_ylabel('Illumination fraction'); axes[1].set_xlabel('Time [s]'); axes[1].set_ylim(-.05,1.05)
    finalize_figure(fig, plots/'environment_sun_direction_and_eclipse_timeline', formats=formats, dpi=dpi)

    colors = [PALETTE['green_3'], PALETTE['red_strong'], PALETTE['blue_main']]
    fig, axes = create_subplots(figsize=(9.6,5.4)); ax=axes[0]
    for b,c in zip(BRANCHES,colors): ax.plot(t, results[b]['true_point'], label=LABELS[b], color=c)
    add_fault_span(ax,fs,fe)
    if np.any(eclipse):
        # Shade contiguous umbra periods.
        starts=np.where(np.diff(np.r_[False,eclipse])==1)[0]; ends=np.where(np.diff(np.r_[eclipse,False])==-1)[0]
        for s,e in zip(starts,ends): ax.axvspan(t[s],t[e],color='0.25',alpha=.12)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('True Sun-pointing error [deg]'); ax.set_title('Sun pointing: healthy, faulted, and ML recovery with eclipse'); ax.legend()
    finalize_figure(fig, plots/'comparison_true_sun_pointing_error_with_eclipse', formats=formats, dpi=dpi)

    fig, axes = create_subplots(figsize=(8.8,4.9)); ax=axes[0]
    x=np.arange(3); vals=summary.pointing_rmse_eclipse_deg.to_numpy(); ax.bar(x, vals, color=colors, edgecolor='black'); ax.set_xticks(x,list(LABELS.values()),rotation=15); ax.set_ylabel('Eclipse pointing RMSE [deg]'); ax.set_title('Sun-pointing performance during full eclipse')
    finalize_figure(fig, plots/'comparison_eclipse_pointing_rmse_by_branch', formats=formats, dpi=dpi)

    # ADCS operational mode timeline for each comparison branch.
    fig, axes = create_subplots(3, 1, figsize=(9.8, 7.8), sharex=True)
    for ax, b in zip(axes, BRANCHES):
        ax.step(t, results[b]['mode_code'], where='post', color=PALETTE['blue_main'])
        ax.set_yticks(list(MODE_CODE.values()), list(MODE_CODE.keys()))
        ax.set_ylabel(LABELS[b])
        ax.grid(True, axis='x', alpha=.2)
    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title('ADCS mode state machine: acquisition, pointing, eclipse drift, reacquisition')
    finalize_figure(fig, plots/'adcs_mode_timeline_by_comparison_branch', formats=formats, dpi=dpi)

    # Control torque magnitude highlights that Sun-pointing torque is disabled in eclipse drift.
    fig, axes = create_subplots(figsize=(9.6, 5.0)); ax=axes[0]
    for b,c in zip(BRANCHES,colors):
        ax.plot(t, np.linalg.norm(results[b]['tau'],axis=1), label=LABELS[b], color=c)
    if np.any(eclipse):
        starts=np.where(np.diff(np.r_[False,eclipse])==1)[0]; ends=np.where(np.diff(np.r_[eclipse,False])==-1)[0]
        for ss,ee in zip(starts,ends): ax.axvspan(t[ss],t[ee],color='0.25',alpha=.12)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Commanded torque magnitude [N m]'); ax.set_title('Sun-pointing control torque and eclipse drift intervals'); ax.legend()
    finalize_figure(fig, plots/'adcs_control_torque_with_eclipse_drift', formats=formats, dpi=dpi)

    # Learning curve.
    lc=pd.read_csv(models/'virtual_sun_sensor'/'learning_curve.csv'); fig,axes=create_subplots(figsize=(7.8,4.8)); ax=axes[0]
    ax.plot(lc.epoch,lc.train_loss,label='Training loss',color=PALETTE['blue_main']); ax.plot(lc.epoch,lc.val_loss,label='Validation loss',color=PALETTE['red_strong']); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.set_title('Virtual Sun-sensor MLP learning curve'); ax.legend()
    finalize_figure(fig, plots/'ml_virtual_sun_sensor_learning_curve', formats=formats, dpi=dpi)

    # Synchronized HTML animations with dynamic Sun vector and eclipse overlay.
    ac=global_cfg.get('animation',{}); sac=cfg.get('animation',{})
    if bool(ac.get('enabled',False)) and bool(sac.get('enabled',True)):
        ad=out/'animations'; ad.mkdir(exist_ok=True); oc=global_cfg['orbit']
        kwargs={k:v for k,v in ac.items() if k not in ('enabled','phase4','phase5')}
        kwargs.update(body_pointing_axis=axis, sun_i=sun_series, mag_i=mag_i, show_sun_pointing_error=True,
                      eclipse_mask=eclipse, position_eci_m=orbit_states[:,:3], velocity_eci_mps=orbit_states[:,3:],
                      earth_epoch_utc=str(cfg.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00')))
        for b in BRANCHES:
            r=results[b]
            create_orbit_attitude_animation(
                ad/f'{b}_sun_pointing_comparison.html', t, r['truth'][:,:4], r['qhat'],
                altitude_m=oc['altitude_m'], inclination_deg=oc['inclination_deg'], raan_deg=oc['raan_deg'], phase_deg=oc['phase_deg'],
                fault_name='none' if b=='healthy_mekf' else str(cfg.get('fault_type','sun_loss')),
                severity='none' if b=='healthy_mekf' else str(cfg.get('fault_severity','medium')),
                fault_start_s=fs if b!='healthy_mekf' else None, fault_end_s=fe if b!='healthy_mekf' else None,
                title=LABELS[b], mode_series=r['mode'], **kwargs)

    return {
        'scenario':'sun_pointing_comparison', 'branches':3,
        'sun_model':str(cfg.get('illumination',{}).get('sun_model','analytic')),
        'eclipse_duration_s':float(np.sum(eclipse)*dt),
        'first_eclipse_s':meta['first_eclipse_s'], 'last_eclipse_s':meta['last_eclipse_s'],
        'virtual_sensor_validation_rmse_deg':float(vm['validation_angular_rmse_deg']),
    }
