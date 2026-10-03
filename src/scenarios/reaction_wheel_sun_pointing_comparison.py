from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.simulation.attitude import (
    propagate_attitude_controlled, quat_angle_error_deg, quat_to_dcm,
    quat_multiply, small_angle_quat, quat_normalize,
)
from src.simulation.control import (
    sun_pointing_initial_quaternion, sun_pointing_torque, sun_pointing_error_deg,
    desired_sun_orbit_dcm, full_attitude_pd_torque,
)
from src.simulation.orbit_profiles import resolve_orbit_profile
from src.simulation.orbit_model import propagate_circular_two_body
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.adcs_modes import (
    SunPointingModeManager, DETUMBLING, ECLIPSE_RATE_DAMPING, SUN_REACQUISITION,
    SUN_POINTING, SUN_ACQUISITION, MODE_CODE, sun_pointing_control_enabled,
    magnetic_rate_damping_enabled,
)
from src.simulation.spacecraft import inertia_from_config
from src.simulation.magnetic_field import magnetic_field_eci
from src.simulation.magnetorquer import (
    dipole_for_desired_torque, magnetic_torque, first_order_dipole_response, bdot_dipole_command,
)
from src.simulation.disturbances import gravity_gradient_torque_body
from src.estimation.mekf import MEKF
from src.ml.virtual_sensor import train_virtual_sun_model
from src.visualization.publication import PALETTE, create_subplots, finalize_figure, add_fault_span
from src.visualization.animation import create_orbit_attitude_animation
from src.scenarios.sun_pointing_comparison import _apply_sun_fault, _settling_time, BRANCHES, LABELS


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _parse_epoch(text):
    from datetime import datetime
    return datetime.fromisoformat(str(text).replace('Z', '+00:00'))


def _sensor_measurement(q, omega, gyro_bias, gyro_noise, sun_noise, mag_noise,
                        sun_i, B_i_t, sun_visible=True):
    C = quat_to_dcm(q)
    g = np.asarray(omega) + np.asarray(gyro_bias) + np.asarray(gyro_noise)
    s = _unit(C.T @ _unit(sun_i) + sun_noise) if sun_visible else np.full(3, np.nan)
    mag_ref_i = _unit(B_i_t)
    m = _unit(C.T @ mag_ref_i + mag_noise)
    return g, s, m




def _rotation_matrix_xyz_deg(angles_deg):
    a=np.deg2rad(np.asarray(angles_deg,float)); cx,cy,cz=np.cos(a); sx,sy,sz=np.sin(a)
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx


def _apply_extra_sensor_faults(gyro_meas, mag_meas, t_s, fault_case, active, rng):
    """Apply configurable gyro and magnetometer faults to one sensor sample.

    ``fault_case`` is a plain dictionary.  Supported gyro fault types are
    ``none``, ``bias_step``, ``drift``, ``noise``, ``scale``, ``stuck`` and ``loss``.
    Supported magnetometer fault types are ``none``, ``bias``, ``noise``, ``scale``,
    ``misalignment``, ``stuck``, ``dropout`` and ``loss``.
    """
    g=np.asarray(gyro_meas,float).copy(); m=np.asarray(mag_meas,float).copy()
    if not active or not fault_case:
        return g,m,False,False
    gf=fault_case.get('gyro',{}) or {}; mf=fault_case.get('mag',{}) or {}
    gtype=str(gf.get('type','none')).lower(); mtype=str(mf.get('type','none')).lower()
    g_fault=gtype!='none'; m_fault=mtype!='none'
    if gtype=='bias_step':
        g += np.asarray(gf.get('bias_rad_s',[0.003,-0.002,0.0015]),float)
    elif gtype=='drift':
        elapsed=max(0.0,float(t_s)-float(fault_case.get('start_s',0.0)))
        g += np.asarray(gf.get('drift_rad_s2',[2e-6,-1.5e-6,1e-6]),float)*elapsed
    elif gtype=='noise':
        g += rng.normal(0,float(gf.get('noise_std_rad_s',0.002)),3)
    elif gtype=='scale':
        g *= np.asarray(gf.get('scale',[1.15,0.85,1.10]),float)
    elif gtype=='stuck':
        g[:] = np.asarray(gf.get('value_rad_s',[0.02,-0.015,0.01]),float)
    elif gtype=='loss':
        # Treat a lost gyro as a held last/zero-like invalid measurement only when the
        # estimator cannot consume NaNs. Here a configured fallback can emulate the
        # flight software's fail-safe substituted rate.
        g[:] = np.asarray(gf.get('fallback_rad_s',[0.0,0.0,0.0]),float)
    elif gtype!='none':
        raise ValueError(f'Unsupported gyro fault type: {gtype}')

    if mtype=='bias':
        m = _unit(m + np.asarray(mf.get('bias_body',[0.05,-0.03,0.04]),float))
    elif mtype=='noise':
        m = _unit(m + rng.normal(0,float(mf.get('noise_std',0.03)),3))
    elif mtype=='scale':
        scale=np.asarray(mf.get('scale',[1.10,0.90,1.05]),float); m=_unit(m*scale)
    elif mtype=='misalignment':
        m=_unit(_rotation_matrix_xyz_deg(mf.get('angles_deg',[3.0,-2.0,4.0]))@m)
    elif mtype=='stuck':
        m=_unit(np.asarray(mf.get('body_vector',[1.0,0.0,0.0]),float))
    elif mtype=='dropout':
        if rng.random() < float(mf.get('probability',0.5)): m[:] = np.nan
    elif mtype=='loss':
        m[:] = np.nan
    elif mtype!='none':
        raise ValueError(f'Unsupported magnetometer fault type: {mtype}')
    return g,m,g_fault,m_fault

def _make_environment(t, cfg, global_cfg):
    illum = cfg.get('illumination', {})
    sun_model = str(illum.get('sun_model', 'analytic'))
    epoch_text = str(illum.get('epoch_utc', '2026-01-01T00:00:00+00:00'))
    sun_dir, sun_pos = sun_direction_series(t, sun_model, epoch=epoch_text,
                                            constant_vector=cfg.get('sun_vector_eci', [1.0, 0.2, 0.1]))
    oc = resolve_orbit_profile(str(cfg.get('orbit_profile', 'global')), global_cfg['orbit'], epoch_utc=epoch_text,
                               high_beta_cfg=cfg.get('high_beta_orbit', {}))
    states = propagate_circular_two_body(t, float(oc['altitude_m']), float(oc['inclination_deg']),
                                         float(oc['raan_deg']), float(oc['phase_deg']))
    if bool(illum.get('eclipse_enabled', True)):
        eclipse_state, illumination = eclipse_geometry(states[:, :3], sun_pos)
    else:
        eclipse_state = np.zeros(len(t), int)
        illumination = np.ones(len(t), float)
    mag_cfg = cfg.get('magnetic_field', {})
    B_i = magnetic_field_eci(states[:, :3], t, epoch_utc=_parse_epoch(epoch_text),
                             model=str(mag_cfg.get('model', 'igrf14')))
    return sun_dir, sun_pos, states, eclipse_state, illumination, B_i, oc


def _clip_norm(v, limit):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    lim = float(limit)
    if lim <= 0 or n <= lim or n == 0:
        return v
    return v * (lim / n)


def _rw_allocate(tau_cmd, h_prev, dt, max_torque_nm, max_momentum_nms):
    """Axis-wise torque limiting with wheel momentum saturation."""
    tau = np.asarray(tau_cmd, float).copy()
    tmax = np.broadcast_to(np.asarray(max_torque_nm, float), (3,))
    hmax = np.broadcast_to(np.asarray(max_momentum_nms, float), (3,))
    tau = np.clip(tau, -tmax, tmax)
    h_next = np.asarray(h_prev, float).copy()
    tau_eff = tau.copy()
    for i in range(3):
        tentative = h_prev[i] - tau_eff[i] * dt
        if tentative > hmax[i]:
            tau_eff[i] = (h_prev[i] - hmax[i]) / dt
            tentative = hmax[i]
        elif tentative < -hmax[i]:
            tau_eff[i] = (h_prev[i] + hmax[i]) / dt
            tentative = -hmax[i]
        h_next[i] = tentative
    return tau_eff, h_next


def _one_step_real(dt, state, inertia_diag, tau_rw_body, dipole_actual_am2, B_i_t, position_eci_m,
                   gravity_gradient_enabled=True):
    def torque_cb(tt, q, w):
        B_body = quat_to_dcm(q).T @ B_i_t
        tau_mtq = magnetic_torque(dipole_actual_am2, B_body)
        tau = np.asarray(tau_rw_body, float) + tau_mtq
        if gravity_gradient_enabled:
            tau = tau + gravity_gradient_torque_body(q, position_eci_m, inertia_diag)
        return tau
    tr, _ = propagate_attitude_controlled(np.array([0.0, float(dt)]), state[:4], state[4:], inertia_diag, torque_cb)
    q0 = state[:4]
    B_body0 = quat_to_dcm(q0).T @ B_i_t
    tau_mtq = magnetic_torque(dipole_actual_am2, B_body0)
    tau_gg = gravity_gradient_torque_body(q0, position_eci_m, inertia_diag) if gravity_gradient_enabled else np.zeros(3)
    return tr[-1], tau_mtq, tau_gg


def _run_branch(branch, cfg, global_cfg, t, q0, omega0, sun_series, B_i_series,
                orbit_states, eclipse_state, noise, predictor=None, fault_case=None):
    sc = global_cfg['sensors']
    dt = float(t[1] - t[0])
    n = len(t)
    spacecraft = cfg.get('spacecraft', {})
    inertia = inertia_from_config(spacecraft)
    gravity_gradient = bool(cfg.get('disturbances', {}).get('gravity_gradient', True))

    rw_cfg = cfg.get('reaction_wheels', {})
    max_rw_torque = np.asarray(rw_cfg.get('max_torque_nm', [2.3e-4, 2.3e-4, 2.3e-4]), float)
    max_rw_momentum = np.asarray(rw_cfg.get('max_momentum_nms', [1.8e-3, 1.8e-3, 1.8e-3]), float)
    wheel_inertia = np.asarray(rw_cfg.get('wheel_inertia_kg_m2', [1.0e-6, 1.0e-6, 1.0e-6]), float)
    rpm_limit = np.asarray(rw_cfg.get('max_speed_rpm', [6000.0, 6000.0, 6000.0]), float)

    mtq_cfg = cfg.get('magnetorquers', {})
    max_dipole = np.asarray(mtq_cfg.get('max_dipole_am2', [0.2, 0.2, 0.2]), float)
    torquer_tau = float(mtq_cfg.get('time_constant_s', 0.0065))

    ctrl = cfg.get('controller', {})
    axis = str(cfg.get('pointing_axis', 'x'))
    acq = ctrl.get('acquisition', {})
    hold = ctrl.get('hold', {})
    reacq = ctrl.get('reacquisition', acq)
    eclipse_hold = ctrl.get('eclipse_hold', {})
    det = ctrl.get('detumbling', {})

    acq_kp = float(acq.get('kp', 7.0e-4))
    acq_kd = float(acq.get('kd', 9.0e-3))
    acq_limit = float(acq.get('torque_limit_nm', 1.6e-4))
    hold_kp = float(hold.get('kp', 3.0e-4))
    hold_kd = float(hold.get('kd', 8.0e-3))
    hold_limit = float(hold.get('torque_limit_nm', 8.0e-5))
    reacq_kp = float(reacq.get('kp', acq_kp))
    reacq_kd = float(reacq.get('kd', acq_kd))
    reacq_limit = float(reacq.get('torque_limit_nm', acq_limit))
    eclipse_kd = float(eclipse_hold.get('kd', 4.0e-3))
    eclipse_limit = float(eclipse_hold.get('torque_limit_nm', 6.0e-5))
    use_full_hold = bool(hold.get('use_full_attitude', True))

    bdot_gain = float(det.get('bdot_gain_am2_s_per_t', 1.5e5))
    bdot_deadband = float(det.get('bdot_deadband_t_per_s', 0.0))
    bdot_filter_tau = float(det.get('bdot_filter_tau_s', 2.0))

    dump = cfg.get('momentum_dumping', {})
    dump_enable = bool(dump.get('enabled', True))
    dump_start = float(dump.get('start_fraction', 0.65))
    dump_stop = float(dump.get('stop_fraction', 0.35))
    dump_gain = float(dump.get('gain_per_s', 0.12))
    dump_torque_limit = float(dump.get('torque_limit_nm', 5.0e-6))

    fc = fault_case or {}
    fs = float(fc.get('start_s', cfg.get('fault_start_s', 250.0)))
    fe = float(fc.get('end_s', cfg.get('fault_end_s', 450.0)))
    sun_cfg = fc.get('sun', {}) or {}
    ft = str(sun_cfg.get('type', cfg.get('fault_type', 'sun_loss')))
    sev = str(sun_cfg.get('severity', cfg.get('fault_severity', 'medium')))
    eclipse_as_loss = bool(cfg.get('illumination', {}).get('eclipse_sun_sensor_unavailable', True))

    mode_cfg = cfg.get('modes', {})
    mgr = SunPointingModeManager(
        threshold_deg=float(mode_cfg.get('pointing_threshold_deg', 2.0)),
        hold_s=float(mode_cfg.get('pointing_hold_s', 15.0)),
        dt_s=dt,
        rate_threshold_deg_s=float(mode_cfg.get('rate_threshold_deg_s', 0.08)),
        rate_hold_s=float(mode_cfg.get('rate_hold_s', 8.0)),
        rate_recovery_trigger_deg_s=float(mode_cfg.get('rate_recovery_trigger_deg_s', 0.5)),
        pointing_exit_threshold_deg=float(mode_cfg.get('pointing_exit_threshold_deg', 5.0)),
        mode=DETUMBLING,
    )
    filt = MEKF(q0=q0, gyro_noise_std=float(sc['gyro_noise_std_rad_s']), vector_noise_std=float(sc['vector_noise_std']))

    truth = np.zeros((n, 7)); qhat = np.zeros((n, 4)); bhat = np.zeros((n, 3))
    gyro = np.zeros((n, 3)); sun_raw = np.full((n, 3), np.nan); sun_used = np.full((n, 3), np.nan); mag = np.zeros((n, 3))
    rw_tau = np.zeros((n, 3)); mtq_dump_cmd = np.zeros((n, 3)); mtq_dump_actual = np.zeros((n, 3))
    tau_mtq = np.zeros((n, 3)); tau_gg = np.zeros((n, 3)); bdot = np.zeros((n, 3))
    wheel_h = np.zeros((n, 3)); wheel_speed_rpm = np.zeros((n, 3)); dump_active = np.zeros(n, bool)
    ml_used = np.zeros(n, bool); ml_candidate = np.zeros(n, bool); ml_rejected = np.zeros(n, bool); sensor_faulted = np.zeros(n, bool)
    gyro_faulted=np.zeros(n,bool); mag_faulted=np.zeros(n,bool)
    mode = np.empty(n, dtype=object); mode_code = np.zeros(n, int); control_on = np.zeros(n, bool); rate_damping_on = np.zeros(n, bool)

    truth[0, :4] = q0; truth[0, 4:] = omega0; qhat[0] = filt.q; bhat[0] = filt.b
    visible0 = not (eclipse_as_loss and eclipse_state[0] == 2)
    gyro[0], sun_raw[0], mag[0] = _sensor_measurement(q0, omega0, sc['gyro_bias_rad_s'], noise['gyro'][0], noise['sun'][0], noise['mag'][0], sun_series[0], B_i_series[0], visible0)
    sun_used[0] = sun_raw[0]
    B_ctrl_filtered = mag[0] * np.linalg.norm(B_i_series[0])
    anchor = sun_raw[0].copy() if np.all(np.isfinite(sun_raw[0])) else quat_to_dcm(qhat[0]).T @ sun_series[0]
    initial_est = sun_pointing_error_deg(qhat[0], axis, sun_series[0])
    initial_rate_deg_s = float(np.degrees(np.linalg.norm(gyro[0] - filt.b)))
    mode[0] = mgr.update(in_full_eclipse=bool(eclipse_state[0] == 2), real_sun_available=bool(np.all(np.isfinite(sun_raw[0]))), estimated_pointing_error_deg=initial_est, estimated_rate_deg_s=initial_rate_deg_s)
    mode_code[0] = MODE_CODE[mode[0]]; control_on[0] = sun_pointing_control_enabled(mode[0]); rate_damping_on[0] = magnetic_rate_damping_enabled(mode[0])
    frng = np.random.default_rng(int(cfg.get('fault_seed', 4501)))
    previous_mode = mode[0]
    is_dumping = False

    for k in range(1, n):
        sun_i = sun_series[k-1]; B_i = B_i_series[k-1]
        omega_est = gyro[k-1] - bhat[k-1]
        C_est = quat_to_dcm(qhat[k-1])
        B_body_est = C_est.T @ B_i

        B_meas_for_bdot = mag[k-1] * np.linalg.norm(B_i_series[k-1])
        B_prev_filtered = B_ctrl_filtered.copy()
        if bdot_filter_tau <= 0:
            B_ctrl_filtered = B_meas_for_bdot.copy()
        else:
            alpha_b = 1.0 - np.exp(-dt / bdot_filter_tau)
            B_ctrl_filtered = B_ctrl_filtered + alpha_b * (B_meas_for_bdot - B_ctrl_filtered)

        tau_body_cmd = np.zeros(3)
        if sun_pointing_control_enabled(mode[k-1]):
            if mode[k-1] == SUN_POINTING:
                kp, kd, tau_lim = hold_kp, hold_kd, hold_limit
            elif mode[k-1] == SUN_REACQUISITION:
                kp, kd, tau_lim = reacq_kp, reacq_kd, reacq_limit
            else:
                kp, kd, tau_lim = acq_kp, acq_kd, acq_limit
            if use_full_hold:
                Cdes = desired_sun_orbit_dcm(sun_i, orbit_states[k-1, :3], orbit_states[k-1, 3:], axis)
                tau_body_cmd, _ = full_attitude_pd_torque(qhat[k-1], omega_est, Cdes, kp, kd, tau_lim)
            else:
                tau_body_cmd = sun_pointing_torque(qhat[k-1], omega_est, axis, sun_i, kp, kd, tau_lim)
        elif mode[k-1] == ECLIPSE_RATE_DAMPING:
            tau_body_cmd = _clip_norm(-eclipse_kd * np.asarray(omega_est, float), eclipse_limit)
        elif mode[k-1] == DETUMBLING:
            tau_body_cmd = np.zeros(3)
        else:
            tau_body_cmd = np.zeros(3)

        # Deployment-only detumbling via B-dot magnetorquers.
        mtq_det_cmd = np.zeros(3)
        if mode[k-1] == DETUMBLING:
            mtq_det_cmd, bdot[k-1] = bdot_dipole_command(B_ctrl_filtered, B_prev_filtered, dt, bdot_gain, max_dipole, bdot_deadband)

        # Reaction-wheel torque allocation with optional momentum dumping.
        hw_prev = wheel_h[k-1]
        if dump_enable:
            frac = float(np.max(np.abs(hw_prev / np.maximum(max_rw_momentum, 1e-12))))
            if is_dumping:
                is_dumping = frac > dump_stop
            else:
                is_dumping = frac > dump_start
        else:
            is_dumping = False
        dump_active[k-1] = is_dumping

        mtq_dump_desired_tau = np.zeros(3)
        if is_dumping and np.linalg.norm(B_body_est) > 1e-12:
            mtq_dump_desired_tau = _clip_norm(-dump_gain * hw_prev, dump_torque_limit)
            mtq_dump_cmd[k-1], _ = dipole_for_desired_torque(mtq_dump_desired_tau, B_body_est, max_dipole)
        else:
            mtq_dump_cmd[k-1] = np.zeros(3)

        mtq_cmd_total = mtq_det_cmd + mtq_dump_cmd[k-1]
        mtq_cmd_total = np.clip(mtq_cmd_total, -max_dipole, max_dipole)
        mtq_dump_actual[k-1] = first_order_dipole_response(mtq_dump_actual[k-2] if k > 1 else np.zeros(3), mtq_cmd_total, dt, torquer_tau)
        tau_mtq_expected = magnetic_torque(mtq_dump_actual[k-1], B_body_est)

        # Compensate the commanded body torque with the expected external MTQ torque.
        tau_rw_cmd = tau_body_cmd - tau_mtq_expected
        tau_rw_cmd = _clip_norm(tau_rw_cmd, float(np.linalg.norm(max_rw_torque)))
        rw_tau[k-1], wheel_h[k] = _rw_allocate(tau_rw_cmd, hw_prev, dt, max_rw_torque, max_rw_momentum)
        wheel_speed_rpm[k] = np.clip((wheel_h[k] / np.maximum(wheel_inertia, 1e-12)) * 60.0 / (2 * np.pi), -rpm_limit, rpm_limit)

        truth[k], tau_mtq[k-1], tau_gg[k-1] = _one_step_real(dt, truth[k-1], inertia, rw_tau[k-1], mtq_dump_actual[k-1], B_i, orbit_states[k-1, :3], gravity_gradient)

        sun_visible = not (eclipse_as_loss and eclipse_state[k] == 2)
        gyro[k], sun_raw[k], mag[k] = _sensor_measurement(truth[k, :4], truth[k, 4:], sc['gyro_bias_rad_s'], noise['gyro'][k], noise['sun'][k], noise['mag'][k], sun_series[k], B_i_series[k], sun_visible)
        fault_active = (t[k] >= fs and t[k] <= fe)
        if branch != 'healthy_mekf':
            gyro[k],mag[k],gyro_faulted[k],mag_faulted[k]=_apply_extra_sensor_faults(gyro[k],mag[k],t[k],fc,fault_active,frng)
        measured = sun_raw[k].copy()
        if branch != 'healthy_mekf' and ft != 'none':
            measured, faulted = _apply_sun_fault(measured, ft, sev, fault_active, frng)
            sensor_faulted[k] = faulted or (fault_active and ft == 'sun_noise')

        physical_sun_visible = not (eclipse_as_loss and eclipse_state[k] == 2)
        needs_recovery = physical_sun_visible and (not np.all(np.isfinite(measured)) or (branch == 'ml_recovery' and fault_active and ft == 'sun_noise'))
        if branch == 'ml_recovery' and needs_recovery:
            omega_virtual = gyro[k-1] - bhat[k-1]
            candidate = predictor.predict(anchor, omega_virtual, mag[k], dt); ml_candidate[k] = True
            qpred = quat_normalize(quat_multiply(qhat[k-1], small_angle_quat(omega_virtual * dt)))
            predicted_sun = quat_to_dcm(qpred).T @ sun_series[k]
            disagreement = np.degrees(np.arccos(np.clip(np.dot(_unit(candidate), _unit(predicted_sun)), -1, 1)))
            if disagreement <= float(cfg.get('ml_gate_deg', 2.0)):
                blend = float(cfg.get('ml_blend', 0.25)); measured = _unit((1-blend) * predicted_sun + blend * candidate)
                ml_used[k] = True; anchor = measured.copy()
            else:
                measured = np.full(3, np.nan); ml_rejected[k] = True; anchor = _unit(candidate)
        elif np.all(np.isfinite(measured)):
            anchor = measured.copy()
        sun_used[k] = measured

        sun_std = float(cfg.get('ml_virtual_sensor_noise_std', 0.01)) if ml_used[k] else None
        q, b, _ = filt.step(gyro[k-1], sun_used[k], mag[k], sun_series[k], _unit(B_i_series[k]), dt, sun_noise_std=sun_std)
        qhat[k] = q; bhat[k] = b
        est_point = sun_pointing_error_deg(qhat[k], axis, sun_series[k])
        real_sun_available = bool(np.all(np.isfinite(measured))) and not bool(ml_used[k])
        est_rate_deg_s = float(np.degrees(np.linalg.norm(gyro[k] - bhat[k])))
        mode[k] = mgr.update(in_full_eclipse=bool(eclipse_state[k] == 2), real_sun_available=real_sun_available, estimated_pointing_error_deg=est_point, estimated_rate_deg_s=est_rate_deg_s)
        previous_mode = mode[k]
        mode_code[k] = MODE_CODE[mode[k]]; control_on[k] = sun_pointing_control_enabled(mode[k]); rate_damping_on[k] = magnetic_rate_damping_enabled(mode[k])

    rw_tau[-1] = rw_tau[-2]; mtq_dump_cmd[-1] = mtq_dump_cmd[-2]; mtq_dump_actual[-1] = mtq_dump_actual[-2]
    tau_mtq[-1] = tau_mtq[-2]; tau_gg[-1] = tau_gg[-2]; bdot[-1] = bdot[-2]; dump_active[-1] = dump_active[-2]
    true_point = np.array([sun_pointing_error_deg(q, axis, s) for q, s in zip(truth[:, :4], sun_series)])
    est_point = np.array([sun_pointing_error_deg(q, axis, s) for q, s in zip(qhat, sun_series)])
    att_err = np.array([quat_angle_error_deg(qe, qt) for qe, qt in zip(qhat, truth[:, :4])])
    sun_rec_err = np.full(n, np.nan)
    for k in range(n):
        if np.all(np.isfinite(sun_used[k])):
            target = _unit(quat_to_dcm(truth[k, :4]).T @ sun_series[k])
            sun_rec_err[k] = np.degrees(np.arccos(np.clip(np.dot(_unit(sun_used[k]), target), -1, 1)))

    return dict(
        truth=truth, qhat=qhat, bhat=bhat, gyro=gyro, sun_raw=sun_raw, sun_used=sun_used, mag=mag,
        rw_tau=rw_tau, mtq_dump_cmd=mtq_dump_cmd, mtq_dump_actual=mtq_dump_actual, tau_mtq=tau_mtq, tau_gg=tau_gg,
        bdot=bdot, wheel_h=wheel_h, wheel_speed_rpm=wheel_speed_rpm, dump_active=dump_active,
        ml_used=ml_used, ml_candidate=ml_candidate, ml_rejected=ml_rejected, sensor_faulted=sensor_faulted, gyro_faulted=gyro_faulted, mag_faulted=mag_faulted,
        mode=mode, mode_code=mode_code, control_on=control_on, rate_damping_on=rate_damping_on,
        true_point=true_point, est_point=est_point, att_err=att_err, sun_rec_err=sun_rec_err, inertia=inertia,
    )


def run(cfg, global_cfg, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    plots = out / 'plots'; plots.mkdir(exist_ok=True)
    models = out / 'models'; models.mkdir(exist_ok=True)
    duration = float(cfg.get('duration_s', 6000.0)); dt = float(cfg.get('dt_s', 0.1)); t = np.arange(0, duration + dt/2, dt)
    sun_series, sun_pos, orbit_states, eclipse_state, illumination, B_i, orbit_meta = _make_environment(t, cfg, global_cfg)
    axis = str(cfg.get('pointing_axis', 'x'))
    q0 = sun_pointing_initial_quaternion(axis, sun_series[0], float(cfg.get('initial_error_deg', 30.0)))
    omega0 = np.asarray(cfg.get('omega0_rad_s', [0, 0, 0]), float)

    sc = global_cfg['sensors']; rng = np.random.default_rng(int(cfg.get('comparison_seed', 7301))); n = len(t)
    noise = {
        'gyro': rng.normal(0, float(sc['gyro_noise_std_rad_s']), (n, 3)),
        'sun': rng.normal(0, float(sc['vector_noise_std']), (n, 3)),
        'mag': rng.normal(0, float(sc['vector_noise_std']), (n, 3)),
    }
    vc = cfg.get('virtual_sensor', {})
    predictor, vm = train_virtual_sun_model(
        models / 'virtual_sun_sensor', runs=int(vc.get('train_runs', 24)), duration_s=float(vc.get('train_duration_s', 40.0)), dt=dt,
        hidden=tuple(vc.get('hidden', [48, 24])), epochs=int(vc.get('epochs', 18)), batch_size=int(vc.get('batch_size', 256)), seed=int(vc.get('seed', 8128)),
        gyro_noise_std=float(sc['gyro_noise_std_rad_s']), vector_noise_std=float(sc['vector_noise_std'])
    )

    results = {b: _run_branch(b, cfg, global_cfg, t, q0, omega0, sun_series, B_i, orbit_states, eclipse_state, noise, predictor if b == 'ml_recovery' else None) for b in BRANCHES}

    fs = float(cfg.get('fault_start_s', 250.0)); fe = float(cfg.get('fault_end_s', 450.0))
    active = (t >= fs) & (t <= fe); eclipse = eclipse_state == 2; sunlight = ~eclipse
    mode_cfg = cfg.get('modes', {})
    th = float(mode_cfg.get('pointing_threshold_deg', 2.0)); hold = float(mode_cfg.get('pointing_hold_s', 15.0))
    Bmag = np.linalg.norm(B_i, axis=1)
    rows = []; frames = []
    for b, r in results.items():
        def rmse(mask, x):
            arr = np.asarray(x)
            return float(np.sqrt(np.mean(arr[mask]**2))) if np.any(mask) else np.nan
        rows.append({
            'branch': b, 'label': LABELS[b], 'fault_type': 'none' if b == 'healthy_mekf' else cfg.get('fault_type', 'sun_loss'),
            'orbit_profile': orbit_meta['profile'], 'orbit_beta_deg': orbit_meta['beta_angle_deg'],
            'pointing_rmse_all_deg': rmse(np.ones(n, bool), r['true_point']),
            'pointing_rmse_fault_deg': rmse(active, r['true_point']),
            'pointing_rmse_eclipse_deg': rmse(eclipse, r['true_point']),
            'pointing_rmse_sunlight_deg': rmse(sunlight, r['true_point']),
            'attitude_rmse_all_deg': rmse(np.ones(n, bool), r['att_err']),
            'final_pointing_error_deg': float(r['true_point'][-1]),
            'initial_acquisition_time_s': _settling_time(t, r['true_point'], th, hold),
            'reaction_wheel_hold_duration_s': float(np.sum(r['mode'] == SUN_POINTING) * dt),
            'max_reaction_wheel_torque_uNm': float(np.max(np.linalg.norm(r['rw_tau'], axis=1)) * 1e6),
            'max_wheel_momentum_mNms': float(np.max(np.linalg.norm(r['wheel_h'], axis=1)) * 1e3),
            'max_wheel_speed_rpm': float(np.max(np.linalg.norm(r['wheel_speed_rpm'], axis=1))),
            'max_magnetic_dump_torque_uNm': float(np.max(np.linalg.norm(r['tau_mtq'], axis=1)) * 1e6),
            'dump_active_fraction_pct': float(100 * np.mean(r['dump_active'])),
            'ml_recovery_usage_pct_fault': float(100 * np.mean(r['ml_used'][active])) if np.any(active) else 0.0,
        })
        frames.append(pd.DataFrame({
            'time_s': t, 'branch': b,
            'sun_eci_x': sun_series[:,0], 'sun_eci_y': sun_series[:,1], 'sun_eci_z': sun_series[:,2],
            'B_eci_x_T': B_i[:,0], 'B_eci_y_T': B_i[:,1], 'B_eci_z_T': B_i[:,2], 'B_magnitude_uT': Bmag * 1e6,
            'eclipse_state': eclipse_state, 'illumination_fraction': illumination,
            'true_sun_pointing_error_deg': r['true_point'], 'estimated_sun_pointing_error_deg': r['est_point'], 'attitude_estimation_error_deg': r['att_err'],
            'adcs_mode': r['mode'], 'adcs_mode_code': r['mode_code'],
            'rw_torque_x_Nm': r['rw_tau'][:,0], 'rw_torque_y_Nm': r['rw_tau'][:,1], 'rw_torque_z_Nm': r['rw_tau'][:,2],
            'wheel_momentum_x_Nms': r['wheel_h'][:,0], 'wheel_momentum_y_Nms': r['wheel_h'][:,1], 'wheel_momentum_z_Nms': r['wheel_h'][:,2],
            'wheel_speed_x_rpm': r['wheel_speed_rpm'][:,0], 'wheel_speed_y_rpm': r['wheel_speed_rpm'][:,1], 'wheel_speed_z_rpm': r['wheel_speed_rpm'][:,2],
            'dump_active': r['dump_active'].astype(int),
            'dump_dipole_x_Am2': r['mtq_dump_actual'][:,0], 'dump_dipole_y_Am2': r['mtq_dump_actual'][:,1], 'dump_dipole_z_Am2': r['mtq_dump_actual'][:,2],
            'mtq_torque_x_Nm': r['tau_mtq'][:,0], 'mtq_torque_y_Nm': r['tau_mtq'][:,1], 'mtq_torque_z_Nm': r['tau_mtq'][:,2],
            'gravity_gradient_x_Nm': r['tau_gg'][:,0], 'gravity_gradient_y_Nm': r['tau_gg'][:,1], 'gravity_gradient_z_Nm': r['tau_gg'][:,2],
            'omega_x_rad_s': r['truth'][:,4], 'omega_y_rad_s': r['truth'][:,5], 'omega_z_rad_s': r['truth'][:,6],
            'body_rate_deg_s': np.degrees(np.linalg.norm(r['truth'][:,4:], axis=1)),
            'ml_recovery_active': r['ml_used'].astype(int), 'sensor_fault_active': r['sensor_faulted'].astype(int),
        }))

    summary = pd.DataFrame(rows); summary.to_csv(out / 'reaction_wheel_sun_pointing_comparison_summary.csv', index=False)
    pd.concat(frames).to_csv(out / 'reaction_wheel_sun_pointing_comparison_timeseries.csv', index=False)

    meta = {
        'scenario': 'reaction_wheel_sun_pointing_comparison', 'branches': list(BRANCHES),
        'spacecraft': cfg.get('spacecraft', {}), 'reaction_wheels': cfg.get('reaction_wheels', {}),
        'magnetorquers': cfg.get('magnetorquers', {}), 'controller': cfg.get('controller', {}),
        'orbit': orbit_meta, 'modes': cfg.get('modes', {}), 'virtual_sensor': vm,
    }
    (out / 'scenario_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    formats = tuple(global_cfg['visualization']['formats']); dpi = int(global_cfg['visualization']['dpi'])
    colors = [PALETTE['green_3'], PALETTE['red_strong'], PALETTE['blue_main']]

    fig, axes = create_subplots(figsize=(10.0, 5.2)); ax = axes[0]
    for b, c in zip(BRANCHES, colors):
        ax.plot(t, results[b]['true_point'], label=LABELS[b], color=c)
    add_fault_span(ax, fs, fe)
    if np.any(eclipse):
        starts = np.where(np.diff(np.r_[False, eclipse]) == 1)[0]; ends = np.where(np.diff(np.r_[eclipse, False]) == -1)[0]
        for s, e in zip(starts, ends):
            ax.axvspan(t[s], t[e], color='0.25', alpha=.12)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('True Sun-pointing error [deg]'); ax.set_title('Reaction-wheel Sun acquisition / pointing / eclipse / reacquisition'); ax.legend()
    finalize_figure(fig, plots / 'rw_aocs_sun_pointing_error_healthy_faulty_ml', formats=formats, dpi=dpi)

    fig, axes = create_subplots(3, 1, figsize=(10.0, 8.2), sharex=True)
    r = results['healthy_mekf']
    for i, lab in enumerate('xyz'):
        axes[0].plot(t, r['rw_tau'][:,i] * 1e6, label=f'τrw {lab}')
        axes[1].plot(t, r['wheel_h'][:,i] * 1e3, label=f'H {lab}')
        axes[2].plot(t, r['wheel_speed_rpm'][:,i], label=f'ωw {lab}')
    axes[0].set_ylabel('RW torque [µN m]'); axes[0].set_title('Healthy branch reaction-wheel actuation'); axes[0].legend(ncol=3)
    axes[1].set_ylabel('Wheel momentum [mN s]'); axes[1].legend(ncol=3)
    axes[2].set_ylabel('Wheel speed [rpm]'); axes[2].set_xlabel('Time [s]'); axes[2].legend(ncol=3)
    finalize_figure(fig, plots / 'rw_aocs_wheel_torque_momentum_speed', formats=formats, dpi=dpi)

    fig, axes = create_subplots(2, 1, figsize=(10.0, 7.0), sharex=True)
    axes[0].plot(t, np.linalg.norm(r['tau_mtq'], axis=1) * 1e6, color=PALETTE['teal'], label='Magnetorquer dump torque')
    axes[0].plot(t, np.linalg.norm(r['tau_gg'], axis=1) * 1e6, color=PALETTE['violet'], label='Gravity-gradient torque')
    axes[0].set_ylabel('Torque [µN m]'); axes[0].set_title('External torques in healthy branch'); axes[0].legend()
    axes[1].step(t, r['dump_active'].astype(float), where='post', color=PALETTE['red_strong'], label='Momentum dumping active')
    axes[1].set_ylabel('Active'); axes[1].set_xlabel('Time [s]'); axes[1].set_ylim(-0.05, 1.05); axes[1].legend()
    finalize_figure(fig, plots / 'rw_aocs_momentum_dumping_torque_and_activity', formats=formats, dpi=dpi)

    fig, axes = create_subplots(figsize=(10.0, 5.2)); ax = axes[0]
    ax.plot(t, np.degrees(np.linalg.norm(r['truth'][:,4:], axis=1)), color=PALETTE['violet'], label='Body-rate magnitude')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Body-rate [deg/s]'); ax.set_title('Healthy branch body-rate evolution'); ax.legend()
    finalize_figure(fig, plots / 'rw_aocs_body_rate_magnitude', formats=formats, dpi=dpi)

    fig, axes = create_subplots(3, 1, figsize=(10.0, 7.8), sharex=True)
    for ax, b in zip(axes, BRANCHES):
        ax.step(t, results[b]['mode_code'], where='post', color=PALETTE['blue_main'])
        ax.set_yticks(list(MODE_CODE.values()), list(MODE_CODE.keys())); ax.set_ylabel(LABELS[b])
    axes[-1].set_xlabel('Time [s]'); axes[0].set_title('Reaction-wheel AOCS operational modes by branch')
    finalize_figure(fig, plots / 'rw_aocs_mode_timeline_by_branch', formats=formats, dpi=dpi)

    ac = global_cfg.get('animation', {}); sac = cfg.get('animation', {})
    if bool(ac.get('enabled', False)) and bool(sac.get('enabled', True)):
        ad = out / 'animations'; ad.mkdir(exist_ok=True)
        kwargs = {k: v for k, v in ac.items() if k not in ('enabled', 'phase4', 'phase5')}
        kwargs.update(body_pointing_axis=axis, sun_i=sun_series, mag_i=B_i, show_sun_pointing_error=True, eclipse_mask=eclipse,
                      position_eci_m=orbit_states[:, :3], velocity_eci_mps=orbit_states[:, 3:], earth_epoch_utc=str(cfg.get('illumination', {}).get('epoch_utc', '2026-01-01T00:00:00+00:00')))
        for b in BRANCHES:
            r = results[b]
            create_orbit_attitude_animation(
                ad / f'{b}_reaction_wheel_sun_pointing.html', t, r['truth'][:, :4], r['qhat'],
                altitude_m=float(orbit_meta['altitude_m']), inclination_deg=float(orbit_meta['inclination_deg']),
                raan_deg=float(orbit_meta['raan_deg']), phase_deg=float(orbit_meta.get('phase_deg', 0.0)),
                fault_name='none' if b == 'healthy_mekf' else str(cfg.get('fault_type', 'sun_loss')),
                severity='none' if b == 'healthy_mekf' else str(cfg.get('fault_severity', 'medium')),
                fault_start_s=fs if b != 'healthy_mekf' else None, fault_end_s=fe if b != 'healthy_mekf' else None,
                title=LABELS[b] + ' — reaction-wheel + magnetorquer desaturation AOCS', mode_series=r['mode'],
                body_rate_deg_s=np.degrees(np.linalg.norm(r['truth'][:, 4:], axis=1)), **kwargs
            )

    return {
        'scenario': 'reaction_wheel_sun_pointing_comparison', 'branches': 3,
        'max_pointing_rmse_deg': float(summary['pointing_rmse_all_deg'].max()),
        'virtual_sensor_validation_rmse_deg': float(vm['validation_angular_rmse_deg'])
    }
