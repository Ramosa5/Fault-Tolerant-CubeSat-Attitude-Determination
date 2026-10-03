from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

from src.simulation.attitude import (
    propagate_attitude_controlled, quat_angle_error_deg, quat_to_dcm,
    quat_multiply, small_angle_quat, quat_normalize,
)
from src.simulation.control import (sun_pointing_initial_quaternion, sun_pointing_torque, sun_pointing_error_deg,
    desired_sun_orbit_dcm, full_attitude_pd_torque, full_attitude_error_vector_body, full_attitude_hold_torque)
from src.simulation.orbit_model import propagate_circular_two_body
from src.simulation.orbit_profiles import resolve_orbit_profile
from src.simulation.illumination import sun_direction_series, eclipse_geometry
from src.simulation.adcs_modes import (
    SunPointingModeManager, DETUMBLING, ECLIPSE_RATE_DAMPING, SUN_REACQUISITION, SUN_POINTING, SUN_ACQUISITION,
    MODE_CODE, control_enabled, sun_pointing_control_enabled, magnetic_rate_damping_enabled,
)
from src.simulation.spacecraft import inertia_from_config
from src.simulation.magnetic_field import magnetic_field_eci
from src.simulation.magnetorquer import (dipole_for_desired_torque, magnetic_torque, first_order_dipole_response, bdot_dipole_command)
from src.simulation.disturbances import gravity_gradient_torque_body
from src.simulation.magnetic_ltv import authority_scheduled_dipole
from src.estimation.mekf import MEKF
from src.ml.virtual_sensor import train_virtual_sun_model
from src.visualization.publication import PALETTE, create_subplots, finalize_figure, add_fault_span
from src.visualization.animation import create_orbit_attitude_animation
from src.scenarios.sun_pointing_comparison import _apply_sun_fault, _settling_time, BRANCHES, LABELS


def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v/n if n else v


def _parse_epoch(text):
    return datetime.fromisoformat(str(text).replace('Z','+00:00'))


def _sensor_measurement(q, omega, gyro_bias, gyro_noise, sun_noise, mag_noise,
                        sun_i, B_i_t, sun_visible=True):
    C = quat_to_dcm(q)
    g = np.asarray(omega) + np.asarray(gyro_bias) + np.asarray(gyro_noise)
    s = _unit(C.T @ _unit(sun_i) + sun_noise) if sun_visible else np.full(3, np.nan)
    mag_ref_i = _unit(B_i_t)
    m = _unit(C.T @ mag_ref_i + mag_noise)
    return g, s, m


def _make_environment(t, cfg, global_cfg):
    illum = cfg.get('illumination', {})
    sun_model = str(illum.get('sun_model', 'analytic'))
    epoch_text = str(illum.get('epoch_utc', '2026-01-01T00:00:00+00:00'))
    sun_dir, sun_pos = sun_direction_series(t, sun_model, epoch=epoch_text,
                                             constant_vector=cfg.get('sun_vector_eci',[1,.2,.1]))
    oc = resolve_orbit_profile(str(cfg.get('orbit_profile','global')), global_cfg['orbit'], epoch_utc=epoch_text,
                               high_beta_cfg=cfg.get('high_beta_orbit',{}))
    states = propagate_circular_two_body(t, float(oc['altitude_m']), float(oc['inclination_deg']),
                                         float(oc['raan_deg']), float(oc['phase_deg']))
    if bool(illum.get('eclipse_enabled', True)):
        eclipse_state, illumination = eclipse_geometry(states[:, :3], sun_pos)
    else:
        eclipse_state = np.zeros(len(t), int); illumination = np.ones(len(t), float)
    mag_cfg = cfg.get('magnetic_field', {})
    B_i = magnetic_field_eci(states[:, :3], t, epoch_utc=_parse_epoch(epoch_text),
                             model=str(mag_cfg.get('model','igrf14')))
    return sun_dir, sun_pos, states, eclipse_state, illumination, B_i, oc


def _one_step_real(dt, state, inertia_diag, dipole_actual_am2, B_i_t, position_eci_m,
                   gravity_gradient_enabled=True):
    def torque_cb(tt, q, w):
        B_body = quat_to_dcm(q).T @ B_i_t
        tau = magnetic_torque(dipole_actual_am2, B_body)
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
                orbit_states, eclipse_state, noise, predictor=None):
    sc = global_cfg['sensors']; dt=float(t[1]-t[0]); n=len(t)
    spacecraft = cfg.get('spacecraft', {})
    inertia = inertia_from_config(spacecraft)
    mtq_cfg = cfg.get('magnetorquers', {})
    max_dipole = np.asarray(mtq_cfg.get('max_dipole_am2',[0.4,0.4,0.4]), float)
    torquer_tau = float(mtq_cfg.get('time_constant_s', .0065))
    gravity_gradient = bool(cfg.get('disturbances',{}).get('gravity_gradient', True))
    ctrl = cfg.get('controller', {})
    controller_type=str(ctrl.get('type','projected_pd')).lower()
    target_mode=str(ctrl.get('target_mode','sun_orbit')).lower()
    ltv=ctrl.get('ltv',{})
    ltv_horizon_s=float(ltv.get('horizon_s',120.0)); ltv_step_s=float(ltv.get('step_s',5.0)); ltv_update_s=float(ltv.get('update_interval_s',2.0))
    ltv_regularization=float(ltv.get('regularization',0.35)); ltv_max_comp=float(ltv.get('max_compensation',2.0))
    # Backward-compatible global defaults plus mode-specific gains.
    base_kp=float(ctrl.get('kp',8e-6)); base_kd=float(ctrl.get('kd',3e-4)); base_limit=float(ctrl.get('desired_torque_limit_nm',2.5e-5))
    acq=ctrl.get('acquisition', {})
    point=ctrl.get('pointing', {})
    reacq=ctrl.get('reacquisition', acq)
    det=ctrl.get('detumbling', {})
    acq_kp=float(acq.get('kp',base_kp)); acq_kd=float(acq.get('kd',max(base_kd,8e-4))); acq_limit=float(acq.get('desired_torque_limit_nm',base_limit))
    capture_error_deg=float(acq.get('capture_error_deg',8.0)); capture_kp=float(acq.get('capture_kp',2.0e-6)); capture_kd=float(acq.get('capture_kd',4.0e-3)); capture_limit=float(acq.get('capture_torque_limit_nm',1.2e-5))
    point_kp=float(point.get('kp',max(base_kp*0.35,1e-8))); point_kd=float(point.get('kd',max(base_kd,1.0e-3))); point_limit=float(point.get('desired_torque_limit_nm',base_limit*0.6))
    hold=ctrl.get('hold', point)
    hold_kp=float(hold.get('kp',point_kp)); hold_kd=float(hold.get('kd',max(point_kd,3.0e-3))); hold_ki=float(hold.get('ki',2.0e-8))
    hold_limit=float(hold.get('desired_torque_limit_nm',point_limit)); hold_integral_limit=float(hold.get('integral_limit_rad_s',20.0))
    hold_integral_leak=float(hold.get('integral_leak_per_s',0.01)); hold_rate_priority=float(np.deg2rad(float(hold.get('rate_priority_deg_s',0.04)))); hold_error_weights=np.asarray(hold.get('error_weights',[0.1,1.0,1.0]),float); hold_objective=str(hold.get('objective','sun_primary')).lower(); hold_rate_only_above=float(np.deg2rad(float(hold.get('rate_only_above_deg_s',0.02)))); hold_rate_damping_kd=float(hold.get('rate_damping_kd',2.0e-2))
    reacq_kp=float(reacq.get('kp',acq_kp)); reacq_kd=float(reacq.get('kd',acq_kd)); reacq_limit=float(reacq.get('desired_torque_limit_nm',acq_limit))
    damping_method=str(det.get('method','rate_damping')).lower()
    damping_kd=float(det.get('rate_damping_kd_nm_per_rad_s',3.0e-3))
    damping_tau_limit=float(det.get('rate_damping_torque_limit_nm',1.0e-5))
    bdot_gain=float(det.get('bdot_gain_am2_s_per_t',2.0e5))
    bdot_deadband=float(det.get('bdot_deadband_t_per_s',0.0))
    bdot_filter_tau=float(det.get('bdot_filter_tau_s',2.0))
    axis=str(cfg.get('pointing_axis','x'))
    fs=float(cfg.get('fault_start_s',250.)); fe=float(cfg.get('fault_end_s',450.))
    ft=str(cfg.get('fault_type','sun_loss')); sev=str(cfg.get('fault_severity','medium'))
    eclipse_as_loss=bool(cfg.get('illumination',{}).get('eclipse_sun_sensor_unavailable',True))
    mode_cfg=cfg.get('modes',{})
    rate_recovery_trigger=float(mode_cfg.get('rate_recovery_trigger_deg_s',0.20))
    mgr=SunPointingModeManager(threshold_deg=float(mode_cfg.get('pointing_threshold_deg',2.0)),
                               hold_s=float(mode_cfg.get('pointing_hold_s',20.0)), dt_s=dt,
                               rate_threshold_deg_s=float(mode_cfg.get('rate_threshold_deg_s',0.05)),
                               rate_hold_s=float(mode_cfg.get('rate_hold_s',10.0)),
                               rate_recovery_trigger_deg_s=rate_recovery_trigger,
                               pointing_exit_threshold_deg=float(mode_cfg.get('pointing_exit_threshold_deg',4.0)), mode=DETUMBLING)
    filt=MEKF(q0=q0, gyro_noise_std=float(sc['gyro_noise_std_rad_s']), vector_noise_std=float(sc['vector_noise_std']))

    truth=np.zeros((n,7)); qhat=np.zeros((n,4)); bhat=np.zeros((n,3))
    gyro=np.zeros((n,3)); sun_raw=np.full((n,3),np.nan); sun_used=np.full((n,3),np.nan); mag=np.zeros((n,3))
    dip_cmd=np.zeros((n,3)); dip_actual=np.zeros((n,3)); tau_des=np.zeros((n,3)); tau_mtq=np.zeros((n,3)); tau_gg=np.zeros((n,3)); bdot=np.zeros((n,3))
    ml_used=np.zeros(n,bool); ml_candidate=np.zeros(n,bool); ml_rejected=np.zeros(n,bool); sensor_faulted=np.zeros(n,bool)
    mode=np.empty(n,dtype=object); mode_code=np.zeros(n,int); control_on=np.zeros(n,bool); rate_damping_on=np.zeros(n,bool)
    truth[0,:4]=q0; truth[0,4:]=omega0; qhat[0]=filt.q; bhat[0]=filt.b
    visible0=not(eclipse_as_loss and eclipse_state[0]==2)
    gyro[0],sun_raw[0],mag[0]=_sensor_measurement(q0,omega0,sc['gyro_bias_rad_s'],noise['gyro'][0],noise['sun'][0],noise['mag'][0],sun_series[0],B_i_series[0],visible0)
    # B-dot uses the body-frame magnetometer measurement (with physical field magnitude),
    # low-pass filtered before differentiation to avoid amplifying sensor noise.
    B_ctrl_filtered = mag[0] * np.linalg.norm(B_i_series[0])
    sun_used[0]=sun_raw[0]
    anchor=sun_raw[0].copy() if np.all(np.isfinite(sun_raw[0])) else quat_to_dcm(qhat[0]).T@sun_series[0]
    initial_est=sun_pointing_error_deg(qhat[0],axis,sun_series[0])
    initial_rate_deg_s=float(np.degrees(np.linalg.norm(gyro[0]-filt.b)))
    mode[0]=mgr.update(in_full_eclipse=bool(eclipse_state[0]==2),real_sun_available=bool(np.all(np.isfinite(sun_raw[0]))),estimated_pointing_error_deg=initial_est,estimated_rate_deg_s=initial_rate_deg_s)
    mode_code[0]=MODE_CODE[mode[0]]; control_on[0]=sun_pointing_control_enabled(mode[0]); rate_damping_on[0]=magnetic_rate_damping_enabled(mode[0])
    frng=np.random.default_rng(int(cfg.get('fault_seed',4501)))
    ltv_future_cache=None; ltv_next_update=0
    hold_integral=np.zeros(3); previous_mode=mode[0]

    for k in range(1,n):
        sun_i=sun_series[k-1]; B_i=B_i_series[k-1]
        omega_est=gyro[k-1]-bhat[k-1]
        B_body_est=quat_to_dcm(qhat[k-1]).T@B_i
        # Continuously maintain a filtered magnetometer signal so eclipse/detumble
        # entry does not create a derivative spike.
        B_meas_for_bdot = mag[k-1] * np.linalg.norm(B_i_series[k-1])
        B_prev_filtered = B_ctrl_filtered.copy()
        if bdot_filter_tau <= 0:
            B_ctrl_filtered = B_meas_for_bdot.copy()
        else:
            alpha_b = 1.0 - np.exp(-dt / bdot_filter_tau)
            B_ctrl_filtered = B_ctrl_filtered + alpha_b * (B_meas_for_bdot - B_ctrl_filtered)
        if sun_pointing_control_enabled(mode[k-1]):
            hold_active = mode[k-1] == SUN_POINTING
            if hold_active:
                kp, kd, desired_tau_limit = hold_kp, hold_kd, hold_limit
            elif mode[k-1] == SUN_REACQUISITION:
                kp, kd, desired_tau_limit = reacq_kp, reacq_kd, reacq_limit
            else:
                kp, kd, desired_tau_limit = acq_kp, acq_kd, acq_limit
            # Deployment detumbling is never re-entered after the initial release.
            # If acquisition/reacquisition has excessive body rate, retain the Sun-
            # seeking mode but temporarily strengthen the damping term.  This avoids
            # repeatedly abandoning Sun reacquisition while still removing the
            # angular momentum that would otherwise cause another overshoot.
            current_rate_deg_s = float(np.degrees(np.linalg.norm(omega_est)))
            current_point_deg = sun_pointing_error_deg(qhat[k-1],axis,sun_i)
            if mode[k-1] in (SUN_ACQUISITION, SUN_REACQUISITION) and current_point_deg <= capture_error_deg:
                # Capture/braking region: once close to the Sun, stop accelerating
                # toward the target and remove angular momentum before hold handover.
                kp = min(kp, capture_kp); kd = max(kd, capture_kd); desired_tau_limit = min(desired_tau_limit, capture_limit)
            if mode[k-1] in (SUN_ACQUISITION, SUN_REACQUISITION) and current_rate_deg_s > rate_recovery_trigger:
                kd = max(kd, damping_kd)
            if target_mode == 'sun_orbit':
                Cdes = desired_sun_orbit_dcm(sun_i,orbit_states[k-1,:3],orbit_states[k-1,3:],axis)
                if hold_active:
                    if np.linalg.norm(omega_est) > hold_rate_only_above:
                        # First remove residual angular momentum; this prevents the
                        # spacecraft from flying through the Sun target after acquisition.
                        tau_des[k-1] = -hold_rate_damping_kd*np.asarray(omega_est,float)
                        tn=np.linalg.norm(tau_des[k-1])
                        if tn>hold_limit and tn>0:
                            tau_des[k-1]*=hold_limit/tn
                        att_e = full_attitude_error_vector_body(qhat[k-1],Cdes)
                        hold_integral *= max(0.0, 1.0-5.0*hold_integral_leak*dt)
                    elif hold_objective == 'sun_primary':
                        # Primary mission objective: keep the designated body axis on the Sun.
                        tau_des[k-1] = sun_pointing_torque(qhat[k-1],omega_est,axis,sun_i,hold_kp,hold_kd,hold_limit)
                        att_e = full_attitude_error_vector_body(qhat[k-1],Cdes)
                        hold_integral *= max(0.0, 1.0-5.0*hold_integral_leak*dt)
                    else:
                        # Optional full-attitude PI-D hold for experiments where roll is equally important.
                        e_now = full_attitude_error_vector_body(qhat[k-1],Cdes)
                        hold_integral *= max(0.0, 1.0-hold_integral_leak*dt)
                        hold_integral += e_now*dt
                        hi=np.linalg.norm(hold_integral)
                        if hi>hold_integral_limit and hi>0: hold_integral*=hold_integral_limit/hi
                        tau_des[k-1], att_e = full_attitude_hold_torque(qhat[k-1],omega_est,Cdes,hold_integral,
                            hold_kp,hold_kd,hold_ki,hold_limit,hold_rate_priority,hold_error_weights)
                else:
                    hold_integral *= max(0.0, 1.0-5.0*hold_integral_leak*dt)
                    tau_des[k-1], att_e = full_attitude_pd_torque(qhat[k-1],omega_est,Cdes,kp,kd,desired_tau_limit)
            else:
                Cdes = None
                att_e = None
                tau_des[k-1]=sun_pointing_torque(qhat[k-1],omega_est,axis,sun_i,kp,kd,desired_tau_limit)
            if controller_type == 'magnetic_ltv':
                if Cdes is None:
                    # LTV regulator requires a complete attitude target; promote the
                    # operational target to the Sun+orbit-normal frame.
                    Cdes = desired_sun_orbit_dcm(sun_i,orbit_states[k-1,:3],orbit_states[k-1,3:],axis)
                if att_e is None:
                    att_e = full_attitude_error_vector_body(qhat[k-1],Cdes)
                if ltv_future_cache is None or (k-1) >= ltv_next_update:
                    stride=max(1,int(round(ltv_step_s/dt))); nh=max(2,int(np.ceil(ltv_horizon_s/ltv_step_s)))
                    inds=np.minimum((k-1)+stride*np.arange(nh),n-1)
                    future_B=[]
                    for j in inds:
                        Cdj=desired_sun_orbit_dcm(sun_series[j],orbit_states[j,:3],orbit_states[j,3:],axis)
                        future_B.append(Cdj.T@B_i_series[j])
                    ltv_future_cache=np.asarray(future_B)
                    ltv_next_update=(k-1)+max(1,int(round(ltv_update_s/dt)))
                dip_cmd[k-1], tau_sched, _ = authority_scheduled_dipole(tau_des[k-1],B_body_est,ltv_future_cache,max_dipole,
                    torque_limit_nm=desired_tau_limit,regularization=ltv_regularization,max_compensation=ltv_max_comp)
            else:
                dip_cmd[k-1], _ = dipole_for_desired_torque(tau_des[k-1],B_body_est,max_dipole)
        elif magnetic_rate_damping_enabled(mode[k-1]):
            if damping_method == 'bdot':
                # Classical B-dot is retained for dedicated detumbling experiments.
                dip_cmd[k-1], bdot[k-1] = bdot_dipole_command(B_ctrl_filtered,B_prev_filtered,dt,bdot_gain,max_dipole,bdot_deadband)
                tau_des[k-1]=0.0
            else:
                # Default: gyro-based magnetic rate damping.  The desired damping
                # torque is still projected through the real m x B actuator geometry.
                tau_des[k-1] = -damping_kd * np.asarray(omega_est,float)
                tn=np.linalg.norm(tau_des[k-1])
                if tn > damping_tau_limit and tn > 0:
                    tau_des[k-1] *= damping_tau_limit/tn
                dip_cmd[k-1], _ = dipole_for_desired_torque(tau_des[k-1],B_body_est,max_dipole)
        else:
            dip_cmd[k-1]=0.0; tau_des[k-1]=0.0
        dip_actual[k-1]=first_order_dipole_response(dip_actual[k-2] if k>1 else np.zeros(3),dip_cmd[k-1],dt,torquer_tau)
        truth[k],tau_mtq[k-1],tau_gg[k-1]=_one_step_real(dt,truth[k-1],inertia,dip_actual[k-1],B_i,orbit_states[k-1,:3],gravity_gradient)

        sun_visible=not(eclipse_as_loss and eclipse_state[k]==2)
        gyro[k],sun_raw[k],mag[k]=_sensor_measurement(truth[k,:4],truth[k,4:],sc['gyro_bias_rad_s'],noise['gyro'][k],noise['sun'][k],noise['mag'][k],sun_series[k],B_i_series[k],sun_visible)
        fault_active=(t[k]>=fs and t[k]<=fe)
        measured=sun_raw[k].copy()
        if branch!='healthy_mekf':
            measured,faulted=_apply_sun_fault(measured,ft,sev,fault_active,frng)
            sensor_faulted[k]=faulted or (fault_active and ft=='sun_noise')

        physical_sun_visible=not(eclipse_as_loss and eclipse_state[k]==2)
        needs_recovery=physical_sun_visible and (not np.all(np.isfinite(measured)) or (branch=='ml_recovery' and fault_active and ft=='sun_noise'))
        if branch=='ml_recovery' and needs_recovery:
            omega_virtual=gyro[k-1]-bhat[k-1]
            candidate=predictor.predict(anchor,omega_virtual,mag[k],dt); ml_candidate[k]=True
            qpred=quat_normalize(quat_multiply(qhat[k-1],small_angle_quat(omega_virtual*dt)))
            predicted_sun=quat_to_dcm(qpred).T@sun_series[k]
            disagreement=np.degrees(np.arccos(np.clip(np.dot(_unit(candidate),_unit(predicted_sun)),-1,1)))
            if disagreement<=float(cfg.get('ml_gate_deg',2.0)):
                blend=float(cfg.get('ml_blend',.25)); measured=_unit((1-blend)*predicted_sun+blend*candidate)
                ml_used[k]=True; anchor=measured.copy()
            else:
                measured=np.full(3,np.nan); ml_rejected[k]=True; anchor=_unit(candidate)
        elif np.all(np.isfinite(measured)):
            anchor=measured.copy()
        sun_used[k]=measured
        sun_std=float(cfg.get('ml_virtual_sensor_noise_std',.01)) if ml_used[k] else None
        q,b,_=filt.step(gyro[k-1],sun_used[k],mag[k],sun_series[k],_unit(B_i_series[k]),dt,sun_noise_std=sun_std)
        qhat[k]=q; bhat[k]=b
        est_point=sun_pointing_error_deg(qhat[k],axis,sun_series[k])
        real_sun_available=bool(np.all(np.isfinite(measured))) and not bool(ml_used[k])
        est_rate_deg_s=float(np.degrees(np.linalg.norm(gyro[k]-bhat[k])))
        mode[k]=mgr.update(in_full_eclipse=bool(eclipse_state[k]==2),real_sun_available=real_sun_available,estimated_pointing_error_deg=est_point,estimated_rate_deg_s=est_rate_deg_s)
        if mode[k] != previous_mode:
            # Integral action belongs only to steady Sun hold.  Reset it whenever
            # we leave hold so eclipse/acquisition errors cannot wind it up.
            if mode[k] != SUN_POINTING:
                hold_integral[:] = 0.0
            previous_mode = mode[k]
        mode_code[k]=MODE_CODE[mode[k]]; control_on[k]=sun_pointing_control_enabled(mode[k]); rate_damping_on[k]=magnetic_rate_damping_enabled(mode[k])

    dip_cmd[-1]=dip_cmd[-2]; dip_actual[-1]=dip_actual[-2]; tau_des[-1]=tau_des[-2]; tau_mtq[-1]=tau_mtq[-2]; tau_gg[-1]=tau_gg[-2]; bdot[-1]=bdot[-2]
    true_point=np.array([sun_pointing_error_deg(q,axis,s) for q,s in zip(truth[:,:4],sun_series)])
    est_point=np.array([sun_pointing_error_deg(q,axis,s) for q,s in zip(qhat,sun_series)])
    att_err=np.array([quat_angle_error_deg(qe,qt) for qe,qt in zip(qhat,truth[:,:4])])
    sun_rec_err=np.full(n,np.nan)
    for k in range(n):
        if np.all(np.isfinite(sun_used[k])):
            target=_unit(quat_to_dcm(truth[k,:4]).T@sun_series[k])
            sun_rec_err[k]=np.degrees(np.arccos(np.clip(np.dot(_unit(sun_used[k]),target),-1,1)))
    return dict(truth=truth,qhat=qhat,bhat=bhat,gyro=gyro,sun_raw=sun_raw,sun_used=sun_used,mag=mag,
                dip_cmd=dip_cmd,dip_actual=dip_actual,tau_des=tau_des,tau_mtq=tau_mtq,tau_gg=tau_gg,bdot=bdot,
                ml_used=ml_used,ml_candidate=ml_candidate,ml_rejected=ml_rejected,sensor_faulted=sensor_faulted,
                mode=mode,mode_code=mode_code,control_on=control_on,rate_damping_on=rate_damping_on,true_point=true_point,est_point=est_point,
                att_err=att_err,sun_rec_err=sun_rec_err,inertia=inertia,controller_type=controller_type,target_mode=target_mode)


def run(cfg, global_cfg, out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); plots=out/'plots'; plots.mkdir(exist_ok=True); models=out/'models'; models.mkdir(exist_ok=True)
    duration=float(cfg.get('duration_s',6000.)); dt=float(cfg.get('dt_s',.1)); t=np.arange(0,duration+dt/2,dt)
    sun_series,sun_pos,orbit_states,eclipse_state,illumination,B_i,orbit_meta=_make_environment(t,cfg,global_cfg)
    axis=str(cfg.get('pointing_axis','x')); q0=sun_pointing_initial_quaternion(axis,sun_series[0],float(cfg.get('initial_error_deg',30.)))
    omega0=np.asarray(cfg.get('omega0_rad_s',[0,0,0]),float)
    sc=global_cfg['sensors']; rng=np.random.default_rng(int(cfg.get('comparison_seed',7301))); n=len(t)
    noise={'gyro':rng.normal(0,float(sc['gyro_noise_std_rad_s']),(n,3)),
           'sun':rng.normal(0,float(sc['vector_noise_std']),(n,3)),
           'mag':rng.normal(0,float(sc['vector_noise_std']),(n,3))}
    vc=cfg.get('virtual_sensor',{})
    predictor,vm=train_virtual_sun_model(models/'virtual_sun_sensor',runs=int(vc.get('train_runs',24)),duration_s=float(vc.get('train_duration_s',40.)),dt=dt,
                                         hidden=tuple(vc.get('hidden',[48,24])),epochs=int(vc.get('epochs',18)),batch_size=int(vc.get('batch_size',256)),seed=int(vc.get('seed',8128)),
                                         gyro_noise_std=float(sc['gyro_noise_std_rad_s']),vector_noise_std=float(sc['vector_noise_std']))
    results={b:_run_branch(b,cfg,global_cfg,t,q0,omega0,sun_series,B_i,orbit_states,eclipse_state,noise,predictor if b=='ml_recovery' else None) for b in BRANCHES}

    fs=float(cfg.get('fault_start_s',250.)); fe=float(cfg.get('fault_end_s',450.)); active=(t>=fs)&(t<=fe); eclipse=eclipse_state==2; sunlight=~eclipse
    mode_cfg=cfg.get('modes',{}); th=float(mode_cfg.get('pointing_threshold_deg',2.)); hold=float(mode_cfg.get('pointing_hold_s',20.)); rate_th=float(mode_cfg.get('rate_threshold_deg_s',0.05)); rate_hold=float(mode_cfg.get('rate_hold_s',10.0))
    Bmag=np.linalg.norm(B_i,axis=1)
    rows=[]; frames=[]
    for b,r in results.items():
        def rmse(mask,x): return float(np.sqrt(np.mean(np.asarray(x)[mask]**2))) if np.any(mask) else np.nan
        rows.append({
            'branch':b,'label':LABELS[b],'fault_type':'none' if b=='healthy_mekf' else cfg.get('fault_type','sun_loss'),
            'spacecraft_mass_kg':float(cfg['spacecraft']['mass_kg']),'controller_type':r['controller_type'],'orbit_profile':orbit_meta['profile'],'orbit_beta_deg':orbit_meta['beta_angle_deg'],
            'inertia_x_kg_m2':r['inertia'][0],'inertia_y_kg_m2':r['inertia'][1],'inertia_z_kg_m2':r['inertia'][2],
            'pointing_rmse_all_deg':rmse(np.ones(n,bool),r['true_point']),'pointing_rmse_fault_deg':rmse(active,r['true_point']),
            'pointing_rmse_eclipse_deg':rmse(eclipse,r['true_point']),'attitude_rmse_all_deg':rmse(np.ones(n,bool),r['att_err']),
            'attitude_rmse_eclipse_deg':rmse(eclipse,r['att_err']),'final_pointing_error_deg':float(r['true_point'][-1]),
            'initial_acquisition_time_s':_settling_time(t,r['true_point'],th,hold),
            'max_dipole_used_am2':float(np.max(np.linalg.norm(r['dip_actual'],axis=1))),
            'max_magnetic_torque_nm':float(np.max(np.linalg.norm(r['tau_mtq'],axis=1))),
            'mean_field_uT':float(np.mean(Bmag)*1e6),'min_field_uT':float(np.min(Bmag)*1e6),'max_field_uT':float(np.max(Bmag)*1e6),'eclipse_fraction_pct':float(100*np.mean(eclipse)),
            'detumbling_duration_s':float(np.sum(r['mode']==DETUMBLING)*dt),'eclipse_rate_damping_duration_s':float(np.sum(r['mode']==ECLIPSE_RATE_DAMPING)*dt),'reacquisition_duration_s':float(np.sum(r['mode']==SUN_REACQUISITION)*dt),
            'final_body_rate_deg_s':float(np.degrees(np.linalg.norm(r['truth'][-1,4:]))),'max_body_rate_deg_s':float(np.max(np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)))),
            'ml_recovery_usage_pct_fault':float(100*np.mean(r['ml_used'][active])) if np.any(active) else 0.,
        })
        frames.append(pd.DataFrame({
            'time_s':t,'branch':b,'sun_eci_x':sun_series[:,0],'sun_eci_y':sun_series[:,1],'sun_eci_z':sun_series[:,2],
            'B_eci_x_T':B_i[:,0],'B_eci_y_T':B_i[:,1],'B_eci_z_T':B_i[:,2],'B_magnitude_uT':Bmag*1e6,
            'eclipse_state':eclipse_state,'illumination_fraction':illumination,'true_sun_pointing_error_deg':r['true_point'],
            'estimated_sun_pointing_error_deg':r['est_point'],'attitude_estimation_error_deg':r['att_err'],
            'adcs_mode':r['mode'],'adcs_mode_code':r['mode_code'],'sun_pointing_control_enabled':r['control_on'].astype(int),'magnetic_rate_damping_enabled':r['rate_damping_on'].astype(int),
            'dipole_cmd_x_Am2':r['dip_cmd'][:,0],'dipole_cmd_y_Am2':r['dip_cmd'][:,1],'dipole_cmd_z_Am2':r['dip_cmd'][:,2],
            'dipole_actual_x_Am2':r['dip_actual'][:,0],'dipole_actual_y_Am2':r['dip_actual'][:,1],'dipole_actual_z_Am2':r['dip_actual'][:,2],
            'mtq_torque_x_Nm':r['tau_mtq'][:,0],'mtq_torque_y_Nm':r['tau_mtq'][:,1],'mtq_torque_z_Nm':r['tau_mtq'][:,2],
            'gravity_gradient_x_Nm':r['tau_gg'][:,0],'gravity_gradient_y_Nm':r['tau_gg'][:,1],'gravity_gradient_z_Nm':r['tau_gg'][:,2],
            'omega_x_rad_s':r['truth'][:,4],'omega_y_rad_s':r['truth'][:,5],'omega_z_rad_s':r['truth'][:,6],
            'body_rate_deg_s':np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),
            'bdot_x_T_s':r['bdot'][:,0],'bdot_y_T_s':r['bdot'][:,1],'bdot_z_T_s':r['bdot'][:,2],
            'ml_recovery_active':r['ml_used'].astype(int),'sensor_fault_active':r['sensor_faulted'].astype(int),
        }))
    summary=pd.DataFrame(rows); summary.to_csv(out/'magnetic_sun_pointing_comparison_summary.csv',index=False)
    pd.concat(frames).to_csv(out/'magnetic_sun_pointing_comparison_timeseries.csv',index=False)

    eidx=np.where(eclipse)[0]
    meta={'scenario':'magnetic_sun_pointing_comparison','branches':list(BRANCHES),'spacecraft':cfg.get('spacecraft',{}),'magnetorquers':cfg.get('magnetorquers',{}),
          'controller':cfg.get('controller',{}),'orbit':orbit_meta,'modes':cfg.get('modes',{}),'magnetic_field':cfg.get('magnetic_field',{}),'disturbances':cfg.get('disturbances',{}),
          'first_eclipse_s':float(t[eidx[0]]) if len(eidx) else None,'last_eclipse_s':float(t[eidx[-1]]) if len(eidx) else None,'virtual_sensor':vm}
    (out/'scenario_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    formats=tuple(global_cfg['visualization']['formats']); dpi=int(global_cfg['visualization']['dpi']); colors=[PALETTE['green_3'],PALETTE['red_strong'],PALETTE['blue_main']]
    fig,axes=create_subplots(2,1,figsize=(9.8,7.0),sharex=True)
    axes[0].plot(t,Bmag*1e6,color=PALETTE['teal']); axes[0].set_ylabel('|B| [µT]'); axes[0].set_title('IGRF geomagnetic field along the CubeSat orbit')
    for i,lab in enumerate('xyz'): axes[1].plot(t,B_i[:,i]*1e6,label=f'B ECI {lab}')
    axes[1].set_ylabel('Field component [µT]'); axes[1].set_xlabel('Time [s]'); axes[1].legend(ncol=3)
    finalize_figure(fig,plots/'environment_geomagnetic_field_magnitude_and_eci_components',formats=formats,dpi=dpi)

    fig,axes=create_subplots(figsize=(9.8,5.2)); ax=axes[0]
    for b,c in zip(BRANCHES,colors): ax.plot(t,results[b]['true_point'],label=LABELS[b],color=c)
    add_fault_span(ax,fs,fe)
    if np.any(eclipse):
        starts=np.where(np.diff(np.r_[False,eclipse])==1)[0]; ends=np.where(np.diff(np.r_[eclipse,False])==-1)[0]
        for s,e in zip(starts,ends): ax.axvspan(t[s],t[e],color='0.25',alpha=.12)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('True Sun-pointing error [deg]'); ax.set_title('Magnetorquer-only detumbling, Sun acquisition, eclipse rate damping, and reacquisition'); ax.legend()
    finalize_figure(fig,plots/'real_aocs_sun_pointing_error_healthy_faulty_ml',formats=formats,dpi=dpi)

    fig,axes=create_subplots(2,1,figsize=(9.8,7.0),sharex=True)
    r=results['healthy_mekf']
    for i,lab in enumerate('xyz'): axes[0].plot(t,r['dip_actual'][:,i],label=f'm{lab}')
    axes[0].set_ylabel('Dipole [A m²]'); axes[0].set_title('Mode-dependent magnetorquer actuation during healthy branch'); axes[0].legend(ncol=3)
    axes[1].plot(t,np.linalg.norm(r['tau_mtq'],axis=1)*1e6,label='Magnetic torque')
    axes[1].plot(t,np.linalg.norm(r['tau_gg'],axis=1)*1e6,label='Gravity-gradient torque')
    axes[1].set_ylabel('Torque [µN m]'); axes[1].set_xlabel('Time [s]'); axes[1].legend()
    finalize_figure(fig,plots/'real_aocs_magnetorquer_dipole_and_physical_torques',formats=formats,dpi=dpi)

    fig,axes=create_subplots(figsize=(9.8,5.2)); ax=axes[0]
    rate_deg=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1))
    ax.plot(t,rate_deg,color=PALETTE['violet'],label='Body-rate magnitude')
    ax.axhline(rate_th,color=PALETTE['red_strong'],linestyle='--',label=f'Pointing-mode rate threshold ({rate_th:g}°/s)')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Body-rate magnitude [deg/s]')
    ax.set_title('Healthy branch body-rate damping and mode-transition threshold'); ax.legend()
    finalize_figure(fig,plots/'real_aocs_body_rate_magnitude_and_transition_threshold',formats=formats,dpi=dpi)

    fig,axes=create_subplots(3,1,figsize=(9.8,7.8),sharex=True)
    for ax,b in zip(axes,BRANCHES):
        ax.step(t,results[b]['mode_code'],where='post',color=PALETTE['blue_main']); ax.set_yticks(list(MODE_CODE.values()),list(MODE_CODE.keys())); ax.set_ylabel(LABELS[b])
    axes[-1].set_xlabel('Time [s]'); axes[0].set_title('Real AOCS operational modes with detumbling and eclipse rate damping')
    finalize_figure(fig,plots/'real_aocs_mode_timeline_by_branch',formats=formats,dpi=dpi)

    ac=global_cfg.get('animation',{}); sac=cfg.get('animation',{})
    if bool(ac.get('enabled',False)) and bool(sac.get('enabled',True)):
        ad=out/'animations'; ad.mkdir(exist_ok=True); oc=orbit_meta; kwargs={k:v for k,v in ac.items() if k not in ('enabled','phase4','phase5')}
        kwargs.update(body_pointing_axis=axis,sun_i=sun_series,mag_i=B_i,show_sun_pointing_error=True,eclipse_mask=eclipse,
                      position_eci_m=orbit_states[:,:3],velocity_eci_mps=orbit_states[:,3:],
                      earth_epoch_utc=str(cfg.get('illumination',{}).get('epoch_utc','2026-01-01T00:00:00+00:00')))
        for b in BRANCHES:
            r=results[b]
            create_orbit_attitude_animation(ad/f'{b}_real_magnetorquer_sun_pointing.html',t,r['truth'][:,:4],r['qhat'],
                altitude_m=oc['altitude_m'],inclination_deg=oc['inclination_deg'],raan_deg=oc['raan_deg'],phase_deg=oc['phase_deg'],
                fault_name='none' if b=='healthy_mekf' else str(cfg.get('fault_type','sun_loss')),severity='none' if b=='healthy_mekf' else str(cfg.get('fault_severity','medium')),
                fault_start_s=fs if b!='healthy_mekf' else None,fault_end_s=fe if b!='healthy_mekf' else None,title=LABELS[b]+' — physical magnetorquer AOCS',mode_series=r['mode'],body_rate_deg_s=np.degrees(np.linalg.norm(r['truth'][:,4:],axis=1)),**kwargs)

    return {'scenario':'magnetic_sun_pointing_comparison','branches':3,'magnetic_field_model':str(cfg.get('magnetic_field',{}).get('model','igrf14')),
            'mean_field_uT':float(np.mean(Bmag)*1e6),'spacecraft_mass_kg':float(cfg['spacecraft']['mass_kg']),'controller_type':str(cfg.get('controller',{}).get('type','projected_pd')),'orbit_profile':orbit_meta['profile'],'orbit_beta_deg':orbit_meta['beta_angle_deg'],
            'virtual_sensor_validation_rmse_deg':float(vm['validation_angular_rmse_deg'])}
