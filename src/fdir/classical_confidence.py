from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.simulation.attitude import quat_to_dcm, quat_multiply, quat_normalize, small_angle_quat


HEALTHY = 'HEALTHY'
DEGRADED = 'DEGRADED'
FAULTY = 'FAULTY'
LOST = 'LOST'
EXPECTED_UNAVAILABLE = 'EXPECTED_UNAVAILABLE'


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return np.full(3, np.nan)
    return v / n


def _angle_deg(a, b):
    a = _unit(a); b = _unit(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _rotation_vector_from_dcm(R):
    R = np.asarray(R, float)
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(tr))
    if angle < 1e-10:
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], float)
    s = np.sin(angle)
    if abs(s) < 1e-10:
        return np.zeros(3)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], float) / (2.0*s)
    return axis * angle


def _quat_from_dcm(R):
    """Scalar-first quaternion from a proper body-to-inertial DCM."""
    R=np.asarray(R,float); tr=float(np.trace(R))
    if tr>0:
        S=np.sqrt(tr+1.0)*2.0; w=0.25*S
        x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        S=np.sqrt(max(1.0+R[0,0]-R[1,1]-R[2,2],0.0))*2.0; S=max(S,1e-12)
        w=(R[2,1]-R[1,2])/S; x=0.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1]>R[2,2]:
        S=np.sqrt(max(1.0+R[1,1]-R[0,0]-R[2,2],0.0))*2.0; S=max(S,1e-12)
        w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=0.25*S; z=(R[1,2]+R[2,1])/S
    else:
        S=np.sqrt(max(1.0+R[2,2]-R[0,0]-R[1,1],0.0))*2.0; S=max(S,1e-12)
        w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=0.25*S
    return quat_normalize(np.array([w,x,y,z],float))


def _triad_dcm_body_to_inertial(sun_body, mag_body, sun_i, mag_i):
    b1=_unit(sun_body); b2=_unit(mag_body); r1=_unit(sun_i); r2=_unit(mag_i)
    cb=np.cross(b1,b2); cr=np.cross(r1,r2)
    if np.linalg.norm(cb)<1e-5 or np.linalg.norm(cr)<1e-5:
        return None
    tb2=_unit(cb); tb3=_unit(np.cross(b1,tb2)); tr2=_unit(cr); tr3=_unit(np.cross(r1,tr2))
    B=np.column_stack((b1,tb2,tb3)); RR=np.column_stack((r1,tr2,tr3))
    return RR@B.T


def _rate_from_dcm_pair(C0,C1,dt):
    if C0 is None or C1 is None or dt<=0: return None
    D=np.asarray(C0,float).T@np.asarray(C1,float)
    return _rotation_vector_from_dcm(D)/float(dt)


def _quality(value, warn, fault):
    if not np.isfinite(value):
        return 0.0
    value = float(max(value, 0.0)); warn = float(max(warn, 0.0)); fault = float(max(fault, warn + 1e-12))
    if value <= warn:
        return 1.0
    if value >= fault:
        return 0.0
    return float((fault - value) / (fault - warn))


@dataclass
class FDIRResult:
    confidence: dict
    state: dict
    score: dict
    diagnostics: dict
    hard_unavailable: dict
    expected_unavailable: dict

    def is_bad(self, sensor: str, reconstruction_threshold: float = 0.25) -> bool:
        if bool(self.expected_unavailable.get(sensor, False)):
            return False
        if bool(self.hard_unavailable.get(sensor, False)):
            return True
        return self.state[sensor] in (FAULTY, LOST) or float(self.confidence[sensor]) < float(reconstruction_threshold)


class ClassicalConfidenceFDIR:
    """Classical confidence-based FDIR v2.

    The detector separates three concepts:
      1) hard sample availability (loss/dropout/missing data),
      2) expected unavailability (Sun sensor in eclipse),
      3) soft degradation/isolation (bias, drift, misalignment, noise).

    The detector does not reconstruct data. It only produces sensor states,
    confidence and reconstruction requests for the downstream fallback / ML layer.
    Injected fault labels are never used.
    """

    def __init__(self, cfg: dict | None = None, dt: float = 0.1):
        c = cfg or {}
        self.dt = float(dt)
        self.alpha = float(np.clip(c.get('confidence_smoothing_alpha', 0.50), 0.0, 1.0))
        self.degraded_threshold = float(c.get('degraded_confidence_threshold', 0.65))
        self.faulty_threshold = float(c.get('faulty_confidence_threshold', 0.45))
        self.healthy_recovery_threshold = float(c.get('healthy_recovery_confidence_threshold', 0.80))
        self.trusted_update_threshold = float(c.get('trusted_reference_update_confidence', 0.95))
        self.fault_persistence = max(1, int(c.get('fault_persistence_samples', 3)))
        self.recovery_persistence = max(1, int(c.get('recovery_persistence_samples', 20)))

        # Hard availability / stuck logic. In this simulator gyro loss/dropout uses
        # an exact fallback vector (default zero), so exact-zero samples are treated
        # as unavailable. This is configurable for real-hardware studies.
        self.gyro_zero_is_unavailable = bool(c.get('gyro_zero_is_unavailable', False))
        self.gyro_zero_threshold = float(c.get('gyro_zero_sample_threshold_rad_s', 1e-10))
        self.gyro_stuck_delta = float(c.get('gyro_stuck_delta_threshold_rad_s', 1e-10))
        self.gyro_stuck_samples = max(2, int(c.get('gyro_stuck_samples', 5)))
        self.vector_stuck_delta_deg = float(c.get('vector_stuck_delta_threshold_deg', 1e-7))
        self.vector_stuck_samples = max(2, int(c.get('vector_stuck_samples', 8)))
        self.vector_stuck_motion_min_deg = float(c.get('vector_stuck_motion_min_deg', 0.05))
        self.availability_recovery_grace_samples = max(0, int(c.get('availability_recovery_grace_samples', 5)))
        self.gyro_reacquisition_consistency_samples = max(1, int(c.get('gyro_reacquisition_consistency_samples', 20)))

        # Gyro residual thresholds.
        self.gyro_rate_warn = float(c.get('gyro_rate_residual_warn_rad_s', 0.0025))
        self.gyro_rate_fault = float(c.get('gyro_rate_residual_fault_rad_s', 0.010))
        self.gyro_abs_limit = float(c.get('gyro_absolute_limit_rad_s', 0.35))
        self.gyro_cross_warn = float(c.get('gyro_cross_sensor_warn_deg', 1.5))
        self.gyro_cross_fault = float(c.get('gyro_cross_sensor_fault_deg', 8.0))
        self.gyro_cross_min_observable_rate = float(c.get('gyro_cross_min_observable_rate_rad_s', 2e-4))
        self.vector_rate_window_samples = max(2, int(c.get('vector_rate_window_samples', 10)))
        self.vector_rate_limit = float(c.get('vector_rate_limit_rad_s', 0.20))
        self.gyro_track_reanchor_samples = max(1, int(c.get('gyro_track_reanchor_samples', 200)))

        # Vector innovation thresholds.
        self.sun_innov_warn = float(c.get('sun_innovation_warn_deg', 2.0))
        self.sun_innov_fault = float(c.get('sun_innovation_fault_deg', 12.0))
        self.mag_innov_warn = float(c.get('mag_innovation_warn_deg', 3.0))
        self.mag_innov_fault = float(c.get('mag_innovation_fault_deg', 15.0))

        # Vector temporal / geometric checks.
        self.vector_temporal_margin_deg = float(c.get('vector_temporal_margin_deg', 0.8))
        self.vector_temporal_fault_deg = float(c.get('vector_temporal_fault_deg', 8.0))
        self.vector_norm_warn = float(c.get('vector_norm_warn', 0.08))
        self.vector_norm_fault = float(c.get('vector_norm_fault', 0.35))
        self.vector_geometry_warn_deg = float(c.get('vector_geometry_warn_deg', 2.0))
        self.vector_geometry_fault_deg = float(c.get('vector_geometry_fault_deg', 10.0))
        self.vector_isolation_margin_deg = float(c.get('vector_isolation_margin_deg', 3.0))
        self.single_vector_innovation_floor = float(c.get('single_vector_innovation_quality_floor', 0.80))
        self.geometry_consistent_innovation_floor = float(c.get('geometry_consistent_innovation_quality_floor', 0.90))
        self.ambiguous_geometry_quality_floor = float(c.get('ambiguous_geometry_quality_floor', 0.45))

        self.weights = {
            'range': float(c.get('weight_range', 0.20)),
            'innovation': float(c.get('weight_innovation', 0.35)),
            'temporal': float(c.get('weight_temporal', 0.20)),
            'cross': float(c.get('weight_cross_sensor', 0.25)),
        }
        sw = sum(max(v, 0.0) for v in self.weights.values())
        self.weights = ({'range': .2, 'innovation': .35, 'temporal': .2, 'cross': .25}
                        if sw <= 0 else {k: max(v, 0.0) / sw for k, v in self.weights.items()})

        self.reset()

    def reset(self):
        self.conf = {'gyro': 1.0, 'sun': 1.0, 'mag': 1.0}
        self.state = {'gyro': HEALTHY, 'sun': HEALTHY, 'mag': HEALTHY}
        self.bad_count = {'gyro': 0, 'sun': 0, 'mag': 0}
        self.good_count = {'gyro': 0, 'sun': 0, 'mag': 0}
        self.prev = {'gyro': None, 'sun': None, 'mag': None}
        self.stuck_count = {'gyro': 0, 'sun': 0, 'mag': 0}
        self.last_trusted_rate = np.zeros(3)
        self.q_gyro_track = None
        self.step_count = 0
        self.prev_q_est = None
        self.prev_hard = {'gyro': False, 'sun': False, 'mag': False}
        self.recovery_grace = {'gyro': 0, 'sun': 0, 'mag': 0}
        self.gyro_reacquiring = False
        self.gyro_reacquire_good_count = 0
        self.triad_anchor_C = None
        self.triad_anchor_step = 0
        self.coarse_vector_rate = None

    def _state_update(self, sensor, raw_conf, hard_unavailable=False, expected_unavailable=False):
        if expected_unavailable:
            # This is a known environmental condition, not a sensor fault.
            self.state[sensor] = EXPECTED_UNAVAILABLE
            self.bad_count[sensor] = 0
            self.good_count[sensor] = 0
            return

        if hard_unavailable:
            self.conf[sensor] = 0.0
            self.state[sensor] = LOST
            self.bad_count[sensor] = self.fault_persistence
            self.good_count[sensor] = 0
            return

        old_state = self.state[sensor]
        old = float(self.conf[sensor])
        # A valid packet after an availability-only LOST state may recover immediately;
        # soft FAULTY states still require persistence to recover.
        if old_state in (LOST, EXPECTED_UNAVAILABLE):
            old = max(old, self.healthy_recovery_threshold)
        self.conf[sensor] = float(np.clip((1.0 - self.alpha) * old + self.alpha * raw_conf, 0.0, 1.0))
        c = self.conf[sensor]

        if c < self.faulty_threshold:
            self.bad_count[sensor] += 1; self.good_count[sensor] = 0
        elif c >= self.healthy_recovery_threshold:
            self.good_count[sensor] += 1; self.bad_count[sensor] = 0
        else:
            self.good_count[sensor] = 0
            if c >= self.degraded_threshold:
                self.bad_count[sensor] = 0

        if old_state == LOST:
            self.state[sensor] = HEALTHY if c >= self.degraded_threshold else DEGRADED
        elif old_state == EXPECTED_UNAVAILABLE:
            self.state[sensor] = HEALTHY if c >= self.degraded_threshold else DEGRADED
        elif old_state == FAULTY:
            if self.good_count[sensor] >= self.recovery_persistence:
                self.state[sensor] = HEALTHY
        else:
            if self.bad_count[sensor] >= self.fault_persistence:
                self.state[sensor] = FAULTY
            elif c < self.degraded_threshold:
                self.state[sensor] = DEGRADED
            else:
                self.state[sensor] = HEALTHY

    def _update_stuck_counter(self, sensor, z, available, vector=False):
        if not available or self.prev[sensor] is None or not np.all(np.isfinite(self.prev[sensor])):
            self.stuck_count[sensor] = 0
            return False
        if vector:
            delta = _angle_deg(z, self.prev[sensor])
            same = np.isfinite(delta) and delta <= self.vector_stuck_delta_deg
        else:
            delta = float(np.linalg.norm(np.asarray(z, float) - np.asarray(self.prev[sensor], float)))
            same = np.isfinite(delta) and delta <= self.gyro_stuck_delta
        self.stuck_count[sensor] = self.stuck_count[sensor] + 1 if same else 0
        return self.stuck_count[sensor] >= self.gyro_stuck_samples if sensor == 'gyro' else self.stuck_count[sensor] >= self.vector_stuck_samples

    def update(self, gyro, sun, mag, q_est, bias_est, sun_i, mag_i, sun_visible=True):
        gyro = np.asarray(gyro, float); sun = np.asarray(sun, float); mag = np.asarray(mag, float)
        q_est = np.asarray(q_est, float); bias_est = np.asarray(bias_est, float)
        C = quat_to_dcm(q_est)

        gyro_finite = bool(np.all(np.isfinite(gyro)))
        sun_finite = bool(np.all(np.isfinite(sun)))
        mag_finite = bool(np.all(np.isfinite(mag)))
        expected_sun_unavailable = bool(not sun_visible)

        gyro_zero = bool(gyro_finite and np.linalg.norm(gyro) <= self.gyro_zero_threshold)
        gyro_hard_unavailable = bool((not gyro_finite) or (self.gyro_zero_is_unavailable and gyro_zero))
        sun_hard_unavailable = bool(sun_visible and not sun_finite)
        mag_hard_unavailable = bool(not mag_finite)

        hard_now = {'gyro': gyro_hard_unavailable, 'sun': sun_hard_unavailable, 'mag': mag_hard_unavailable}
        for sensor in ('gyro','sun','mag'):
            if self.prev_hard.get(sensor, False) and not hard_now[sensor]:
                self.recovery_grace[sensor] = self.availability_recovery_grace_samples
        # A gyro track accumulated before a hard loss is stale when samples return.
        # Re-anchor on the current estimator state before evaluating soft consistency.
        if self.prev_hard.get('gyro', False) and not gyro_hard_unavailable:
            self.q_gyro_track = quat_normalize(q_est)
            self.gyro_reacquiring = True
            self.gyro_reacquire_good_count = 0

        # Stuck-output detection is deliberately independent of the MEKF.
        gyro_stuck = self._update_stuck_counter('gyro', gyro, gyro_finite and not gyro_hard_unavailable, vector=False)
        sun_stuck = self._update_stuck_counter('sun', sun, sun_visible and sun_finite, vector=True)
        mag_stuck = self._update_stuck_counter('mag', mag, mag_finite, vector=True)

        gyro_available = bool(gyro_finite and not gyro_hard_unavailable)
        sun_available = bool(sun_visible and sun_finite)
        mag_available = bool(mag_finite)

        prior_conf = dict(self.conf)
        corrected_rate = gyro - bias_est if gyro_available else np.full(3, np.nan)
        rate_norm = float(np.linalg.norm(corrected_rate)) if gyro_available else np.nan

        # Model innovations against MEKF attitude. They are only one source of evidence;
        # v2 explicitly prevents them from accusing the last remaining healthy vector.
        sun_pred = _unit(C.T @ np.asarray(sun_i, float))
        mag_pred = _unit(C.T @ _unit(mag_i))
        sun_innov = _angle_deg(sun, sun_pred) if sun_available else np.nan
        mag_innov = _angle_deg(mag, mag_pred) if mag_available else np.nan

        # Independent gyro attitude track.
        self.step_count += 1
        if self.q_gyro_track is None:
            self.q_gyro_track = quat_normalize(q_est)
        if gyro_available and (self.recovery_grace['gyro'] > 0 or self.gyro_reacquiring):
            # During packet reacquisition the estimator is itself recovering from a
            # period without gyro data. Re-anchor each sample so stale track error is
            # not misinterpreted as a fresh gyro fault.
            self.q_gyro_track = quat_normalize(q_est)
        if gyro_available:
            self.q_gyro_track = quat_normalize(quat_multiply(self.q_gyro_track, small_angle_quat(corrected_rate * self.dt)))
        Cp = quat_to_dcm(self.q_gyro_track)
        sun_track_err = _angle_deg(sun, Cp.T @ np.asarray(sun_i, float)) if sun_available else np.nan
        mag_track_err = _angle_deg(mag, Cp.T @ _unit(mag_i)) if mag_available else np.nan

        # Gyro cross-sensor evidence uses the BEST consistent absolute reference, not
        # the worst one. Therefore one faulty vector cannot by itself falsely accuse
        # an otherwise healthy gyro.
        gyro_cross_refs = []
        sun_track_pred = _unit(Cp.T @ np.asarray(sun_i, float))
        mag_track_pred = _unit(Cp.T @ _unit(mag_i))
        sun_observability = float(np.linalg.norm(np.cross(corrected_rate, sun_track_pred))) if gyro_available and np.all(np.isfinite(sun_track_pred)) else 0.0
        mag_observability = float(np.linalg.norm(np.cross(corrected_rate, mag_track_pred))) if gyro_available and np.all(np.isfinite(mag_track_pred)) else 0.0
        if sun_available and prior_conf['sun'] >= self.faulty_threshold and np.isfinite(sun_track_err) and sun_observability >= self.gyro_cross_min_observable_rate:
            gyro_cross_refs.append(sun_track_err)
        if mag_available and prior_conf['mag'] >= self.faulty_threshold and np.isfinite(mag_track_err) and mag_observability >= self.gyro_cross_min_observable_rate:
            gyro_cross_refs.append(mag_track_err)
        # Use the best *observable* absolute reference. A vector parallel to the
        # candidate rotation axis contains essentially no information about that rate.
        gyro_cross = min(gyro_cross_refs) if gyro_cross_refs else np.nan

        # Estimator-independent Sun/B geometry.
        geometry_error = np.nan
        geometry_consistent = False
        if sun_available and mag_available:
            geometry_error = abs(_angle_deg(sun, mag) - _angle_deg(sun_i, mag_i))
            geometry_consistent = bool(np.isfinite(geometry_error) and geometry_error <= self.vector_geometry_warn_deg)
        geom_q = _quality(geometry_error, self.vector_geometry_warn_deg, self.vector_geometry_fault_deg) if np.isfinite(geometry_error) else 1.0

        # Estimator-independent coarse body-rate estimate from two mutually consistent
        # absolute vectors. It is intentionally windowed because vector-sensor noise
        # makes a one-sample finite difference unnecessarily noisy.
        Ctriad = _triad_dcm_body_to_inertial(sun, mag, sun_i, mag_i) if geometry_consistent else None
        if Ctriad is not None:
            if self.triad_anchor_C is None:
                self.triad_anchor_C = Ctriad; self.triad_anchor_step = self.step_count
            elif (self.step_count-self.triad_anchor_step) >= self.vector_rate_window_samples:
                elapsed=(self.step_count-self.triad_anchor_step)*self.dt
                wr=_rate_from_dcm_pair(self.triad_anchor_C,Ctriad,elapsed)
                if wr is not None and np.all(np.isfinite(wr)):
                    wn=float(np.linalg.norm(wr))
                    if wn>self.vector_rate_limit and wn>0: wr=wr*(self.vector_rate_limit/wn)
                    self.coarse_vector_rate=np.asarray(wr,float)
                self.triad_anchor_C=Ctriad; self.triad_anchor_step=self.step_count
        else:
            # Do not form a two-vector rate while the pair is geometrically inconsistent.
            self.triad_anchor_C=None; self.coarse_vector_rate=None; self.triad_anchor_step=self.step_count
        gyro_vector_rate_residual = (float(np.linalg.norm(corrected_rate-self.coarse_vector_rate))
                                     if gyro_available and self.coarse_vector_rate is not None else np.nan)

        # Temporal consistency. If gyro itself is unavailable/untrusted, do not use its
        # rate to accuse vector sensors; retain temporal evidence only as diagnostic.
        expected_step_deg = float(np.degrees(max(rate_norm, 0.0) * self.dt)) if np.isfinite(rate_norm) else 0.0
        # A constant vector is only suspicious when independent dynamics imply that
        # the body-frame vector should actually be moving. Constant pointing at near
        # zero rate is physically valid and must not be called a stuck sensor.
        # Vector stuck detection is kept diagnostic-only in v2. A vector can be
        # physically constant during nominal pointing, so hard isolation requires
        # additional mission/controller context not assumed by this generic detector.
        sun_stuck_effective = False
        mag_stuck_effective = False
        temporal = {}
        observed_step = {}
        for name, z, available in (('sun', sun, sun_available), ('mag', mag, mag_available)):
            if available and self.prev[name] is not None and np.all(np.isfinite(self.prev[name])):
                observed = _angle_deg(z, self.prev[name]); observed_step[name] = observed
                temporal[name] = max(0.0, observed - expected_step_deg - self.vector_temporal_margin_deg)
            else:
                observed_step[name] = np.nan
                temporal[name] = 0.0 if available else np.nan

        independent_vector_motion = max([x for x in observed_step.values() if np.isfinite(x)], default=0.0)
        gyro_stuck_effective = bool(gyro_stuck and independent_vector_motion >= self.vector_stuck_motion_min_deg)

        gyro_rate_residual = float(np.linalg.norm(corrected_rate - self.last_trusted_rate)) if gyro_available else np.nan
        estimator_rate_residual = np.nan
        if gyro_available and self.prev_q_est is not None:
            Cprev = quat_to_dcm(self.prev_q_est); Ccur = quat_to_dcm(q_est)
            rv = _rotation_vector_from_dcm(Cprev.T @ Ccur)
            est_rate = rv / max(self.dt, 1e-12)
            estimator_rate_residual = float(np.linalg.norm(corrected_rate - est_rate))
        gyro_abs_excess = max(0.0, rate_norm - self.gyro_abs_limit) if np.isfinite(rate_norm) else np.nan

        sun_norm_err = abs(float(np.linalg.norm(sun)) - 1.0) if sun_available else np.nan
        mag_norm_err = abs(float(np.linalg.norm(mag)) - 1.0) if mag_available else np.nan

        # Cross-sensor isolation. When geometry disagrees and gyro is still trusted,
        # use the gyro-propagated attitude to identify which vector is more suspicious.
        # If isolation is ambiguous, degrade both only mildly rather than letting the
        # (possibly corrupt) MEKF innovation choose a victim.
        sun_cross_q = 1.0; mag_cross_q = 1.0
        isolated_vector = 'none'
        if sun_available and mag_available and np.isfinite(geometry_error) and not geometry_consistent:
            if prior_conf['gyro'] >= self.degraded_threshold and np.isfinite(sun_track_err) and np.isfinite(mag_track_err):
                d = mag_track_err - sun_track_err
                if d >= self.vector_isolation_margin_deg:
                    mag_cross_q = geom_q; isolated_vector = 'mag'
                elif d <= -self.vector_isolation_margin_deg:
                    sun_cross_q = geom_q; isolated_vector = 'sun'
                else:
                    floor = self.ambiguous_geometry_quality_floor
                    sun_cross_q = max(geom_q, floor); mag_cross_q = max(geom_q, floor); isolated_vector = 'ambiguous'
            else:
                floor = self.ambiguous_geometry_quality_floor
                sun_cross_q = max(geom_q, floor); mag_cross_q = max(geom_q, floor); isolated_vector = 'ambiguous'

        sun_innov_q = _quality(sun_innov, self.sun_innov_warn, self.sun_innov_fault) if sun_available else 0.0
        mag_innov_q = _quality(mag_innov, self.mag_innov_warn, self.mag_innov_fault) if mag_available else 0.0

        # Protect a mutually consistent pair from a drifting MEKF estimate.
        if geometry_consistent:
            sun_innov_q = max(sun_innov_q, self.geometry_consistent_innovation_floor)
            mag_innov_q = max(mag_innov_q, self.geometry_consistent_innovation_floor)

        # Protect the last remaining absolute vector when the gyro is unavailable or
        # already untrusted. A single vector cannot be isolated as faulty solely from
        # an MEKF innovation that may itself result from loss of observability.
        if sun_available and not mag_available and (gyro_hard_unavailable or prior_conf['gyro'] < self.degraded_threshold):
            sun_innov_q = max(sun_innov_q, self.single_vector_innovation_floor)
        if mag_available and not sun_available and (gyro_hard_unavailable or prior_conf['gyro'] < self.degraded_threshold):
            mag_innov_q = max(mag_innov_q, self.single_vector_innovation_floor)

        # If the independent gyro track strongly supports a vector, do not let MEKF
        # innovation alone mark it faulty.
        if sun_available and prior_conf['gyro'] >= self.degraded_threshold and np.isfinite(sun_track_err) and sun_track_err <= self.sun_innov_warn:
            sun_innov_q = max(sun_innov_q, self.geometry_consistent_innovation_floor)
        if mag_available and prior_conf['gyro'] >= self.degraded_threshold and np.isfinite(mag_track_err) and mag_track_err <= self.mag_innov_warn:
            mag_innov_q = max(mag_innov_q, self.geometry_consistent_innovation_floor)

        # Temporal evidence is disabled as an accusing term when gyro is unavailable or
        # low-confidence; otherwise a healthy vector's natural motion would look faulty.
        gyro_support_for_temporal = gyro_available and prior_conf['gyro'] >= self.faulty_threshold
        sun_temporal_q = (_quality(temporal['sun'], self.vector_temporal_margin_deg, self.vector_temporal_fault_deg)
                          if sun_available and gyro_support_for_temporal else 1.0)
        mag_temporal_q = (_quality(temporal['mag'], self.vector_temporal_margin_deg, self.vector_temporal_fault_deg)
                          if mag_available and gyro_support_for_temporal else 1.0)

        sun_q = {
            'range': _quality(sun_norm_err, self.vector_norm_warn, self.vector_norm_fault),
            'innovation': sun_innov_q,
            'temporal': 0.0 if sun_stuck_effective else sun_temporal_q,
            'cross': sun_cross_q,
        }
        mag_q = {
            'range': _quality(mag_norm_err, self.vector_norm_warn, self.vector_norm_fault),
            'innovation': mag_innov_q,
            'temporal': 0.0 if mag_stuck_effective else mag_temporal_q,
            'cross': mag_cross_q,
        }
        gyro_cross_q = _quality(gyro_cross, self.gyro_cross_warn, self.gyro_cross_fault) if np.isfinite(gyro_cross) else 1.0
        gyro_rate_q = (_quality(gyro_vector_rate_residual, self.gyro_rate_warn, self.gyro_rate_fault)
                       if np.isfinite(gyro_vector_rate_residual) else
                       (_quality(gyro_rate_residual, self.gyro_rate_warn, self.gyro_rate_fault) if gyro_available else 0.0))
        gyro_q = {
            'range': 1.0 if gyro_available and rate_norm <= self.gyro_abs_limit else _quality(gyro_abs_excess, 0.0, max(self.gyro_abs_limit, 1e-6)),
            'innovation': gyro_cross_q,
            'temporal': 0.0 if gyro_stuck_effective else gyro_rate_q,
            'cross': gyro_cross_q,
        }

        def combine(q):
            return float(np.clip(sum(self.weights[k] * q[k] for k in self.weights), 0.0, 1.0))

        raw = {
            'gyro': combine(gyro_q) if gyro_available else 0.0,
            'sun': combine(sun_q) if sun_available else 0.0,
            'mag': combine(mag_q) if mag_available else 0.0,
        }

        # Soft-fault recovery uses the estimator-independent two-vector rate. Once
        # Sun and magnetic geometry is again consistent and the current gyro agrees
        # with the TRIAD-derived rate, a stale gyro-track disagreement accumulated
        # during the fault is cleared and the track is re-anchored from TRIAD.
        vector_rate_support = (geometry_consistent and np.isfinite(gyro_vector_rate_residual) and
                               gyro_vector_rate_residual <= 2.0*self.gyro_rate_warn)
        if self.state['gyro'] in (FAULTY, DEGRADED) and vector_rate_support:
            raw['gyro'] = max(raw['gyro'], self.healthy_recovery_threshold + 0.08)
            if Ctriad is not None:
                self.q_gyro_track = _quat_from_dcm(Ctriad)

        # Stuck outputs are soft faults unless they coincide with hard loss encoding.
        if gyro_stuck_effective and gyro_available: raw['gyro'] = min(raw['gyro'], self.faulty_threshold * 0.5)
        if sun_stuck_effective and sun_available: raw['sun'] = min(raw['sun'], self.faulty_threshold * 0.5)
        if mag_stuck_effective and mag_available: raw['mag'] = min(raw['mag'], self.faulty_threshold * 0.5)

        # A gyro returning after hard loss is provisionally trusted until its
        # independent vector consistency has re-established for several samples.
        # This avoids blaming a newly recovered gyro for estimator error accumulated
        # while that gyro was absent.
        if self.gyro_reacquiring and gyro_available:
            raw['gyro'] = max(raw['gyro'], 0.98)
            if np.isfinite(gyro_cross) and gyro_cross <= self.gyro_cross_warn:
                self.gyro_reacquire_good_count += 1
            else:
                self.gyro_reacquire_good_count = 0
            if self.gyro_reacquire_good_count >= self.gyro_reacquisition_consistency_samples:
                self.gyro_reacquiring = False
                self.gyro_reacquire_good_count = 0

        # Valid packets immediately following a hard availability outage receive a
        # short re-entry grace period. This prevents stale estimator/gyro-track errors
        # accumulated during the outage from turning into a long false alarm.
        for sensor, available in (('gyro',gyro_available),('sun',sun_available),('mag',mag_available)):
            if available and self.recovery_grace[sensor] > 0:
                raw[sensor] = max(raw[sensor], self.healthy_recovery_threshold + 0.05)
                self.recovery_grace[sensor] -= 1

        self._state_update('gyro', raw['gyro'], hard_unavailable=gyro_hard_unavailable, expected_unavailable=False)
        self._state_update('sun', raw['sun'], hard_unavailable=sun_hard_unavailable, expected_unavailable=expected_sun_unavailable)
        self._state_update('mag', raw['mag'], hard_unavailable=mag_hard_unavailable, expected_unavailable=False)

        # Re-anchor the independent gyro track only when the gyro is confidently healthy.
        vector_support = geometry_consistent or (
            (sun_available and np.isfinite(sun_track_err) and sun_track_err <= self.sun_innov_warn) or
            (mag_available and np.isfinite(mag_track_err) and mag_track_err <= self.mag_innov_warn)
        )
        if self.state['gyro'] == HEALTHY and self.conf['gyro'] >= self.trusted_update_threshold and vector_support:
            if self.step_count % self.gyro_track_reanchor_samples == 0:
                self.q_gyro_track = quat_normalize(q_est)
            self.last_trusted_rate = corrected_rate.copy() if gyro_available else self.last_trusted_rate

        for name, z, available in (('gyro', gyro, gyro_finite), ('sun', sun, sun_available), ('mag', mag, mag_available)):
            if available:
                self.prev[name] = np.asarray(z, float).copy()
        self.prev_q_est = quat_normalize(q_est)
        self.prev_hard = dict(hard_now)

        score = {k: 1.0 - raw[k] for k in raw}
        hard = {'gyro': gyro_hard_unavailable, 'sun': sun_hard_unavailable, 'mag': mag_hard_unavailable}
        expected = {'gyro': False, 'sun': expected_sun_unavailable, 'mag': False}
        diagnostics = {
            'sun_innovation_deg': sun_innov,
            'mag_innovation_deg': mag_innov,
            'gyro_rate_residual_rad_s': gyro_rate_residual,
            'gyro_estimator_rate_residual_rad_s': estimator_rate_residual,
            'gyro_vector_rate_residual_rad_s': gyro_vector_rate_residual,
            'gyro_cross_sensor_deg': gyro_cross,
            'sun_rate_observability_rad_s': sun_observability,
            'mag_rate_observability_rad_s': mag_observability,
            'sun_gyro_track_error_deg': sun_track_err,
            'mag_gyro_track_error_deg': mag_track_err,
            'sun_temporal_excess_deg': temporal['sun'],
            'mag_temporal_excess_deg': temporal['mag'],
            'sun_mag_geometry_error_deg': geometry_error,
            'geometry_consistent': float(geometry_consistent),
            'isolated_vector': isolated_vector,
            'gyro_stuck': float(gyro_stuck_effective),
            'sun_stuck': float(sun_stuck_effective),
            'mag_stuck': float(mag_stuck_effective),
            'gyro_reacquiring': float(self.gyro_reacquiring),
            'raw_gyro_confidence': raw['gyro'],
            'raw_sun_confidence': raw['sun'],
            'raw_mag_confidence': raw['mag'],
        }
        return FDIRResult(confidence=dict(self.conf), state=dict(self.state), score=score,
                          diagnostics=diagnostics, hard_unavailable=hard, expected_unavailable=expected)
