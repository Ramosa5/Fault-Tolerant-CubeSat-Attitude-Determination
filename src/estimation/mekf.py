import numpy as np
from src.simulation.attitude import quat_normalize, quat_multiply, quat_to_dcm, small_angle_quat


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 and np.isfinite(n) else v


class MEKF:
    """Stateful 6-state multiplicative EKF for online closed-loop simulations.

    Quaternion convention matches the rest of the project: DCM maps body -> inertial.
    The ``step`` method propagates with the previous gyro sample over ``dt`` and then
    updates with the current Sun/magnetic vector measurements. NaN vectors are skipped.

    Numerical robustness safeguards are intentionally separate from fault rejection:
    they prevent covariance overflow/NaN during severe stress-test faults, but do not
    automatically declare an inconsistent sensor healthy or replace it. Consequently,
    a faulty-MEKF baseline can still diverge physically (large attitude error) without
    making the simulation numerically undefined.
    """

    def __init__(self, q0=(1, 0, 0, 0), bias0=(0, 0, 0), gyro_noise_std=5e-5,
                 bias_rw_std=2e-7, vector_noise_std=2e-3,
                 max_bias_norm_rad_s=0.35, covariance_eig_floor=1e-14,
                 covariance_eig_ceiling=1e3, max_update_angle_rad=np.deg2rad(45.0)):
        self.q = quat_normalize(q0)
        self.b = np.asarray(bias0, float).copy()
        self.P = np.diag([np.deg2rad(5) ** 2] * 3 + [5e-4 ** 2] * 3)
        self.gyro_noise_std = float(gyro_noise_std)
        self.bias_rw_std = float(bias_rw_std)
        self.vector_noise_std = float(vector_noise_std)
        self.last_gyro = np.zeros(3)

        self.max_bias_norm_rad_s = float(max_bias_norm_rad_s)
        self.covariance_eig_floor = float(covariance_eig_floor)
        self.covariance_eig_ceiling = float(covariance_eig_ceiling)
        self.max_update_angle_rad = float(max_update_angle_rad)

    def _repair_covariance(self, P):
        """Return a finite, symmetric positive-semidefinite covariance matrix."""
        P = np.asarray(P, float)
        if not np.all(np.isfinite(P)):
            # Retain large uncertainty instead of poisoning all later calculations.
            diag = np.diag(self.P).copy() if np.all(np.isfinite(self.P)) else np.ones(6)
            diag = np.clip(diag, self.covariance_eig_floor, self.covariance_eig_ceiling)
            return np.diag(diag)
        P = 0.5 * (P + P.T)
        try:
            vals, vecs = np.linalg.eigh(P)
            vals = np.clip(vals, self.covariance_eig_floor, self.covariance_eig_ceiling)
            P = (vecs * vals) @ vecs.T
        except np.linalg.LinAlgError:
            d = np.clip(np.diag(P), self.covariance_eig_floor, self.covariance_eig_ceiling)
            P = np.diag(d)
        return 0.5 * (P + P.T)

    def _clip_bias(self):
        if not np.all(np.isfinite(self.b)):
            self.b = np.zeros(3)
            return
        n = float(np.linalg.norm(self.b))
        if n > self.max_bias_norm_rad_s and n > 0:
            self.b *= self.max_bias_norm_rad_s / n

    def _propagate_covariance(self, F, dt, Q):
        """Stable covariance propagation using sub-stepped first-order transition."""
        # A severe gyro/bias fault can make ||F dt|| large. Sub-stepping prevents a
        # single enormous first-order Phi from producing immediate overflow.
        scale = float(np.linalg.norm(F, ord=np.inf) * abs(dt))
        nsub = max(1, int(np.ceil(scale / 0.25)))
        nsub = min(nsub, 200)
        h = float(dt) / nsub
        Phi = np.eye(6) + F * h
        Qs = Q / nsub
        P = self.P
        for _ in range(nsub):
            P = Phi @ P @ Phi.T + Qs
            P = self._repair_covariance(P)
        self.P = P

    def step(self, gyro_prev, sun_current, mag_current, sun_i, mag_i, dt,
             sun_noise_std=None, mag_noise_std=None):
        I3 = np.eye(3)
        g = np.asarray(gyro_prev, float)
        if np.all(np.isfinite(g)):
            self.last_gyro = g.copy()
        else:
            g = self.last_gyro.copy()

        self._clip_bias()
        w = g - self.b
        if not np.all(np.isfinite(w)):
            w = np.zeros(3)

        self.q = quat_normalize(quat_multiply(self.q, small_angle_quat(w * float(dt))))
        F = np.block([[-skew(w), -I3], [np.zeros((3, 3)), np.zeros((3, 3))]])
        Q = np.diag([self.gyro_noise_std ** 2] * 3 + [self.bias_rw_std ** 2] * 3) * float(dt)
        self._propagate_covariance(F, float(dt), Q)

        diagnostics = {}
        for name, z, ref in (("sun", sun_current, sun_i), ("mag", mag_current, mag_i)):
            z = np.asarray(z, float)
            if not np.all(np.isfinite(z)):
                diagnostics[f'{name}_innovation'] = np.full(3, np.nan)
                diagnostics[f'{name}_available'] = False
                diagnostics[f'{name}_update_used'] = False
                continue

            # All vector sensors represent directions. Re-normalizing protects the
            # filter from pathological fault values while preserving their direction.
            z = _unit(z)
            ref = _unit(ref)
            pred = _unit(quat_to_dcm(self.q).T @ ref)
            innovation = z - pred
            H = np.hstack((skew(pred), np.zeros((3, 3))))
            override = sun_noise_std if name == 'sun' else mag_noise_std
            meas_std = self.vector_noise_std if override is None else float(override)
            meas_std = max(meas_std, 1e-9)
            R = (meas_std ** 2) * I3

            S = H @ self.P @ H.T + R
            S = 0.5 * (S + S.T)
            if not np.all(np.isfinite(S)):
                diagnostics[f'{name}_innovation'] = innovation
                diagnostics[f'{name}_available'] = True
                diagnostics[f'{name}_update_used'] = False
                continue

            try:
                # solve(S, H P)^T is preferable to explicitly forming inv(S)
                K = np.linalg.solve(S, H @ self.P).T
            except np.linalg.LinAlgError:
                K = self.P @ H.T @ np.linalg.pinv(S, rcond=1e-12)

            if not np.all(np.isfinite(K)):
                diagnostics[f'{name}_innovation'] = innovation
                diagnostics[f'{name}_available'] = True
                diagnostics[f'{name}_update_used'] = False
                continue

            dx = K @ innovation
            if not np.all(np.isfinite(dx)):
                diagnostics[f'{name}_innovation'] = innovation
                diagnostics[f'{name}_available'] = True
                diagnostics[f'{name}_update_used'] = False
                continue

            # Limit a single correction step. This is a numerical safeguard, not an
            # innovation gate: repeated faulty measurements can still drive the
            # estimator far away over multiple samples.
            a = float(np.linalg.norm(dx[:3]))
            if a > self.max_update_angle_rad and a > 0:
                dx[:3] *= self.max_update_angle_rad / a

            self.q = quat_normalize(quat_multiply(self.q, small_angle_quat(dx[:3])))
            self.b = self.b + dx[3:]
            self._clip_bias()

            KH = K @ H
            A = np.eye(6) - KH
            self.P = A @ self.P @ A.T + K @ R @ K.T
            self.P = self._repair_covariance(self.P)

            diagnostics[f'{name}_innovation'] = innovation
            diagnostics[f'{name}_available'] = True
            diagnostics[f'{name}_update_used'] = True

        return self.q.copy(), self.b.copy(), diagnostics


def run_mekf(times_s, gyro, sun_b, mag_b, sun_i, mag_i,
             q0=(1, 0, 0, 0), bias0=(0, 0, 0), gyro_noise_std=5e-5,
             bias_rw_std=2e-7, vector_noise_std=2e-3, return_diagnostics=False):
    """Batch convenience wrapper around :class:`MEKF`."""
    t = np.asarray(times_s, float)
    n = len(t)
    filt = MEKF(q0=q0, bias0=bias0, gyro_noise_std=gyro_noise_std,
                bias_rw_std=bias_rw_std, vector_noise_std=vector_noise_std)
    qs = np.zeros((n, 4)); bs = np.zeros((n, 3)); qs[0] = filt.q; bs[0] = filt.b
    sun_innov = np.full((n, 3), np.nan); mag_innov = np.full((n, 3), np.nan)
    sun_available = np.all(np.isfinite(sun_b), axis=1); mag_available = np.all(np.isfinite(mag_b), axis=1)
    for k in range(1, n):
        q, b, d = filt.step(gyro[k-1], sun_b[k], mag_b[k], sun_i, mag_i, t[k] - t[k-1])
        qs[k] = q; bs[k] = b
        sun_innov[k] = d['sun_innovation']; mag_innov[k] = d['mag_innovation']
    if not return_diagnostics:
        return qs, bs
    diagnostics = {'sun_innovation': sun_innov, 'mag_innovation': mag_innov,
                   'sun_available': sun_available, 'mag_available': mag_available}
    return qs, bs, diagnostics
