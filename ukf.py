"""
Unscented Kalman Filter (UKF) for Nonlinear State Estimation
=============================================================
Implements the Unscented Kalman Filter from scratch (sigma-point generation,
unscented transform, predict/update cycle) and benchmarks it against an
Extended Kalman Filter (EKF) baseline on a nonlinear tracking problem:
estimating position and velocity of a target from noisy range-and-bearing
sensor measurements.

Ported/re-implemented in Python from an original MATLAB implementation.

Author: Arvind Kanagasabapathi Chandirakala
"""

from __future__ import annotations

import time
import numpy as np


# ---------------------------------------------------------------------------
# System model: constant-velocity target, nonlinear range/bearing sensor
# ---------------------------------------------------------------------------

DT = 0.1  # s, time step
N_STATES = 4  # [x, y, vx, vy]


def state_transition(x: np.ndarray, dt: float = DT) -> np.ndarray:
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    return F @ x


def measurement_model(x: np.ndarray) -> np.ndarray:
    """Nonlinear range/bearing measurement from a sensor at the origin."""
    px, py = x[0], x[1]
    r = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return np.array([r, theta])


def measurement_jacobian(x: np.ndarray) -> np.ndarray:
    px, py = x[0], x[1]
    r2 = px**2 + py**2
    r = np.sqrt(r2)
    return np.array([
        [px / r, py / r, 0, 0],
        [-py / r2, px / r2, 0, 0],
    ])


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    def __init__(self, x0, P0, Q, R, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = len(x0)
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q
        self.R = R
        self.alpha, self.beta, self.kappa = alpha, beta, kappa
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self._compute_weights()

    def _compute_weights(self):
        n, lam = self.n, self.lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha**2 + self.beta)

    def _sigma_points(self, x, P):
        n, lam = self.n, self.lam
        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        S = np.linalg.cholesky((n + lam) * (P + 1e-9 * np.eye(n)))
        for i in range(n):
            sigmas[i + 1] = x + S[:, i]
            sigmas[n + i + 1] = x - S[:, i]
        return sigmas

    def predict(self, dt=DT):
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_f = np.array([state_transition(s, dt) for s in sigmas])
        x_pred = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        P_pred = self.Q.copy()
        for i in range(len(sigmas_f)):
            dx = sigmas_f[i] - x_pred
            P_pred += self.Wc[i] * np.outer(dx, dx)
        self._sigmas_f = sigmas_f
        self.x, self.P = x_pred, P_pred

    def update(self, z):
        sigmas_f = self._sigmas_f
        sigmas_h = np.array([measurement_model(s) for s in sigmas_f])
        z_pred = np.sum(self.Wm[:, None] * sigmas_h, axis=0)

        Pzz = self.R.copy()
        Pxz = np.zeros((self.n, len(z)))
        for i in range(len(sigmas_f)):
            dz = sigmas_h[i] - z_pred
            dz[1] = (dz[1] + np.pi) % (2 * np.pi) - np.pi  # wrap bearing residual
            dx = sigmas_f[i] - self.x
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pzz)
        innovation = z - z_pred
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi
        self.x = self.x + K @ innovation
        self.P = self.P - K @ Pzz @ K.T


# ---------------------------------------------------------------------------
# Extended Kalman Filter baseline
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R):
        self.x, self.P, self.Q, self.R = x0.copy(), P0.copy(), Q, R

    def predict(self, dt=DT):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = measurement_jacobian(self.x)
        z_pred = measurement_model(self.x)
        y = z - z_pred
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P


# ---------------------------------------------------------------------------
# Simulation / benchmark
# ---------------------------------------------------------------------------

def run_benchmark(n_steps=300, seed=0):
    rng = np.random.default_rng(seed)

    x_true = np.array([100.0, 50.0, 2.0, -1.0])
    Q = np.diag([0.01, 0.01, 0.05, 0.05])
    R = np.diag([1.0, np.deg2rad(1.0) ** 2])

    x0_est = x_true + rng.normal(0, [5, 5, 1, 1])
    P0 = np.diag([25.0, 25.0, 4.0, 4.0])

    ukf = UnscentedKalmanFilter(x0_est, P0, Q, R)
    ekf = ExtendedKalmanFilter(x0_est, P0, Q, R)

    ukf_errors, ekf_errors = [], []

    t0 = time.perf_counter()
    for _ in range(n_steps):
        x_true = state_transition(x_true) + rng.multivariate_normal(np.zeros(4), Q)
        z = measurement_model(x_true) + rng.multivariate_normal(np.zeros(2), R)

        ukf.predict()
        ukf.update(z)
        ukf_errors.append(np.linalg.norm(ukf.x[:2] - x_true[:2]))

        ekf.predict()
        ekf.update(z)
        ekf_errors.append(np.linalg.norm(ekf.x[:2] - x_true[:2]))
    elapsed = time.perf_counter() - t0

    return np.array(ukf_errors), np.array(ekf_errors), elapsed


def main():
    ukf_err, ekf_err, elapsed = run_benchmark()
    print(f"Ran {len(ukf_err)} filter steps in {elapsed*1000:.2f} ms")
    print(f"UKF mean position error: {ukf_err.mean():.3f} m (final: {ukf_err[-1]:.3f} m)")
    print(f"EKF mean position error: {ekf_err.mean():.3f} m (final: {ekf_err[-1]:.3f} m)")
    print(f"UKF converges within first 20 steps to mean error "
          f"{ukf_err[20:].mean():.3f} m vs EKF {ekf_err[20:].mean():.3f} m")


if __name__ == "__main__":
    main()
