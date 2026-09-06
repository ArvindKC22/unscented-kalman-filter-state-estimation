# Unscented Kalman Filter (UKF) State Estimation

A from-scratch implementation of the **Unscented Kalman Filter** (sigma-point generation, unscented transform, predict/update) for nonlinear state estimation, benchmarked against an **Extended Kalman Filter (EKF)** baseline.

Originally developed in MATLAB for graduate coursework in Optimal Estimation and Kalman Filtering at the University of Florida; this repo is a Python re-implementation for portability and testing.

## Problem

Estimate the position and velocity of a constant-velocity target from noisy **range/bearing** measurements taken by a sensor at the origin — a classic nonlinear estimation problem where the measurement model `h(x) = [√(x²+y²), atan2(y,x)]` is not linear in the state.

## What's implemented

- `UnscentedKalmanFilter` — sigma-point generation via Cholesky decomposition, unscented-transform predict step, and measurement update with bearing-angle wraparound handling.
- `ExtendedKalmanFilter` — analytic Jacobian linearization baseline.
- A Monte-Carlo benchmark harness comparing tracking accuracy and runtime.

## Usage

```bash
pip install -r requirements.txt
python ukf.py
```

Example output:

```
Ran 300 filter steps in 226.83 ms
UKF mean position error: 0.795 m (final: 0.231 m)
EKF mean position error: 0.794 m (final: 0.233 m)
UKF converges within first 20 steps to mean error 0.754 m vs EKF 0.754 m
```

The UKF matches or beats EKF tracking accuracy on this problem without ever forming a Jacobian — it only needs the nonlinear functions `f(x)` and `h(x)` themselves, which makes it easier to extend to more complex nonlinear/non-differential sensor models.

## Files

- `ukf.py` — UKF, EKF, and benchmark harness
- `requirements.txt` — dependencies (numpy)

## Author

Arvind Kanagasabapathi Chandirakala — M.S. Aerospace & Mechanical Engineering, University of Florida
[LinkedIn](https://www.linkedin.com/in/arvind-kc-/)
