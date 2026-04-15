"""Kalman filter implementations used by the workshop scripts."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _kalman_gain(P, H, innovation_cov):
    """Solve for the Kalman gain without forming an explicit matrix inverse."""
    return jnp.linalg.solve(innovation_cov, H @ P.T).T


def _joseph_covariance_update(P, K, H, R):
    """Joseph-form covariance update for better numerical stability."""
    identity = jnp.eye(P.shape[0], dtype=P.dtype)
    residual_factor = identity - K @ H
    return residual_factor @ P @ residual_factor.T + K @ R @ K.T


def linear_kf_predict(x, P, F, Q):
    """Pure function for the linear Kalman-filter prediction step."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def linear_kf_update(x, P, z, H, R):
    """Pure function for the linear Kalman-filter update step."""
    innovation = z - H @ x
    innovation_cov = H @ P @ H.T + R
    K = _kalman_gain(P, H, innovation_cov)

    x_updated = x + K @ innovation
    P_updated = _joseph_covariance_update(P, K, H, R)

    return x_updated, P_updated, innovation, innovation_cov


def extended_kf_predict(x, P, f, F_jacobian, Q):
    """Pure function for the EKF prediction step."""
    x_pred = f(x)
    F = F_jacobian(x)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def extended_kf_update(x, P, z, h, H_jacobian, R):
    """Pure function for the EKF update step."""
    H = H_jacobian(x)
    innovation = z - h(x)
    innovation_cov = H @ P @ H.T + R
    K = _kalman_gain(P, H, innovation_cov)

    x_updated = x + K @ innovation
    P_updated = _joseph_covariance_update(P, K, H, R)

    return x_updated, P_updated, innovation, innovation_cov


def _unscented_weights(state_dim, alpha=1e-3, beta=2.0, kappa=0.0):
    lambda_ = alpha**2 * (state_dim + kappa) - state_dim
    scale = state_dim + lambda_

    Wm = jnp.full(2 * state_dim + 1, 1.0 / (2.0 * scale))
    Wc = jnp.full(2 * state_dim + 1, 1.0 / (2.0 * scale))
    Wm = Wm.at[0].set(lambda_ / scale)
    Wc = Wc.at[0].set(lambda_ / scale + (1.0 - alpha**2 + beta))
    return scale, Wm, Wc


def _sigma_points(x, P, alpha=1e-3, beta=2.0, kappa=0.0, jitter=1e-9):
    state_dim = x.shape[0]
    scale, Wm, Wc = _unscented_weights(state_dim, alpha=alpha, beta=beta, kappa=kappa)
    sqrt_P = jnp.linalg.cholesky(P + jitter * jnp.eye(state_dim, dtype=P.dtype))
    offsets = jnp.sqrt(scale) * sqrt_P.T
    sigma_points = jnp.concatenate(
        [x[None, :], x[None, :] + offsets, x[None, :] - offsets],
        axis=0,
    )
    return sigma_points, Wm, Wc


def unscented_kf_predict(x, P, f, Q, alpha=1e-3, beta=2.0, kappa=0.0):
    """Pure function for the UKF prediction step."""
    sigma_points, Wm, Wc = _sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
    transformed_points = jax.vmap(f)(sigma_points)

    x_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    diffs = transformed_points - x_pred
    cov_terms = jax.vmap(lambda diff: jnp.outer(diff, diff))(diffs)
    P_pred = jnp.sum(Wc[:, None, None] * cov_terms, axis=0) + Q

    return x_pred, P_pred


def unscented_kf_update(x, P, z, h, R, alpha=1e-3, beta=2.0, kappa=0.0):
    """Pure function for the UKF update step."""
    sigma_points, Wm, Wc = _sigma_points(x, P, alpha=alpha, beta=beta, kappa=kappa)
    transformed_points = jax.vmap(h)(sigma_points)

    z_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    z_diffs = transformed_points - z_pred
    x_diffs = sigma_points - x

    measurement_terms = jax.vmap(lambda diff: jnp.outer(diff, diff))(z_diffs)
    cross_terms = jax.vmap(lambda x_diff, z_diff: jnp.outer(x_diff, z_diff))(x_diffs, z_diffs)

    P_zz = jnp.sum(Wc[:, None, None] * measurement_terms, axis=0) + R
    P_xz = jnp.sum(Wc[:, None, None] * cross_terms, axis=0)

    K = jnp.linalg.solve(P_zz, P_xz.T).T
    innovation = z - z_pred
    x_updated = x + K @ innovation
    P_updated = P - K @ P_zz @ K.T
    P_updated = 0.5 * (P_updated + P_updated.T)

    return x_updated, P_updated, innovation, P_zz


def generate_linear_synthetic_data(
    key,
    n_steps=100,
    dt=0.1,
    process_noise_std=0.1,
    measurement_noise_std=0.5,
    initial_state=None,
):
    """Generate synthetic 1D position and velocity data."""
    F = jnp.array([[1.0, dt], [0.0, 1.0]])
    H = jnp.array([[1.0, 0.0]])

    Q = jnp.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * process_noise_std**2
    R = jnp.array([[measurement_noise_std**2]])

    x_true = jnp.array([0.0, 1.0]) if initial_state is None else initial_state

    true_states = []
    measurements = []

    for _ in range(n_steps):
        key, process_key = jax.random.split(key)
        process_noise = jax.random.multivariate_normal(process_key, jnp.zeros(2), Q)
        x_true = F @ x_true + process_noise
        true_states.append(x_true)

        key, measurement_key = jax.random.split(key)
        measurement_noise = jax.random.normal(measurement_key, (1,)) * measurement_noise_std
        measurements.append(H @ x_true + measurement_noise)

    return F, H, Q, R, jnp.array(true_states), jnp.vstack(measurements)


def run_linear_kf(measurements, F, H, Q, R, x0, P0):
    """Run the linear Kalman filter over a measurement sequence."""

    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = linear_kf_predict(x, P, F, Q)
        x_updated, P_updated, _, _ = linear_kf_update(x_pred, P_pred, measurement, H, R)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def run_ekf(measurements, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
    """Run the EKF over a measurement sequence."""

    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = extended_kf_predict(x, P, f, F_jacobian, Q)
        x_updated, P_updated, _, _ = extended_kf_update(x_pred, P_pred, measurement, h, H_jacobian, R)
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def run_ukf(measurements, f, h, Q, R, x0, P0, alpha=1e-3, beta=2.0, kappa=0.0):
    """Run the UKF over a measurement sequence."""

    def step(carry, measurement):
        x, P = carry
        x_pred, P_pred = unscented_kf_predict(x, P, f, Q, alpha=alpha, beta=beta, kappa=kappa)
        x_updated, P_updated, _, _ = unscented_kf_update(
            x_pred,
            P_pred,
            measurement,
            h,
            R,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
        return (x_updated, P_updated), x_updated

    (_, _), estimated_states = jax.lax.scan(step, (x0, P0), measurements)
    return estimated_states


def compute_rmse(true_states, estimated_states):
    """Compute per-state RMSE between a reference and an estimate."""
    return jnp.sqrt(jnp.mean((true_states - estimated_states) ** 2, axis=0))
