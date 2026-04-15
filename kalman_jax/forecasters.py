"""Generic one-step-ahead Kalman-style forecasters with learned dynamics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad

from .filters import _kalman_gain, _joseph_covariance_update
from .learned_dynamics import mlp


def transition_predict(x_prev, u, params, A, B):
    """Predict the next latent state from the previous state and exogenous inputs."""
    learned_residual = mlp(params, jnp.concatenate([x_prev, u]))
    return A @ x_prev + B @ u + learned_residual


def measurement_update(x_pred, P_pred, y, C, R):
    """Apply the Kalman measurement update to a predicted latent state."""
    innovation = y - C @ x_pred
    innovation_cov = C @ P_pred @ C.T + R
    K = _kalman_gain(P_pred, C, innovation_cov)

    x_new = x_pred + K @ innovation
    P_new = _joseph_covariance_update(P_pred, K, C, R)

    return x_new, P_new, innovation_cov


def forecasting_loss_fn(params, A, B, C, Q, R, x0, P0, us, ys):
    """One-step-ahead forecast loss with filtering only after scoring the forecast."""

    def step(carry, inputs):
        x, P = carry
        u, y = inputs

        x_pred = transition_predict(x, u, params, A, B)
        P_pred = A @ P @ A.T + Q
        y_pred = C @ x_pred

        x_new, P_new, _ = measurement_update(x_pred, P_pred, y, C, R)
        return (x_new, P_new), y_pred

    (_, _), y_preds = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - y_preds) ** 2)


def forecast_with_kalman(params, A, B, C, Q, R, x0, P0, us, ys):
    """Generate one-step-ahead forecasts and filtered estimates across a sequence."""

    def step(carry, inputs):
        x, P = carry
        u, y = inputs

        x_pred = transition_predict(x, u, params, A, B)
        P_pred = A @ P @ A.T + Q
        y_pred = C @ x_pred
        S_pred = C @ P_pred @ C.T + R

        x_new, P_new, _ = measurement_update(x_pred, P_pred, y, C, R)
        y_filtered = C @ x_new
        S_filtered = C @ P_new @ C.T + R

        outputs = (y_pred, y_filtered, S_pred, S_filtered)
        return (x_new, P_new), outputs

    (_, _), outputs = jax.lax.scan(step, (x0, P0), (us, ys))
    return outputs


def train_kalman_forecaster(
    params,
    optimizer,
    *,
    A,
    B,
    C,
    Q,
    R,
    x0,
    P0,
    us,
    ys,
    num_steps=1000,
    log_every=None,
):
    """Train the learned transition model using one-step-ahead forecast loss."""
    opt_state = optimizer.init(params)

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = value_and_grad(forecasting_loss_fn)(
            current_params,
            A,
            B,
            C,
            Q,
            R,
            x0,
            P0,
            us,
            ys,
        )
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    losses = []
    for step in range(num_steps):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if log_every is not None and step % log_every == 0:
            print(f"Step {step}, Forecast Loss: {float(loss):.6f}")

    return params, opt_state, jnp.stack(losses)
