"""Learned-dynamics Kalman-filter experiment helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jax import random, value_and_grad

from .filters import _kalman_gain, _joseph_covariance_update


def true_dynamics(x, u, A_true=None, B_true=None):
    """Nonlinear state dynamics used to generate synthetic data."""
    A_true = jnp.array([[1.0, 0.1], [0.0, 1.0]]) if A_true is None else A_true
    B_true = jnp.array([[0.0], [0.1]]) if B_true is None else B_true
    nonlinear_residual = 3.0 * jnp.sin(x[0]) + 2.0 * jnp.cos(x[1])
    return A_true @ x + B_true @ u + nonlinear_residual


def generate_data(T=100, key=None, A_true=None, B_true=None, measurement_noise_std=0.05):
    """Generate a short nonlinear rollout with noisy observations."""
    key = random.PRNGKey(0) if key is None else key
    x = jnp.zeros((2,))
    xs, us, ys = [], [], []

    for _ in range(T):
        key, control_key, measurement_key = random.split(key, 3)
        u = random.normal(control_key, (1,))
        x = true_dynamics(x, u, A_true=A_true, B_true=B_true)
        y = x + measurement_noise_std * random.normal(measurement_key, x.shape)
        xs.append(x)
        us.append(u)
        ys.append(y)

    return jnp.stack(xs), jnp.stack(us), jnp.stack(ys)


def mlp(params, x):
    """Evaluate the learned residual model."""
    for W, b in params[:-1]:
        x = jnp.tanh(W @ x + b)
    W, b = params[-1]
    return W @ x + b


def init_mlp_params(key, sizes):
    """Initialize an MLP with distinct random keys for each weight and bias."""
    num_layers = len(sizes) - 1
    keys = random.split(key, num_layers * 2)
    params = []
    for layer_index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        W_key = keys[2 * layer_index]
        b_key = keys[2 * layer_index + 1]
        params.append(
            (
                random.normal(W_key, (n_out, n_in)) * 0.1,
                random.normal(b_key, (n_out,)) * 0.1,
            )
        )
    return params


def kalman_filter_step(x_prev, u, y, params, A, B, C, Q, R, P_prev):
    """One predict-update step with a learned dynamics residual."""
    learned_residual = mlp(params, jnp.concatenate([x_prev, u]))
    x_pred = A @ x_prev + B @ u + learned_residual
    P_pred = A @ P_prev @ A.T + Q

    innovation_cov = C @ P_pred @ C.T + R
    K = _kalman_gain(P_pred, C, innovation_cov)
    x_new = x_pred + K @ (y - C @ x_pred)
    P_new = _joseph_covariance_update(P_pred, K, C, R)

    return x_new, P_new


def loss_fn(params, A, B, C, Q, R, x0, P0, us, ys):
    """Observation-space MSE accumulated across a full rollout."""

    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter_step(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), C @ x_new

    (_, _), y_preds = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - y_preds) ** 2)


def predict_with_kalman(us, ys, params, A, B, C, Q, R, x0, P0):
    """Run the learned Kalman-filter model and return predictions and covariances."""

    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter_step(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), (C @ x_new, P_new)

    (_, _), (y_preds, Ps) = jax.lax.scan(step, (x0, P0), (us, ys))
    return y_preds, Ps


def train_learned_kalman(
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
    """Train the learned residual model inside the Kalman loop."""
    opt_state = optimizer.init(params)

    @jax.jit
    def update(current_params, current_opt_state):
        loss, grads = value_and_grad(loss_fn)(current_params, A, B, C, Q, R, x0, P0, us, ys)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss

    losses = []
    for step in range(num_steps):
        params, opt_state, loss = update(params, opt_state)
        losses.append(loss)
        if log_every is not None and step % log_every == 0:
            print(f"Step {step}, Loss: {float(loss):.6f}")

    return params, opt_state, jnp.stack(losses)


def compute_dynamics_error(xs, us, params, A, B, A_true=None, B_true=None):
    """Compare true nonlinear dynamics against the learned residual model."""
    real_dynamics = jax.vmap(lambda x, u: true_dynamics(x, u, A_true=A_true, B_true=B_true))(xs, us)
    learned_dynamics = jax.vmap(
        lambda x, u: A @ x + B @ u + mlp(params, jnp.concatenate([x, u]))
    )(xs, us)
    error = real_dynamics - learned_dynamics
    return error, real_dynamics, learned_dynamics
