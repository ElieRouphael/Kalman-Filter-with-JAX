import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from kalman_jax.filters import (
    generate_linear_synthetic_data,
    run_ekf,
    run_linear_kf,
    run_ukf,
    unscented_kf_predict,
)


def test_unscented_predict_preserves_identity_dynamics():
    x = jnp.array([1.0, -2.0])
    P = jnp.array([[2.0, 0.6], [0.6, 1.0]])
    Q = jnp.zeros((2, 2))

    x_pred, P_pred = unscented_kf_predict(x, P, lambda state: state, Q)

    assert jnp.allclose(x_pred, x, atol=1e-6)
    assert jnp.allclose(P_pred, P, atol=1e-6)


def test_linear_ekf_and_ukf_match_on_linear_system():
    key = jax.random.PRNGKey(0)
    F, H, Q, R, _, measurements = generate_linear_synthetic_data(
        key,
        n_steps=50,
        process_noise_std=0.05,
        measurement_noise_std=0.1,
    )
    x0 = jnp.array([0.0, 0.0])
    P0 = jnp.eye(2)

    f = lambda state: F @ state
    h = lambda state: H @ state
    F_jacobian = lambda _: F
    H_jacobian = lambda _: H

    linear_states = run_linear_kf(measurements, F, H, Q, R, x0, P0)
    ekf_states = run_ekf(measurements, f, h, F_jacobian, H_jacobian, Q, R, x0, P0)
    ukf_states = run_ukf(measurements, f, h, Q, R, x0, P0)

    assert jnp.allclose(linear_states, ekf_states, atol=1e-6)
    assert jnp.allclose(linear_states, ukf_states, atol=1e-4)
