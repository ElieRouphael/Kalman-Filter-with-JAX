import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kalman_jax.filters import (
    compute_rmse,
    generate_linear_synthetic_data,
    run_ekf,
    run_linear_kf,
    run_ukf,
)


def main():
    key = jax.random.PRNGKey(42)
    F, H, Q, R, true_states, measurements = generate_linear_synthetic_data(key, n_steps=1000)

    x0 = jnp.array([0.0, 0.0])
    P0 = jnp.eye(2)

    def f_linear(x):
        return F @ x

    def h_linear(x):
        return H @ x

    def F_jacobian(_):
        return F

    def H_jacobian(_):
        return H

    linear_states = run_linear_kf(measurements, F, H, Q, R, x0, P0)
    ekf_states = run_ekf(measurements, f_linear, h_linear, F_jacobian, H_jacobian, Q, R, x0, P0)
    ukf_states = run_ukf(measurements, f_linear, h_linear, Q, R, x0, P0)

    linear_rmse = compute_rmse(true_states, linear_states)
    ekf_rmse = compute_rmse(true_states, ekf_states)
    ukf_rmse = compute_rmse(true_states, ukf_states)

    print("RMSE Comparison:")
    print(f"Linear KF - Position: {linear_rmse[0]:.4f}, Velocity: {linear_rmse[1]:.4f}")
    print(f"EKF - Position: {ekf_rmse[0]:.4f}, Velocity: {ekf_rmse[1]:.4f}")
    print(f"UKF - Position: {ukf_rmse[0]:.4f}, Velocity: {ukf_rmse[1]:.4f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(true_states[:, 0], "k-", label="True Position")
    plt.plot(measurements, "r.", label="Measurements", markersize=4)
    plt.plot(linear_states[:, 0], "b-", label="Linear KF")
    plt.plot(ekf_states[:, 0], "g--", label="EKF")
    plt.plot(ukf_states[:, 0], "m:", label="UKF")
    plt.ylabel("Position")
    plt.title("Position Tracking")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(true_states[:, 1], "k-", label="True Velocity")
    plt.plot(linear_states[:, 1], "b-", label="Linear KF")
    plt.plot(ekf_states[:, 1], "g--", label="EKF")
    plt.plot(ukf_states[:, 1], "m:", label="UKF")
    plt.ylabel("Velocity")
    plt.title("Velocity Tracking")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
