import jax
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

## Pure functional Kalman Filter implementations

def linear_kf_predict(x, P, F, Q):
    """Pure function for linear KF prediction step"""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def linear_kf_update(x, P, z, H, R):
    """Pure function for linear KF update step"""
    y = z - H @ x  # Innovation/residual
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ jnp.linalg.inv(S)  # Kalman gain
    
    x_updated = x + K @ y
    P_updated = (jnp.eye(P.shape[0]) - K @ H) @ P
    
    return x_updated, P_updated, y, S

def extended_kf_predict(x, P, f, F_jacobian, Q):
    """Pure function for EKF prediction step"""
    x_pred = f(x)
    F = F_jacobian(x)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def extended_kf_update(x, P, z, h, H_jacobian, R):
    """Pure function for EKF update step"""
    H = H_jacobian(x)
    y = z - h(x)  # Innovation/residual
    S = H @ P @ H.T + R  # Innovation covariance
    K = P @ H.T @ jnp.linalg.inv(S)  # Kalman gain
    
    x_updated = x + K @ y
    P_updated = (jnp.eye(P.shape[0]) - K @ H) @ P
    
    return x_updated, P_updated, y, S

def unscented_kf_predict(x, P, f, Q, alpha=1e-4, beta=2, kappa=0):
    """Pure function for UKF prediction step"""
    n = x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    gamma = jnp.sqrt(n + lambda_)
    
    # Generate sigma points
    sqrt_P = jnp.linalg.cholesky(P)
    sigma_points = jnp.zeros((2 * n + 1, n))
    sigma_points = sigma_points.at[0].set(x)
    for i in range(n):
        sigma_points = sigma_points.at[i + 1].set(x + gamma * sqrt_P[i])
        sigma_points = sigma_points.at[i + 1 + n].set(x - gamma * sqrt_P[i])
    
    # Transform sigma points
    transformed_points = jax.vmap(f)(sigma_points)
    
    # Compute weights
    Wm = jnp.zeros(2 * n + 1)
    Wc = jnp.zeros(2 * n + 1)
    Wm = Wm.at[0].set(lambda_ / (n + lambda_))
    Wc = Wc.at[0].set(lambda_ / (n + lambda_) + (1 - alpha**2 + beta))
    for i in range(1, 2 * n + 1):
        Wm = Wm.at[i].set(1 / (2 * (n + lambda_)))
        Wc = Wc.at[i].set(1 / (2 * (n + lambda_)))
    
    # Compute predicted state and covariance
    x_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    P_pred = jnp.zeros((n, n))
    for i in range(2 * n + 1):
        diff = transformed_points[i] - x_pred
        P_pred += Wc[i] * jnp.outer(diff, diff)
    P_pred += Q
    
    return x_pred, P_pred

def unscented_kf_update(x, P, z, h, R, alpha=1e-4, beta=2, kappa=0):
    """Pure function for UKF update step"""
    n = x.shape[0]
    m = z.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n
    gamma = jnp.sqrt(n + lambda_)
    
    # Generate sigma points
    sqrt_P = jnp.linalg.cholesky(P + 1e-6 * jnp.eye(P.shape[0]))
    sigma_points = jnp.zeros((2 * n + 1, n))
    sigma_points = sigma_points.at[0].set(x)
    for i in range(n):
        sigma_points = sigma_points.at[i + 1].set(x + gamma * sqrt_P[i])
        sigma_points = sigma_points.at[i + 1 + n].set(x - gamma * sqrt_P[i])
    
    # Transform sigma points through observation function
    transformed_points = jax.vmap(h)(sigma_points)
    
    # Compute weights
    Wm = jnp.zeros(2 * n + 1)
    Wc = jnp.zeros(2 * n + 1)
    Wm = Wm.at[0].set(lambda_ / (n + lambda_))
    Wc = Wc.at[0].set(lambda_ / (n + lambda_) + (1 - alpha**2 + beta))
    for i in range(1, 2 * n + 1):
        Wm = Wm.at[i].set(1 / (2 * (n + lambda_)))
        Wc = Wc.at[i].set(1 / (2 * (n + lambda_)))
    
    # Compute predicted measurement and covariance
    z_pred = jnp.sum(Wm[:, None] * transformed_points, axis=0)
    P_zz = jnp.zeros((m, m))
    P_xz = jnp.zeros((n, m))
    for i in range(2 * n + 1):
        z_diff = transformed_points[i] - z_pred
        x_diff = sigma_points[i] - x
        P_zz += Wc[i] * jnp.outer(z_diff, z_diff)
        P_xz += Wc[i] * jnp.outer(x_diff, z_diff)
    P_zz += R
    
    # Compute Kalman gain and update
    P_zz += 1e-6 * jnp.eye(P_zz.shape[0])
    K = P_xz @ jnp.linalg.inv(P_zz)
    y = z - z_pred
    x_updated = x + K @ y
    P_updated = P - K @ P_zz @ K.T
    
    return x_updated, P_updated, y, P_zz

## Test on Synthetic Data (same as before)

def generate_synthetic_data(key, n_steps=100):
    """Generate synthetic 1D position and velocity data."""
    dt = 0.1
    F = jnp.array([[1, dt], [0, 1]])
    H = jnp.array([[1, 0]])
    
    process_noise_std = 0.1
    measurement_noise_std = 0.5
    
    Q = jnp.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]) * process_noise_std**2
    R = jnp.array([[measurement_noise_std**2]])
    
    x_true = jnp.array([0.0, 1.0])
    
    true_states = []
    measurements = []
    
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        process_noise = jax.random.multivariate_normal(subkey, jnp.zeros(2), Q)
        x_true = F @ x_true + process_noise
        true_states.append(x_true)
        
        key, subkey = jax.random.split(key)
        measurement_noise = jax.random.normal(subkey, (1,)) * measurement_noise_std
        z = H @ x_true + measurement_noise
        measurements.append(z)
    
    true_states = jnp.array(true_states)
    measurements = jnp.vstack(measurements)
    
    return F, H, Q, R, true_states, measurements

# Generate synthetic data
F, H, Q, R, true_states, measurements = generate_synthetic_data(key)

## Run Filters (pure functional style)

def run_linear_kf(measurements, F, H, Q, R, x0, P0):
    """Run linear KF on measurements"""
    x, P = x0, P0
    estimated_states = []
    
    for z in measurements:
        x, P = linear_kf_predict(x, P, F, Q)
        x, P, _, _ = linear_kf_update(x, P, z, H, R)
        estimated_states.append(x)
    
    return jnp.array(estimated_states)

def run_ekf(measurements, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
    """Run EKF on measurements"""
    x, P = x0, P0
    estimated_states = []
    
    for z in measurements:
        x, P = extended_kf_predict(x, P, f, F_jacobian, Q)
        x, P, _, _ = extended_kf_update(x, P, z, h, H_jacobian, R)
        estimated_states.append(x)
    
    return jnp.array(estimated_states)

def run_ukf(measurements, f, h, Q, R, x0, P0):
    """Run UKF on measurements"""
    x_pred, P_pred = x0, P0
    x, P = x0, P0
    estimated_states = []
    
    for z in measurements:
        x_pred, P_pred = unscented_kf_predict(x, P, f, Q)
        x, P, _, _ = unscented_kf_update(x_pred, P_pred, z, h, R)
        estimated_states.append(x)
    
    return jnp.array(estimated_states)

# Initial state and covariance
x0 = jnp.array([0.0, 0.0])
P0 = jnp.eye(2) * 1.0

# For EKF/UKF we'll use the same linear functions for comparison
def f_ekf(x): return F @ x
def h_ekf(x): return H @ x
def F_jacobian(x): return F
def H_jacobian(x): return H

# Run all filters
lkf_states = run_linear_kf(measurements, F, H, Q, R, x0, P0)
ekf_states = run_ekf(measurements, f_ekf, h_ekf, F_jacobian, H_jacobian, Q, R, x0, P0)
ukf_states = run_ukf(measurements, f_ekf, h_ekf, Q, R, x0, P0)

## Compare Results

def compute_rmse(true, est):
    """Compute RMSE between true and estimated states."""
    return jnp.sqrt(jnp.mean((true - est)**2, axis=0))

lkf_rmse = compute_rmse(true_states, lkf_states)
ekf_rmse = compute_rmse(true_states, ekf_states)
ukf_rmse = compute_rmse(true_states, ukf_states)

print("RMSE Comparison:")
print(f"Linear KF - Position: {lkf_rmse[0]:.4f}, Velocity: {lkf_rmse[1]:.4f}")
print(f"EKF - Position: {ekf_rmse[0]:.4f}, Velocity: {ekf_rmse[1]:.4f}")
print(f"UKF - Position: {ukf_rmse[0]:.4f}, Velocity: {ukf_rmse[1]:.4f}")

## Plot Results

plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(true_states[:, 0], 'k-', label='True Position')
plt.plot(measurements, 'r.', label='Measurements', markersize=4)
plt.plot(lkf_states[:, 0], 'b-', label='Linear KF')
plt.plot(ekf_states[:, 0], 'g--', label='EKF')
plt.plot(ukf_states[:, 0], 'm:', label='UKF')
plt.ylabel('Position')
plt.title('Position Tracking')
plt.legend()

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(true_states[:, 1], 'k-', label='True Velocity')
plt.plot(lkf_states[:, 1], 'b-', label='Linear KF')
plt.plot(ekf_states[:, 1], 'g--', label='EKF')
plt.plot(ukf_states[:, 1], 'm:', label='UKF')
plt.ylabel('Velocity')
plt.title('Velocity Tracking')
plt.legend()

plt.tight_layout()
plt.show()