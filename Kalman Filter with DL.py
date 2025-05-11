import jax
import jax.numpy as jnp
import optax
from jax import random, jit, grad, value_and_grad
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# ---- System Dynamics ---- #
def true_dynamics(x, u):
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [0.1]])
    return A @ x + B @ u + 3 * jnp.sin(x[0]) + 2 * jnp.cos(x[1])

def generate_data(T=100):
    x = jnp.zeros((2,))
    xs, us, ys = [], [], []
    key = random.PRNGKey(0)
    for t in range(T):
        key, subkey1, subkey2 = random.split(key, 3)
        u = random.normal(subkey1, (1,))
        x = true_dynamics(x, u)
        y = x + 0.05 * random.normal(subkey2, x.shape)
        xs.append(x)
        us.append(u)
        ys.append(y)
    return jnp.stack(xs), jnp.stack(us), jnp.stack(ys)

xs, us, ys = generate_data(50)

# ---- MLP ---- #
def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.tanh(W @ x + b)
    W, b = params[-1]
    return W @ x + b

def init_mlp_params(key, sizes):
    keys = random.split(key, len(sizes) * 2)
    return [(random.normal(keys[i], (n_out, n_in)) * 0.1,
             random.normal(keys[i + 1], (n_out,)) * 0.1)
            for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]))]

# ---- Kalman Filter ---- #
def kalman_filter(x_prev, u, y, params, A, B, C, Q, R, P_prev):
    f_theta = mlp(params, jnp.concatenate([x_prev, u]))
    x_pred = A @ x_prev + B @ u + f_theta
    P_pred = A @ P_prev @ A.T + Q
    K = P_pred @ C.T @ jnp.linalg.inv(C @ P_pred @ C.T + R)
    x_new = x_pred + K @ (y - C @ x_pred)
    P_new = (jnp.eye(len(x_prev)) - K @ C) @ P_pred
    return x_new, P_new

# ---- Loss ---- #
def loss_fn(params, A, B, C, Q, R, x0, P0, us, ys):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), C @ x_new

    (_, _), y_preds = jax.lax.scan(step, (x0, P0), (us, ys))
    return jnp.mean((ys - y_preds) ** 2)

def predict_with_kalman(us, ys, params, A, B, C, Q, R, x0, P0):
    def step(carry, inputs):
        x, P = carry
        u, y = inputs
        x_new, P_new = kalman_filter(x, u, y, params, A, B, C, Q, R, P)
        return (x_new, P_new), (C @ x_new, P_new)

    (_, _), (y_preds, Ps) = jax.lax.scan(step, (x0, P0), (us, ys))
    return y_preds, Ps

# ---- Initialize Parameters ---- #
key_init = random.PRNGKey(42)
params = init_mlp_params(key_init, [3, 16, 16, 16, 16, 2])  # x(2) + u(1) = 3 input, 2 output

A = jnp.array([[5.0, 0.1], [0.1, 2.0]])
B = jnp.array([[0.0], [0.2]])
C = jnp.eye(2)
Q = 0.1 * jnp.eye(2)
R = 0.5 * jnp.eye(2)
x0 = jnp.zeros(2)
P0 = jnp.eye(2)

# ---- Optimizer Setup ---- #
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# ---- Training Step ---- #
@jit
def update(params, opt_state):
    loss, grads = value_and_grad(loss_fn)(params, A, B, C, Q, R, x0, P0, us, ys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---- Training Loop ---- #
for step in range(1000):
    params, opt_state, loss = update(params, opt_state)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")
        
y_preds_kf, Ps = predict_with_kalman(us, ys, params, A, B, C, Q, R, x0, P0)
# Convert JAX arrays to NumPy for plotting
y_preds_kf = np.array(y_preds_kf)
Ps = np.array(Ps)
stds = np.sqrt(np.array([np.diag(P) for P in Ps]))  # shape (T, 2)

def plot_kalman_margins(preds, targets, stds, factor=1.96):
    timesteps = np.arange(len(preds))
    for i in range(preds.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.plot(timesteps, targets[:, i], label=f'True x[{i}]')
        plt.plot(timesteps, preds[:, i], label=f'KF Pred x[{i}]')
        plt.fill_between(timesteps,
                         preds[:, i] - factor * stds[:, i],
                         preds[:, i] + factor * stds[:, i],
                         alpha=0.3, label=f'Margin ±{factor}σ')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Kalman Prediction x[{i}] with ±{factor}σ Margin')
        plt.legend()
        plt.grid(True)
        plt.show()

        
# ---- Plotting ---- #

# ---- Function to plot loss during training ---- #
def plot_loss(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(losses)), losses, label="Training Loss")
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# ---- Function to plot error ---- #
def plot_error(xs, us, params, A, B):
    # Compute the real dynamics and the learned dynamics
    real_dynamics = [true_dynamics(x, u) for x, t, u in zip(xs, range(len(xs)), us)]
    learned_dynamics = [A @ x + B @ u + mlp(params, jnp.concatenate([x, u])) for x, u in zip(xs, us)]
    
    # Calculate the error between the real and learned dynamics
    real_dynamics = np.array(real_dynamics)
    learned_dynamics = np.array(learned_dynamics)
    error = real_dynamics - learned_dynamics
    
    # Plot the error
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(xs)), error[:, 0], label="Error in x[0]")
    plt.plot(np.arange(len(xs)), error[:, 1], label="Error in x[1]")
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.title('Error Between Real and Learned Dynamics')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return error, real_dynamics, learned_dynamics

# ---- Training Loop ---- #
losses = []
for step in range(1000):
    params, opt_state, loss = update(params, opt_state)
    losses.append(loss)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.6f}")

# ---- Plot Training Loss ---- #
plot_loss(losses)

# ---- Plot Error ---- #
error, Rd, Ld = plot_error(xs, us, params, A, B)
# ---- Plot Kalman Margins ---- #
plot_kalman_margins(y_preds_kf, ys, stds)

'''
plt.plot(np.arange(len(xs)), Rd[:, 0], label="Real Dynamics x[0]")
plt.plot(np.arange(len(xs)), Ld[:, 0], label="Learned Dynamics x[0]")
plt.xlabel('Time Step')
plt.ylabel('Dynamics')
plt.title('Real vs Learned Dynamics x[0]')
plt.legend()
plt.grid(True)
plt.show() 

plt.plot(np.arange(len(xs)), Rd[:, 1], label="Real Dynamics x[1]")
plt.plot(np.arange(len(xs)), Ld[:, 1], label="Learned Dynamics x[1]")
plt.xlabel('Time Step')
plt.ylabel('Dynamics')
plt.title('Real vs Learned Dynamics x[1]')
plt.legend()
plt.grid(True)
plt.show()
'''