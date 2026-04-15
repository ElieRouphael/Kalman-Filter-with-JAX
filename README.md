# Learning Dynamics with Kalman Filters in JAX

This project explores **nonlinear system identification** using a hybrid approach: combining **Kalman filtering** with a **neural network (MLP)** to model unknown nonlinearities in a state-space system. It’s implemented entirely in **JAX** for speed and automatic differentiation.

---

## 📚 Objectives

- Learn a **nonlinear component** in system dynamics using a trainable MLP.
- Integrate this learned model within a **Kalman filter** for robust state estimation.
- Visualize model performance, uncertainty margins, and approximation error.
- Provide a pedagogical example for **hybrid filtering and machine learning** in dynamical systems.

---

## 🛠️ Key Components

### System Dynamics

We define a nonlinear system with additive sinusoidal dynamics:

x_{t+1} = A x_t + B u_t + f(x_t, u_t)

where `f(x, u)` includes `sin(x)` and `cos(x)` terms, making it **nonlinear**.

### Neural Network (MLP)

A multilayer perceptron is used to learn the unknown nonlinear function `f(x, u)`:

- **Input**: `[x_t, u_t]`
- **Output**: `f_theta(x_t, u_t)`
- **Architecture**: 3 → 16 → 16 → 16 → 16 → 2 (tanh activations)

### Kalman Filter

The Kalman filter uses the learned dynamics to perform state estimation:

- **Prediction step** includes the MLP output.
- **Update step** uses standard KF equations with fixed matrices `C`, `Q`, `R`.

### Loss Function

We minimize the **mean squared error** between the observed output and the Kalman-predicted state:

loss = mean((y_true - y_pred)^2)

## 📖 Credits
Developed as a pedagogical example by a control systems lecturer and researcher in AI and system identification.

## 🤝 Contributing
If you're a student, educator, or researcher interested in:

- Control theory

- Hybrid learning models

- Kalman filtering with neural nets

Feel free to fork, explore, and contribute ideas via pull requests!

## 📬 Contact
For questions, suggestions, or collaboration opportunities, please reach out via GitHub issues or contact me directly on [LinkedIn](https://www.linkedin.com/in/elie-rouphael/).
## Updated Guide

This repository contains two small workshop-style experiments around Kalman filtering in JAX:

- `Kalman Filter with JAX.py` compares a linear Kalman filter, EKF, and UKF on synthetic position/velocity data.
- `Kalman Filter with DL.py` learns a nonlinear residual term with an MLP and plugs it into a Kalman-style state estimator.

The code is now split into a small reusable package, `kalman_jax/`, so the scripts remain easy to run while the core math is easier to test and improve.

## What the repo is doing

### 1. Classical filtering comparison

The linear-filter script generates synthetic motion data and evaluates:

- A standard linear Kalman filter
- An extended Kalman filter using the same linear dynamics
- An unscented Kalman filter using the same linear dynamics

This is mainly a sanity check and teaching example for comparing filter implementations on the same problem.

### 2. Learned nonlinear residuals

The learned-dynamics script simulates a nonlinear system of the form:

`x_(t+1) = A x_t + B u_t + f(x_t, u_t)`

where `f(x_t, u_t)` is approximated by an MLP. The filter keeps the known linear dynamics in `A` and `B`, then learns the nonlinear residual that remains.

## Project layout

- `kalman_jax/filters.py` contains the linear KF, EKF, UKF, rollout helpers, and synthetic linear-data generator.
- `kalman_jax/learned_dynamics.py` contains the nonlinear data generator, MLP helpers, training loop, and learned-Kalman rollout code.
- `tests/test_filters.py` contains regression tests for the filtering math.

## Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the examples

Run the classical filtering comparison:

```bash
python "Kalman Filter with JAX.py"
```

Run the learned-dynamics experiment:

```bash
python "Kalman Filter with DL.py"
```

Run the tests:

```bash
pytest
```

## Improvements made

- Fixed the UKF sigma-point construction so it uses covariance-root columns instead of rows.
- Replaced explicit matrix inverses with linear solves in the Kalman updates.
- Switched the linear and EKF covariance updates to Joseph form for better numerical stability.
- Fixed the MLP parameter initialization so each weight and bias uses a distinct PRNG key.
- Removed the duplicated training pass from the learned-dynamics workflow.
- Aligned the learned model's known linear dynamics with the simulator so the MLP focuses on the nonlinear residual.
- Added a minimal package layout, dependency file, `.gitignore`, and tests.

## Real-Data Air Quality Experiment

The repo now also supports a real dataset workflow using the UCI / Kaggle air-quality dataset. The new script is:

```bash
python "Air Quality with JAX.py"
```

It treats the problem as next-hour `NO2(GT)` forecasting from the current sensor array and weather readings:

- `PT08.S1(CO)`
- `PT08.S2(NMHC)`
- `PT08.S3(NOx)`
- `PT08.S4(NO2)`
- `PT08.S5(O3)`
- `T`
- `RH`
- `AH`

The script compares:

- a persistence baseline
- a linear regression baseline
- a Kalman-style learned forecaster in JAX

Place the CSV at `data/raw/AirQualityUCI.csv`, or pass `--csv` with the file path.
