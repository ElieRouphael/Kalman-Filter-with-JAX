# 🧠 Learning Dynamics with Kalman Filters in JAX

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
