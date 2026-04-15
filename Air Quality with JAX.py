from __future__ import annotations

import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import random

from kalman_jax.air_quality import (
    DEFAULT_INPUT_COLUMNS,
    build_no2_forecasting_dataset,
    fit_linear_regression,
    load_air_quality_frame,
    mae,
    persistence_forecast,
    predict_linear_regression,
    resolve_air_quality_csv,
    rmse,
)
from kalman_jax.forecasters import forecast_with_kalman, train_kalman_forecaster
from kalman_jax.learned_dynamics import init_mlp_params


def parse_args():
    parser = argparse.ArgumentParser(description="Train a JAX Kalman-style forecaster on air-quality data.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to AirQualityUCI.csv. Defaults to data/raw/AirQualityUCI.csv if present.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of optimization steps for the learned forecaster.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in hours. Default: 1.",
    )
    return parser.parse_args()


def evaluate_split(name, split, *, linear_weights, target_scaler, forecaster_outputs):
    y_forecast_scaled, _, S_forecast, _ = forecaster_outputs
    y_true = target_scaler.inverse_transform(split["future_target"])
    y_forecast = target_scaler.inverse_transform(np.array(y_forecast_scaled))
    y_linear = target_scaler.inverse_transform(predict_linear_regression(linear_weights, split["controls"]))
    y_persistence = target_scaler.inverse_transform(persistence_forecast(split["current_target"]))

    print(f"\n{name} metrics for next-hour NO2 prediction:")
    print(f"  Persistence RMSE: {rmse(y_true, y_persistence):.3f} | MAE: {mae(y_true, y_persistence):.3f}")
    print(f"  Linear      RMSE: {rmse(y_true, y_linear):.3f} | MAE: {mae(y_true, y_linear):.3f}")
    print(f"  Kalman-MLP  RMSE: {rmse(y_true, y_forecast):.3f} | MAE: {mae(y_true, y_forecast):.3f}")

    return y_true, y_persistence, y_linear, y_forecast, np.array(S_forecast)


def plot_results(split, y_true, y_persistence, y_linear, y_forecast, S_forecast, target_scale_std):
    timestamps = split["timestamps"]
    forecast_std = np.sqrt(np.maximum(S_forecast[:, 0, 0], 1e-8)) * float(target_scale_std)

    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, y_true[:, 0], label="True next-hour NO2")
    plt.plot(timestamps, y_persistence[:, 0], label="Persistence baseline", alpha=0.8)
    plt.plot(timestamps, y_linear[:, 0], label="Linear baseline", alpha=0.8)
    plt.plot(timestamps, y_forecast[:, 0], label="Kalman-MLP forecast", linewidth=2)
    plt.fill_between(
        timestamps,
        y_forecast[:, 0] - 1.96 * forecast_std,
        y_forecast[:, 0] + 1.96 * forecast_std,
        alpha=0.2,
        label="Kalman-MLP 95% interval",
    )
    plt.title("Air-quality forecasting on the NO2 target")
    plt.ylabel("NO2(GT)")
    plt.xlabel("Timestamp")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    csv_path = resolve_air_quality_csv(args.csv)

    frame = load_air_quality_frame(csv_path)
    dataset = build_no2_forecasting_dataset(frame, input_columns=DEFAULT_INPUT_COLUMNS, horizon=args.horizon)

    train_split = dataset["splits"]["train"]
    val_split = dataset["splits"]["val"]
    test_split = dataset["splits"]["test"]

    print(f"Loaded dataset: {csv_path}")
    print(f"Rows after cleaning and horizon shift: {len(train_split['controls']) + len(val_split['controls']) + len(test_split['controls'])}")
    print(f"Input columns: {', '.join(dataset['input_columns'])}")
    print(f"Target column: {dataset['target_column']}")
    print(
        "Split sizes -> "
        f"train: {len(train_split['controls'])}, "
        f"val: {len(val_split['controls'])}, "
        f"test: {len(test_split['controls'])}"
    )

    linear_weights = fit_linear_regression(train_split["controls"], train_split["future_target"])

    input_dim = train_split["controls"].shape[1]
    params = init_mlp_params(random.PRNGKey(42), [1 + input_dim, 32, 32, 1])

    A = jnp.array([[0.9]], dtype=jnp.float32)
    B = jnp.zeros((1, input_dim), dtype=jnp.float32)
    C = jnp.array([[1.0]], dtype=jnp.float32)
    Q = jnp.array([[0.05]], dtype=jnp.float32)
    R = jnp.array([[0.25]], dtype=jnp.float32)
    P0 = jnp.eye(1, dtype=jnp.float32)

    optimizer = optax.adam(1e-3)
    params, _, losses = train_kalman_forecaster(
        params,
        optimizer,
        A=A,
        B=B,
        C=C,
        Q=Q,
        R=R,
        x0=jnp.asarray(train_split["current_target"][0]),
        P0=P0,
        us=jnp.asarray(train_split["controls"]),
        ys=jnp.asarray(train_split["future_target"]),
        num_steps=args.steps,
        log_every=100,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(np.array(losses))
    plt.title("Kalman-MLP forecast loss")
    plt.xlabel("Training step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()

    target_scaler = dataset["target_scaler"]

    val_outputs = forecast_with_kalman(
        params,
        A,
        B,
        C,
        Q,
        R,
        jnp.asarray(val_split["current_target"][0]),
        P0,
        jnp.asarray(val_split["controls"]),
        jnp.asarray(val_split["future_target"]),
    )
    evaluate_split(
        "Validation",
        val_split,
        linear_weights=linear_weights,
        target_scaler=target_scaler,
        forecaster_outputs=val_outputs,
    )

    test_outputs = forecast_with_kalman(
        params,
        A,
        B,
        C,
        Q,
        R,
        jnp.asarray(test_split["current_target"][0]),
        P0,
        jnp.asarray(test_split["controls"]),
        jnp.asarray(test_split["future_target"]),
    )
    y_true, y_persistence, y_linear, y_forecast, S_forecast = evaluate_split(
        "Test",
        test_split,
        linear_weights=linear_weights,
        target_scaler=target_scaler,
        forecaster_outputs=test_outputs,
    )

    plot_results(test_split, y_true, y_persistence, y_linear, y_forecast, S_forecast, target_scaler.std[0, 0])


if __name__ == "__main__":
    main()
