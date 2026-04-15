"""Utilities for the UCI / Kaggle air-quality time-series dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TARGET_COLUMN = "NO2(GT)"
DEFAULT_INPUT_COLUMNS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


@dataclass(frozen=True)
class Standardizer:
    """Simple mean/std standardizer used for train-only normalization."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float32)
        return (values * self.std + self.mean).astype(np.float32)


def resolve_air_quality_csv(csv_path=None, search_root="data/raw"):
    """Resolve a dataset path or search the default raw-data directory."""
    if csv_path is not None:
        return Path(csv_path)

    root = Path(search_root)
    candidates = [
        root / "AirQualityUCI.csv",
        *sorted(root.glob("*Air*Quality*.csv")),
        *sorted(root.glob("*.csv")),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the air-quality CSV. Download the dataset and place the file "
        "at data/raw/AirQualityUCI.csv, or pass --csv with the dataset path."
    )


def _read_air_quality_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    parse_attempts = (
        {"sep": ";", "decimal": ","},
        {"sep": ",", "decimal": "."},
    )
    expected_columns = {"Date", "Time", DEFAULT_TARGET_COLUMN}

    for options in parse_attempts:
        frame = pd.read_csv(csv_path, **options)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = frame.loc[:, [column for column in frame.columns if column and not column.startswith("Unnamed")]]
        if expected_columns.issubset(frame.columns):
            return frame

    raise ValueError(
        "Could not parse the air-quality CSV. Expected Date, Time, and NO2(GT) columns "
        f"in {csv_path}."
    )


def load_air_quality_frame(csv_path):
    """Load and clean the air-quality frame from the UCI / Kaggle CSV."""
    frame = _read_air_quality_csv(csv_path).copy()

    numeric_columns = [column for column in frame.columns if column not in {"Date", "Time"}]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[numeric_columns] = frame[numeric_columns].replace(-200, np.nan)

    date_str = frame["Date"].astype(str).str.strip()
    time_str = frame["Time"].astype(str).str.strip().str.replace(".", ":", regex=False)
    frame["timestamp"] = pd.to_datetime(
        date_str + " " + time_str,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _fit_standardizer(values):
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _time_split_bounds(n_rows, train_fraction, val_fraction):
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1.")

    train_end = int(n_rows * train_fraction)
    val_end = train_end + int(n_rows * val_fraction)
    if train_end < 2 or val_end <= train_end or val_end >= n_rows:
        raise ValueError("Not enough rows to form train/validation/test time splits.")
    return train_end, val_end


def build_no2_forecasting_dataset(
    frame,
    *,
    input_columns=None,
    target_column=DEFAULT_TARGET_COLUMN,
    horizon=1,
    train_fraction=0.7,
    val_fraction=0.15,
):
    """Build time-ordered train/validation/test splits for one-step NO2 forecasting."""
    input_columns = DEFAULT_INPUT_COLUMNS if input_columns is None else list(input_columns)
    required_columns = ["timestamp", *input_columns, target_column]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Frame is missing required columns: {missing_columns}")

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    model_frame = frame.dropna(subset=input_columns + [target_column]).copy()
    model_frame["target_next"] = model_frame[target_column].shift(-horizon)
    model_frame = model_frame.dropna(subset=["target_next"]).reset_index(drop=True)

    controls_raw = model_frame[input_columns].to_numpy(dtype=np.float32)
    current_target_raw = model_frame[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)
    future_target_raw = model_frame["target_next"].to_numpy(dtype=np.float32).reshape(-1, 1)
    timestamps = model_frame["timestamp"].to_numpy()

    train_end, val_end = _time_split_bounds(len(model_frame), train_fraction, val_fraction)

    control_scaler = _fit_standardizer(controls_raw[:train_end])
    target_scaler = _fit_standardizer(current_target_raw[:train_end])

    controls = control_scaler.transform(controls_raw)
    current_target = target_scaler.transform(current_target_raw)
    future_target = target_scaler.transform(future_target_raw)

    train_effective_end = train_end - horizon
    val_effective_end = val_end - horizon
    if train_effective_end <= 0 or val_effective_end <= train_end:
        raise ValueError("Forecast horizon is too large for the chosen split sizes.")

    def make_split(start, end):
        return {
            "timestamps": timestamps[start:end],
            "controls_raw": controls_raw[start:end],
            "controls": controls[start:end],
            "current_target_raw": current_target_raw[start:end],
            "current_target": current_target[start:end],
            "future_target_raw": future_target_raw[start:end],
            "future_target": future_target[start:end],
        }

    return {
        "target_column": target_column,
        "input_columns": input_columns,
        "horizon": horizon,
        "control_scaler": control_scaler,
        "target_scaler": target_scaler,
        "splits": {
            "train": make_split(0, train_effective_end),
            "val": make_split(train_end, val_effective_end),
            "test": make_split(val_end, len(model_frame)),
        },
    }


def fit_linear_regression(features, targets):
    """Fit a simple least-squares baseline without introducing sklearn."""
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    design = np.concatenate([features, np.ones((len(features), 1), dtype=np.float32)], axis=1)
    weights, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return weights.astype(np.float32)


def predict_linear_regression(weights, features):
    """Predict with the least-squares baseline."""
    features = np.asarray(features, dtype=np.float32)
    design = np.concatenate([features, np.ones((len(features), 1), dtype=np.float32)], axis=1)
    return design @ weights


def persistence_forecast(current_target):
    """A strong baseline for hourly pollutant series: next value equals current value."""
    return np.asarray(current_target, dtype=np.float32)


def rmse(y_true, y_pred):
    """Root-mean-squared error."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))
