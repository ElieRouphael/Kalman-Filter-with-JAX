# Data Files

Place the downloaded air-quality CSV in `data/raw/`.

Expected filename:

```text
data/raw/AirQualityUCI.csv
```

You can also point the real-data script at a different location:

```bash
python "Air Quality with JAX.py" --csv "path/to/AirQualityUCI.csv"
```

The air-quality experiment expects the UCI / Kaggle air-quality dataset with columns such as:

- `NO2(GT)`
- `PT08.S1(CO)`
- `PT08.S2(NMHC)`
- `PT08.S3(NOx)`
- `PT08.S4(NO2)`
- `PT08.S5(O3)`
- `T`
- `RH`
- `AH`
