import pytest

pandas = pytest.importorskip("pandas")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from kalman_jax.air_quality import DEFAULT_INPUT_COLUMNS, build_no2_forecasting_dataset, load_air_quality_frame
from kalman_jax.forecasters import forecast_with_kalman


def test_load_air_quality_frame_parses_semicolon_decimal_and_missing_values(tmp_path):
    csv_path = tmp_path / "AirQualityUCI.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;",
                "10/03/2004;18.00.00;2,6;1360;150;11,9;1046;166;1056;113;1692;1268;13,6;48,9;0,7578;;",
                "10/03/2004;19.00.00;2,0;-200;112;9,4;955;103;1174;92;1559;972;13,3;47,7;0,7255;;",
                "10/03/2004;20.00.00;2,2;1402;88;9,0;939;131;1140;114;1555;1074;11,9;54,0;0,7502;;",
                "10/03/2004;21.00.00;2,2;1376;80;9,2;948;172;1092;122;1584;1203;11,0;60,0;0,7867;;",
                "10/03/2004;22.00.00;1,6;1272;51;6,5;836;131;1205;116;1490;1110;11,2;59,6;0,7888;;",
                "10/03/2004;23.00.00;1,2;1197;38;4,7;750;89;1337;96;1393;949;11,2;59,2;0,7848;;",
                "11/03/2004;00.00.00;1,2;1185;31;3,6;690;62;1462;77;1333;733;11,3;56,8;0,7603;;",
                "11/03/2004;01.00.00;1,0;1136;31;3,3;672;62;1453;76;1333;730;10,7;60,0;0,7702;;",
                "11/03/2004;02.00.00;0,9;1094;24;2,8;609;45;1579;60;1276;620;10,7;59,7;0,7648;;",
            ]
        ),
        encoding="utf-8",
    )

    frame = load_air_quality_frame(csv_path)

    assert frame["timestamp"].is_monotonic_increasing
    assert frame["PT08.S1(CO)"].isna().sum() == 1

    dataset = build_no2_forecasting_dataset(frame, input_columns=DEFAULT_INPUT_COLUMNS, train_fraction=0.5, val_fraction=0.25)
    assert dataset["splits"]["train"]["controls"].shape[1] == len(DEFAULT_INPUT_COLUMNS)


def test_forecast_with_kalman_returns_prior_prediction_before_update():
    params = [(jnp.zeros((1, 2)), jnp.zeros((1,)))]
    A = jnp.array([[1.0]])
    B = jnp.array([[0.0]])
    C = jnp.array([[1.0]])
    Q = jnp.zeros((1, 1))
    R = jnp.array([[1e-3]])
    x0 = jnp.array([0.0])
    P0 = jnp.eye(1)
    us = jnp.zeros((2, 1))
    ys = jnp.array([[1.0], [2.0]])

    y_forecast, y_filtered, _, _ = forecast_with_kalman(params, A, B, C, Q, R, x0, P0, us, ys)

    assert jnp.allclose(y_forecast[0], jnp.array([0.0]))
    assert float(y_filtered[0][0]) > 0.5
