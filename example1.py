import pandas as pd
import matplotlib.pyplot as plt

from rls_assimilation.RLSAssimilation import RLSAssimilation
from helpers import (
    plot_data,
    print_metrics,
)


def demo_assimilation_and_plot(
    all_data_df,
    variable,
    ax_data,
    s_in1,
    s_in2,
    s_out,
    col_in1,
    col_in2,
    obs_source_title,
    obs_source_color,
    with_legend,
):
    if s_in1 == s_in2:
        scenario = "DA2"
    elif s_out == s_in1:
        scenario = f"DA3 (Model -> {obs_source_title})"
    elif s_out == s_in2:
        scenario = f"DA3 ({obs_source_title} -> Model)"
    else:
        raise ValueError("Unsupported testing parameters")

    observations_source1 = all_data_df[col_in1].values  # observations from source 1
    observations_source2 = all_data_df[col_in2].values  # observations from source 2
    n_observations = len(observations_source1)

    # assimilated (weighted) values and errors
    assimilated = []
    err_assimilated = []

    assimilator = RLSAssimilation(
        t_in1="hourly",
        t_in2="hourly",
        s_in1=s_in1,
        s_in2=s_in2,
        t_out="hourly",
        s_out=s_out,
    )

    # assimilate
    for k in range(n_observations):
        # Step 1: Obtain raw observations from 2 sources
        latest_observation_sensor1 = observations_source1[k]
        latest_observation_sensor2 = observations_source2[k]

        # Step 2: Assimilate
        (
            assimilated_obs_calibrated,
            err_assimilated_obs_calibrated,
        ) = assimilator.assimilate(
            latest_observation_sensor1,
            latest_observation_sensor2,
        )
        assimilated.append(assimilated_obs_calibrated)
        err_assimilated.append(err_assimilated_obs_calibrated)

    # plot and print metrics
    ax_data = plot_data(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated, index=all_data_df.index),
        variable,
        ax_data,
        scenario,
        obs_source_title,
        obs_source_color,
        with_legend,
    )

    err1_r = None
    err2_r = None
    if scenario == f"DA3 (Model -> {obs_source_title})":
        err1_ar = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors(force_ar_errors=True)
        err2_r = assimilator.source2.get_all_errors()
    elif scenario == f"DA3 ({obs_source_title} -> Model)":
        err1_ar = assimilator.source1.get_all_errors(force_ar_errors=True)
        err1_r = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors()
    else:
        err1_ar = assimilator.source1.get_all_errors()
        err2_ar = assimilator.source2.get_all_errors()
    err_r = err1_r if err1_r else err2_r

    print(f"{variable} metrics")
    print_metrics(
        observations_source1,
        observations_source2,
        assimilated,
        err1_ar,
        err2_ar,
        err_r,
        err_assimilated,
        scenario,
        obs_source_title,
    )

    return (ax_data,)


def test_scenario(s_in1, s_in2, s_out, scenario_id):
    data_path = "data/liivalaia_aq_meas_with_forecast.csv"  # for autumn, or liivalaia_aq_meas_with_forecast.csv - for winter
    all_data_df = pd.read_csv(data_path, index_col=0)
    all_data_df.index = pd.to_datetime(
        list(all_data_df.index), format="%Y-%m-%d %H:%M:%S"
    )
    all_data_df = all_data_df.sort_index()

    variables = ["CO", "NO2", "O3", "SO2", "PM2.5", "PM10"]

    fig_data, axs_data = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    for idx, variable in enumerate(variables):
        (axs_data[idx % 3, idx % 2],) = demo_assimilation_and_plot(
            all_data_df,
            variable,
            axs_data[idx % 3, idx % 2],
            s_in1,
            s_in2,
            s_out,
            f"{variable}",
            f"{variable}_model",
            "Station",
            "red",
            variable == "CO",
        )
    fig_data.savefig(
        f"plots/Liivalaia/data-{scenario_id}.jpg"
    )  # the autumn data, directory Liivalaia2 is used for the winter data


def test_scenario_iot(s_in1, s_in2, s_out, scenario_id, sensor_col):
    data_path = "data/liivalaia_pm10_iot.csv"
    all_data_df = pd.read_csv(data_path, index_col=0)
    all_data_df.index = pd.to_datetime(
        list(all_data_df.index), format="%Y-%m-%d %H:%M:%S"
    )
    all_data_df = all_data_df.sort_index()

    variables = ["PM10"]

    fig_data, axs_data = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

    for idx, variable in enumerate(variables):
        (axs_data) = demo_assimilation_and_plot(
            all_data_df,
            variable,
            axs_data,
            s_in1,
            s_in2,
            s_out,
            sensor_col,
            "Model",
            "Sensor",
            "orange" if sensor_col == "60m" else "orchid",
            True,
        )
    fig_data.savefig(
        f"plots/Liivalaia/IoT/data-pm10-iot-{sensor_col}-{scenario_id}.jpg"
    )


# # NB: keep model as the second source (s_in2, not s_in1)
# TESTS with station data as observations:
print("Liivalaia station")
test_scenario(s_in1="obs", s_in2="obs", s_out="obs", scenario_id="da2")  # DA2
test_scenario(
    s_in1="obs", s_in2="model", s_out="obs", scenario_id="da3-1"
)  # DA3: model -> obs
test_scenario(
    s_in1="obs", s_in2="model", s_out="model", scenario_id="da3-2"
)  # DA3: obs -> model

# TESTS with IoT sensors as observations:
# 60 meters away from the station:
print("IoT - 60 meters away from the station")
test_scenario_iot(
    s_in1="obs", s_in2="obs", s_out="obs", scenario_id="da2", sensor_col="60m"
)  # DA2
test_scenario_iot(
    s_in1="obs", s_in2="model", s_out="obs", scenario_id="da3-1", sensor_col="60m"
)  # DA3: model -> obs
test_scenario_iot(
    s_in1="obs", s_in2="model", s_out="model", scenario_id="da3-2", sensor_col="60m"
)  # DA3: obs -> model

# 700 meters away from the station:
print("IoT - 700 meters away from the station")
test_scenario_iot(
    s_in1="obs", s_in2="obs", s_out="obs", scenario_id="da2", sensor_col="700m"
)  # DA2
test_scenario_iot(
    s_in1="obs", s_in2="model", s_out="obs", scenario_id="da3-1", sensor_col="700m"
)  # DA3: model -> obs
test_scenario_iot(
    s_in1="obs", s_in2="model", s_out="model", scenario_id="da3-2", sensor_col="700m"
)  # DA3: obs -> model
