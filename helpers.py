from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 22})


def get_rmse(arr1, arr2):
    return np.sqrt(((arr1 - arr2) ** 2).mean()).round(2)


def get_uncertainty_stats(err):
    return f"{np.mean(err).round(2)} ± {np.std(err).round(2)} [{np.min(err).round(2)}; {np.max(err).round(2)}]"


def print_stats_from_array(arr, title):
    arr_mean = round(np.mean(arr), 3)
    arr_sd = round(np.std(arr), 3)
    arr_min = round(np.min(arr), 3)
    arr_max = round(np.max(arr), 3)
    print(f"{title}: {arr_mean} ± {arr_sd} [{arr_min};{arr_max}]")


def print_metrics(
    s1,
    s2,
    assimilated,
    err1_ar,
    err2_ar,
    err_r,
    err_assimilated,
    scenario,
    obs_source_title="Station",
):
    s1[np.isnan(s1)] = 0
    s2[np.isnan(s2)] = 0

    # Root Mean Squared Errors
    print(f"RMSE ({obs_source_title} and Model): {get_rmse(s1, s2)}")
    print(f"RMSE ({obs_source_title} and {scenario}): {get_rmse(s1, assimilated)}")
    print(f"RMSE (Model and {scenario}): {get_rmse(s2, assimilated)}")

    # Mean Absolute Uncertainties
    print(f"MAU ({obs_source_title}): {np.mean(np.abs(err1_ar)).round(2)}")
    print(f"MAU (Model): {np.mean(np.abs(err2_ar)).round(2)}")

    if scenario == f"DA3 (Model -> {obs_source_title})":
        print(f"MAU (Model calibrated): {np.mean(np.abs(err_r)).round(2)}")
    elif scenario == f"DA3 ({obs_source_title} -> Model)":
        print(f"MAU ({obs_source_title} calibrated): {np.mean(np.abs(err_r)).round(2)}")

    print(f"MAU ({scenario}): {np.mean(np.abs(err_assimilated)).round(2)}")


def plot_data(
    s1,
    s2,
    assimilated,
    variable,
    ax_data,
    scenario,
    obs_source_title="Station",
    obs_source_color="red",
    with_legend=True,
):
    ax_data.set_title(f"{variable}")
    ax_data.plot(
        s1.index,
        s1.values,
        color=obs_source_color,
        marker="X",
        linewidth=0,
        markersize=6,
    )
    ax_data.plot(
        s2.index,
        s2.values,
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_data.plot(
        assimilated.index,
        assimilated.values,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax_data.set_xlabel("Date")
    ax_data.set_ylabel("Concentration (ug/m³)")
    ax_data.grid()

    every_nth = 2
    for n, label in enumerate(ax_data.xaxis.get_ticklabels()):
        if n % every_nth != 1:
            label.set_visible(False)

    if with_legend:
        ax_data.legend(
            [
                obs_source_title,
                "Model",
                scenario,
            ]
        )

    return ax_data


def read_data(data_path):
    all_data_df = pd.read_csv(data_path, index_col=0)
    all_data_df.index = pd.to_datetime(
        list(all_data_df.index), format="%Y-%m-%d %H:%M:%S"
    )
    all_data_df = all_data_df.sort_index()
    return all_data_df


def prepare_daily_data(variable, data_path):
    all_data_df = read_data(data_path)
    daily_means1 = all_data_df[f"{variable}"].resample("D").mean()
    daily_means1.index = daily_means1.index + timedelta(days=1)
    observations_source1_daily_and_hourly = pd.concat(
        [all_data_df[f"{variable}"][23:], daily_means1], axis=1
    ).ffill()
    observations_source1_daily_and_hourly.columns = [
        f"{variable}_obs_hourly",
        f"{variable}_obs_daily",
    ]

    daily_means2 = all_data_df[f"{variable}_model"].resample("D").mean()
    daily_means2.index = daily_means2.index + timedelta(days=1)
    observations_source2_daily_and_hourly = pd.concat(
        [all_data_df[f"{variable}_model"][23:], daily_means2], axis=1
    ).ffill()
    observations_source2_daily_and_hourly.columns = [
        f"{variable}_model_hourly",
        f"{variable}_model_daily",
    ]

    concatenated_sources_daily_and_hourly = pd.concat(
        [observations_source1_daily_and_hourly, observations_source2_daily_and_hourly],
        axis=1,
    )

    return concatenated_sources_daily_and_hourly


def plot_data_seq(
    s1,
    s2,
    da_assimilated,
    seq_assimilated,
    variable,
    ax_data,
    da_scenario,
    seq_scenario,
    location_label,
):
    ax_data.set_title(f"{variable} - {location_label}")
    ax_data.plot(
        s1.index,
        s1.values,
        color="red",
        marker="X",
        linewidth=0,
        markersize=6,
    )
    ax_data.plot(
        s2.index,
        s2.values,
        color="blue",
        linestyle="-",
        linewidth=4,
    )
    ax_data.plot(
        da_assimilated.index,
        da_assimilated.values,
        color="black",
        linestyle="-",
        linewidth=2,
    )
    ax_data.plot(
        seq_assimilated.index,
        seq_assimilated.values,
        color="gray",
        linestyle="-",
        linewidth=2,
    )
    ax_data.set_xlabel("Date")
    ax_data.set_ylabel("Concentration (ug/m³)")
    ax_data.grid()

    every_nth = 2
    for n, label in enumerate(ax_data.xaxis.get_ticklabels()):
        if n % every_nth != 1:
            label.set_visible(False)

    if variable == "CO":
        ax_data.legend(
            [
                "Station",
                "Model",
                da_scenario,
                seq_scenario,
            ]
        )

    return ax_data


def print_metrics_seq(
    s1,
    s2,
    da_assimilated,
    da_err_assimilated,
    seq_assimilated,
    seq_err_assimilated,
    da_scenario,
    seq_scenario,
):
    s1[np.isnan(s1)] = 0
    s2[np.isnan(s2)] = 0

    # Root Mean Squared Errors
    print(f"RMSE (Station and Model): {get_rmse(s1, s2)}")
    print(f"RMSE (Station and {da_scenario}): {get_rmse(s1, da_assimilated)}")
    print(f"RMSE (Station and {seq_scenario}): {get_rmse(s1, seq_assimilated)}")
    print(f"RMSE (Model and {da_scenario}): {get_rmse(s2, da_assimilated)}")
    print(f"RMSE (Model and {seq_scenario}): {get_rmse(s2, seq_assimilated)}")

    # Mean Absolute Uncertainties
    print(f"MAU ({da_scenario}): {np.mean(np.abs(da_err_assimilated)).round(2)}")
    print(f"MAU ({seq_scenario}): {np.mean(np.abs(seq_err_assimilated)).round(2)}")
