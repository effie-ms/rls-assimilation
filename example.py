import pandas as pd
import matplotlib.pyplot as plt

from RLSAssimilation import RLSAssimilation
from helpers import plot_data, plot_errors, plot_errors_scatter


def demo_assimilation_and_plot(
    all_data_df,
    variable,
    ax,
    ax_err,
    bx_err,
    ax_err_scatter,
):
    observations_source1 = all_data_df[
        f"{variable}"
    ].values  # observations from source 1
    observations_source2 = all_data_df[
        f"{variable}_model"
    ].values  # observations from source 2
    n_observations = len(observations_source1)

    # assimilated (weighted) values and errors
    assimilated_calibrated = []
    err_assimilated_calibrated = []

    assimilated_uncalibrated = []
    err_assimilated_uncalibrated = []

    assimilator_with_calibration = RLSAssimilation(do_calibration=True)
    assimilator_without_calibration = RLSAssimilation(do_calibration=False)

    # assimilate
    for k in range(n_observations):
        # Step 1: Obtain raw observations from 2 sources
        latest_observation_sensor1 = observations_source1[k]
        latest_observation_sensor2 = observations_source2[k]

        # Step 2: Assimilate
        (
            assimilated_obs_calibrated,
            err_assimilated_obs_calibrated,
        ) = assimilator_with_calibration.assimilate(
            latest_observation_sensor1,
            latest_observation_sensor2,
        )
        assimilated_calibrated.append(assimilated_obs_calibrated)
        err_assimilated_calibrated.append(err_assimilated_obs_calibrated)

        (
            assimilated_obs_uncalibrated,
            err_assimilated_obs_uncalibrated,
        ) = assimilator_without_calibration.assimilate(
            latest_observation_sensor1,
            latest_observation_sensor2,
        )
        assimilated_uncalibrated.append(assimilated_obs_uncalibrated)
        err_assimilated_uncalibrated.append(err_assimilated_obs_uncalibrated)

    # plot and print metrics

    # for calibrated:
    ax = plot_data(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated_calibrated, index=all_data_df.index),
        pd.Series(assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax,
    )

    err1_calibrated, err2_calibrated = assimilator_with_calibration.get_assimilation_errors()
    err1_uncalibrated, err2_uncalibrated = assimilator_without_calibration.get_assimilation_errors()

    ax_err, bx_err = plot_errors(
        pd.Series(err1_calibrated, index=all_data_df.index),
        pd.Series(err2_calibrated, index=all_data_df.index),
        pd.Series(err1_uncalibrated, index=all_data_df.index),
        pd.Series(err2_uncalibrated, index=all_data_df.index),
        pd.Series(err_assimilated_calibrated, index=all_data_df.index),
        pd.Series(err_assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax_err,
        bx_err,
    )

    assimilator_with_calibration.print_assimilation_metrics(
        observations_source1,
        observations_source2,
        assimilated_calibrated,
        err_assimilated_calibrated,
        variable,
        'calibrated',
    )
    assimilator_without_calibration.print_assimilation_metrics(
        observations_source1,
        observations_source2,
        assimilated_uncalibrated,
        err_assimilated_uncalibrated,
        variable,
        'uncalibrated',
    )

    ax_err_scatter = plot_errors_scatter(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated_calibrated, index=all_data_df.index),
        pd.Series(assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax_err_scatter,
    )

    return (
        ax,
        ax_err,
        bx_err,
        ax_err_scatter,
    )


data_path = "data/liivalaia_aq_meas_with_forecast.csv"
all_data_df = pd.read_csv(data_path, index_col=0)
all_data_df.index = pd.to_datetime(list(all_data_df.index), format="%Y-%m-%d %H:%M:%S")
all_data_df = all_data_df.sort_index()

variables = ["CO", "NO2", "O3", "SO2", "PM2.5", "PM10"]
fig_data, axs_data = plt.subplots(
    nrows=3, ncols=2, figsize=(60, 45)
)
fig_err, axs_err = plt.subplots(
    nrows=6, ncols=2, figsize=(60, 90)
)
fig_err_scatter, axs_err_scatter = plt.subplots(nrows=3, ncols=2, figsize=(40, 30))
for idx, variable in enumerate(variables):
    (
        axs_data[idx % 3, idx % 2],
        axs_err[idx, 0],
        axs_err[idx, 1],
        axs_err_scatter[idx % 3, idx % 2],
    ) = demo_assimilation_and_plot(
        all_data_df,
        variable,
        axs_data[idx % 3, idx % 2],
        axs_err[idx, 0],
        axs_err[idx, 1],
        axs_err_scatter[idx % 3, idx % 2],
    )

fig_data.savefig("plots/data.png")
fig_err.savefig("plots/uncertainties.png")
fig_err_scatter.savefig("plots/errors.png")
