import pandas as pd
import matplotlib.pyplot as plt

from rls_assimilation import RLSAssimilation
from helpers import (
    plot_data,
    plot_uncertainties,
    plot_data_pdf,
    plot_uncertainty_pdf,
    plot_data_cdf,
    plot_uncertainty_cdf,
    print_metrics,
)


def demo_assimilation_and_plot(
    all_data_df,
    variable,
    ax,
    ax_err,
    ax_data_pdf,
    ax_unc_pdf,
    ax_data_cdf,
    ax_unc_cdf,
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

    observations_source2_calibrated = (
        assimilator_with_calibration.source2.get_corrected_data()
    )

    # plot and print metrics
    ax = plot_data(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2_calibrated, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated_calibrated, index=all_data_df.index),
        pd.Series(assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax,
    )

    err1_calibrated = assimilator_with_calibration.source1.get_all_errors()
    err2_calibrated = assimilator_with_calibration.source2.get_all_errors()

    err1_uncalibrated = assimilator_without_calibration.source1.get_all_errors()
    err2_uncalibrated = assimilator_without_calibration.source2.get_all_errors()

    ax_err = plot_uncertainties(
        pd.Series(err1_uncalibrated, index=all_data_df.index),
        pd.Series(err2_calibrated, index=all_data_df.index),
        pd.Series(err2_uncalibrated, index=all_data_df.index),
        pd.Series(err_assimilated_calibrated, index=all_data_df.index),
        pd.Series(err_assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax_err,
    )

    print(f"{variable} calibrated")
    print_metrics(
        observations_source1,
        observations_source2,
        assimilated_calibrated,
        err1_calibrated,
        err2_calibrated,
        err_assimilated_calibrated,
    )

    print(f"{variable} uncalibrated")
    print_metrics(
        observations_source1,
        observations_source2,
        assimilated_uncalibrated,
        err1_uncalibrated,
        err2_uncalibrated,
        err_assimilated_uncalibrated,
    )

    ax_data_pdf = plot_data_pdf(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2_calibrated, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated_calibrated, index=all_data_df.index),
        pd.Series(assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax_data_pdf,
    )

    ax_unc_pdf = plot_uncertainty_pdf(
        pd.Series(err1_uncalibrated, index=all_data_df.index),
        pd.Series(err2_uncalibrated, index=all_data_df.index),
        pd.Series(err2_calibrated, index=all_data_df.index),
        pd.Series(err_assimilated_uncalibrated, index=all_data_df.index),
        pd.Series(err_assimilated_calibrated, index=all_data_df.index),
        variable,
        ax_unc_pdf,
    )

    ax_data_cdf = plot_data_cdf(
        pd.Series(observations_source1, index=all_data_df.index),
        pd.Series(observations_source2_calibrated, index=all_data_df.index),
        pd.Series(observations_source2, index=all_data_df.index),
        pd.Series(assimilated_calibrated, index=all_data_df.index),
        pd.Series(assimilated_uncalibrated, index=all_data_df.index),
        variable,
        ax_data_cdf,
    )

    ax_unc_cdf = plot_uncertainty_cdf(
        pd.Series(err1_uncalibrated, index=all_data_df.index),
        pd.Series(err2_uncalibrated, index=all_data_df.index),
        pd.Series(err2_calibrated, index=all_data_df.index),
        pd.Series(err_assimilated_uncalibrated, index=all_data_df.index),
        pd.Series(err_assimilated_calibrated, index=all_data_df.index),
        variable,
        ax_unc_cdf,
    )

    return (
        ax,
        ax_err,
        ax_data_pdf,
        ax_unc_pdf,
        ax_data_cdf,
        ax_unc_cdf,
    )


data_path = "data/liivalaia_aq_meas_with_forecast.csv"
all_data_df = pd.read_csv(data_path, index_col=0)
all_data_df.index = pd.to_datetime(list(all_data_df.index), format="%Y-%m-%d %H:%M:%S")
all_data_df = all_data_df.sort_index()

variables = ["CO", "NO2", "O3", "SO2", "PM2.5", "PM10"]
fig_data, axs_data = plt.subplots(nrows=3, ncols=2, figsize=(60, 25))
fig_err, axs_err = plt.subplots(nrows=3, ncols=2, figsize=(60, 25))
fig_data_pdf, axs_data_pdf = plt.subplots(nrows=3, ncols=2, figsize=(35, 45))
fig_unc_pdf, axs_unc_pdf = plt.subplots(nrows=3, ncols=2, figsize=(35, 45))
fig_data_cdf, axs_data_cdf = plt.subplots(nrows=3, ncols=2, figsize=(35, 45))
fig_unc_cdf, axs_unc_cdf = plt.subplots(nrows=3, ncols=2, figsize=(35, 45))

for idx, variable in enumerate(variables):
    (
        axs_data[idx % 3, idx % 2],
        axs_err[idx % 3, idx % 2],
        axs_data_pdf[idx % 3, idx % 2],
        axs_unc_pdf[idx % 3, idx % 2],
        axs_data_cdf[idx % 3, idx % 2],
        axs_unc_cdf[idx % 3, idx % 2],
    ) = demo_assimilation_and_plot(
        all_data_df,
        variable,
        axs_data[idx % 3, idx % 2],
        axs_err[idx % 3, idx % 2],
        axs_data_pdf[idx % 3, idx % 2],
        axs_unc_pdf[idx % 3, idx % 2],
        axs_data_cdf[idx % 3, idx % 2],
        axs_unc_cdf[idx % 3, idx % 2],
    )

fig_data.savefig("plots/data.jpg")
fig_err.savefig("plots/uncertainties.jpg")
fig_data_pdf.savefig("plots/data-pdf.jpg")
fig_unc_pdf.savefig("plots/uncertainties-pdf.jpg")
fig_data_cdf.savefig("plots/data-cdf.jpg")
fig_unc_cdf.savefig("plots/uncertainties-cdf.jpg")
