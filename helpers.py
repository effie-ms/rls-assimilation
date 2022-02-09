import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({"font.size": 22})


def get_rmse(arr1, arr2):
    return np.sqrt(((arr1 - arr2) ** 2).mean()).round(2)


def get_uncertainty_stats(err):
    return f"{np.mean(err).round(2)} ± {np.std(err).round(2)} [{np.min(err).round(2)}; {np.max(err).round(2)}]"


def print_metrics(s1, s2, assimilated, err1, err2, err_assimilated):
    s1[np.isnan(s1)] = 0
    s2[np.isnan(s2)] = 0

    print(f"RMSE (measured and model): {get_rmse(s1, s2)}")
    print(f"RMSE (measured and assimilated): {get_rmse(s1, assimilated)}")
    print(f"RMSE (model and assimilated): {get_rmse(s2, assimilated)}")

    print(f"Error (measured - model): {get_uncertainty_stats(s1 - s2)}")
    print(f"Error (measured - assimilated): {get_uncertainty_stats(s1 - assimilated)}")
    print(f"Error (model - assimilated): {get_uncertainty_stats(s2 - assimilated)}")

    print(f"Uncertainty (measured): {get_uncertainty_stats(np.abs(err1))}")
    print(f"Uncertainty (model): {get_uncertainty_stats(np.abs(err2))}")
    print(
        f"Uncertainty (assimilated): {get_uncertainty_stats(np.abs(err_assimilated))}"
    )


def plot_data(
    s1,
    s2_calibrated,
    s2_uncalibrated,
    assimilated_calibrated,
    assimilated_uncalibrated,
    variable,
    ax,
):
    ax.set_title(f"Least-squares data assimilation: {variable}")
    ax.plot(
        s1.index,
        s1.values,
        color="red",
        marker="X",
        linewidth=0,
        markersize=10,
    )
    ax.plot(
        s2_uncalibrated.index,
        s2_uncalibrated.values,
        color="gray",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        s2_calibrated.index,
        s2_calibrated.values,
        color="gray",
        linestyle=":",
        linewidth=3,
    )
    ax.plot(
        assimilated_uncalibrated.index,
        assimilated_uncalibrated.values,
        color="black",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        assimilated_calibrated.index,
        assimilated_calibrated.values,
        color="black",
        linestyle=":",
        linewidth=3,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("ug/m³")
    ax.grid()
    ax.legend(
        [
            "Station",
            "Model (uncalibrated)",
            "Model (calibrated)",
            "Assimilated (uncalibrated)",
            "Assimilated (calibrated)",
        ]
    )
    return ax


def plot_uncertainties(
    err1,
    err2_uncalibrated,
    err2_calibrated,
    err_assimilated_uncalibrated,
    err_assimilated_calibrated,
    variable,
    ax,
):
    ax.set_title(
        f"Least-squares data assimilation: {variable} uncertainties (absolute values)"
    )
    ax.plot(
        err1.index,
        np.abs(err1.values),
        color="red",
        marker="X",
        linewidth=0,
        markersize=10,
    )
    ax.plot(
        err2_uncalibrated.index,
        np.abs(err2_uncalibrated.values),
        color="gray",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        err2_calibrated.index,
        np.abs(err2_calibrated.values),
        color="gray",
        linestyle=":",
        linewidth=3,
    )
    ax.plot(
        err_assimilated_uncalibrated.index,
        np.abs(err_assimilated_uncalibrated.values),
        color="black",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        err_assimilated_calibrated.index,
        np.abs(err_assimilated_calibrated.values),
        color="black",
        linestyle=":",
        linewidth=3,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("ug/m³")
    ax.grid()
    ax.legend(
        [
            "Station",
            "Model (uncalibrated)",
            "Model (calibrated)",
            "Assimilated (uncalibrated)",
            "Assimilated (calibrated)",
        ]
    )
    return ax


def plot_data_pdf(
    s1,
    s2_calibrated,
    s2_uncalibrated,
    assimilated_calibrated,
    assimilated_uncalibrated,
    variable,
    ax,
):
    df = pd.concat(
        [
            s1,
            s2_uncalibrated,
            s2_calibrated,
            assimilated_uncalibrated,
            assimilated_calibrated,
        ],
        axis=1,
    )
    df.columns = [
        "Station",
        "Model (uncalibrated)",
        "Model (calibrated)",
        "Assimilated (uncalibrated)",
        "Assimilated (calibrated)",
    ]

    ax.set_title(f"Least-squares data assimilation (PDF): {variable}")
    sns.kdeplot(
        data=df["Station"],
        ax=ax,
        linewidth=0,
        linestyle="-",
        marker="X",
        markersize=10,
        color="red",
    )
    sns.kdeplot(
        data=df["Model (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.kdeplot(
        data=df["Model (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.kdeplot(
        data=df["Assimilated (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="black",
    )
    sns.kdeplot(
        data=df["Assimilated (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="black",
    )

    ax.set_xlabel("ug/m³")
    ax.grid()
    ax.legend(df.columns)
    return ax


def plot_uncertainty_pdf(
    err1,
    err2_uncalibrated,
    err2_calibrated,
    err_assimilated_uncalibrated,
    err_assimilated_calibrated,
    variable,
    ax,
):
    df = pd.concat(
        [
            np.abs(err1),
            np.abs(err2_uncalibrated),
            np.abs(err2_calibrated),
            np.abs(err_assimilated_uncalibrated),
            np.abs(err_assimilated_calibrated),
        ],
        axis=1,
    )
    df.columns = [
        "Station",
        "Model (uncalibrated)",
        "Model (calibrated)",
        "Assimilated (uncalibrated)",
        "Assimilated (calibrated)",
    ]

    ax.set_title(
        f"Least-squares data assimilation (PDF): {variable} uncertainties (absolute values)"
    )
    sns.kdeplot(
        data=df["Station"],
        ax=ax,
        linewidth=0,
        linestyle="-",
        marker="X",
        markersize=10,
        color="red",
    )
    sns.kdeplot(
        data=df["Model (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.kdeplot(
        data=df["Model (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.kdeplot(
        data=df["Assimilated (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="black",
    )
    sns.kdeplot(
        data=df["Assimilated (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="black",
    )

    ax.set_xlabel("ug/m³")
    ax.grid()
    ax.legend(df.columns)
    return ax


def plot_data_cdf(
    s1,
    s2_calibrated,
    s2_uncalibrated,
    assimilated_calibrated,
    assimilated_uncalibrated,
    variable,
    ax,
):
    df = pd.concat(
        [
            s1,
            s2_uncalibrated,
            s2_calibrated,
            assimilated_uncalibrated,
            assimilated_calibrated,
        ],
        axis=1,
    )
    df.columns = [
        "Station",
        "Model (uncalibrated)",
        "Model (calibrated)",
        "Assimilated (uncalibrated)",
        "Assimilated (calibrated)",
    ]

    ax.set_title(f"Least-squares data assimilation (CDF): {variable}")
    sns.ecdfplot(
        data=df["Station"],
        ax=ax,
        linewidth=0,
        linestyle="-",
        marker="X",
        markersize=10,
        color="red",
    )
    sns.ecdfplot(
        data=df["Model (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.ecdfplot(
        data=df["Model (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.ecdfplot(
        data=df["Assimilated (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="black",
    )
    sns.ecdfplot(
        data=df["Assimilated (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="black",
    )

    ax.set_xlabel("ug/m³")
    ax.grid()
    ax.legend(df.columns)
    return ax


def plot_uncertainty_cdf(
    err1,
    err2_uncalibrated,
    err2_calibrated,
    err_assimilated_uncalibrated,
    err_assimilated_calibrated,
    variable,
    ax,
):
    df = pd.concat(
        [
            np.abs(err1),
            np.abs(err2_uncalibrated),
            np.abs(err2_calibrated),
            np.abs(err_assimilated_uncalibrated),
            np.abs(err_assimilated_calibrated),
        ],
        axis=1,
    )
    df.columns = [
        "Station",
        "Model (uncalibrated)",
        "Model (calibrated)",
        "Assimilated (uncalibrated)",
        "Assimilated (calibrated)",
    ]

    ax.set_title(
        f"Least-squares data assimilation (CDF): {variable} uncertainties (absolute values)"
    )
    sns.ecdfplot(
        data=df["Station"],
        ax=ax,
        linewidth=0,
        linestyle="-",
        marker="X",
        markersize=10,
        color="red",
    )
    sns.ecdfplot(
        data=df["Model (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.ecdfplot(
        data=df["Model (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="gray",
    )
    sns.ecdfplot(
        data=df["Assimilated (uncalibrated)"],
        ax=ax,
        linewidth=3,
        linestyle="--",
        marker=None,
        markersize=0,
        color="black",
    )
    sns.ecdfplot(
        data=df["Assimilated (calibrated)"],
        ax=ax,
        linewidth=3,
        linestyle=":",
        marker=None,
        markersize=0,
        color="black",
    )

    ax.set_xlabel("ug/m³")
    ax.grid()
    ax.legend(df.columns)
    return ax
