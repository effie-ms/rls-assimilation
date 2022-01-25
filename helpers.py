import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


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


def plot_data(s1, s2, assimilated_calibrated, assimilated_uncalibrated, variable, ax):
    ax.set_title(f"Least-squares data assimilation: {variable}")
    ax.plot(
        s1.index,
        s1.values,
        color="red",
        linestyle="-",
        linewidth=3,
        marker=".",
        markersize=12,
    )
    ax.plot(s2.index, s2.values, color="gray", linestyle="-", linewidth=3)
    ax.plot(
        assimilated_calibrated.index,
        assimilated_calibrated.values,
        color="black",
        linestyle="--",
        linewidth=3,
    )
    ax.plot(
        assimilated_uncalibrated.index,
        assimilated_uncalibrated.values,
        color="black",
        linestyle=":",
        linewidth=3,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("ug/m³")
    ax.grid()
    ax.legend(
        ["Station", "Model", "Assimilated (calibrated)", "Assimilated (uncalibrated)"]
    )
    return ax


def plot_uncertainty_ranges(
    s1,
    s2,
    s2_calibrated,
    assimilated_uncalibrated,
    assimilated_calibrated,
    err1_uncalibrated,
    err2_uncalibrated,
    err2_calibrated,
    assimilated_err_uncalibrated,
    assimilated_err_calibrated,
    variable,
):
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(28, 21))
    ax1.set_title(f"{variable}: Station and model with AR(1) uncertainties")
    ax1.plot(
        s1.index,
        s1.values,
        color="red",
        linestyle="-",
        linewidth=1,
    )
    ax1.fill_between(
        s1.index,
        s1.values - np.abs(err1_uncalibrated),
        s1.values + np.abs(err1_uncalibrated),
        alpha=0.3,
        facecolor="red",
    )
    ax1.plot(
        s2.index,
        s2.values,
        color="gray",
        linestyle=":",
        linewidth=1,
    )
    ax1.fill_between(
        s2.index,
        s2.values - np.abs(err2_uncalibrated),
        s2.values + np.abs(err2_uncalibrated),
        alpha=0.3,
        facecolor="gray",
    )
    ax1.set_ylabel("ug/m³")
    ax1.legend(["Station ± AR.Ɛ", "Model ± AR.Ɛ"])

    ax2.set_title(
        f"{variable}: Uncalibrated model with AR(1) and calibrated model with R(1) uncertainties"
    )
    ax2.plot(
        s2.index,
        s2.values,
        color="gray",
        linestyle=":",
        linewidth=1,
    )
    ax2.fill_between(
        s2.index,
        s2.values - np.abs(err2_uncalibrated),
        s2.values + np.abs(err2_uncalibrated),
        alpha=0.3,
        facecolor="gray",
    )
    ax2.plot(
        s2_calibrated.index,
        s2_calibrated.values,
        color="gray",
        linestyle="--",
        linewidth=1,
    )
    ax2.fill_between(
        s2_calibrated.index,
        s2_calibrated.values - np.abs(err2_calibrated),
        s2_calibrated.values + np.abs(err2_calibrated),
        alpha=0.3,
        facecolor="gray",
    )
    ax2.set_ylabel("ug/m³")
    ax2.legend(["Model ± AR.Ɛ", "Calibrated model ± R.Ɛ"])

    ax3.set_title(
        f"{variable}: Assimilated values and their uncertainties with and without calibration"
    )
    ax3.plot(
        assimilated_uncalibrated.index,
        assimilated_uncalibrated.values,
        color="black",
        linestyle=":",
        linewidth=1,
    )
    ax3.fill_between(
        assimilated_uncalibrated.index,
        assimilated_uncalibrated.values - np.abs(assimilated_err_uncalibrated),
        assimilated_uncalibrated.values + np.abs(assimilated_err_uncalibrated),
        alpha=0.3,
        facecolor="black",
    )
    ax3.plot(
        assimilated_calibrated.index,
        assimilated_calibrated.values,
        color="black",
        linestyle="--",
        linewidth=1,
    )
    ax3.fill_between(
        assimilated_calibrated.index,
        assimilated_calibrated.values - np.abs(assimilated_err_calibrated),
        assimilated_calibrated.values + np.abs(assimilated_err_calibrated),
        alpha=0.3,
        facecolor="black",
    )

    ax3.set_xlabel("Date")
    ax3.set_ylabel("ug/m³")
    ax3.legend(["Assimilated (uncalibrated) ± Ɛ", "Assimilated (calibrated) ± Ɛ"])

    plt.savefig(f"plots/uncertainties/{variable}-uncertainty-ranges.jpg")


def plot_errors(
    s1,
    s2,
    s2_calibrated,
    assimilated_calibrated,
    assimilated_uncalibrated,
    err1_calibrated,
    err2_calibrated,
    err1_uncalibrated,
    err2_uncalibrated,
    assimilated_err_calibrated,
    assimilated_err_uncalibrated,
    variable,
    ax,
    bx,
):
    print(variable)
    plot_uncertainty_ranges(
        s1,
        s2,
        s2_calibrated,
        assimilated_uncalibrated,
        assimilated_calibrated,
        err1_uncalibrated,
        err2_uncalibrated,
        err2_calibrated,
        assimilated_err_uncalibrated,
        assimilated_err_calibrated,
        variable,
    )

    ax.axis("equal")
    ax.set_title(
        f"The first difference dx/dt vs AR(1) uncertainties: {variable} [ug/m³]"
    )
    ax.scatter(
        np.array(np.diff(s1.values)),
        np.array(err1_uncalibrated.values[1:]),
        c="red",
        s=12,
        label="Station",
    )
    z = np.polyfit(np.diff(s1.values), err1_uncalibrated.values[1:], 1)
    p = np.poly1d(z)
    ax.plot(np.diff(s1.values), p(np.diff(s1.values)), color="red")

    ax.scatter(
        np.array(np.diff(s2.values)),
        np.array(err2_uncalibrated.values[1:]),
        c="gray",
        s=12,
        label="Model",
    )
    z = np.polyfit(np.diff(s2.values), err2_uncalibrated.values[1:], 1)
    p = np.poly1d(z)
    ax.plot(np.diff(s2.values), p(np.diff(s2.values)), color="gray")
    print(
        f"Correlation between dx/dt and AR(1) errors: station: "
        f"{np.corrcoef(np.diff(s1.values), err1_uncalibrated.values[1:])[0, 1]}, "
        f"model: {np.corrcoef(np.diff(s2.values), err2_uncalibrated.values[1:])[0, 1]}"
    )

    ax.legend()
    ax.set_xlabel("dx/dt")
    ax.set_ylabel("AR.Ɛ")
    ax.grid()

    bx.set_title(
        f"Difference between station and model VS R(1) uncertainties: {variable} [ug/m³]"
    )
    bx.scatter(
        np.array(s1.values - s2.values),
        np.array(err2_calibrated.values),
        c="gray",
        s=12,
    )
    z = np.polyfit(np.array(s1.values - s2.values), err2_calibrated.values, 1)
    p = np.poly1d(z)
    bx.plot(
        np.array(s1.values - s2.values),
        p(np.array(s1.values - s2.values)),
        color="gray",
    )
    print(
        f"Correlation between (dx2/dt + d(x1-x2)/dt) and R(1) errors: "
        f"{np.corrcoef(np.array(np.diff(s2.values) + np.array(s1.values - s2.values)[1:]), np.array(err2_calibrated.values[1:]))[0, 1]}"
    )

    bx.set_xlabel("dx2/dt + d(x1-x2)/dt")
    bx.set_ylabel("R.Ɛ")
    bx.grid()

    return ax, bx


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Source: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        linewidth=3,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ellipse


def plot_errors_scatter(
    s1, s2, assimilated_calibrated, assimilated_uncalibrated, variable, ax
):
    ax.axis("equal")
    ax.set_title(f"Errors between observations/model and assimilated: {variable}")
    ax.scatter(
        np.array(s1.values - assimilated_calibrated.values),
        np.array(s2.values - assimilated_calibrated.values),
        c="gray",
        s=12,
    )
    ax.add_patch(
        confidence_ellipse(
            np.array(s1.values - assimilated_calibrated.values),
            np.array(s2.values - assimilated_calibrated.values),
            ax,
            edgecolor="gray",
        )
    )

    ax.scatter(
        np.array(s1.values - assimilated_uncalibrated.values),
        np.array(s2.values - assimilated_uncalibrated.values),
        c="black",
        s=12,
    )
    ax.add_patch(
        confidence_ellipse(
            np.array(s1.values - assimilated_uncalibrated.values),
            np.array(s2.values - assimilated_uncalibrated.values),
            ax,
            edgecolor="black",
        )
    )

    ax.legend(["Calibrated", "Uncalibrated"])
    ax.set_xlabel("Station - Assimilated [ug/m³]")
    ax.set_ylabel("Model - Assimilated [ug/m³]")
    ax.grid()
    return ax
