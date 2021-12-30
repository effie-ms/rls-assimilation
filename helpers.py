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

    print(f"Absolute error (measured - model): {get_uncertainty_stats(np.abs(s1 - s2))}")
    print(f"Absolute error (measured - assimilated): {get_uncertainty_stats(np.abs(s1 - err_assimilated))}")
    print(f"Absolute error (model - assimilated): {get_uncertainty_stats(np.abs(s2 - err_assimilated))}")

    print(f"Uncertainty (measured): {get_uncertainty_stats(err1)}")
    print(f"Uncertainty (model): {get_uncertainty_stats(err2)}")
    print(f"Uncertainty (assimilated): {get_uncertainty_stats(err_assimilated)}")


def plot_data(s1, s2, assimilated_calibrated, assimilated_uncalibrated, variable, ax):
    ax.set_title(f"Data assimilation: {variable}")
    ax.plot(s1.index, s1.values, color='red', linestyle='-', linewidth=3, marker='.', markersize=12)
    ax.plot(s2.index, s2.values, color='gray', linestyle='-', linewidth=3)
    ax.plot(assimilated_calibrated.index, assimilated_calibrated.values, color='black', linestyle='--', linewidth=3)
    ax.plot(assimilated_uncalibrated.index, assimilated_uncalibrated.values, color='black', linestyle=':', linewidth=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("ug/m³")
    ax.grid()
    ax.legend(["Station", "Model", "Assimilated (calibrated)", "Assimilated (uncalibrated)"])
    return ax


def plot_errors(err1_calibrated, err2_calibrated, err1_uncalibrated, err2_uncalibrated, assimilated_err_calibrated, assimilated_err_uncalibrated, variable, ax, bx):
    ax.set_title(f"Regression-based uncertainties of data assimilation (uncalibrated): {variable}")
    ax.plot(err1_uncalibrated.index, err1_uncalibrated.values, color='red', linestyle='-', linewidth=3, marker='.', markersize=12)
    ax.plot(err2_uncalibrated.index, err2_uncalibrated.values, color='gray', linestyle=':', linewidth=3)
    ax.plot(assimilated_err_uncalibrated.index, assimilated_err_uncalibrated.values, color='black', linestyle=':', linewidth=3)
    ax.set_xlabel("Date")
    ax.set_ylabel("ug/m³")
    ax.grid()
    ax.legend(["Station", "Model (uncalibrated)", "Assimilated (uncalibrated)"])

    bx.set_title(f"Regression-based uncertainties of data assimilation (calibrated): {variable}")
    bx.plot(err1_calibrated.index, err1_calibrated.values, color='red', linestyle='-', linewidth=3, marker='.', markersize=12)
    bx.plot(err2_calibrated.index, err2_calibrated.values, color='gray', linestyle='--', linewidth=3)
    bx.plot(assimilated_err_calibrated.index, assimilated_err_calibrated.values, color='black', linestyle='--', linewidth=3)
    bx.set_xlabel("Date")
    bx.set_ylabel("ug/m³")
    bx.grid()
    bx.legend(["Station", "Model (calibrated)", "Assimilated (calibrated)"])

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
    ax.axis('equal')
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

    ax.legend(['Calibrated', 'Uncalibrated'])
    ax.set_xlabel("Station - Assimilated [ug/m³]")
    ax.set_ylabel("Model - Assimilated [ug/m³]")
    ax.grid()
    return ax
