import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 22})


def assimilate(obs1, err1, obs2, err2, use_err_max=True):
    if err1 + err2 == 0:
        assimilation_coef = 1
    else:
        assimilation_coef = err2 / (err1 + err2)

    assimilated_obs = assimilation_coef * obs1 + (1 - assimilation_coef) * obs2

    min_err_assimilated_obs = np.sqrt(
        (assimilation_coef * err1) ** 2 + ((1 - assimilation_coef) * err2) ** 2
    )
    max_err_assimilated_obs = assimilation_coef * err1 + (1 - assimilation_coef) * err2

    if use_err_max:
        return assimilated_obs, max_err_assimilated_obs

    return assimilated_obs, min_err_assimilated_obs


def get_rmse(arr1, arr2):
    return np.sqrt(((arr1 - arr2) ** 2).mean())


def get_uncertainty_stats(err):
    return f"{np.mean(err).round(2)} ± {np.std(err).round(2)} [{np.min(err).round(2)}; {np.max(err).round(2)}]"


def print_metrics(s1, s2, assimilated, err1, err2, err_assimilated):
    s1[np.isnan(s1)] = 0
    s2[np.isnan(s2)] = 0

    print(f"RMSE (measured and model): {get_rmse(s1, s2)}")
    print(f"RMSE (measured and assimilated): {get_rmse(s1, assimilated)}")
    print(f"RMSE (model and assimilated): {get_rmse(s2, assimilated)}")

    print(f"Uncertainty (measured): {get_uncertainty_stats(err1)}")
    print(f"Uncertainty (model): {get_uncertainty_stats(err2)}")
    print(f"Uncertainty (assimilated): {get_uncertainty_stats(err_assimilated)}")


def plot_data(s1, s2, assimilated):
    plt.figure(figsize=(30, 10))
    plt.title("Data assimilation: estimates")
    s1.plot()
    s2.plot()
    assimilated.plot()
    plt.xlabel("Date")
    plt.ylabel("ug/m³")
    plt.grid()
    plt.legend(["Source 1", "Source 2", "Assimilated"])
    plt.savefig("plots/data.png")


def plot_errors(err1, err2, assimilated_err):
    plt.figure(figsize=(30, 10))
    plt.title("Data assimilation: estimates")
    err1.plot()
    err2.plot()
    assimilated_err.plot()
    plt.xlabel("Date")
    plt.ylabel("ug/m³")
    plt.grid()
    plt.legend(["Error of source 1", "Error of source 2", "Error of assimilated"])
    plt.savefig("plots/errors.png")
