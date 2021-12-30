import pandas as pd
import numpy as np

from Sensor import Sensor
from helpers import plot_errors, print_metrics


class RLSAssimilation:
    """
    Assimilation of data from 2 data sources with RLS-based uncertainty estimation

    :param do_calibration: True if the second data source is less accurate and should be calibrated, otherwise False
    """

    def __init__(self, do_calibration):
        self.cov = 0
        self.n_obs = 0
        self.sensor1 = Sensor(biased=False)  # accurate source of data
        self.sensor2 = Sensor(biased=do_calibration)
        self.last_avg_err1 = 0
        self.last_avg_err2 = 0
        self.last_covariance = 0

    def update_err_covariance(self, err1, err2):
        if self.n_obs == 1:
            self.last_avg_err1 = err1
            self.last_avg_err2 = err2
            self.last_covariance = 0
        else:
            self.last_avg_err1 = (
                self.n_obs - 1
            ) / self.n_obs * self.last_avg_err1 + 1 / self.n_obs * err1
            self.last_avg_err2 = (
                self.n_obs - 1
            ) / self.n_obs * self.last_avg_err2 + 1 / self.n_obs * err2
            self.last_covariance = (
                self.n_obs - 1
            ) / self.n_obs * self.last_covariance + (1 / (self.n_obs - 1)) * (
                err1 - self.last_avg_err1
            ) * (
                err2 - self.last_avg_err2
            )

        self.last_covariance = np.abs(self.last_covariance)

    def get_L1_uncertainty(self, err1, err2, assimilation_coef):
        """
        Maximum assimilation uncertainty
        """
        return assimilation_coef * err1 + (1 - assimilation_coef) * err2

    def get_L2_uncertainty(self, err1, err2, assimilation_coef):
        """
        Assimilation uncertainty of uncorrelated sources
        """
        return np.sqrt(
            (assimilation_coef * err1) ** 2 + ((1 - assimilation_coef) * err2) ** 2
        )

    def get_uncertainty(self, err1, err2, assimilation_coef):
        """
        General recursive estimation of assimilation uncertainty
        """
        return np.sqrt(
            (assimilation_coef * err1) ** 2
            + ((1 - assimilation_coef) * err2) ** 2
            + 2 * assimilation_coef * (1 - assimilation_coef) * self.last_covariance
        )

    def get_uncertainty_L1L2(self, err1, err2, assimilation_coef):
        """
        General recursive estimation of assimilation uncertainty (equal to get_uncertainty)
        """
        l1 = self.get_L1_uncertainty(err1, err2, assimilation_coef)
        l2 = self.get_L2_uncertainty(err1, err2, assimilation_coef)

        if l1 == l2:
            alpha = 1
        else:
            alpha = (
                np.sqrt(
                    l2 ** 2
                    + 2
                    * assimilation_coef
                    * (1 - assimilation_coef)
                    * self.last_covariance
                )
                - l1
            ) / (l2 - l1)

        return alpha * l2 + (1 - alpha) * l1

    def assimilate(self, obs1, obs2):
        self.n_obs += 1

        # Step 1: Pre-process observations and estimate errors
        self.sensor1.sense(obs1)
        sensor1_obs = self.sensor1.get_latest_observation()
        err_sensor1 = self.sensor1.get_latest_error()

        accurate_observation = sensor1_obs
        self.sensor2.sense(obs2, accurate_observation)
        sensor2_obs = self.sensor2.get_latest_observation()
        err_sensor2 = self.sensor2.get_latest_error()

        # Step 2: Assimilate
        if err_sensor1 + err_sensor2 == 0:
            assimilation_coef = 1
        else:
            assimilation_coef = err_sensor2 / (err_sensor1 + err_sensor2)

        assimilated_obs = (
            assimilation_coef * sensor1_obs + (1 - assimilation_coef) * sensor2_obs
        )

        self.update_err_covariance(err_sensor1, err_sensor2)

        err_assimilated_obs = self.get_uncertainty(
            err_sensor1, err_sensor2, assimilation_coef
        )
        # err_assimilated_obs = self.get_uncertainty_L1L2(err_sensor1, err_sensor2, assimilation_coef)

        return assimilated_obs, err_assimilated_obs

    def get_assimilation_errors(self):
        return self.sensor1.get_all_errors(), self.sensor2.get_all_errors()

    def print_assimilation_metrics(
        self,
        observations_source1,
        observations_source2,
        assimilated,
        err_assimilated,
        variable,
        type,
    ):
        print(f'{variable} {type}')
        print_metrics(
            observations_source1,
            observations_source2,
            assimilated,
            self.sensor1.get_all_errors(),
            self.sensor2.get_all_errors(),
            err_assimilated,
        )
