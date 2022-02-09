import numpy as np

from rls_assimilation.DataSource import DataSource


class RLSAssimilation:
    """
    Least-squares assimilation of data from 2 data sources

    :param do_calibration: True if the second data source is should be calibrated (linearly mapped) to the first one, otherwise False
    """

    def __init__(self, do_calibration):
        self.source1 = DataSource(
            do_calibration=False
        )  # data source of the needed scale
        self.source2 = DataSource(do_calibration=do_calibration)
        self.cov = 0  # covariance of errors of 2 data sources (sample)
        self.t = 0  # number of acquired data points

    def assimilate(self, obs1, obs2):
        self.t += 1

        # Step 1: Pre-process observations and estimate errors
        self.source1.estimate(obs1)
        source1_obs = self.source1.get_latest_data_point()
        err_source1 = self.source1.get_latest_error()
        avg_err_source1 = self.source1.get_latest_avg_error()
        err_var_source1 = self.source1.get_latest_error_variance()

        self.source2.estimate(obs2, source1_obs)
        source2_obs = self.source2.get_latest_data_point()
        err_source2 = self.source2.get_latest_error()
        avg_err_source2 = self.source2.get_latest_avg_error()
        err_var_source2 = self.source2.get_latest_error_variance()

        corr = 0
        # Step 2: Assimilate
        if self.t > 1:
            # update sample covariance
            self.cov = (self.t - 1) / self.t * self.cov + (1 / (self.t - 1)) * (
                err_source1 - avg_err_source1
            ) * (err_source2 - avg_err_source2)
            if err_var_source1 == 0 or err_var_source2 == 0:
                corr = 0
            else:
                corr = self.cov / (np.sqrt(err_var_source1) * np.sqrt(err_var_source2))

        # find assimilation coefficient k
        cov = corr * err_source1 * err_source2  # NB: cov != self.cov
        try:
            k = (err_source2 ** 2 - cov) / (
                err_source1 ** 2 + err_source2 ** 2 - 2 * cov
            )

            # force min (0) and max (1) values in case of exceedance
            k = max(min(k, 1), 0)

        except ZeroDivisionError:
            k = 1

        assimilated_obs = k * source1_obs + (1 - k) * source2_obs

        err_assimilated_obs = np.sqrt(
            np.abs(
                (k * err_source1) ** 2
                + ((1 - k) * err_source2) ** 2
                + 2 * k * (1 - k) * cov
            )
        )

        return assimilated_obs, err_assimilated_obs
