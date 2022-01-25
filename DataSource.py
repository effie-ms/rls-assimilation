from typing import List
import numpy as np

from RLS import RLS


class DataSource:
    """
    Implements AR(1) and R(1) algorithms
    """

    def __init__(self, do_calibration=False):
        """
        :param do_calibration: True, if R(1) algorithm should be used to estimate the uncertainty, otherwise False
        """

        # stored for plotting
        self.x_all = []  # raw values
        self.x_corr_all = []  # x_all with filled missing values (if there are any)
        self.x_calibrated_all = []  # R(1) model predictions
        self.ar_errors = []  # AR(1) modelling errors
        self.r_errors = []  # R(1) modelling errors

        self.ar_avg_err = 0
        self.ar_err_var = 0
        self.r_avg_err = 0
        self.r_err_var = 0
        self.ar_model = None  # AR(1) model
        self.r_model = RLS() if do_calibration else None  # R(1) model

    def get_latest_data_point(self) -> float:
        return self.x_calibrated_all[-1] if self.r_model else self.x_corr_all[-1]

    def get_latest_error(self) -> float:
        return self.r_errors[-1] if self.r_model else self.ar_errors[-1]

    def get_latest_avg_error(self) -> float:
        return self.r_avg_err if self.r_model else self.ar_avg_err

    def get_latest_error_variance(self) -> float:
        return self.r_err_var if self.r_model else self.ar_err_var

    def get_all_errors(self) -> List[float]:
        return self.r_errors if self.r_model else self.ar_errors

    def get_raw_data(self) -> List[float]:
        return self.x_all

    def get_corrected_data(self) -> List[float]:
        return self.x_calibrated_all if self.r_model else self.x_corr_all

    def impute(self, x_past: float):
        if not self.ar_model:
            if np.isnan(x_past):
                return 0
            else:
                return x_past

        return self.ar_model.predict(x_past)

    def calibrate(self, x_corr: float, err: float, x_ref: float):
        """
        Run R(1) calibration
        """
        if len(self.x_calibrated_all) < 1:
            self.x_calibrated_all.append(x_corr)
            self.r_errors.append(err)
        else:
            self.x_calibrated_all.append(self.r_model.predict(x_corr))
            ar_r_err = float(np.abs(self.r_model.w[1]) * err) + self.r_model.error
            self.r_errors.append(ar_r_err)

            self.r_model.update(x_corr, x_ref)

    def estimate(self, x_new: float, x_ref: float = np.nan):
        """
        Runs AR(1) and if do_calibration=True, R(1) uncertainty estimation

        :param: x_new - the latest value from the data source
        :param: x_ref - the reference value for calibration (if needed)
        """

        self.x_all.append(x_new)  # save a raw observation
        t = len(self.x_all)  # the number of acquired data points

        # Run AR(1) estimation
        x_past = self.x_corr_all[-1] if t > 1 else np.nan
        if np.isnan(x_new):
            x_corr = self.impute(x_past)  # impute (predict) if missing
        else:
            x_corr = x_new
            if not np.isnan(x_past):
                if not self.ar_model:
                    self.ar_model = RLS()  # initialise when data gets available

                self.ar_model.update(x_past, x_corr)

        self.x_corr_all.append(x_corr)  # save a corrected (raw or imputed) data value

        # obtain error of the AR(1) model
        err = self.ar_model.error if self.ar_model else 0
        self.ar_errors.append(err)  # save the latest error
        self.ar_avg_err = ((t - 1) / t) * self.ar_avg_err + (1 / t) * err
        if t > 1:
            self.ar_err_var = ((t - 1) / t) * self.ar_err_var + (1 / (t - 1)) * (
                err - self.ar_avg_err
            ) ** 2
        else:
            self.ar_err_var = 0

        # if do calibration
        if self.r_model:
            self.calibrate(x_corr, err, x_ref)
            self.r_avg_err = ((t - 1) / t) * self.r_avg_err + (
                1 / t
            ) * self.r_model.error
            if t > 1:
                self.r_err_var = ((t - 1) / t) * self.r_err_var + (1 / (t - 1)) * (
                    self.r_model.error - self.r_avg_err
                ) ** 2
            else:
                self.r_err_var = 0
