import numpy as np

from RLS import RLS


class Sensor:
    def __init__(self, location=None, biased=False):
        self.location = location
        self.observations = []  # observations
        self.corrections = (
            []
        )  # observations with filled missing values (if there are any)
        self.mapping = (
            []
        )  # mapped values to more accurate observations, used instead of corrections (biased=True)
        self.internal_errors = (
            []
        )  # regression-based errors of observations (from the internal correction model)
        self.external_errors = (
            []
        )  # regression-based errors of observations (from the external mapping model)

        # Internal model (for imputation and uncertainty estimation)
        self.backup_model = None

        # External model (mapping to a more accurate observation)
        self.bias_model = RLS() if biased else None

    def get_location(self):
        return self.location

    def get_latest_observation(self):
        return self.mapping[-1] if self.bias_model else self.corrections[-1]

    def get_latest_error(self):
        return self.external_errors[-1] if self.bias_model else self.internal_errors[-1]

    def get_all_errors(self):
        return self.external_errors if self.bias_model else self.internal_errors

    def get_all_raw_observations(self):
        return self.observations

    def get_all_corrected_observations(self):
        return self.mapping if self.bias_model else self.corrections

    def init_internal_model(self):
        self.backup_model = RLS()

    def update_internal_model(self, latest_correction, past_observation):
        if len(self.corrections) > 0:
            if not self.backup_model:
                self.init_internal_model()

            self.backup_model.update(past_observation, latest_correction)

    def impute_if_missing(self, new_obs, past_obs):
        if np.isnan(new_obs):
            if not self.backup_model or np.isnan(past_obs):
                # use the last corrected value as a placeholder for a missing value or 0
                placeholder_value = (
                    self.corrections[-1] if len(self.corrections) > 0 else 0
                )
                return placeholder_value

            return self.backup_model.predict(past_obs)

        return new_obs

    def sense(self, latest_observation, accurate_observation=None):
        # save a raw observation
        self.observations.append(latest_observation)

        # impute (predict) if missing
        past_observation = self.corrections[-1] if len(self.corrections) > 0 else np.nan
        latest_correction = self.impute_if_missing(latest_observation, past_observation)

        # obtain error of the internal model
        latest_error = self.backup_model.error if self.backup_model else 0

        # save an imputed observation
        self.corrections.append(latest_correction)

        # save the latest error
        self.internal_errors.append(latest_error)

        # update the internal model
        if not np.isnan(latest_observation) and not np.isnan(past_observation):
            self.update_internal_model(latest_correction, past_observation)

        # if observation should be mapped to a more accurate observation
        if self.bias_model:
            if len(self.mapping) < 1:
                # use the observation as it is (without mapping) if no past data
                self.mapping.append(self.corrections[-1])
                self.external_errors.append(latest_error)
            else:
                input_observation = self.corrections[-1]
                output_observation = accurate_observation

                self.mapping.append(self.bias_model.predict(input_observation))
                self.external_errors.append(self.bias_model.error)

                # update the external model
                self.bias_model.update(input_observation, output_observation)
