from typing import Optional
import numpy as np

from rls_assimilation.RLS import RLS
from rls_assimilation.DataSource import DataSourceAR1
from rls_assimilation.RLSAssimilation import RLSAssimilation


class SequentialRLSAssimilationOneSource:
    def __init__(self):
        self.source: DataSourceAR1 = DataSourceAR1()
        self.ar_model = None
        self.last_assimilated = None
        self.last_err_assimilated = None

    def seq_assimilate(self, new_obs, err_new_obs):
        if self.last_assimilated is None or self.last_err_assimilated is None:
            self.last_assimilated = new_obs
            self.last_err_assimilated = err_new_obs
            return new_obs, err_new_obs
        elif self.ar_model is None:
            self.ar_model = RLS()
            self.ar_model.update(self.last_assimilated, new_obs)
            self.last_assimilated = new_obs
            self.last_err_assimilated = err_new_obs
            return new_obs, err_new_obs
        else:
            pred_assimilated = float(self.ar_model.predict(self.last_assimilated))
            pred_err_assimilated = float(
                self.last_err_assimilated * np.abs(self.ar_model.w[1])
                + self.ar_model.error
            )
            self.ar_model.update(self.last_assimilated, new_obs)

            try:
                k = pred_err_assimilated**2 / (
                    pred_err_assimilated**2 + err_new_obs**2
                )
            except (ZeroDivisionError, FloatingPointError):
                k = 1

            assimilated_obs = k * new_obs + (1 - k) * pred_assimilated
            err_assimilated_obs = np.sqrt(
                (k * err_new_obs) ** 2 + ((1 - k) * pred_err_assimilated) ** 2
            )
            self.last_assimilated = assimilated_obs
            self.last_err_assimilated = err_assimilated_obs
            return assimilated_obs, err_assimilated_obs

    def assimilate(self, obs: Optional[float]):
        source1_obs, err_source1 = self.source.estimate(obs)
        assimilated_obs, err_assimilated_obs = self.seq_assimilate(
            source1_obs, err_source1
        )
        return assimilated_obs, err_assimilated_obs


class SequentialRLSAssimilationTwoSources(
    RLSAssimilation, SequentialRLSAssimilationOneSource
):
    """
    Sequential least-squares assimilation of data from 2 data sources

    :param t_in1: temporal scale of source1 (str, "hourly" or "daily")
    :param t_in2: temporal scale of source2 (str, "hourly" or "daily")
    :param s_in1: spatial scale of source1 (str)
    :param s_in2: spatial scale of source1  (str)
    :param t_out: temporal scale of assimilation output (str, "hourly" or "daily")
    :param s_out: spatial scale of assimilation output (str)
    """

    def __init__(
        self, t_in1: str, t_in2: str, s_in1: str, s_in2: str, t_out: str, s_out: str
    ):
        RLSAssimilation.__init__(self, t_in1, t_in2, s_in1, s_in2, t_out, s_out)
        SequentialRLSAssimilationOneSource.__init__(self)

    def assimilate(self, obs1: Optional[float], obs2: Optional[float]):
        assimilated_obs, err_assimilated_obs = RLSAssimilation.assimilate(
            self, obs1, obs2
        )
        (
            assimilated_obs,
            err_assimilated_obs,
        ) = SequentialRLSAssimilationOneSource.seq_assimilate(
            self, assimilated_obs, err_assimilated_obs
        )
        return assimilated_obs, err_assimilated_obs
