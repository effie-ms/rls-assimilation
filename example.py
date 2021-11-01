import pandas as pd

from Sensor import Sensor
from helpers import assimilate, plot_data, plot_errors, print_metrics


data_path = "data/Tallinn_CO_all.csv"
all_data_df = pd.read_csv(data_path, index_col=0)
all_data_df.index = pd.to_datetime(list(all_data_df.index), format="%Y-%m-%d %H:%M:%S")
all_data_df = all_data_df.sort_index()

observations_source1 = all_data_df["Liivalaia"].values  # observations from source 1
observations_source2 = all_data_df["SILAM"].values  # observations from source 2
n_observations = len(observations_source1)

# assimilated (weighted) values and errors
assimilated = []
err_assimilated = []

sensor1 = Sensor(biased=False)
sensor2 = Sensor(
    biased=True
)  # True if the source is known to be less accurate than the other source

for k in range(n_observations):
    # Step 1: Pre-process observations and estimate errors
    latest_observation_sensor1 = observations_source1[k]
    sensor1.sense(latest_observation_sensor1)

    latest_observation_sensor2 = observations_source2[k]
    accurate_observation = sensor1.get_latest_observation()
    sensor2.sense(latest_observation_sensor2, accurate_observation)

    # Step 2: Assimilate
    sensor1_obs = sensor1.get_latest_observation()
    sensor2_obs = sensor2.get_latest_observation()

    err_sensor1 = sensor1.get_latest_error()
    err_sensor2 = sensor2.get_latest_error()

    assimilated_obs, err_assimilated_obs = assimilate(
        sensor1_obs,
        err_sensor1,
        sensor2_obs,
        err_sensor2,
        use_err_max=False,
    )

    assimilated.append(assimilated_obs)
    err_assimilated.append(err_assimilated_obs)

plot_data(
    pd.Series(observations_source1, index=all_data_df.index),
    pd.Series(observations_source2, index=all_data_df.index),
    pd.Series(assimilated, index=all_data_df.index),
)
plot_errors(
    pd.Series(sensor1.get_all_errors(), index=all_data_df.index),
    pd.Series(sensor2.get_all_errors(), index=all_data_df.index),
    pd.Series(err_assimilated, index=all_data_df.index),
)

print_metrics(
    observations_source1,
    observations_source2,
    assimilated,
    sensor1.get_all_errors(),
    sensor2.get_all_errors(),
    err_assimilated,
)
