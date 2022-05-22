import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rls_assimilation.RLSAssimilation import RLSAssimilation
from rls_assimilation.SequentialRLSAssimilation import (
    SequentialRLSAssimilationOneSource,
    SequentialRLSAssimilationTwoSources,
)
from helpers import (
    plot_data_seq,
    print_metrics_seq,
    get_rmse,
    print_stats_from_array,
    read_data,
    prepare_daily_data,
)


np.seterr(all="raise")


def run_assimilation(df, variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out):
    is_multi_t = t_in1 != t_out or t_in2 != t_out  # multi-temporal data assimilation
    is_one_seq_source = not is_multi_t

    assimilator = RLSAssimilation(
        t_in1=t_in1,
        t_in2=t_in2,
        s_in1=s_in1,
        s_in2=s_in2,
        t_out=t_out,
        s_out=s_out,
    )
    if is_one_seq_source:
        seq_assimilator = SequentialRLSAssimilationOneSource()
    else:
        seq_assimilator = SequentialRLSAssimilationTwoSources(
            t_in1=t_in1,
            t_in2=t_in2,
            s_in1=s_in1,
            s_in2=s_in2,
            t_out=t_out,
            s_out=s_out,
        )

    if is_one_seq_source:
        source1_col = f"{variable}"
        source2_col = f"{variable}_model"
        seq_source_col = source1_col if s_out == s_in1 else source2_col
    else:
        source1_col = f"{variable}_{s_in1}_{t_in1}"
        source2_col = f"{variable}_{s_in2}_{t_in2}"
        seq_source_col = None

    assimilated = []
    err_assimilated = []
    seq_assimilated = []
    seq_err_assimilated = []

    for k in range(len(df)):
        # Step 1: Obtain raw observations from 2 sources
        latest_observation_source1 = df[source1_col].values[k]
        latest_observation_source2 = df[source2_col].values[k]

        # Step 2: Assimilate
        analysis, err_analysis = assimilator.assimilate(
            latest_observation_source1,
            latest_observation_source2,
        )
        assimilated.append(analysis)
        err_assimilated.append(err_analysis)

        if is_one_seq_source:
            latest_observation_source = df[seq_source_col][k]
            seq_analysis, seq_err_analysis = seq_assimilator.assimilate(
                latest_observation_source
            )
        else:
            seq_analysis, seq_err_analysis = seq_assimilator.assimilate(
                latest_observation_source1,
                latest_observation_source2,
            )

        seq_assimilated.append(seq_analysis)
        seq_err_assimilated.append(seq_err_analysis)

    df["Assimilated"] = assimilated
    df["Seq_Assimilated"] = seq_assimilated
    df = df.dropna()

    # Step 3: Get metrics
    # Uncertainties
    mean_unc_da = np.mean(err_assimilated)
    mean_unc_seq = np.mean(seq_err_assimilated)

    # Get a ratio of mean uncertainties for DA and sequential DA
    try:
        err_seq_da_ratio = mean_unc_seq / mean_unc_da
    except (ZeroDivisionError, FloatingPointError):
        err_seq_da_ratio = 1

    # RMSE between values
    if seq_source_col:
        rmse_seq = get_rmse(
            df["Seq_Assimilated"].values,
            df[seq_source_col].values,
        )
        rmse_da = get_rmse(
            df[f"Assimilated"].values,
            df[seq_source_col].values,
        )
        try:
            seq_da_ratio = rmse_seq / rmse_da
        except (ZeroDivisionError, FloatingPointError):
            seq_da_ratio = 1

        return (
            seq_da_ratio,
            None,
            err_seq_da_ratio,
            df,
            err_assimilated,
            seq_err_assimilated,
        )

    # Compare errors of assimilated from actual hourly reference
    rmse_da_h = get_rmse(
        df["Assimilated"].values,
        df[f"{variable}_{s_out}_hourly"].values,
    )
    rmse_seq_h = get_rmse(
        df["Seq_Assimilated"].values,
        df[f"{variable}_{s_out}_hourly"].values,
    )
    rmse_dh = get_rmse(
        df[f"{variable}_{s_out}_daily"].values,
        df[f"{variable}_{s_out}_hourly"].values,
    )

    try:
        da_dh_ratio = rmse_da_h / rmse_dh
    except (ZeroDivisionError, FloatingPointError):
        da_dh_ratio = 1

    try:
        seq_dh_ratio = rmse_seq_h / rmse_dh
    except (ZeroDivisionError, FloatingPointError):
        seq_dh_ratio = 1

    return (
        da_dh_ratio,
        seq_dh_ratio,
        err_seq_da_ratio,
        df,
        err_assimilated,
        seq_err_assimilated,
    )


def test_Liivalaia(t_in1, t_in2, s_in1, s_in2, t_out, s_out):
    is_multi_t = t_in1 != t_out or t_in2 != t_out

    data_path = "data/liivalaia_aq_meas_with_forecast.csv"

    variables = ["CO", "NO2", "O3", "SO2", "PM2.5", "PM10"]

    fig_data, axs_data = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    for idx, variable in enumerate(variables):
        print(variable)
        if not is_multi_t:
            df = read_data(data_path)
            (
                seq_da_ratio,
                _,
                err_seq_da_ratio,
                df,
                err_assimilated,
                seq_err_assimilated,
            ) = run_assimilation(df, variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out)
        else:
            df = prepare_daily_data(variable, data_path)
            (
                da_dh_ratio,
                seq_dh_ratio,
                err_seq_da_ratio,
                df,
                err_assimilated,
                seq_err_assimilated,
            ) = run_assimilation(df, variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out)

        da_scenario = f"DA{'3' if not is_multi_t else '4'} ({'Model' if s_out == s_in1 else 'Station'} -> {'Station' if s_out == s_in1 else 'Model'})"
        seq_scenario = (
            f"Sequential DA ({'Station' if s_in1 == s_out else 'Model'})"
            if not is_multi_t
            else f"Sequential DA ({'Model' if s_out == s_in1 else 'Station'} -> {'Station' if s_out == s_in1 else 'Model'})"
        )

        if not is_multi_t:
            source1_col = f"{variable}"
            source2_col = f"{variable}_model"
        else:
            source1_col = f"{variable}_{s_in1}_{t_in1}"
            source2_col = f"{variable}_{s_in2}_{t_in2}"

        axs_data[idx % 3, idx % 2] = plot_data_seq(
            pd.Series(df[source1_col], index=df.index),
            pd.Series(df[source2_col], index=df.index),
            pd.Series(df["Assimilated"], index=df.index),
            pd.Series(df["Seq_Assimilated"], index=df.index),
            variable,
            axs_data[idx % 3, idx % 2],
            da_scenario,
            seq_scenario,
        )

        print_metrics_seq(
            df[source1_col].values,
            df[source2_col].values,
            df["Assimilated"].values,
            err_assimilated,
            df["Seq_Assimilated"].values,
            seq_err_assimilated,
            da_scenario,
            seq_scenario,
        )

    scenario_id = f"da{'3' if not is_multi_t else '4'}-{'1' if s_out == 'obs' else '2'}"
    fig_data.savefig(f"plots/Liivalaia/Sequential/data-{scenario_id}.jpg")


def test_variable_Europe_AQ(
    variable,
    t_in1,
    t_in2,
    s_in1,
    s_in2,
    t_out,
    s_out,
):
    is_multi_t = t_in1 != t_out or t_in2 != t_out
    data_path_dir = f"data/Europe_AQ/combined_{variable}"
    unc_ratios = []

    if not is_multi_t:
        seq_da_ratios = []
        for filename in os.listdir(data_path_dir):
            df = read_data(f"{data_path_dir}/{filename}")
            (
                seq_da_ratio,
                _,
                err_seq_da_ratio,
                _,
                _,
                _,
            ) = run_assimilation(df, variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out)
            seq_da_ratios.append(seq_da_ratio)
            unc_ratios.append(err_seq_da_ratio)

        print_stats_from_array(seq_da_ratios, "RMSE ratio (Sequential/Non-sequential)")
    else:
        da_dh_ratios = []
        seq_dh_ratios = []
        for filename in os.listdir(data_path_dir):
            data_path = f"{data_path_dir}/{filename}"
            df = prepare_daily_data(variable, data_path)
            (
                da_dh_ratio,
                seq_dh_ratio,
                err_seq_da_ratio,
                _,
                _,
                _,
            ) = run_assimilation(df, variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out)
            da_dh_ratios.append(da_dh_ratio)
            seq_dh_ratios.append(seq_dh_ratio)
            unc_ratios.append(err_seq_da_ratio)

        print_stats_from_array(
            da_dh_ratios,
            "RMSE ratio from hourly reference (Non-sequential assimilated / Daily reference)",
        )
        print_stats_from_array(
            seq_dh_ratios,
            "RMSE ratio from hourly reference (Sequential assimilated / Daily reference)",
        )

    print_stats_from_array(unc_ratios, "MAU ratio (Sequential/Non-Sequential)")


def generate_tests(is_multi_t, s_out):
    s_in1 = "obs"
    s_in2 = "model"

    t_in1 = "daily" if is_multi_t and s_in1 == s_out else "hourly"
    t_in2 = "daily" if is_multi_t and s_in2 == s_out else "hourly"
    t_out = "hourly"

    print(
        f"Scales: {'hourly' if not is_multi_t else 'daily to hourly'}, {'model to station' if s_out == 'obs' else 'station to model'}"
    )

    # For Liivalaia
    # print('Liivalaia')
    # test_Liivalaia(t_in1, t_in2, s_in1, s_in2, t_out, s_out)

    # For Europe AQ dataset
    print("European AQ")
    variables = ["CO", "NO2", "O3", "SO2", "PM25", "PM10"]
    for variable in variables:
        print(variable)
        test_variable_Europe_AQ(variable, t_in1, t_in2, s_in1, s_in2, t_out, s_out)


# Test 1-source sequential VS 2-source non-sequential (the same temporal scales)
generate_tests(False, "obs")
# generate_tests(False, "model")

# Test 2-source non-sequential VS 2-source sequential (different temporal scales)
# generate_tests(True, "obs")
generate_tests(True, "model")
