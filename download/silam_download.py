import numpy as np
import pandas as pd
import datetime
import xarray as xr
import boto3
from botocore.handlers import disable_signing
import os


def download(bucket_name, key, dst_root="/tmp"):
    """Download zarr directory from S3"""
    resource = boto3.resource("s3")
    resource.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)

    bucket = resource.Bucket(bucket_name)
    for object in bucket.objects.filter(Prefix=key):
        dst = dst_root + "/" + object.key
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))
        resource.meta.client.download_file(bucket_name, object.key, dst)


def get_forecast_from_silam(date_str, modality, day, version="v5_7_1", fmt="zarr"):
    """
    Source: https://en.ilmatieteenlaitos.fi/silam-opendata-on-aws-s3
    datestr: 8 digits (YYYYMMDD), e.g. 21210101
    modality: CO, NO2, NO, O3, PM10, PM25, SO2, airdens
    day: one of 0, 1, 2, 3, 4
    version: check if anything changed http://fmi-opendata-silam-surface-zarr.s3-website-eu-west-1.amazonaws.com/?prefix=global/
    fmt: zarr or netcdf
    """
    bucket_name = f"fmi-opendata-silam-surface-{fmt}"
    key = f"global/{date_str}/silam_glob_{version}_{date_str}_{modality}_d{day}.{fmt}"

    tmp_dir = "/tmp"
    tmp_file = tmp_dir + "/" + key

    if not os.path.exists(os.path.dirname(tmp_file)):
        os.makedirs(os.path.dirname(tmp_file))

    download(bucket_name, key)

    print(tmp_file)

    ds = xr.open_zarr(tmp_file)

    return ds


def get_date_str(start_date):
    month_str = f'{start_date.month if len(str(start_date.month)) == 2 else f"0{start_date.month}"}'
    day_str = (
        f'{start_date.day if len(str(start_date.day)) == 2 else f"0{start_date.day}"}'
    )

    return f"{start_date.year}{month_str}{day_str}"


def find_closest_to(arr, val):
    return arr.flat[np.abs(arr - val).argmin()]


def get_curve_series(ds, modality, approx_lat, approx_lon):
    lat = find_closest_to(ds[modality].lat.values, approx_lat)
    lon = find_closest_to(ds[modality].lon.values, approx_lon)

    times = [val.values for val in list(ds[modality].time)]
    data = ds[modality].sel(lat=lat, lon=lon).values

    return pd.Series(index=times, data=data)


def get_var_meta(ds, var):
    for name, da in ds.variables.items():
        if name == var:
            return dict(list(da.attrs.items()))

    return None


def get_all_days_series(start_date, modality, fmt, lat, lon):
    series_list = []
    for d in range(5):
        ds = get_forecast_from_silam(get_date_str(start_date), modality, day=d, fmt=fmt)
        ts = get_curve_series(ds, modality, lat, lon)
        series_list.append(ts)

    return pd.concat(series_list, axis=0)


def collect_latest_available_data(modality, lat, lon, max_days=30):
    all_series = []
    for offset_days in range(0, max_days + 1):
        start_date = datetime.datetime.now() - datetime.timedelta(offset_days)
        series = get_all_days_series(start_date, modality, "zarr", lat, lon)
        all_series.append(series)

    return all_series


def get_silam_ts(modality, lat, lon, max_days=30):
    all_series = collect_latest_available_data(modality, lat, lon, max_days)

    # build a dataframe from 5-day forecasts
    df = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    for ts in all_series:
        for idx, val in ts.items():
            if idx in list(df.index):
                for col in df.columns:
                    if np.isnan(df.loc[idx, col]):
                        df.loc[idx, col] = val
                        break
            else:
                df.loc[idx, 0] = val

    df = df.sort_index()

    # take series of the latest available measurements
    for idx, row in df.iterrows():
        for col in df.columns:
            if col != "0":
                if not np.isnan(row[col]):
                    df.loc[idx, "0"] = row[col]
                else:
                    break

    silam_ts = df["0"]
    format = "%Y-%m-%d %H:%M:%S"
    silam_ts.index = pd.to_datetime(list(silam_ts.index), format=format)

    return silam_ts
