from silam_download import get_silam_ts

# get all available data from location (30 days of model data + future forecasts)
# ts = get_silam_ts('CO', 59.431, 24.760)

# get the latest available forecast for a location
ts = get_silam_ts("CO", 59.431, 24.760, max_days=0)

print(ts)
