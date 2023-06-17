# RLS-based data assimilation with unknown uncertainty

Implementation of the Recursive Least Squares (RLS)-based uncertainty quantification and least-squares data assimilation
algorithms.
 
The algorithms are described in the following papers:
 * [Lightweight Assimilation of Open Urban Ambient Air Quality Monitoring Data and Numerical Simulations 
with Unknown Uncertainty](https://link.springer.com/article/10.1007/s10666-023-09909-x), DOI: 10.1007/s10666-023-09909-x 
(experiments from `example1.py`).
 * [Lightweight Open Data Assimilation of Pan-European Urban Air Quality](#TODO) 
 (experiments from `example2.py`).


Python: 3.*

## Testing

To be able to reproduce the results presented in the paper, install requirements from `requirements.txt`.
It will install the packages needed to read and plot the data.

Project package dependencies can be installed using `pip` and a virtual environment manager:

Create and activate a virtual environment:

    python -m venv rls-venv
    source rls-venv/bin/activate

In the virtual environment, install requirements from the `requirements.txt` file:

    pip install -r requirements.txt
 
### Lightweight Assimilation of Open Urban Ambient Air Quality Monitoring Data and Numerical Simulations with Unknown Uncertainty 

After installing the requirements, run the script `example1.py`:
 
    python example1.py

The script will generate plots in `plots/Liivalaia/` and print metrics for the autumn Tallinn dataset 
`data/liivalaia_aq_meas_with_forecast.csv`. Modify plots' location to `plots/Liivalaia2/` for the winter Tallinn dataset 
`data/liivalaia_aq_meas_with_forecast2.csv`.

The IoT PM10 sensor data corresponds to the `data/liivalaia_pm10_iot.csv` dataset. The plots for the sensor data are
located in the `plots/Liivalaia/IoT` directory.

### Sequential Assimilation of Open European Urban Ambient Air Quality Data of Different Scales with Unknown Uncertainty

After installing the requirements, run the script `example2.py`:
 
    python example2.py
    
The script run the experiments and prints the validation results described in the paper.
The plots are generated for the `data/eu-eq.csv` dataset file. The statistics are collected for the datasets from the 
`data/Europe_AQ/` directory.
    
## Repository content

`rls_assimilatiion` is the lightweight package for uncertainty quantification and least-squares data 
assimilation.

The used data is stored in the `data/` directory, plots are generated to `plots/` directory.

Directory `download/` contains script to download data from the SILAM cloud storage.
