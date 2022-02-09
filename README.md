# RLS-based data assimilation with unknown uncertainty

Implementation of the Recursive Least Squares (RLS)-based uncertainty quantification and least-squares data assimilation
algorithm described in paper *Efficient Urban Ambient Air Quality Assimilation Using Open Environmental Monitoring Data 
and Simulations with Unknown Uncertainty*.

Python: 3.*

## Usage

1) Install the wheel from the `dist` directory with `pip install rls_assimilation-*.whl`.
It installs the dependencies needed to run the uncertainty quantification and least squares data assimilation 
algorithms. *NB*: it installs only the minimum requirements, see **Testing** section to see how to reproduce the 
results presented in the paper.
2) The simplest usage of the algorithm is as follows:

```
from rls_assimilation import RLSAssimilation 

# input and initialisation
data_source1 = [1, 2, 3]
data_source2 = [3, 5, 6]
assimilator_with_calibration = RLSAssimilation(do_calibration=True)

# assimilate
for i in range(len(data_source1)):
    x1 = data_source1[i]
    x2 = data_source2[i]

    assimilated, uncertainty = assimilator_with_calibration.assimilate(x1, x2)
```

See `example.py` for other examples.


## Testing

To be able to reproduce the results presented in the paper, install requirements from `requirements.txt`.
It will install the packages needed to read and plot the data.

Project package dependencies can be installed using `pip` and a virtual environment manager:

Create and activate a virtual environment (refer to https://docs.python.org/3/library/venv.html for more 
information):

    python -m venv rls-venv
    source rls-venv/bin/activate

In the virtual environment, install requirements from the `requirements.txt` file:

    pip install -r requirements.txt
   
After installing the requirements, run the script `example.py`:
 
    python example.py

The script will generate plots in `plots/` and print metrics for the Tallinn dataset 
`data/liivalaia_aq_meas_with_forecast.csv`.
    
## Repository content

`rls_assimilatiion` is the lightweight package for uncertainty quantification and least-squares data 
assimilation. `pyproject.toml` includes the package details, the wheel is generated with `poetry build` to `dist/` 
directory.

The used data is stored in the `data/` directory, plots are generated to `plots/` directory.

Directory `download/` contains script to download data from the SILAM cloud storage.


