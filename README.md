# RLS assimilation

Implementation of the RLS (Recursive Least Squares)-based uncertainty quantification and assimilation
algorithm described in paper `TODO`.

Python: 3.8

## Installation

Project package dependencies can be installed using `pip` and a virtual environment manager:

Create and activate a virtual environment (refer to https://docs.python.org/3/library/venv.html for more 
information):

    python -m venv rls-venv
    source rls-venv/bin/activate

Install requirements from the `requirements.txt` file:

    pip install -r requirements.txt
    
    
## Repository content

The used data is stored in the `data/` directory, plots are generated to `plots/` directory.

Directory `download/` contains script to download data from the SILAM cloud storage.


## Usage

To run the example script `example.py`:
 
    python example.py
