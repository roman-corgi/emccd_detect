# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image. 

This project is managed simultaneously in both Python and Matlab. The Python version is located in emccd\_detect, while the Matlab version is located in emccd\_detect\_m.


## Getting Started - Python
### Installing

This package requires Python 3.6 or higher. To install emccd\_detect, navigate to the emccd\_detect directory (where setup.py is located). (If you are using conda, activate the environment where you would like it installed). Then use

	pip install .

This will install the dependencies, which are as follows:

* arcticpy
* astropy
* matplotlib
* numpy
* scipy
* pynufft
* pyyaml


### Usage

For an example of how to use emccd\_detect, see example_script.py.

## Getting Started - Matlab
### Installing

To use emccd\_detect in one of your matlab projects, either add emccd\_detect\_m to your path or copy the emccd\_detect\_m directory to your project.

### Usage

For an example of how to use emccd\_detect, see scripts/example_script.m.

## Authors

* Bijan Nemati (<bijan.nemati@uah.edu>)
* Sam Miller (<sam.miller@uah.edu>)

