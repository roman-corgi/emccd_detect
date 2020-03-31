# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image. 

This project is managed simultaneously in both Python and Matlab. The Python version is located in emccd\_detect, while the Matlab version is located in emccd\_detect\_m.

This is a preliminary version; future updates will have fast EMCCD capability (for quick approximation accross a large set of frames), as well more finely tuned error simulations (ie. fixed pattern, cosmics, etc.).

## Getting Started - Python
### Installing

This package requires Python 3.7. To install emccd\_detect, navigate to the root emccd\_detect directory where setup.py is located. (If you are using conda, activate the environment where you would like it installed). Then use

	pip install .

This will install the dependencies, which are as follows:

* astropy
* matplotlib
* numpy
* scipy


### Usage

For an example of how to use emccd\_detect, see main.py.

## Getting Started - Matlab
### Installing

To use emccd\_detect in one of your matlab projects, either add emccd\_detect\_m to your path or copy the emccd\_detect\_m directory to your project.

### Usage

For an example of how to use emccd\_detect, see scripts/main.m.

## Authors

* Bijan Nemati (<bijan.nemati@uah.edu>)
* Sam Miller (<sm0204@uah.edu>)

