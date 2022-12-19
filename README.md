# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image.


# Version

The latest version of emccd\_detect is 2.2.5.



## Getting Started
### Installing

This package requires Python version 3.6 or higher.  To install emccd\_detect, navigate to the emccd\_detect directory where setup.py is located and use

	pip install .

This will install emccd\_detect and its dependencies, which are as follows:

* arcticpy==1.1
* astropy
* matplotlib
* numpy
* scipy
* pynufft==2020.0.0
* pyyaml

If your installation fails, pay attention to what the output says.  It may require an update of C++ tools, and it will probably tell you how to update.

### Usage

For an example of how to use emccd\_detect, see example_script.py.


## Authors

* Bijan Nemati (<bijan.nemati@tellus1.com>)
* Sam Miller (<sam.miller@uah.edu>)
* Kevin Ludwick (<kevin.ludwick@uah.edu>)

