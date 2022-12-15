# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image.


# Version

The latest version of emccd\_detect is 2.2.5.



## Getting Started
### Installing

This package requires Python between versions 3.6 and 3.8, inclusive.  (emccd\_detect currently uses arcticpy, which cannot run on more recent Python versions.  The update to that package, called arCTIc, can run on newer versions of Python, but it requires C++ libraries to run.)  To install emccd\_detect, navigate to the emccd\_detect directory where setup.py is located and use

	pip install .

This will install emccd\_detect and its dependencies, which are as follows:

* arcticpy
* astropy
* matplotlib
* numpy
* scipy
* pynufft==2020.0.0
* pyyaml


### Usage

For an example of how to use emccd\_detect, see example_script.py.


## Authors

* Bijan Nemati (<bijan.nemati@tellus1.com>)
* Sam Miller (<sam.miller@uah.edu>)
* Kevin Ludwick (<kevin.ludwick@uah.edu>)

