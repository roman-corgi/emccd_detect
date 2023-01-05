# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image.


# Version

The latest version of emccd\_detect is 2.2.5.


## Getting Started
### Installing

This package requires Python version 3.6 or higher.  If the user wants the ability to apply charge transfer inefficiency (CTI) to detector frames using the optional tool provided in emccd\_detect, then the Python version should be >=3.6 and <=3.9.

To install emccd\_detect, navigate to the emccd\_detect directory where setup.py is located and use

	pip install .

This will install emccd\_detect and its dependencies, which are as follows:

* astropy
* matplotlib
* numpy
* scipy
* pynufft==2020.0.0
* pyyaml

To optionally implement CTI capabilities, navigate to the arcticpy directory, and there will be a file called setup.py in that directory.  Use

	pip install .

This will install arcticpy version 1.0, which is an older version of arcticpy which runs purely on Python (<https://github.com/jkeger/arcticpy/tree/row_wise/arcticpy>).  If
you have Python>=3.10, the CTI functionality will not work if you are using the arcticpy installation that was included with this emccd_detect package, but everything else will work fine.


### Usage

For an example of how to use emccd\_detect, see example_script.py.


## Authors

* Bijan Nemati (<bijan.nemati@tellus1.com>)
* Sam Miller (<sam.miller@uah.edu>)
* Kevin Ludwick (<kevin.ludwick@uah.edu>)

