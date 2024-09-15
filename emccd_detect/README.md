# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image.  Website:  (<https://github.com/roman-corgi/emccd_detect/tree/master/emccd_detect>)


# Version

The latest version of emccd\_detect is 2.4.0.  Main differences from previous version: the ability to implement readout nonlinearity and the latest version of arcticpy for charge transfer inefficiency implementation.


## Getting Started
### Installing

This package requires Python version 3.6 or higher.  If the user wants the ability to apply charge transfer inefficiency (CTI) to detector frames using the optional tool (older version of arcticpy which is pure Python) provided in emccd\_detect, then the Python version should be >=3.6 and <=3.9.  If the newer version of arcticpy (wrapper around C++ code) is installed, there is no upper limit restriction for Python version. For installation instructions and documentation for the newer arcticpy, see <https://github.com/jkeger/arctic>. emccd\_detect works apart from arcticpy and does not require it.

emccd\_detect is available on PyPI.org, so the following command will install the module (without CTI capabilities):

	pip install emccd-detect

To install emccd\_detect instead from this package download, after downloading, navigate to the emccd\_detect directory where setup.py is located and use

	pip install .

This will install emccd\_detect and its dependencies, which are as follows:

* astropy
* matplotlib
* numpy
* scipy
* pynufft==2020.0.0
* pyyaml

To optionally implement CTI capabilities with the pure-Python arcticpy, navigate to the arcticpy directory (<https://github.com/roman-corgi/emccd_detect/tree/master/arcticpy_folder>), and there will be a file called setup.py in that directory.  Use

	pip install .

This will install arcticpy version 1.0.  See (<https://github.com/jkeger/arcticpy/tree/row_wise/arcticpy>) for documentation.  If
you have Python>3.9, the CTI functionality will not work if you are using the arcticpy installation that was included with this emccd_detect package, but everything else will work fine.


### Usage

For an example of how to use emccd\_detect, see example_script.py.


## Authors

* Bijan Nemati (<bijan.nemati@tellus1.com>)
* Sam Miller (<sam.miller@uah.edu>)
* Kevin Ludwick (<kevin.ludwick@uah.edu>)

