# EMCCD Detect

Given an input fluxmap, emccd_detect will return a simulated EMCCD detector image. 

This is a preliminary version; future updates will have fast EMCCD capability (for quick approximation accross a large set of frames), as well more finely tuned error simulations (ie. fixed pattern, cosmics, etc.).

## Getting Started
### Installing

This package requires Python 2.7. To install emccd_detect, use 

	python setup.py install

This will install the dependencies, which are as follows:

* matplotlib
* numpy
* scipy

**Note for Linux users:** Upon runtime you may get the error

	ImportError: No module named _tkinter, please install the python-tk package

If so, use

	sudo apt-get install python-tk

to install Tkinter on your machine.

### Usage

For an example of how to use emccd\_detect, run example\_script.

## Authors

* Bijan Nemati
* Sam Miller

