# -*- coding: utf-8 -*-
from io import open
from os import path
from setuptools import setup, find_packages, find_namespace_packages
#import emccd_detect

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='emccd_detect',
    version='2.4.0', #emccd_detect.__version__,
    description='EMCCD detector image simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect',
    author='Bijan Nemati, Sam Miller, Kevin Ludwick',
    author_email='bijan.nemati@tellus1.com, sam.miller@uah.edu, kevin.ludwick@uah.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    #packages=find_namespace_packages('arcticpy.src', 'arcticpy.include'),
    packages=find_packages(exclude=["arcticpy_folder.*", "arcticpy_folder", "arcticpy.*", "arcticpy"]),
    #packages=find_namespace_packages(),
    #packages=['emccd_detect'],
    package_data={'': ['metadata.yaml']},
    include_package_data=True,
    #exclude_package_data={'':['arcticpy']},
    python_requires= '>=3.6',
    install_requires=[
        #'arcticpy @ git+https://github.com/jkeger/arcticpy@row_wise#egg=arcticpy',
        #'arcticpy @ git+https://github.com/jkeger/arcticpy.git',
        #'arcticpy @ git+https://github.com/jkeger/arctic.git',
        #'arctic',
        'astropy',
        'matplotlib',
        'numpy',
        'scipy',
        'pynufft==2020.0.0',
        'pyyaml'
    ]
)
