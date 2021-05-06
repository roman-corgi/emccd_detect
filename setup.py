# -*- coding: utf-8 -*-
from io import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='emccd_detect',
    version='2.1.0',
    description='EMCCD detector image simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect',
    author='Bijan Nemati, Sam Miller',
    author_email='bijan.nemati@uah.edu, sam.miller@uah.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'arcticpy',
        'astropy',
        'matplotlib',
        'numpy',
        'scipy',
        'pynufft==2020.0.0',
        'pyyaml'
    ]
)
