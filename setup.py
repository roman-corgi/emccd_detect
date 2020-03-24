# -*- coding: utf-8 -*-
from io import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='emccd_detect',
    version='0.0.1',
    description='An EMCCD Detector image simulator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect',
    author='Bijan Nemati, Sam Miller',
    author_email='bn0021@uah.edu, sm0204@uah.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
        'scipy'
    ]
)
