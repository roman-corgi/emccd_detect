# -*- coding: utf-8 -*-
from io import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='emccd_detect',
    version='1.0.2',
    description='EMCCD detector image simulation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.jpl.nasa.gov/WFIRST-CGI/emccd_detect',
    author='Bijan Nemati, Sam Miller',
    author_email='bijan.nemati@uah.edu, sm0204@uah.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.7',
    install_requires=[
        'astropy',
        'matplotlib',
        'numpy',
        'scipy'
    ]
)
