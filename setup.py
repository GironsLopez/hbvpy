#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HBVpy: A module for interacting with the HBV-light hydrological model.

HBVpy is a package designed to interact with the command line version of the
HBV-light hydrological model. It allows generate the necessary configuration
files adapted for the different modifications, to run the model,
and to process the results.

"""
from setuptools import setup, find_packages


def readme():
    with open('README.md') as rm:
        return rm.read()


setup(
    name='hbvpy',
    version='0.1',
    description='Python functions to interact with HBV-light',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Operating System :: Microsoft :: Windows'
    ],
    keywords='HBV rainfall-runoff hydrology model',
    url='https://github.com/GironsLopez/hbvpy',
    author='Marc Girons Lopez',
    author_email='m.girons@gmail.com',
    license='BSD',
    packages=find_packages(),
    install_requires=[
        'lxml', 'netCDF4', 'numpy', 'gdal',
        'pandas', 'pyproj', 'scipy'
    ],
    zip_safe=False,
)
