#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(name='saturation',
      version='0.1.0',
      description='Crater saturation equilibrium simulation and modeling',
      author='Sean Mason',
      author_email='seanmason2023@u.northwestern.edu',
      url='https://github.com/seanmason/saturation/',
      packages=find_packages(include=['saturation']),
      install_requires=[
            'pandas==1.4.3',
            'numpy==1.22.4',
            'pytest==7.1.2',
            'seaborn==0.11.2',
            'matplotlib==3.5.2',
            'scikit-learn==1.1.1',
            'numba'
      ]
      )
