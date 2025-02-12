#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(name="saturation",
      version="0.1.0",
      description="Crater saturation equilibrium simulation and modeling",
      author="Sean Mason",
      author_email="seanmason2023@u.northwestern.edu",
      url="https://github.com/seanmason/saturation/",
      packages=find_packages(include=["saturation"]),
      install_requires=[
            "pandas",
            "numpy",
            "pytest",
            "seaborn",
            "matplotlib",
            "scikit-learn",
            "numba",
            "sortedcontainers",
            "pillow",
            "pyarrow",
            "pyyaml",
            "statsmodels",
            "pyspark",
            "plotly",
            "jupyterlab",
            "joblib"
      ]
      )
