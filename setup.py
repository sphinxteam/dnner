#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name='dnner',
      version='0.0',
      install_requires=['numpy', 'matplotlib', 'scipy', 'nose2'],
      description='DNNs Entropy from Replicas',
      packages=find_packages(),
      ext_modules=cythonize("dnner/activations/*.pyx")
     )
