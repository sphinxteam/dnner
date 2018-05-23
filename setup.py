#!/usr/bin/env python

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(name='dnner',
      version='0.1',
      install_requires=['numpy', 'matplotlib', 'scipy', 'nose2'],
      description='DNNs Entropy from Replicas',
      author='Andre Manoel, Marylou Gabrie, Florent Krzakala',
      author_email='andremanoel@gmail.com',
      packages=find_packages(),
      ext_modules=cythonize("dnner/activations/*.pyx")
     )
