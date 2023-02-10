#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = ("A set of utilities for working with slurm-administered clusters.")

dist = setup(
    name="experiments_utils",
    version="0.0.0dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=["experiments_utils"],
    install_requires=["pandas", "joblib", "cloudpickle"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.8",
)
