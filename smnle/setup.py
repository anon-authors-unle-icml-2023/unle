#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


dist = setup(
    name='smnle',
    version='0.0.0dev0',
    description='Extended pickling support for Python objects',
    license='BSD 3-Clause License',
    packages=find_packages(),
    long_description='Score Matched Neural Likelihood Estimation',
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',
    ],
    test_suite='tests',
    python_requires='>=3.6',
)
