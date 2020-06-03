#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit data-CAT/__version__.py
version = {}
with open(os.path.join(here, 'dataCAT', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='Data-CAT',
    version=version['__version__'],
    description='A databasing framework for the Compound Attachment Tools package (CAT).',
    long_description=readme + '\n\n',
    author=['Bas van Beek'],
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/data-CAT',
    packages=[
        'dataCAT'
    ],
    package_dir={'dataCAT': 'dataCAT'},
    include_package_data=True,
    license='GNU Lesser General Public License v3 or later',
    zip_safe=False,
    keywords=[
        'database',
        'science',
        'chemistry',
        'python-3',
        'python-3-6',
        'python-3-7',
        'automation'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Database',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    test_suite='tests',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml>=5.1',
        'pymongo',
        'plams@git+https://github.com/SCM-NV/PLAMS@a5696ce62c09153a9fa67b2b03a750913e1d0924',
        'CAT@git+https://github.com/nlesc-nano/CAT@master'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-mock',
        'pycodestyle',
        'CAT@git+https://github.com/nlesc-nano/CAT@devel',
        'AssertionLib@git+https://github.com/nlesc-nano/AssertionLib@master'
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov', 'pytest-mock', 'pycodestyle',
                 'AssertionLib@git+https://github.com/nlesc-nano/AssertionLib@master']
    }
)
