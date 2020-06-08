#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit data-CAT/__version__.py
version = {}
with open(os.path.join(here, 'dataCAT', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), version)

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

tests_require=[
    'pytest>=5.4.0',
    'pytest-cov',
    'flake8>=3.8.0'
    'pyflakes>=2.1.1',
    'pytest-flake8>=1.0.6',
    'pytest-pydocstyle>=2.1',
    'CAT@git+https://github.com/nlesc-nano/CAT@devel',
    'AssertionLib>=2.2.0'
]

setup(
    name='Data-CAT',
    version=version['__version__'],
    description='A databasing framework for the Compound Attachment Tools package (CAT).',
    long_description=f'{readme}\n\n',
    author=['B. F. van Beek'],
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/data-CAT',
    packages=['dataCAT'],
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
        'python-3-8',
        'automation'
    ],
    package_data={'dataCAT': ['py.typed', '*.pyi']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Database',
        'Typing :: Typed'
    ],
    test_suite='tests',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml>=5.1',
        'pymongo',
        'Nano-Utils',
        'plams@git+https://github.com/SCM-NV/PLAMS@a5696ce62c09153a9fa67b2b03a750913e1d0924',
        'CAT@git+https://github.com/nlesc-nano/CAT@master'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=tests_require,
    extras_require={
        'test': tests_require
    }
)
