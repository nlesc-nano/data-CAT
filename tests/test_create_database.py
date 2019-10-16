"""Tests for :mod:`dataCAT.create_database`."""

from os import remove
from os.path import join

import yaml
import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion

from dataCAT.create_database import (_create_csv, _create_hdf5, _create_yaml)

PATH = join('tests', 'test_files')


def test_create_csv() -> None:
    """Test :func:`dataCAT.create_database._create_csv`."""
    path1 = join(PATH, 'ligand_database.csv')
    path2 = join(PATH, 'QD_database.csv')
    dtype1 = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
    dtype2 = {'hdf5 index': int, 'settings': str, 'opt': bool}

    try:
        filename1 = _create_csv(PATH, 'ligand')
        filename2 = _create_csv(PATH, 'QD')
        assertion.eq(filename1, path1)
        assertion.eq(filename2, path2)
        df1 = pd.read_csv(path1, index_col=[0, 1], header=[0, 1], dtype=dtype1)
        df2 = pd.read_csv(path2, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype2)

        assertion.eq(df1.shape, (1, 4))
        assertion.eq(df2.shape, (1, 5))
        assertion.eq(df1.index.names, ['smiles', 'anchor'])
        assertion.eq(df2.index.names, ['core', 'core anchor', 'ligand smiles', 'ligand anchor'])
        assertion.eq(df1.columns.names, ['index', 'sub index'])
        assertion.eq(df2.columns.names, ['index', 'sub index'])
        assertion.contains(df1.index, ('-', '-'))
        assertion.contains(df2.index, ('-', '-', '-', '-'))

        np.testing.assert_array_equal(
            df1.values, np.array([[-1, 'str', False, 'str']], dtype=object)
        )
        np.testing.assert_array_equal(
            df2.values, np.array([[-1, -1, False, 'str', 'str']], dtype=object)
        )

        assertion.assert_(_create_csv, PATH, 'bob', exception=ValueError)
    finally:
        remove(path1)
        remove(path2)


def test_create_hdf5() -> None:
    """Test :func:`dataCAT.create_database._create_hdf5`."""
    path = join(PATH, 'structures.hdf5')
    ref_keys1 = ('QD', 'QD_no_opt', 'core', 'core_no_opt', 'ligand', 'ligand_no_opt')
    ref_keys2 = ('job_settings_BDE', 'job_settings_QD_opt', 'job_settings_crs')

    try:
        filename = _create_hdf5(PATH)
        assertion.eq(filename, path)
        with h5py.File(path, 'r') as f:
            for item in ref_keys1:
                assertion.contains(f.keys(), item)
                assertion.eq(f[item].ndim, 2)
            for item in ref_keys2:
                assertion.contains(f.keys(), item)
                assertion.eq(f[item].ndim, 3)
    finally:
        remove(path)


def test_create_yaml() -> None:
    """Test :func:`dataCAT.create_database._create_yaml`."""
    path = join(PATH, 'job_settings.yaml')

    try:
        filename = _create_yaml(PATH)
        assertion.eq(filename, path)
        with open(path, 'r') as f:
            out = yaml.load(f, Loader=yaml.FullLoader)
        assertion.eq(out, {None: [None]})
    finally:
        remove(path)
