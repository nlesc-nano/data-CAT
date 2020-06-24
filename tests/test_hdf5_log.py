"""Tests for :mod:`dataCAT.hdf5_log`."""

from types import MappingProxyType
from shutil import copyfile

import h5py
import numpy as np

from assertionlib import assertion
from nanoutils import delete_finally
from dataCAT import update_hdf5_log, reset_hdf5_log, log_to_dataframe
from dataCAT.testing_utils import HDF5_READ, HDF5_TMP


@delete_finally(HDF5_TMP)
def test_update_hdf5_log1() -> None:
    """Test :func:`dataCAT.update_hdf5_log`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        group = f['ligand']
        log = f['ligand/logger']

        i = len(log['date'])
        n0 = log.attrs['n']
        assertion.truth(n0)

        i += log.attrs['n_step']
        log.attrs['n'] = 100
        update_hdf5_log(group, idx=[0])

        n1 = log.attrs['n']
        assertion.eq(n1, 101)
        for name, dset in log.items():
            if name != 'version_names':
                assertion.len_eq(dset, i)


@delete_finally(HDF5_TMP)
def test_update_hdf5_log2() -> None:
    """Test :func:`dataCAT.update_hdf5_log`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        group = f['ligand']
        group['logger'].attrs['clear_when_full'] = True

        i = len(group['logger/date'])
        n0 = group['logger'].attrs['n']
        assertion.truth(n0)

        group['logger'].attrs['n'] = 100
        update_hdf5_log(group, idx=[0])

        n1 = group['logger'].attrs['n']
        assertion.eq(n1, 1)
        for name, dset in group['logger'].items():
            if name != 'version_names':
                assertion.len_eq(dset, i)


REF_COLUMNS = MappingProxyType({
    ('CAT', 'major'): np.dtype('int8'),
    ('CAT', 'minor'): np.dtype('int8'),
    ('CAT', 'micro'): np.dtype('int8'),
    ('Nano-CAT', 'major'): np.dtype('int8'),
    ('Nano-CAT', 'minor'): np.dtype('int8'),
    ('Nano-CAT', 'micro'): np.dtype('int8'),
    ('Data-CAT', 'major'): np.dtype('int8'),
    ('Data-CAT', 'minor'): np.dtype('int8'),
    ('Data-CAT', 'micro'): np.dtype('int8'),
    ('message', ''): np.dtype('O'),
    ('index', ''): np.dtype('O')
})


@delete_finally(HDF5_TMP)
def test_log_to_dataframe() -> None:
    """Test :func:`dataCAT.log_to_dataframe`."""
    copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+', libver='latest') as f:
        reset_hdf5_log(f['ligand'])
        df = log_to_dataframe(f['ligand'])
        dct = {k: v.dtype for k, v in df.items()}
        assertion.eq(dct, REF_COLUMNS)
