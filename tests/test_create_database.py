"""Tests for :mod:`dataCAT.create_database`."""

from os.path import join

import h5py

from assertionlib import assertion
from nanoutils import delete_finally

from dataCAT.create_database import create_hdf5

PATH = join('tests', 'test_files')
HDF5_PATH = join(PATH, 'structures.hdf5')


@delete_finally(HDF5_PATH)
def test_create_hdf5() -> None:
    """Test :func:`dataCAT.create_database.create_hdf5`."""
    ref_keys1 = ('qd', 'qd_no_opt', 'core', 'core_no_opt', 'ligand', 'ligand_no_opt')
    ref_keys2 = ('job_settings_BDE', 'job_settings_qd_opt', 'job_settings_crs')

    filename = create_hdf5(PATH)
    assertion.eq(filename, HDF5_PATH)
    with h5py.File(HDF5_PATH, 'r', libver='latest') as f:
        for item in ref_keys1:
            assertion.contains(f.keys(), item)
        for item in ref_keys2:
            assertion.contains(f.keys(), item)
            assertion.eq(f[item].ndim, 3)
