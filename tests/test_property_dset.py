"""Tests for :mod:`dataCAT.property_dset`."""

import shutil

import h5py
import numpy as np

from nanoutils import delete_finally
from assertionlib import assertion
from dataCAT import update_prop_dset
from dataCAT.testing_utils import HDF5_TMP, HDF5_READ


@delete_finally(HDF5_TMP)
def test_update_prop_dset() -> None:
    """Test :func:`dataCAT.update_prop_dset`."""
    shutil.copyfile(HDF5_READ, HDF5_TMP)

    with h5py.File(HDF5_TMP, 'r+') as f:
        dset = f['ligand/properties/formula']

        data1 = np.array(['a', 'b', 'c'], dtype=np.string_)
        update_prop_dset(dset, data1, index=slice(None, 3))
        assertion.eq(data1, dset[:3], post_process=np.all)

        # Resize the index
        idx = f['ligand/index']
        n = len(idx)
        idx.resize(n + 3, axis=0)

        # Test if the automatic resizing works
        data2 = np.array(['d', 'e', 'f'], dtype=np.string_)
        update_prop_dset(dset, data2, index=slice(-3, None))
        assertion.eq(data2, dset[-3:], post_process=np.all)
        assertion.len_eq(dset, n + 3)
