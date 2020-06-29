"""Tests for :mod:`dataCAT.property_dset`."""

import shutil

import h5py
import numpy as np

from nanoutils import delete_finally
from assertionlib import assertion
from dataCAT import update_prop_dset, validate_prop_group, create_prop_group, create_prop_dset
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


@delete_finally(HDF5_TMP)
def test_validate_prop_group() -> None:
    """Test :func:`dataCAT.validate_prop_group`."""
    with h5py.File(HDF5_TMP, 'a') as f:
        scale1 = f.create_dataset('index1', shape=(100,), dtype=int)
        scale1.make_scale('index')

        group = create_prop_group(f, scale1)
        group.create_dataset('test1', shape=(100,), dtype=int)

        try:
            validate_prop_group(group)
        except AssertionError as ex1:
            assertion.contains(str(ex1), 'missing dataset scale')
        else:
            raise AssertionError("Failed to raise an AssertionError")

        del group['test1']
        dset1 = group.create_dataset('test2', shape=(200,), dtype=int)
        dset1.dims[0].label = 'index'
        dset1.dims[0].attach_scale(scale1)

        try:
            validate_prop_group(group)
        except AssertionError as ex2:
            assertion.contains(str(ex2), 'invalid dataset length')
        else:
            raise AssertionError("Failed to raise an AssertionError")

        del group['test2']
        scale2 = f.create_dataset('index2', shape=(100,), dtype=int)
        scale2.make_scale('index')
        dset2 = group.create_dataset('test3', shape=(100,), dtype=int)
        dset2.dims[0].label = 'index'
        dset2.dims[0].attach_scale(scale2)

        try:
            validate_prop_group(group)
        except AssertionError as ex3:
            assertion.contains(str(ex3), 'invalid dataset scale')
        else:
            raise AssertionError("Failed to raise an AssertionError")

        del group['test3']
        create_prop_dset(group, 'test4')
        validate_prop_group(group)
