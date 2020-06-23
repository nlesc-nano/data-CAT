"""Tests for :class:`dataCAT.PDBContainer`."""

import copy
import pickle
from pathlib import Path

import h5py
from assertionlib import assertion
from nanoutils import delete_finally

from dataCAT import PDBContainer
from dataCAT.testing_utils import PDB

PATH = Path('tests') / 'test_files'
HDF5_PATH = PATH / 'database' / 'structures.hdf5'
HDF5_FAIL = PATH / 'tmp.hdf5'


def test_pickle() -> None:
    """Test :meth:`PDBContainer.__reduce__`."""
    dumps = pickle.dumps(PDB)
    loads = pickle.loads(dumps)
    assertion.eq(loads, PDB)


def test_eq() -> None:
    """Test :meth:`PDBContainer.__eq__`."""
    pdb1 = PDB[:]
    pdb2 = PDB[0]

    assertion.eq(pdb1, PDB)
    assertion.ne(pdb2, PDB)
    assertion.ne(1, PDB)

    # We're gona cheat here by manually specifying the object's hash
    pdb3 = pdb2[:]
    pdb3._hash = PDB._hash
    assertion.ne(pdb3, PDB)


def test_hash() -> None:
    """Test :meth:`PDBContainer.__hash__`."""
    pdb = PDB[:]
    assertion.eq(hash(pdb), hash(PDB))


def test_copy() -> None:
    """Test :meth:`PDBContainer.__copy__` and :meth:`PDBContainer.__deepcopy__`."""
    pdb1 = copy.copy(PDB)
    pdb2 = copy.deepcopy(PDB)

    assertion.is_(PDB, pdb1)
    assertion.is_(PDB, pdb2)


def test_init() -> None:
    """Test :meth:`PDBContainer.__init__`."""
    atoms = [(True, 1, 'Cad', 'LIG', 'A', 1, -5.231, 0.808, -0.649, 1, 0, 'C', 0, 0)]
    bonds = [(1, 2, 1)]
    assertion.assert_(PDBContainer, atoms, bonds, 1, 1)


@delete_finally(HDF5_FAIL)
def test_validate() -> None:
    """Test :meth:`PDBContainer.validate_hdf5`."""
    attr_set = {'__doc__', 'date_created', 'version_created'}
    with h5py.File(HDF5_PATH, 'r', libver='latest') as f:
        group = f['ligand']
        PDB.validate_hdf5(group)
        assertion.assert_(set.issubset, attr_set, group.attrs.keys())

    with h5py.File(HDF5_FAIL, 'a', libver='latest') as f:
        group = f.create_group('test')

        try:
            PDB.from_hdf5(group)
        except AssertionError as ex:
            assertion.isinstance(ex.__context__, KeyError)
            group.create_dataset('atoms', data=[1])
        else:
            raise AssertionError("Failed to raise an AssertionError")

        try:
            PDB.to_hdf5(group)
        except AssertionError as ex:
            assertion.isinstance(ex.__context__, IndexError)
        else:
            raise AssertionError("Failed to raise an AssertionError")
