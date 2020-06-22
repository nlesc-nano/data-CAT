"""Tests for :class:`dataCAT.PDBContainer`."""

import copy
import pickle

from assertionlib import assertion

from dataCAT import PDBContainer
from dataCAT.testing_utils import PDB


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
