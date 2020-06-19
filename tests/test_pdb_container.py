"""Tests for :class:`dataCAT.PDBContainer`."""

import os
import pickle
from pathlib import Path

from assertionlib import assertion
from scm.plams import readpdb
from dataCAT import PDBContainer

PATH = Path('tests') / 'test_files' / 'ligand_pdb'
MOL_LIST = tuple(readpdb(str(PATH / f)) for f in os.listdir(PATH))
PDB = PDBContainer.from_molecules(MOL_LIST)


def test_pickle() -> None:
    """Test :meth:`dataCAT.PDBContainer.__reduce__`."""
    dumps = pickle.dumps(PDB)
    loads = pickle.loads(dumps)
    assertion.eq(loads, PDB)


def test_eq() -> None:
    """Test :meth:`dataCAT.PDBContainer.__eq__`."""
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
    """Test :meth:`dataCAT.PDBContainer.__hash__`."""
    pdb = PDB[:]
    assertion.eq(hash(pdb), hash(PDB))
