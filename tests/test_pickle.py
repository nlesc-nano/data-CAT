"""Test pickle."""

import os
import pickle
from pathlib import Path

import numpy as np
from scm.plams import readpdb
from dataCAT import PDBContainer

PATH = Path('tests') / 'test_files' / 'ligand_pdb'
MOL_LIST = tuple(readpdb(str(PATH / f)) for f in os.listdir(PATH)[:3])
PDB = PDBContainer.from_molecules(MOL_LIST)


def test_pickle() -> None:
    """Test pickle."""
    pdb_bytes = pickle.dumps(PDB)
    pdb_copy = pickle.loads(pdb_bytes)

    assert type(PDB) is type(pdb_copy)
    assert hash(PDB) == hash(pdb_copy)
    for name, ar1 in pdb_copy.items():
        ar2 = getattr(PDB, name)
        np.testing.assert_array_equal(ar1, ar2, err_msg=name)
    assert PDB == pdb_copy


def test_eq() -> None:
    """Test pickle."""
    pdb2 = PDB[:]

    assert type(PDB) is type(pdb2)
    assert hash(PDB) == hash(pdb2)
    for name, ar1 in pdb2.items():
        ar2 = getattr(PDB, name)
        np.testing.assert_array_equal(ar1, ar2, err_msg=name)
    assert PDB == pdb2
