"""A module with testing functions for **dataCAT**.

Index
-----
.. currentmodule:: dataCAT.testing_utils
.. autosummary::
    MOL_TUPLE
    MOL
    PDB

API
---
.. autodata:: MOL_TUPLE
    :annotation: : Tuple[Molecule, ...] = ...
.. autodata:: MOL
    :annotation: : Molecule = ...
.. autodata:: PDB
    :annotation: : PDBContainer = ...

"""

from typing import Tuple

from scm.plams import readpdb, Molecule

from .pdb_array import PDBContainer
from .data import PDB_TUPLE

__all__ = ['MOL_TUPLE', 'MOL', 'PDB']

#: A tuple of PLAMS Molecules.
MOL_TUPLE: Tuple[Molecule, ...] = tuple(readpdb(f) for f in PDB_TUPLE)

#: A PLAMS Molecule.
MOL: Molecule = MOL_TUPLE[0]

#: A PDBContainer.
PDB: PDBContainer = PDBContainer.from_molecules(MOL_TUPLE)
