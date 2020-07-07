"""Tests for :mod:`dataCAT.functions`."""

from os.path import join

import numpy as np
import pandas as pd

from rdkit import Chem
from assertionlib import assertion
import scm.plams.interfaces.molecule.rdkit as molkit

from dataCAT.functions import as_pdb_array, from_pdb_array

PATH = join('tests', 'test_files')


def test_as_pdb_array() -> None:
    """Test :func:`dataCAT.functions.as_pdb_array`."""
    mol_list = [molkit.readpdb(join(PATH, 'Methanol.pdb'))]

    out1 = as_pdb_array(mol_list, warn=False)
    ref1 = np.array(
        [['HETATM    1  C1  UNL     1       0.345  -0.116   0.000  1.00  0.00           C  ',
          'HETATM    2  O1  UNL     1      -1.082  -0.219   0.000  1.00  0.00           O  ',
          'HETATM    3  H1  UNL     1       0.733  -1.141   0.000  1.00  0.00           H  ',
          'HETATM    4  H2  UNL     1       0.726   0.399   0.897  1.00  0.00           H  ',
          'HETATM    5  H3  UNL     1       0.726   0.399  -0.897  1.00  0.00           H  ',
          'HETATM    6  H4  UNL     1      -1.448   0.678   0.000  1.00  0.00           H  ',
          'CONECT    1    2    3    4    5', 'CONECT    2    6', 'END']], dtype='|S80'
    )
    np.testing.assert_array_equal(out1, ref1)

    out2 = as_pdb_array(mol_list, min_size=20, warn=False)
    ref2 = np.array(
        [['HETATM    1  C1  UNL     1       0.345  -0.116   0.000  1.00  0.00           C  ',
          'HETATM    2  O1  UNL     1      -1.082  -0.219   0.000  1.00  0.00           O  ',
          'HETATM    3  H1  UNL     1       0.733  -1.141   0.000  1.00  0.00           H  ',
          'HETATM    4  H2  UNL     1       0.726   0.399   0.897  1.00  0.00           H  ',
          'HETATM    5  H3  UNL     1       0.726   0.399  -0.897  1.00  0.00           H  ',
          'HETATM    6  H4  UNL     1      -1.448   0.678   0.000  1.00  0.00           H  ',
          'CONECT    1    2    3    4    5', 'CONECT    2    6', 'END',
          '', '', '', '', '', '', '', '', '', '', '']], dtype='|S80'
    )
    np.testing.assert_array_equal(out2, ref2)


def test_from_pdb_array() -> None:
    """Test :func:`dataCAT.functions.as_pdb_array`."""
    mol = molkit.readpdb(join(PATH, 'Methanol.pdb'))

    pdb_ar = as_pdb_array([mol], warn=False)[0]
    out1 = from_pdb_array(pdb_ar, warn=False)
    assertion.isinstance(out1, Chem.Mol)

    out2 = from_pdb_array(pdb_ar, rdmol=False, warn=False)
    for at1, at2 in zip(out2, mol):
        assertion.eq(at1.coords, at2.coords)
        assertion.eq(at1.atnum, at2.atnum)
