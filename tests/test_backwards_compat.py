"""Test the backwards compatbility of data-CAT ``>= 0.3`` with respect to ``0.2.3``."""

import shutil
from pathlib import Path

import numpy as np

from scm.plams import readpdb
from nanoutils import delete_finally
from dataCAT import Database, PDBContainer

PATH = Path('tests') / 'test_files'
DB_PATH_OLD = PATH / 'database_pre_0_3'
DB_PATH_NEW = PATH / '_database_pre_0_3'


@delete_finally(DB_PATH_NEW)
def test_create_database() -> None:
    """Test the backwards compatiblity of pre-``0.3`` databases."""
    shutil.copytree(DB_PATH_OLD, DB_PATH_NEW)
    db = Database(DB_PATH_NEW)

    with db.hdf5('r', libver='latest') as f:
        pdb = PDBContainer.from_hdf5(f['ligand'])
        mol_list = pdb.to_molecules()

    mol_list_ref = []
    files = ('C3H7O1.pdb', 'C2H5O1.pdb', 'C1H3O1.pdb')
    for _path in files:
        path = str(DB_PATH_NEW / _path)
        mol_list_ref.append(readpdb(path))

    # Compare molecules
    for mol1, mol2 in zip(mol_list, mol_list_ref):
        msg = f'{mol1.get_formula()} & {mol2.get_formula()}'
        np.testing.assert_allclose(mol1, mol2, rtol=0, atol=10**-2, err_msg=msg)
        np.testing.assert_array_equal(mol1.bond_matrix(), mol2.bond_matrix(), err_msg=msg)
