"""Tests for the :class:`dataCAT.Database` class."""

from os.path import join, abspath
import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion

from CAT.workflows import MOL, HDF5_INDEX, OPT
from dataCAT import Database, OpenLig, OpenQD, OpenYaml

PATH = join('tests', 'test_files', 'database')
DB = Database(PATH)


def test_init() -> None:
    """Test :meth:`dataCAT.database.Database.__init__`."""
    assertion.eq(DB.dirname, abspath(PATH))

    assertion.eq(DB.csv_lig.keywords['filename'], abspath(join(PATH, 'ligand_database.csv')))
    assertion.is_(DB.csv_lig.func, OpenLig)

    assertion.eq(DB.csv_qd.keywords['filename'], abspath(join(PATH, 'qd_database.csv')))
    assertion.is_(DB.csv_qd.func, OpenQD)

    assertion.eq(DB.yaml.keywords['filename'], abspath(join(PATH, 'job_settings.yaml')))
    assertion.is_(DB.yaml.func, OpenYaml)

    assertion.eq(DB.hdf5.args[0], abspath(join(PATH, 'structures.hdf5')))
    assertion.is_(DB.hdf5.func, h5py.File)

    assertion.is_(DB.mongodb, None)


def test_eq() -> None:
    """Test :meth:`dataCAT.database.Database.__eq__`."""
    db2 = Database(PATH)
    assertion.eq(db2, DB)


def test_parse_database() -> None:
    """Test :meth:`dataCAT.database.Database._parse_database`."""
    out1 = DB._parse_database('ligand')
    out2 = DB._parse_database('qd')

    assertion.is_(out1, DB.csv_lig)
    assertion.is_(out2, DB.csv_qd)

    assertion.assert_(DB._parse_database, 'bob', exception=ValueError)


def test_hdf5_availability() -> None:
    """Test :meth:`dataCAT.database.Database._parse_database`."""
    filename = join(PATH, 'structures.hdf5')
    with h5py.File(filename, 'r'):
        assertion.assert_(DB.hdf5_availability, 1.0, 2, exception=OSError)


def test_from_hdf5() -> None:
    """Test :meth:`dataCAT.database.Database.from_hdf5`."""
    ref_tup = ('C3H7O1', 'C2H5O1', 'C1H3O1')

    idx1 = slice(0, None)
    mol_list1 = DB.from_hdf5(idx1, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list1, ref_tup):
        assertion.eq(mol.get_formula(), ref)

    idx2 = np.arange(0, 2)
    mol_list2 = DB.from_hdf5(idx2, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list2, ref_tup[0:2]):
        assertion.eq(mol.get_formula(), ref)


def test_from_csv() -> None:
    """Test :meth:`dataCAT.database.Database.from_csv`."""
    columns = pd.MultiIndex.from_tuples(
        [('mol', ''), ('hdf5 index', ''), ('opt', '')]
    )
    idx = pd.MultiIndex.from_tuples(
        [('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4')]
    )
    df = pd.DataFrame(np.nan, index=idx, columns=columns)

    df[MOL] = None
    df[HDF5_INDEX] = np.arange(0, 3, dtype=int)
    df[OPT] = True

    out1 = DB.from_csv(df, 'ligand', get_mol=False)
    assertion.is_(out1, None)

    ref_tup = ('C3H7O1', 'C2H5O1', 'C1H3O1')
    out2: pd.Series = DB.from_csv(df, 'ligand', get_mol=True, inplace=False)
    for mol, ref in zip(out2, ref_tup):
        assertion.eq(mol.get_formula(), ref)
