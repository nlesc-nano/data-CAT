"""Tests for the :class:`dataCAT.database.Database` class."""

from os.path import join
import h5py
import numpy as np
import pandas as pd

from assertionlib import assertion

from dataCAT.database import Database
from dataCAT.context_managers import OpenLig, OpenQD, OpenYaml

PATH = join('tests', 'test_files', 'database')
DB = Database(PATH)

MOL = ('mol', '')
HDF5_INDEX = ('hdf5 index', '')
OPT = ('opt', '')


def test_init() -> None:
    """Test :meth:`dataCAT.database.Database.__init__`."""
    assertion.eq(DB.dirname, PATH)

    assertion.eq(DB.csv_lig.filename, join(PATH, 'ligand_database.csv'))
    assertion.is_(DB.csv_lig.manager, OpenLig)

    assertion.eq(DB.csv_qd.filename, join(PATH, 'QD_database.csv'))
    assertion.is_(DB.csv_qd.manager, OpenQD)

    assertion.eq(DB.yaml.filename, join(PATH, 'job_settings.yaml'))
    assertion.is_(DB.yaml.manager, OpenYaml)

    assertion.eq(DB.hdf5.filename, join(PATH, 'structures.hdf5'))
    assertion.is_(DB.hdf5.manager, h5py.File)

    assertion.is_(DB.mongodb, None)


def test_str() -> None:
    """Test :meth:`dataCAT.database.Database.__str__`."""
    str_list = str(DB).split('\n')

    assertion.eq(str_list[0], 'Database(')
    assertion.eq(str_list[1], f'    dirname = {repr(PATH)},')
    assertion.eq(str_list[-2], '    mongodb = None')
    assertion.eq(str_list[-1], ')')

    for item in str_list[2:-2]:
        assertion.contains(item, 'MetaManager(filename=')


def test_repr() -> None:
    """Test :meth:`dataCAT.database.Database.__repr__`."""
    str_list = repr(DB).split('\n')

    assertion.eq(str_list[0], 'Database(')
    assertion.eq(str_list[1], f'    dirname = {repr(PATH)},')
    assertion.eq(str_list[-2], '    mongodb = None')
    assertion.eq(str_list[-1], ')')

    for item in str_list[2:-2]:
        assertion.contains(item, 'MetaManager(filename=')


def test_eq() -> None:
    """Test :meth:`dataCAT.database.Database.__eq__`."""
    db2 = Database(PATH)
    assertion.eq(db2, DB)


def test_contains() -> None:
    """Test :meth:`dataCAT.database.Database.__contains__`."""
    assertion.contains(DB, 'mongodb')
    assertion.contains(DB, 'bob', invert=True)


def test_parse_database() -> None:
    """Test :meth:`dataCAT.database.Database._parse_database`."""
    out1 = DB._parse_database('ligand')
    out2 = DB._parse_database('QD')

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
    out2 = DB.from_csv(df, 'ligand', get_mol=True, inplace=False)
    for mol, ref in zip(out2, ref_tup):
        assertion.eq(mol.get_formula(), ref)
