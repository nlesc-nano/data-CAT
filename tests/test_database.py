"""Tests for the :class:`dataCAT.Database` class."""

import copy
import shutil
import pickle
import warnings
from types import MappingProxyType
from os.path import join, abspath
from pathlib import Path

import h5py
import numpy as np

from scm.plams import readpdb
from nanoutils import delete_finally
from assertionlib import assertion
from CAT.workflows import MOL, HDF5_INDEX, OPT, JOB_SETTINGS_CDFT

from dataCAT import Database

PATH = Path('tests') / 'test_files'
DB_PATH = PATH / 'database'
DB_PATH_UPDATE = PATH / 'database_update'

DB = Database(DB_PATH)


def test_init() -> None:
    """Test :meth:`dataCAT.Database.__init__`."""
    assertion.eq(DB.dirname, abspath(DB_PATH))

    assertion.eq(DB.hdf5.args[0], abspath(join(DB_PATH, 'structures.hdf5')))
    assertion.is_(DB.hdf5.func, h5py.File)

    assertion.isinstance(DB.mongodb, (type(None), MappingProxyType))


def test_eq() -> None:
    """Test :meth:`dataCAT.Database.__eq__`."""
    db2 = Database(DB_PATH)
    assertion.eq(db2, DB)
    assertion.eq(hash(db2), hash(DB))

    assertion.is_(DB, copy.copy(DB))
    assertion.is_(DB, copy.deepcopy(DB))

    dump = pickle.dumps(DB)
    db3 = pickle.loads(dump)
    assertion.eq(db3, DB)

    db_str = repr(DB)
    assertion.contains(db_str, DB.__class__.__name__)
    for name in ('dirname', 'hdf5'):
        assertion.contains(db_str, str(getattr(DB, name)), message=name)


def test_hdf5_availability() -> None:
    """Test :meth:`dataCAT.Database.hdf5_availability`."""
    with DB.hdf5('r'):
        assertion.assert_(DB.hdf5_availability, 1.0, 2, exception=OSError)


def test_from_hdf5() -> None:
    """Test :meth:`dataCAT.Database.from_hdf5`."""
    ref_tup = ('C1H3O1', 'C2H5O1', 'C3H7O1')

    idx1 = slice(0, None)
    mol_list1 = DB.from_hdf5(idx1, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list1, ref_tup):
        assertion.eq(mol.get_formula(), ref)

    idx2 = np.arange(0, 2)
    mol_list2 = DB.from_hdf5(idx2, 'ligand', rdmol=False)
    for mol, ref in zip(mol_list2, ref_tup[0:2]):
        assertion.eq(mol.get_formula(), ref)


@delete_finally(DB_PATH_UPDATE)
def test_update_hdf5() -> None:
    """Test :meth:`~dataCAT.Database.update_hdf5`."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)

    mol_dict = {
        ('CC(=O)[O-]', 'O4'): readpdb(str(PATH / 'CC[=O][O-]@O4.pdb')),
        ('CCC(=O)[O-]', 'O5'): readpdb(str(PATH / 'CCC[=O][O-]@O5.pdb')),
    }
    df = pd.DataFrame({MOL: mol_dict})
    df[HDF5_INDEX] = -1
    df[OPT] = False

    series1 = db.update_hdf5(df, database='ligand')
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series1, [3, 4], post_process=np.all)

    series2 = db.update_hdf5(df, database='ligand')
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series2, [], post_process=np.all)

    series3 = db.update_hdf5(df, database='ligand', overwrite=True)
    assertion.eq(df[HDF5_INDEX], [3, 4], post_process=np.all)
    assertion.eq(series3, [], post_process=np.all)


@delete_finally(DB_PATH_UPDATE)
def test_update_hdf5_settinga() -> None:
    """Test :meth:`~dataCAT.Database._update_hdf5_settings`."""
    shutil.copytree(DB_PATH, DB_PATH_UPDATE)
    db = Database(DB_PATH_UPDATE)
    job_settings = [
        PATH / 'CDFT' / 'CDFT' / 'CDFT.in',
        PATH / 'CDFT' / 'CDFT.002' / 'CDFT.002.in',
        PATH / 'CDFT' / 'CDFT.003' / 'CDFT.003.in',
        PATH / 'CDFT' / 'CDFT.004' / 'CDFT.004.in'
    ]

    mol_dict = {
        ('CC(=O)[O-]', 'O4'): readpdb(str(PATH / 'CC[=O][O-]@O4.pdb'))
    }
    df = pd.DataFrame({MOL: mol_dict})
    df[HDF5_INDEX] = [3]
    df[JOB_SETTINGS_CDFT] = [job_settings]

    db._update_hdf5_settings(df, 'job_settings_cdft')
    with db.hdf5('r') as f:
        job_ar = f['job_settings_cdft'][-1].astype(str)

    for file, ar in zip(job_settings, job_ar):
        with open(file, 'r') as f:
            try:
                iterator = (i.rstrip('\n') for i in f)
                for i, j in zip(iterator, ar):
                    while not i:
                        i = next(iterator)
                    assertion.eq(i, j)
            except StopIteration:
                pass


def test_update_mongodb() -> None:
    """Test :meth:`~dataCAT.Database.update_mongodb`."""
    if DB.mongodb is None:
        warnings.warn("MongoDB server not found; skipping test", category=RuntimeWarning)
        return

    DB.update_mongodb('ligand')
