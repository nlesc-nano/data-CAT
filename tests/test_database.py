"""Tests for the :class:`dataCAT.Database` class."""

import copy
import pickle
from types import MappingProxyType
from os.path import join, abspath
from pathlib import Path

import h5py

from assertionlib import assertion
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
