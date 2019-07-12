"""
dataCAT.create_database
=======================

A module for creating database files for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.create_database
.. autosummary::
    _create_csv
    _create_csv_lig
    _create_csv_qd
    _create_hdf5
    _create_yaml
    _create_mongodb

API
---
.. autofunction:: _create_csv
.. autofunction:: _create_csv_lig
.. autofunction:: _create_csv_qd
.. autofunction:: _create_hdf5
.. autofunction:: _create_yaml
.. autofunction:: _create_mongodb

"""

from os.path import (join, isfile)
from typing import (Dict, Any, List)

import yaml
import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

from CAT.utils import get_time

__all__: List[str] = []


def _create_csv(path: str,
                database: str = 'ligand') -> str:
    """Create a ligand or QD database (csv format) if it does not yet exist.

    Parameters
    ----------
    path : str
        The path (without filename) of the database.

    database : |str|_
        The type of database, accepted values are ``"ligand"`` and ``"qd"``.

    Returns
    -------
    |str|_
        The absolute path to the ligand or QD database.

    """
    path = join(path, database + '_database.csv')

    # Check if the database exists and has the proper keys; create it if it does not
    if not isfile(path):
        msg = get_time() + '{}_database.csv not found in {}, creating {} database'
        print(msg.format(database, path, database))

        if database == 'ligand':
            _create_csv_lig(path)
        elif database == 'QD':
            _create_csv_qd(path)
        else:
            err = "'{}' is not an accepated value for the 'database' argument"
            raise ValueError(err.format(database))
    return path


def _create_csv_lig(filename: str) -> None:
    """Create a ligand database and and return its absolute path.

    Parameters
    ----------
    path : str
        The path+filename of the ligand database.

    """
    idx = pd.MultiIndex.from_tuples([('-', '-')], names=['smiles', 'anchor'])

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('formula', ''), ('opt', ''), ('settings', 1)],
        names=['index', 'sub index']
    )

    df = pd.DataFrame(None, index=idx, columns=columns)
    df['hdf5 index'] = -1
    df['formula'] = 'str'
    df['settings'] = 'str'
    df['opt'] = False
    df.to_csv(filename)


def _create_csv_qd(filename: str) -> None:
    """Create a QD database and and return its absolute path.

    Parameters
    ----------
    path : str
        The path+filename of the QD database.

    """
    idx = pd.MultiIndex.from_tuples(
        [('-', '-', '-', '-')],
        names=['core', 'core anchor', 'ligand smiles', 'ligand anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('ligand count', ''), ('opt', ''), ('settings', 1), ('settings', 2)],
        names=['index', 'sub index']
    )

    df = pd.DataFrame(None, index=idx, columns=columns)
    df['hdf5 index'] = -1
    df['ligand count'] = -1
    df['settings'] = 'str'
    df['opt'] = False
    df.to_csv(filename)


def _create_hdf5(path: str,
                 name: str = 'structures.hdf5') -> str:
    """Create the .pdb structure database (hdf5 format).

    Parameters
    ----------
    path : str
        The path (without filename) to the database.

    name : str
        The filename of the database (excluding its path).

    Returns
    -------
    |str|_
        The absolute path+filename to the pdb structure database.

    """
    # Define arguments for 2D datasets
    path = join(path, name)
    dataset_names = ('core', 'core_no_opt', 'ligand', 'ligand_no_opt', 'QD', 'QD_no_opt', )
    kwarg = {'chunks': True, 'maxshape': (None, None), 'compression': 'gzip'}

    # Create new 2D datasets
    with h5py.File(path, 'a') as f:
        for name in dataset_names:
            if name not in f:
                f.create_dataset(name=name, data=np.empty((0, 1), dtype='S80'), **kwarg)

    # Define arguments for 3D datasets
    dataset_names_3d = ('job_settings_crs', 'job_settings_QD_opt', 'job_settings_BDE')
    kwarg_3d = {'chunks': True, 'maxshape': (None, None, None), 'compression': 'gzip'}

    # Create new 3D datasets
    with h5py.File(path, 'a') as f:
        for name in dataset_names_3d:
            if name not in f:
                f.create_dataset(name=name, data=np.empty((0, 1, 1), dtype='S120'), **kwarg_3d)

    return path


def _create_yaml(path: str,
                 name: str = 'job_settings.yaml') -> str:
    """Create a job settings database (yaml format).

    Parameters
    ----------
    path : str
        The path (without filenameto the database.

    name : str
        The filename of the database (excluding its path).

    Returns
    -------
    |str|_
        The absolute path+filename to the pdb structure database.

    """
    # Define arguments
    path = join(path, name)

    # Create a new .yaml file if it does not yet exist
    if not isfile(path):
        with open(path, 'w') as f:
            f.write(yaml.dump({None: [None]}, default_flow_style=False, indent=4))
    return path


def _create_mongodb(host: str = 'localhost',
                    port: int = 27017,
                    **kwargs: Dict[str, Any]) -> dict:
    """Create the the MongoDB collections and set their index.

    Paramaters
    ----------
    host : |str|_
        Hostname or IP address or Unix domain socket path of a single mongod or
        mongos instance to connect to, or a mongodb URI, or a list of hostnames mongodb URIs.
        If **host** is an IPv6 literal it must be enclosed in ``"["`` and ``"]"`` characters
        following the RFC2732 URL syntax (e.g. ``"[::1]"`` for localhost).
        Multihomed and round robin DNS addresses are not supported.

    port : |str|_
        port number on which to connect.

    kwargs : |dict|_
        Optional keyword argument for `pymongo.MongoClient <http://api.mongodb.com/python/current/api/pymongo/mongo_client.html>`_.

    Returns
    -------
    |dict|_
        A dictionary with all supplied keyword arguments.

    Raises
    ------
    ServerSelectionTimeoutError
        Raised if no connection can be established with the host.

    """  # noqa
    # Open the client
    client = MongoClient(host, port, serverSelectionTimeoutMS=1000, **kwargs)
    client.server_info()  # Raises an ServerSelectionTimeoutError error if the server is inaccesible

    # Open the database
    db = client.cat_database

    # Open and set the index of the ligand collection
    lig_collection = db.ligand_database
    lig_key = 'smiles_1_anchor_1'
    if lig_key not in lig_collection.index_information():
        lig_collection.create_index([
            ('smiles', ASCENDING),
            ('anchor', ASCENDING)
        ], unique=True)

    # Open and set the index of the QD collection
    qd_collection = db.QD_database
    qd_key = 'core_1_core anchor_1_ligand smiles_1_ligand anchor_1'
    if qd_key not in qd_collection.index_information():
        qd_collection.create_index([
            ('core', ASCENDING),
            ('core anchor', ASCENDING),
            ('ligand smiles', ASCENDING),
            ('ligand anchor', ASCENDING)
        ], unique=True)

    # Return all provided keyword argument
    ret = {'host': host, 'port': port}
    ret.update(kwargs)
    return ret
