"""A module for creating database files for the :class:`.Database` class.

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

from os import PathLike
from os.path import join, isfile
from types import MappingProxyType
from typing import Dict, Any, List, Union, AnyStr, Mapping, overload

import yaml
import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

from nanoutils import Literal, PathType
from CAT.logger import logger

from .dtype import BACKUP_IDX_DTYPE, LIG_IDX_DTYPE, QD_IDX_DTYPE
from .hdf5_log import create_hdf5_log
from .pdb_array import PDBContainer
from .functions import from_pdb_array, _set_index

__all__: List[str] = []

Ligand = Literal['ligand', 'ligand_no_opt']
QD = Literal['qd', 'qd_no_opt']


def _create_csv(path: Union[str, PathLike], database: Union[Ligand, QD] = 'ligand') -> str:
    """Create a ligand or qd database (csv format) if it does not yet exist.

    Parameters
    ----------
    path : str
        The path (without filename) of the database.

    database : |str|_
        The type of database, accepted values are ``"ligand"`` and ``"qd"``.

    Returns
    -------
    |str|_
        The absolute path to the ligand or qd database.

    """
    filename = join(path, f'{database}_database.csv')

    # Check if the database exists and has the proper keys; create it if it does not
    if not isfile(filename):
        if database == 'ligand':
            _create_csv_lig(filename)
        elif database == 'qd':
            _create_csv_qd(filename)
        else:
            raise ValueError(f"{database!r} is not an accepated value for the 'database' argument")
        logger.info(f'{database}_database.csv not found in {path}, creating {database} database')
    return filename


def _create_csv_lig(filename: PathType) -> None:
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


def _create_csv_qd(filename: PathType) -> None:
    """Create a qd database and and return its absolute path.

    Parameters
    ----------
    path : str
        The path+filename of the qd database.

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


IDX_DTYPE: Mapping[str, np.dtype] = MappingProxyType({
    'core': BACKUP_IDX_DTYPE,
    'core_no_opt': BACKUP_IDX_DTYPE,
    'ligand': LIG_IDX_DTYPE,
    'ligand_no_opt': LIG_IDX_DTYPE,
    'qd': QD_IDX_DTYPE,
    'qd_no_opt': QD_IDX_DTYPE
})


@overload
def _create_hdf5(path: Union[AnyStr, 'PathLike[AnyStr]']) -> AnyStr:
    ...
@overload  # noqa: E302
def _create_hdf5(path: Union[AnyStr, 'PathLike[AnyStr]'], name: AnyStr) -> AnyStr:
    ...
def _create_hdf5(path, name='structures.hdf5'):  # noqa: E302
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
    path = join(path, name)

    # Define arguments for 2D datasets
    dataset_names = ('core', 'core_no_opt', 'ligand', 'ligand_no_opt', 'qd', 'qd_no_opt')
    kwargs = {'chunks': True, 'compression': 'gzip'}

    # Define arguments for 3D datasets
    kwargs_3d = {'chunks': True, 'maxshape': (None, None, None), 'compression': 'gzip'}
    dataset_names_3d = (
        'job_settings_crs', 'job_settings_qd_opt', 'job_settings_BDE', 'job_settings_ASA',
        'job_settings_cdft'
    )

    with h5py.File(path, 'a', libver='latest') as f:
        for grp_name in dataset_names:
            # Check for pre-dataCAT-0.3 style databases
            if isinstance(f.get(grp_name), h5py.Dataset):
                logger.info(f'Updating h5py Dataset to data-CAT >= 0.4 style: {grp_name!r}')
                iterator = (from_pdb_array(pdb, rdmol=False, warn=False) for pdb in f[grp_name])
                pdb = PDBContainer.from_molecules(iterator)
                del f[grp_name]
            else:
                pdb = None

            # Create a new group if it does not exist yet
            if grp_name not in f:
                group = PDBContainer.create_hdf5_group(f, grp_name, **kwargs)
            else:
                group = f[grp_name]

            # Check for pre-dataCAT-0.4 style databases
            if 'index' not in group:
                if pdb is None:
                    logger.info(f'Updating h5py Dataset to data-CAT >= 0.4 style: {grp_name!r}')
                i = len(group['atoms'])
                dtype = IDX_DTYPE[grp_name]
                _set_index(PDBContainer, group, dtype, i)

            # Check of the log is present
            if 'logger' not in group:
                create_hdf5_log(group)

            if pdb is not None:
                pdb.to_hdf5(group, mode='append')

        # Create new 3D datasets
        iterator_3d = (grp_name for grp_name in dataset_names_3d if grp_name not in f)
        for grp_name in iterator_3d:
            f.create_dataset(grp_name, data=np.empty((0, 1, 1), dtype='S120'), **kwargs_3d)

    return path


@overload
def _create_yaml(path: Union[AnyStr, 'PathLike[AnyStr]']) -> AnyStr:
    ...
@overload  # noqa: E302
def _create_yaml(path: Union[AnyStr, 'PathLike[AnyStr]'], name: AnyStr) -> AnyStr:
    ...
def _create_yaml(path, name='job_settings.yaml'):  # noqa: E302
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
    filename = join(path, name)

    # Create a new .yaml file if it does not yet exist
    if not isfile(filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump({None: [None]}, default_flow_style=False, indent=4))
    return filename


def _create_mongodb(host: str = 'localhost', port: int = 27017,
                    **kwargs: Any) -> Dict[str, Any]:
    """Create the the MongoDB collections and set their index.

    Paramaters
    ----------
    host : :class:`str`
        Hostname or IP address or Unix domain socket path of a single mongod or
        mongos instance to connect to, or a mongodb URI, or a list of hostnames mongodb URIs.
        If **host** is an IPv6 literal it must be enclosed in ``"["`` and ``"]"`` characters
        following the RFC2732 URL syntax (e.g. ``"[::1]"`` for localhost).
        Multihomed and round robin DNS addresses are not supported.

    port : :class:`int`
        port number on which to connect.

    kwargs : :data:`Any<typing.Any>`
        Optional keyword argument for :class:`pymongo.MongoClient`.

    Returns
    -------
    :class:`Dict[str, Any]<typing.Dict>`
        A dictionary with all supplied keyword arguments.

    Raises
    ------
    :exc:`pymongo.ServerSelectionTimeoutError`
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
