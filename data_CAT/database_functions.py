"""A module for holding functions related to the Database class."""

from os import getcwd
from os.path import (join, isfile, isdir)
from typing import (Optional, Collection, Iterable, Union, Dict, Any, Tuple, TypeVar)

import yaml
import h5py
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING

from scm.plams import (Molecule, Settings)
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit import Chem
from rdkit.Chem import Mol

from .utils import (from_rdmol, get_time, get_template)

__all__ = ['mol_to_file', 'df_to_mongo_dict']


def mol_to_file(mol_list: Iterable[Molecule],
                path: Optional[str] = None,
                overwrite: bool = False,
                mol_format: Collection[str] = ('xyz', 'pdb')) -> None:
    """Export all molecules in **mol_list** to .pdb and/or .xyz files.

    Parameters
    ----------
    mol_list: |list|_ [|plams.Molecule|_]
        An iterable consisting of PLAMS molecules.

    path : str
        Optional: The path to the directory where the molecules will be stored.
        Defaults to the current working directory if ``None``.

    overwrite : bool
        If previously generated files can be overwritten or not.

    mol_format : |list|_ [|str|_]
        A list of strings with the to-be exported file types.
        Accepted values are ``"xyz"`` and/or ``"pdb"``.

    """
    # Set the export path
    path = path or getcwd()
    assert isdir(path)

    if not mol_format:
        return None

    if overwrite:  # Export molecules while allowing for file overriding
        for mol in mol_list:
            mol_path = join(path, mol.properties.name)
            if 'pdb' in mol_format:
                molkit.writepdb(mol, mol_path + '.pdb')
            if 'xyz' in mol_format:
                mol.write(mol_path + '.xyz')

    else:  # Export molecules without allowing for file overriding
        for mol in mol_list:
            mol_path = join(path, mol.properties.name)
            if 'pdb' in mol_format and not isfile(mol_path + '.pdb'):
                molkit.writepdb(mol, mol_path + '.pdb')
            if 'xyz' in mol_format and not isfile(mol_path + '.xyz'):
                mol.write(mol_path + '.xyz')


A = TypeVar('A', str, int, float, frozenset, tuple)  # Immutable objects


def _get_unflattend(input_dict: Dict[Tuple[A], Any]) -> zip:
    """Flatten a dictionary and return a :class:`zip` instance consisting of keys and values.

    Examples
    --------

    .. code:: python

        >>> for key, value in input_dict.items():
        >>>     print('{}: \t{}'.format(str(key), value))
        ('C[O-]', 'O2'):        {('E_solv', 'Acetone'): -56.6, ('E_solv', 'Acetonitrile'): -57.9}
        ('CC[O-]', 'O3'):       {('E_solv', 'Acetone'): -56.5, ('E_solv', 'Acetonitrile'): -57.6}
        ('CCC[O-]', 'O4'):      {('E_solv', 'Acetone'): -57.1, ('E_solv', 'Acetonitrile'): -58.2}

        >>> keys, values = get_unflattend(input_dict)

        >>> print(keys)
        (('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4'))

        >>> print(values)
        ({'E_solv': {'Acetone': -56.6, 'Acetonitrile': -57.9}},
         {'E_solv': {'Acetone': -56.5, 'Acetonitrile': -57.6}},
         {'E_solv': {'Acetone': -57.1, 'Acetonitrile': -58.2}})

    Parameters
    ----------
    input_dict : |dict|_
        A dictionary constructed from a Pandas DataFrame.

    Returns
    -------
    |zip|_
        A :class:`zip` instance that yields a tuple of keys and a tuple of values.

    """
    def _unflatten(input_dict_: Dict[Tuple[A], Any]) -> Dict[A, Dict[A, Any]]:
        """Unflatten a dictionary; dictionary keys are expected to be tuples."""
        ret = Settings()
        for k1, value in input_dict_.items():
            s = ret
            for k2 in k1[:-1]:
                s = s[k2]
            s[k1[-1]] = value
        return ret.as_dict()

    return zip(*[(k, _unflatten(v)) for k, v in input_dict.items()])


def df_to_mongo_dict(df: pd.DataFrame) -> Tuple[dict]:
    """Convert a dataframe into a dictionary suitable for a MongoDB_ database.

    Tuple-keys present in **df** (*i.e.* pd.MultiIndex) are expanded into nested dictionaries.

    .. _MongoDB: https://www.mongodb.com/

    Examples
    --------

    .. code:: python

        >>> print(df)
        index           E_solv
        sub index      Acetone Acetonitrile
        smiles  anchor
        C[O-]   O2       -56.6        -57.9
        CC[O-]  O3       -56.5        -57.6
        CCC[O-] O4       -57.1        -58.2

        >>> output_tuple = df_to_mongo_dict(df)
        >>> for item in output_tuple:
        >>>     print(item)
        {'E_solv': {'Acetone': -56.6, 'Acetonitrile': -57.9}, 'smiles': 'C[O-]', 'anchor': 'O2'}
        {'E_solv': {'Acetone': -56.5, 'Acetonitrile': -57.6}, 'smiles': 'CC[O-]', 'anchor': 'O3'}
        {'E_solv': {'Acetone': -57.1, 'Acetonitrile': -58.2}, 'smiles': 'CCC[O-]', 'anchor': 'O4'}

    Parameters
    ----------
    df : |pd.DataFrame|_
        A Pandas DataFrame whose axis and columns are instance of pd.MultiIndex.

    Returns
    -------
    |tuple|_ [|dict|_]
        A tuple of nested dictionaries construced from **df**.
        Each row in **df** is converted into a single dictionary.
        The to-be returned dictionaries are updated with a dictionary containing their respective
        (multi-)index in **df**.

    """
    if not isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        raise TypeError("df.index and df.columns should be instances of pd.MultiIndex")

    keys, ret = _get_unflattend(df.T.to_dict())
    idx_names = df.index.names

    for item, idx in zip(ret, keys):
        idx_dict = dict(zip(idx_names, idx))
        item.update(idx_dict)

    return ret


def get_nan_row(df: pd.DataFrame) -> list:
    """Return a list of None-esque objects for each column in **df**.

    The object in question depends on the data type of the column.
    Will default to ``None`` if a specific data type is not recognized

        * |np.int64|_: ``-1``
        * |np.float64|_: ``np.nan``
        * |object|_: ``None``
        * |bool|_: ``False``

    Parameters
    ----------
    df : |pd.DataFrame|_
        A dataframe.

    Returns
    -------
    |list|_ [|int|_, |float|_, |bool|_ and/or |None|_]
        A list of none-esque objects, one for each column in **df**.

    """
    dtype_dict = {
        np.dtype('int64'): -1,
        np.dtype('float64'): np.nan,
        np.dtype('O'): None,
        np.dtype('bool'): False
    }

    if not isinstance(df.index, pd.MultiIndex):
        return [dtype_dict[df[i].dtype] for i in df]
    else:
        ret = []
        for _, value in df.items():
            try:
                j = dtype_dict[value.dtype]
            except KeyError:  # dtype is neither int, float nor object
                j = None
            ret.append(j)
        return ret


def as_pdb_array(mol_list: Collection[Molecule],
                 min_size: int = 0) -> np.ndarray:
    """Convert a list of PLAMS molecule into an array of (partially) de-serialized .pdb files.

    Parameters
    ----------
    mol_list: :math:`m` |list|_ [|plams.Molecule|_]
        A list of :math:`m` PLAMS molecules.

    min_size : int
        The minimumum length of the pdb_array.
        The array is padded with empty strings if required.

    Returns
    -------
    :math:`m*n` |np.ndarray|_ [|np.bytes|_ *|S80*]
        An array with :math:`m` partially deserialized .pdb files with up to :math:`n` lines each.

    """
    pdb_list = []
    shape = min_size
    for mol in mol_list:
        pdb_block = Chem.MolToPDBBlock(molkit.to_rdmol(mol)).splitlines()
        pdb_list.append(pdb_block)
        shape_1d = max(shape, len(pdb_block))

    # Construct, fill and return the pdb array
    shape_2d = len(mol_list), shape_1d
    ret = np.zeros(shape_2d, dtype='S80')
    for i, item in enumerate(pdb_list):
        ret[i][:len(item)] = item

    return ret


def from_pdb_array(array: np.ndarray,
                   rdmol: bool = True) -> Union[Molecule, Mol]:
    """Convert an array with a (partially) de-serialized .pdb file into a molecule.

    Parameters
    ----------
    array : :math:`n` |np.ndarray|_ [|np.bytes|_ / S80]
        A (partially) de-serialized .pdb file with :math:`n` lines.

    rdmol : |bool|_
        If ``True``, return an RDKit molecule instead of a PLAMS molecule.

    Returns
    -------
    |plams.Molecule|_ or |rdkit.Chem.Mol|_
        A PLAMS or RDKit molecule build from **array**.

    """
    pdb_str = ''.join([item.decode() + '\n' for item in array if item])
    ret = Chem.MolFromPDBBlock(pdb_str, removeHs=False, proximityBonding=False)
    if not rdmol:
        return molkit.from_rdmol(ret)
    return ret


def sanitize_yaml_settings(settings: Settings,
                           job_type: str) -> Settings:
    """Remove a predetermined set of unwanted keys and values from a settings object.

    Parameters
    ----------
    settings : |plams.Settings|_
        A settings instance with, potentially, undesired keys and values.

    job_type: |str|_
        The name of key in the settings blacklist.

    Returns
    -------
    |plams.Settings|_
        A Settings instance with unwanted keys and values removed.

    """
    def recursive_del(s, s_del):
        for key in s:
            if key in s_del:
                if isinstance(s_del[key], dict):
                    recursive_del(s[key], s_del[key])
                else:
                    del s[key]
            if not s[key]:
                del s[key]

    # Prepare a blacklist of specific keys
    blacklist = get_template('settings_blacklist.yaml')
    settings_del = blacklist['generic']
    settings_del.update(blacklist[job_type])

    # Recursivelly delete all keys from **s** if aforementioned keys are present in the s_del
    recursive_del(settings, settings_del)
    return settings


def even_index(df1: pd.DataFrame,
               df2: pd.DataFrame) -> pd.DataFrame:
    """Ensure that ``df2.index`` is a subset of ``df1.index``.

    Parameters
    ----------
    df1 : |pd.DataFrame|_
        A DataFrame whose index is to-be a superset of ``df2.index``.

    df2 : |pd.DataFrame|_
        A DataFrame whose index is to-be a subset of ``df1.index``.

    Returns
    -------
    |pd.DataFrame|_
        A new dataframe containing all unique elements of ``df1.index`` and ``df2.index``.

    """
    # Figure out if ``df1.index`` is a subset of ``df2.index``
    bool_ar = df2.index.isin(df1.index)
    if bool_ar.all():
        return df1

    # Make ``df1.index`` a subset of ``df2.index``
    nan_row = get_nan_row(df1)
    idx = df2.index[~bool_ar]
    df_tmp = pd.DataFrame(len(idx) * [nan_row], index=idx, columns=df1.columns)
    return df1.append(df_tmp, sort=True)
