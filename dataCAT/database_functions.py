"""A module for holding functions related to the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.database_functions
.. autosummary::
    mol_to_file
    _get_unflattend
    df_to_mongo_dict
    get_nan_row
    as_pdb_array
    from_pdb_array
    sanitize_yaml_settings
    even_index

API
---
.. autofunction:: mol_to_file
.. autofunction:: _get_unflattend
.. autofunction:: df_to_mongo_dict
.. autofunction:: get_nan_row
.. autofunction:: as_pdb_array
.. autofunction:: from_pdb_array
.. autofunction:: sanitize_yaml_settings
.. autofunction:: even_index

"""

from typing import (
    Collection, Union, Sequence, Tuple, List, Generator, Dict, Any, Hashable
)

import numpy as np
import pandas as pd

from scm.plams import Molecule, Settings
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit import Chem
from rdkit.Chem import Mol

from CAT.utils import get_template

__all__ = ['df_to_mongo_dict']


def df_to_mongo_dict(df: pd.DataFrame,
                     as_gen: bool = True) -> Union[Generator, list]:
    """Convert a dataframe into a generator of dictionaries suitable for a MongoDB_ databases.

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

        >>> gen = df_to_mongo_dict(df)
        >>> print(type(gen))
        <class 'generator'>

        >>> for item in gen:
        >>>     print(item)
        {'E_solv': {'Acetone': -56.6, 'Acetonitrile': -57.9}, 'smiles': 'C[O-]', 'anchor': 'O2'}
        {'E_solv': {'Acetone': -56.5, 'Acetonitrile': -57.6}, 'smiles': 'CC[O-]', 'anchor': 'O3'}
        {'E_solv': {'Acetone': -57.1, 'Acetonitrile': -58.2}, 'smiles': 'CCC[O-]', 'anchor': 'O4'}

    Parameters
    ----------
    df : |pd.DataFrame|_
        A Pandas DataFrame whose axis and columns are instance of pd.MultiIndex.

    as_gen : bool
        If ``True``, return a generator of dictionaries rather than a list of dictionaries.

    Returns
    -------
    |Generator|_ [|dict|_] or |list|_ [|dict|_]
        A generator or list of nested dictionaries construced from **df**.
        Each row in **df** is converted into a single dictionary.
        The to-be returned dictionaries are updated with a dictionary containing their respective
        (multi-)index in **df**.

    """
    def _get_dict(idx: Sequence[Hashable],
                  row: pd.Series,
                  idx_names: Sequence[Hashable]) -> dict:
        ret = {i: row[i].to_dict() for i in row.index.levels[0]}  # Add values
        ret.update(dict(zip(idx_names, idx)))  # Add index
        return ret

    if not (isinstance(df.index, pd.MultiIndex) and isinstance(df.columns, pd.MultiIndex)):
        raise TypeError("DataFrame.index and DataFrame.columns should be "
                        "instances of pandas.MultiIndex")

    idx_names = df.index.names
    if as_gen:
        return (_get_dict(idx, row, idx_names) for idx, row in df.iterrows())
    return [_get_dict(idx, row, idx_names) for idx, row in df.iterrows()]


#: A dictionary with NumPy dtypes as keys and matching ``None``-esque items as values
DTYPE_DICT: Dict[np.dtype, Any] = {
    np.dtype('int'): -1,
    np.dtype('float'): np.nan,
    np.dtype('O'): None,
    np.dtype('bool'): False
}


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
    return [(DTYPE_DICT[v.dtype] if v.dtype in DTYPE_DICT else None) for _, v in df.items()]


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
    def _get_value(mol: Molecule) -> Tuple[List[str], int]:
        """Return a partially deserialized .pdb file and the length of aforementioned file."""
        ret = Chem.MolToPDBBlock(molkit.to_rdmol(mol)).splitlines()
        return ret, len(ret)

    pdb_list, shape_list = zip(*[_get_value(mol) for mol in mol_list])

    # Construct, fill and return the pdb array
    shape = len(mol_list), max(min_size, max(shape_list))
    ret = np.zeros(shape, dtype='S80')
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
        A new Settings instance with all unwanted keys and values removed.

    Raises
    ------
    KeyError
        Raised if **jobtype** is not found in .../CAT/data/templates/settings_blacklist.yaml.

    """
    # Prepare a blacklist of specific keys
    blacklist = get_template('settings_blacklist.yaml')
    if job_type not in blacklist:
        return settings.copy()

    settings_del = blacklist['generic']
    settings_del.update(blacklist[job_type])

    # Recursivelly delete all keys from **s** if aforementioned keys are present in the s_del
    ret = settings.copy()
    del_nested(settings, ret, settings_del)
    return ret


def del_nested(s_ref: Settings, s_ret: dict, del_item: dict) -> None:
    """Remove all keys in **del_item** from **collection**: a (nested) dictionary and/or list."""
    empty = Settings()
    iterator = s_ref.items() if isinstance(s_ref, dict) else enumerate(s_ref)

    for key, value in iterator:
        if key in del_item:
            value_del = del_item[key]
            if isinstance(value_del, (dict, list)):
                del_nested(value, s_ret[key], value_del)
            else:
                del s_ret[key]

        if value == empty:  # An empty (leftover) Settings instance: delete it
            del s_ret[key]


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
        A new (sorted) dataframe containing all unique elements of ``df1.index`` and ``df2.index``.
        Returns **df1** if ``df2.index`` is already a subset of ``df1.index``

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
