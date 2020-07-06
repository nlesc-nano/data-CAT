from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from scm.plams import Molecule
from CAT.workflows import MOL, HDF5_INDEX

from dataCAT import PDBContainer
from dataCAT.functions import array_to_index, get_nan_row
from dataCAT.property_dset import _resize_prop_dset

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = 'numpy.typing.ArrayLike'


def df_from_hdf5(mol_group: h5py.Group, index: ArrayLike, *prop_dset: h5py.Dataset,
                 mol_list: Optional[Iterable[Molecule]] = None,
                 read_mol: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""Construct a DataFrame.

    Parameters
    ----------
    mol_group : :class:`h5py.Group`
        The group containing the molecules and the index.
    index : array-like
        An array-like object representing an index.
    \*prop_dset : :class:`h5py.Dataset`
        Zero or more datasets with, to be read, quantum mechanical properties.
    mol_list : :class:`Iterable[Molecule]<typing.Iterable>`, optional
        An iterable of PLAMS molecules whose coordinates will be updated in-place.
        Set to :data:`None` to create new molecules from scratch.
    read_mol : :class:`bool`
        If :data:`True`, set the `mol` and `hdf5_index` columns.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame containing molecules, the hdf5 index and all properties from **prop_dset**.
        The DataFrame's index is an intersection of **index** and :code:`mol_group['index']`.

    """
    # Parse the passed index
    dim0_scale = mol_group['index']
    dim0_name = dim0_scale.name.rsplit('/', 1)[-1]
    index_ = np.asarray(index, dtype=dim0_scale.dtype)

    # Find the intersection between the passed index and the scale
    intersect, i, j = np.intersect1d(dim0_scale[:], index_, assume_unique=True, return_indices=True)
    i.sort()

    # Construct a DataFrame
    multi_index = array_to_index(intersect, name=dim0_name)
    columns = pd.Index([MOL, HDF5_INDEX])
    df = pd.DataFrame(index=multi_index, columns=columns, dtype=object)

    # Fill the DataFrame
    if read_mol:
        pdb = PDBContainer.from_hdf5(mol_group, i)
        df[MOL] = pdb.to_molecules(mol=mol_list)
    df[HDF5_INDEX] = i

    # Fill the DataFrame with other optional properties
    _insert_properties(df, prop_dset, i)

    # Append empty rows
    if len(j) == len(index_):
        ret = df
    else:
        ret = _append_rows(df, index_, j)

    df_bool = pd.DataFrame({name: series.astype(bool) for name, series in df.items()})
    df_bool[HDF5_INDEX] = df[HDF5_INDEX]
    return ret, df_bool


def get_bool_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(bool)


def _insert_properties(df: pd.DataFrame, prop_dset: Iterable[h5py.Dataset], i: np.ndarray) -> None:
    """Add columns to **df** for the various properties in **prop_dset**."""
    for dset in prop_dset:
        _resize_prop_dset(dset)
        name = dset.name.rsplit('/', 1)[-1]

        # It's not a 2D Dataset
        if dset.ndim == 1:
            df[(name, '')] = dset[i]
            continue

        # Construct the new columns
        dim1_scale = dset.dims[1][0]
        dim1 = dim1_scale[:].astype(str)
        columns = pd.MultiIndex.from_product([[name], dim1])

        # Update **df**
        data = dset[i].astype(str) if h5py.check_string_dtype(dset.dtype) else dset[i]
        df_tmp = pd.DataFrame(data, index=df.index, columns=columns)
        df[columns] = df_tmp


def _append_rows(df: pd.DataFrame, index: np.ndarray, j: np.ndarray) -> pd.DataFrame:
    """Append **df** with all (previously missing) indices from **index**."""
    # Invert the indices in `j`
    bool_ar = np.ones_like(index, dtype=bool)
    bool_ar[j] = False
    multi_index2 = array_to_index(index[bool_ar], name=df.index.name)

    # Construct the to-be appended dataframe
    data_list = get_nan_row(df)
    df_append = pd.DataFrame(index=multi_index2)
    for k, data in zip(df.columns, data_list):
        df_append[k] = data

    return df.append(df_append)
