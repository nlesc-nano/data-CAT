from typing import TYPE_CHECKING, Iterable, Optional

import h5py
import numpy as np
import pandas as pd

from scm.plams import Molecule
from CAT.workflows import MOL, HDF5_INDEX, OPT

from dataCAT import PDBContainer
from dataCAT.functions import array_to_index
from dataCAT.property_dset import _resize_prop_dset

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
else:
    ArrayLike = 'numpy.typing.ArrayLike'


def df_from_hdf5(mol_group: h5py.Group, index: ArrayLike, *prop_dset: h5py.Dataset,
                 mol_list: Optional[Iterable[Molecule]] = None,
                 read_mol: bool = True) -> pd.DataFrame:
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
    mol_array = np.array(mol_list, dtype=object) if mol_list is not None else None

    # Parse the passed index
    dim0_scale = mol_group['index']
    dim0_name = dim0_scale.name.rsplit('/', 1)[-1]
    dtype = dim0_scale.dtype
    index_ = np.asarray(index, dtype=dtype)

    # Find the intersection between the passed index and the scale
    _dim0 = np.asarray([dim0_scale[i] for i in dtype.fields.keys()])
    dim0 = np.rec.fromarrays(_dim0.astype(str), dtype=dtype)
    intersect, i, j = np.intersect1d(dim0, index_, assume_unique=True, return_indices=True)
    i.sort()

    # Construct a DataFrame
    multi_index = array_to_index(intersect, name=dim0_name)
    columns = pd.Index([MOL, HDF5_INDEX, OPT]) if read_mol else pd.MultiIndex([(), ()], [(), ()])
    df = pd.DataFrame(index=multi_index, columns=columns, dtype=object)

    # Fill the DataFrame
    if read_mol:
        _set_min_size(mol_group, *PDBContainer.DSET_NAMES.values())
        pdb = PDBContainer.from_hdf5(mol_group, i)
        df[MOL] = pdb.to_molecules(mol=mol_array)
        df[HDF5_INDEX] = i
        df[OPT] = False
        df.loc[pdb.atoms.view(bool).any(axis=1), OPT] = True

        if not df[OPT].all():
            mol_group2 = mol_group.file[mol_group.name.rstrip('/') + '_no_opt']
            j = i[~df[OPT].values]
            pdb2 = PDBContainer.from_hdf5(mol_group2, j)
            if len(df) == 1:
                df.at[df.index[0], MOL] = pdb2.to_molecules(mol=mol_array[j])[0]
            else:
                df.loc[~df[OPT], MOL] = pdb2.to_molecules(mol=mol_array[j])

    # Fill the DataFrame with other optional properties
    _insert_properties(df, prop_dset, i)

    # Append empty rows
    if len(j) == len(index_):
        return df
    else:
        return _append_rows(df, index_, j, len(dim0_scale))


def _set_min_size(group: h5py.Group, *names: str) -> None:
    scale = group['index']
    i = len(scale)
    for n in names:
        dset = group[n]
        if len(dset) < i:
            dset.resize(i, axis=0)


def _insert_properties(df: pd.DataFrame, prop_dset: Iterable[h5py.Dataset], i: np.ndarray) -> None:
    """Add columns to **df** for the various properties in **prop_dset**."""
    for dset in prop_dset:
        _resize_prop_dset(dset)
        name = dset.name.rsplit('/', 1)[-1]

        data = dset[i].astype(str) if h5py.check_string_dtype(dset.dtype) else dset[i]

        # It's not a 2D Dataset
        if dset.ndim == 1:
            df[(name, '')] = data
            continue

        # Construct the new columns
        dim1_scale = dset.dims[1][0]
        dim1 = dim1_scale[:].astype(str)
        columns = pd.MultiIndex.from_product([[name], dim1])

        # Update **df**
        df_tmp = pd.DataFrame(data, index=df.index, columns=columns)
        df[columns] = df_tmp


def _append_rows(df: pd.DataFrame, index: np.ndarray, j: np.ndarray,
                 idx_start: int = 0) -> pd.DataFrame:
    """Append **df** with all (previously missing) indices from **index**."""
    # Invert the indices in `j`
    bool_ar = np.ones_like(index, dtype=bool)
    bool_ar[j] = False
    multi_index2 = array_to_index(index[bool_ar], name=df.index.name)

    # Construct the to-be appended dataframe
    df_append = pd.DataFrame(index=multi_index2)
    for name, series in df.items():
        df_append[name] = np.zeros(1, dtype=series.dtype).take(0)

    if HDF5_INDEX in df_append:
        df_append[HDF5_INDEX] = np.arange(idx_start, idx_start + len(df_append))
    return df.append(df_append)
