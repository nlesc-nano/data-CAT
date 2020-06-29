"""A module for storing quantum mechanical properties in hdf5 format.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    create_prop_group
    create_prop_dset
    update_prop_dset

API
---
.. autofunction:: create_prop_group
.. autofunction:: create_prop_dset
.. autofunction:: update_prop_dset

"""

from typing import Union, Sequence, Any, Optional, TYPE_CHECKING

import h5py
import numpy as np

from assertionlib import assertion

if TYPE_CHECKING:
    from numpy.typing import DtypeLike
else:
    DtypeLike = 'numpy.typing.DtypeLike'

__all__ = ['create_prop_group', 'create_prop_dset', 'update_prop_dset']

PROPERTY_DOC = r"""A h5py Group containing an arbitrary number of quantum-mechanical properties.

Attributes
----------
\*args : dataset
    An arbitrary user-specified property-containing dataset.

index : attribute
    A reference to the dataset used as dimensional scale for all property
    datasets embedded within this group.

"""


def create_prop_group(file: Union[h5py.File, h5py.Group], scale: h5py.Dataset) -> h5py.Group:
    r"""Create a group for holding user-specified properties.

    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import create_prop_group

        >>> hdf5_file = str(...)  # doctest: +SKIP
        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     scale = f.create_dataset('index', data=np.arange(10))
        ...     scale.make_scale('index')
        ...
        ...     group = create_prop_group(f, scale=scale)
        ...     print('group', '=', group)
        group = <HDF5 group "/properties" (0 members)>

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    file : :class:`h5py.File` or :class:`h5py.Group`
        The File or Group where the new ``"properties"`` group should be created.
    scale : :class:`h5py.DataSet`
        The dimensional scale which will be attached to all property datasets
        created by :func:`dataCAT.create_prop_dset`.

    Returns
    -------
    :class:`h5py.Group`
        The newly created group.

    """
    # Construct the group
    grp = file.create_group('properties', track_order=True)
    grp.attrs['index'] = scale.ref
    grp.attrs['__doc__'] = np.string_(PROPERTY_DOC)
    return grp


def create_prop_dset(group: h5py.Group, name: str, dtype: DtypeLike = None,
                     prop_names: Optional[Sequence[str]] = None,
                     **kwargs: Any) -> h5py.Dataset:
    r"""Construct a new dataset for holding a user-defined molecular property.

    Examples
    --------
    In the example below a new dataset is created for storing
    solvation energies in water, methanol and ethanol.

    .. testsetup:: python

        >>> import os
        >>> from shutil import copyfile
        >>> from dataCAT.testing_utils import HDF5_READ, HDF5_TMP as hdf5_file

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)
        >>> _ = copyfile(HDF5_READ, hdf5_file)

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     scale = f.create_dataset('index', data=np.arange(10))
        ...     scale.make_scale('index')
        ...     _ = create_prop_group(f, scale=scale)

    .. code:: python

        >>> import h5py
        >>> from dataCAT import create_prop_dset

        >>> hdf5_file = str(...)  # doctest: +SKIP

        >>> with h5py.File(hdf5_file, 'r+') as f:
        ...     group = f['properties']
        ...     prop_names = ['water', 'methanol', 'ethanol']
        ...
        ...     dset = create_prop_dset(group, 'E_solv', prop_names=prop_names)
        ...     dset_names = group['E_solv_names']
        ...
        ...     print('group', '=', group)
        ...     print('group["E_solv"]', '=', dset)
        ...     print('group["E_solv_names"]', '=', dset_names)
        group = <HDF5 group "/properties" (2 members)>
        group["E_solv"] = <HDF5 dataset "E_solv": shape (10, 3), type "<f4">
        group["E_solv_names"] = <HDF5 dataset "E_solv_names": shape (3,), type "|S8">

    .. testcleanup:: python

        >>> import os

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    Parameters
    ----------
    group : :class:`h5py.Group`
        The ``"properties"`` group where the new dataset will be created.
    name : :class:`str`
        The name of the new dataset.
    prop_names : :class:`Sequence[str]<typing.Sequence>`, optional
        The names of each row in the to-be created dataset.
        Used for defining the length of the second axis and
        will be used as a dimensional scale for aforementioned axis.
        If :data:`None`, create a 1D dataset (with no columns) instead.
    dtype : dtype-like
        The data type of the to-be created dataset.
    \**kwargs : :data:`~Any`
        Further keyword arguments for the h5py :meth:`~h5py.Group.create_dataset` method.

    Returns
    -------
    :class:`h5py.Dataset`
        The newly created dataset.

    """
    scale_name = f'{name}_names'
    index_name = group.attrs['index']
    index = group.file[index_name]
    n = len(index)

    # If no prop_names are specified
    if prop_names is None:
        dset = group.create_dataset(name, shape=(n,), maxshape=(None,), dtype=dtype, **kwargs)
        dset.dims[0].label = 'index'
        dset.dims[0].attach_scale(index)
        return dset

    # Parse the names
    name_array = np.asarray(prop_names, dtype=np.string_)
    if name_array.ndim != 1:
        raise ValueError("'prop_names' expected None or a 1D array-like object; "
                         f"observed dimensionality: {name_array.ndim!r}")

    # Construct the new datasets
    m = len(name_array)
    dset = group.create_dataset(name, shape=(n, m), maxshape=(None, m), dtype=dtype, **kwargs)
    scale = group.create_dataset(scale_name, data=name_array, shape=(m,), dtype=name_array.dtype)
    scale.make_scale(scale_name)

    # Set the dimensional scale
    dset.dims[0].label = 'index'
    dset.dims[0].attach_scale(index)
    dset.dims[1].label = scale_name
    dset.dims[1].attach_scale(scale)
    return dset


def _resize_prop_dset(dset: h5py.Dataset) -> None:
    """Ensure that **dset** is as long as its dimensional scale."""
    scale = dset.dims[0]['index']
    n = len(scale)
    if n > len(dset):
        dset.resize(n, axis=0)


def update_prop_dset(dset: h5py.Dataset, data: np.ndarray,
                     index: Union[None, slice, np.ndarray] = None) -> None:
    """Update **dset** at position **index** with **data**.

    Parameters
    ----------
    dset : :class:`h5py.Dataset`
        The to-be updated h5py dataset.
    data : :class:`numpy.ndarray`
        An array containing the to-be added data.
    index : :class:`slice` or :class:`numpy.ndarray`, optional
        The indices of all to-be updated elements in **dset**.
        **index** either should be of the same length as **data**.


    :rtype: :data:`None`

    """
    idx = slice(None) if index is None else index

    try:
        _resize_prop_dset(dset)
        dset[idx] = data
    except Exception as ex:
        validate_properties(dset.group)
        raise ex


def validate_properties(group: h5py.Group) -> None:
    """Validate the passed hdf5 **group**, ensuring it is compatible with :func:`create_prop_group` and :func:`create_prop_group`.

    This method is called automatically when an exception is raised by :func:`update_prop_dset`.

    Parameters
    ----------
    group : :class:`h5py.Group`
        The to-be validated hdf5 Group.

    Raises
    ------
    :exc:`AssertionError`
        Raised if the validation process fails.

    """  # noqa: E501
    assertion.isinstance(group, h5py.Group)

    assertion.contains(group.attrs, 'index')
    index_ref = group.attrs['index']

    assertion.contains(group.file, index_ref)
    index = group.file[index_ref]

    iterator = ((k, v) for k, v in group.items() if k != 'index' and not k.endswith('_names'))
    for name, dset in iterator:
        assertion.le(len(dset), len(index), message=f'{name!r} invalid dataset length')
        assertion.contains(dset.dims[0], 'index', message=f'{name!r} missing dataset scale')
        assertion.eq(dset.dims[0]['index'], index, message=f'{name!r} invalid dataset scale')
