""" """

from types import MappingProxyType
from typing import Union, Mapping, Any, Sequence, Tuple, Optional, TYPE_CHECKING
from logging import Logger
from datetime import datetime

import h5py
import numpy as np

from . import CAT_VERSION, NANOCAT_VERSION, DATACAT_VERSION

if TYPE_CHECKING:
    from numpy.typing import DtypeLike
else:
    DtypeLike = 'numpy.typing.DtypeLike'

__all__ = ['create_hdf5_log', 'update_hdf5_log']

VERSION = (CAT_VERSION, NANOCAT_VERSION, DATACAT_VERSION)

_DT_MAPPING = {
    'year': 'int16',
    'month': 'int8',
    'day': 'int8',
    'hour': 'int8',
    'minute': 'int8',
    'second': 'int8',
    'microsecond': 'int32'
}
DT_MAPPING: Mapping[str, np.dtype] = MappingProxyType({
    k: np.dtype(v) for k, v in _DT_MAPPING.items()
})
DT_DTYPE = np.dtype(list(DT_MAPPING.items()))

_VERSION_MAPPING = {
    'major': 'int8',
    'minor': 'int8',
    'micro': 'int8'
}
VERSION_MAPPING: Mapping[str, np.dtype] = MappingProxyType({
    k: np.dtype((v, 3)) for k, v in _VERSION_MAPPING.items()
})
VERSION_DTYPE = np.dtype(list(VERSION_MAPPING.items()))

SLICE_DTYPE = h5py.vlen_dtype(np.dtype('int32'))


def _get_dt_tuple():
    date = datetime.now()
    return tuple(getattr(date, key) for key in DT_MAPPING.keys())


def create_hdf5_log(file: Union[h5py.File, h5py.Group],
                    n_entries: int = 100,
                    initial_versions: Sequence[Tuple[int, int, int]] = VERSION
                    ) -> h5py.Group:
    """Placeholder."""
    m = len(initial_versions)

    if n_entries < 1:
        raise ValueError(f"'n_entries' must ba larger than 1; observed value: {n_entries!r}")
    elif m < 1:
        raise ValueError(f"'initial_versions' must be larger than 1")

    date_tuple = _get_dt_tuple()

    grp = file.create_group('logger', track_order=True)
    grp.attrs['__doc__'] = np.string_("A h5py Group for keeping track of database access.")
    grp.attrs['n'] = 0
    grp.attrs['n_step'] = n_entries
    grp.attrs['date_created'] = np.array(date_tuple, dtype=DT_DTYPE)
    grp.attrs['version_created'] = np.array(initial_versions, dtype=VERSION_DTYPE)

    shape1 = (n_entries, )
    shape2 = (n_entries, m)
    grp.create_dataset('date', shape=shape1, maxshape=(None,), dtype=DT_DTYPE, chunks=shape1)
    grp.create_dataset('version', shape=shape2, maxshape=(None, m), dtype=VERSION_DTYPE, chunks=shape2)  # noqa: E501
    grp.create_dataset('index', shape=shape1, maxshape=(None,), dtype=SLICE_DTYPE, chunks=shape1)
    return grp


def update_hdf5_log(file: Union[h5py.Group, h5py.File], idx: np.ndarray,
                    version: Sequence[Tuple[int, int, int]] = VERSION) -> None:
    """Placeholder."""
    group = file['logger']

    n = group.attrs['n']
    n_max = len(group['data'])

    # Increase the size of the datasets by *n_step*
    if n >= n_max:
        n_max += group.attrs['n_step']
        group['date'].resize(n_max, axis=0)
        group['version'].resize(n_max, axis=0)
        group['index'].resize(n_max, axis=0)

    # Parse the passed **idx**
    index = np.array(idx, ndmin=1, copy=False)
    if index.ndim > 1:
        raise ValueError

    generic = index.dtype.type
    if issubclass(generic, np.bool_):
        index, *_ = index.nonzero()
    elif not issubclass(generic, np.integer):
        raise TypeError

    # Update the datasets
    date = _get_dt_tuple()
    group['date'][n] = date
    group['version'][n] = version
    group['index'][n] = index

    group.attrs['n'] += 1
