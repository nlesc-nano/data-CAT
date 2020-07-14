"""A module for holding functions related to the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.functions
.. autosummary::
    df_to_mongo_dict
    hdf5_availability
    scale_to_index
    array_to_index
    if_exception

API
---
.. autofunction:: df_to_mongo_dict
.. autofunction:: hdf5_availability
.. autofunction:: scale_to_index
.. autofunction:: array_to_index
.. autofunction:: if_exception

"""

import warnings
from time import sleep
from functools import wraps
from typing import (
    Collection, Union, Sequence, Tuple, List, Generator, Any, cast,
    Hashable, Optional, Type, TypeVar, Callable, Dict, TYPE_CHECKING
)

import h5py
import numpy as np
import pandas as pd

from scm.plams import Molecule
import scm.plams.interfaces.molecule.rdkit as molkit
from rdkit import Chem
from rdkit.Chem import Mol

from nanoutils import SupportsIndex, PathType

if TYPE_CHECKING:
    from .pdb_array import PDBContainer, IndexLike
    from numpy.typing import DtypeLike, ArrayLike
else:
    PDBContainer = 'dataCAT.PDBContainer'
    IndexLike = 'dataCAT.pdb_array.IndexLike'
    DtypeLike = 'numpy.typing.DtypeLike'
    ArrayLike = 'numpy.typing.ArrayLike'

__all__ = [
    'df_to_mongo_dict', 'int_to_slice', 'hdf5_availability',
    'scale_to_index', 'array_to_index', 'if_exception'
]

FT = TypeVar('FT', bound=Callable[..., Any])


def df_to_mongo_dict(df: pd.DataFrame,
                     as_gen: bool = True) -> Union[Generator, list]:
    """Convert a dataframe into a generator of dictionaries suitable for a MongoDB_ databases.

    Tuple-keys present in **df** (*i.e.* pd.MultiIndex) are expanded into nested dictionaries.

    .. _MongoDB: https://www.mongodb.com/

    Examples
    --------
    .. testsetup:: python

        >>> import pandas as pd

        >>> _columns = [('E_solv', 'Acetone'), ('E_solv', 'Acetonitrile')]
        >>> columns = pd.MultiIndex.from_tuples(_columns, names=['index', 'sub index'])

        >>> _index = [('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4')]
        >>> index = pd.MultiIndex.from_tuples(_index, names=['smiles', 'anchor'])

        >>> df = pd.DataFrame([[-56.6, -57.9],
        ...                    [-56.5, -57.6],
        ...                    [-57.1, -58.2]], index=index, columns=columns)


    .. code:: python

        >>> import pandas as pd
        >>> from dataCAT.functions import df_to_mongo_dict

        >>> df = pd.DataFrame(...)  # doctest: +SKIP
        >>> print(df)  # doctest: +NORMALIZE_WHITESPACE
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
        ...     print(item)
        {'E_solv': {'Acetone': -56.6, 'Acetonitrile': -57.9}, 'smiles': 'C[O-]', 'anchor': 'O2'}
        {'E_solv': {'Acetone': -56.5, 'Acetonitrile': -57.6}, 'smiles': 'CC[O-]', 'anchor': 'O3'}
        {'E_solv': {'Acetone': -57.1, 'Acetonitrile': -58.2}, 'smiles': 'CCC[O-]', 'anchor': 'O4'}

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A Pandas DataFrame whose axis and columns are instance of pd.MultiIndex.
    as_gen : :class:`bool`
        If :class:`True`, return a generator of dictionaries rather than a list of dictionaries.

    Returns
    -------
    :class:`Generator[dict, None, None]<typing.Generator>` or :class:`List[dict]<typing.List>`
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


def as_pdb_array(mol_list: Collection[Molecule], min_size: int = 0,
                 warn: bool = True) -> np.ndarray:
    """Convert a list of PLAMS molecule into an array of (partially) de-serialized .pdb files.

    Parameters
    ----------
    mol_list: :class:`Collection[Molecule]<typing.Collection>`, length :math:`m`
        A collection of :math:`m` PLAMS molecules.
    min_size : :class:`int`
        The minimumum length of the pdb_array.
        The array is padded with empty strings if required.

    Returns
    -------
    :class:`numpy.ndarray[|S80]<numpy.ndarray>`, shape :math:`(m, n)`
        An array with :math:`m` partially deserialized .pdb files with up to :math:`n` lines each.

    """
    if warn:
        msg = DeprecationWarning("'as_pdb_array()' is deprecated")
        warnings.warn(msg, stacklevel=2)

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


def from_pdb_array(array: np.ndarray, rdmol: bool = True,
                   warn: bool = True) -> Union[Molecule, Mol]:
    """Convert an array with a (partially) de-serialized .pdb file into a molecule.

    Parameters
    ----------
    array : :class:`numpy.ndarray[|S80]<numpy.ndarray>`, shape :math:`(n,)`
        A (partially) de-serialized .pdb file with :math:`n` lines.
    rdmol : :class:`bool`
        If :data:`bool`, return an RDKit molecule instead of a PLAMS molecule.

    Returns
    -------
    :class:`plams.Molecule<scm.plams.mol.molecule.Molecule>` or :class:`rdkit.Chem.Mol`
        A PLAMS or RDKit molecule build from **array**.

    """
    if warn:
        msg = DeprecationWarning("'from_pdb_array()' is deprecated")
        warnings.warn(msg, stacklevel=2)

    pdb_str = ''.join([item.decode() + '\n' for item in array if item])
    ret = Chem.MolFromPDBBlock(pdb_str, removeHs=False, proximityBonding=False)
    if not rdmol:
        return molkit.from_rdmol(ret)
    return ret


def int_to_slice(int_like: SupportsIndex, seq_len: int) -> slice:
    """Take an integer-like object and convert it into a :class:`slice`.

    The slice is constructed in such a manner that using it for slicing will
    return the same value as when passing **int_like**,
    expect that the objects dimensionanlity is larger by 1.

    Examples
    --------
    .. code:: python

        >>> import numpy as np
        >>> from dataCAT.functions import int_to_slice

        >>> array = np.ones(10)
        >>> array[0]
        1.0

        >>> idx = int_to_slice(0, len(array))
        >>> array[idx]
        array([1.])


    Parameters
    ----------
    int_like : :class:`int`
        An int-like object.
    seq_len : :class:`int`
        The length of a to-be sliced sequence.

    Returns
    -------
    :class:`slice`
        An object for slicing the sequence associated with **seq_len**.

    """
    integer = int_like.__index__()
    if integer > 0:
        if integer != seq_len:
            return slice(None, integer + 1)
        else:
            return slice(integer - 1, None)

    else:
        if integer == -1:
            return slice(integer, None)
        else:
            return slice(integer, integer + 1)


def hdf5_availability(filename: PathType, timeout: float = 5.0,
                      max_attempts: Optional[int] = 10,
                      **kwargs: Any) -> None:
    r"""Check if a .hdf5 file is opened by another process; return once it is not.

    If two processes attempt to simultaneously open a single hdf5 file then
    h5py will raise an :exc:`OSError`.

    The purpose of this method is ensure that a .hdf5 file is actually closed,
    thus allowing the :meth:`Database.from_hdf5` method to safely access **filename** without
    the risk of raising an :exc:`OSError`.

    Parameters
    ----------
    filename : :class:`str`, :class:`bytes` or :class:`os.PathLike`
        A path-like object pointing to the hdf5 file of interest.
    timeout : :class:`float`
        Time timeout, in seconds, between subsequent attempts of opening **filename**.
    max_attempts : :class:`int`, optional
        Optional: The maximum number attempts for opening **filename**.
        If the maximum number of attempts is exceeded, raise an :exc:`OSError`.
        Setting this value to :data:`None` will set the number of attempts to unlimited.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :class:`h5py.File`.

    Raises
    ------
    :exc:`OSError`
        Raised if **max_attempts** is exceded.

    """
    err = (f"h5py.File({filename!r}) is currently unavailable; "
           f"repeating attempt in {timeout:1.1f} seconds")

    i = max_attempts if max_attempts is not None else np.inf
    if i <= 0:
        raise ValueError(f"'max_attempts' must be larger than 0; observed value: {i!r}")

    while i:
        try:
            with h5py.File(filename, 'r+', **kwargs):
                return None  # the .hdf5 file can safely be opened
        except OSError as ex:  # the .hdf5 file cannot be safely opened yet
            warn = ResourceWarning(err)
            warn.__cause__ = exception = ex
            warnings.warn(warn)
            sleep(timeout)
        i -= 1

    raise exception


def scale_to_index(scale: h5py.Dataset, index: Optional[IndexLike] = None) -> pd.Index:
    """Construct a pandas Index from the passed **scale** Dataset.

    Returns a :class:`pandas.Index` if the **scale** dtype lacks any fields;
    a :class:`pandas.MultiIndex` is returned otherwise.

    Parameters
    ----------
    scale : :class:`h5py.Dataset`
        The to-be converted 1D Dataset.
    index : :class:`int`, 'Sequence[int]<typing.Sequence>' or :class:`slice`, optional
        The indices of the dataset elements of interest.
        Set to :data:`None` to use the entire dataset.

    Returns
    -------
    :class:`pandas.Index` or :class:`pandas.MultiIndex`
        An MultiIndex or Index, depending on whether or not **scale** has a structured dtype.

    """
    i = index if index is not None else slice(None)
    array = scale[i]

    name = scale.name.rsplit('/', 1)[1]
    return array_to_index(array, name)


def array_to_index(array: ArrayLike, name: Optional[str] = None) -> pd.Index:
    """See :func:`scale_to_index`."""
    data_ar = np.asanyarray(array)
    dtype = data_ar.dtype

    # Create an Index
    if data_ar.dtype.fields is None:
        if h5py.check_string_dtype(dtype):
            data = data_ar.astype(str)
        else:
            data = data_ar
        return pd.Index(data, name=name)

    # It's a structured array; create a MultiIndex
    fields = []
    field_names = []
    for name, (field_dtype, *_) in dtype.fields.items():
        # It's a bytes-string; decode it
        if h5py.check_string_dtype(field_dtype):
            ar = data_ar[name].astype(str)

        # It's a h5py `vlen` dtype; convert it into a list of tuples
        elif h5py.check_vlen_dtype(field_dtype):
            ar = _vlen_to_tuples(data_ar[name])

        else:
            ar = data_ar[name]

        fields.append(ar)
        field_names.append(name)
    return pd.MultiIndex.from_arrays(fields, names=field_names)


def _vlen_to_tuples(array: np.ndarray) -> np.ndarray:
    """Convert an (object) array consisting of arrays into an (object) array of tuples."""
    cache: Dict[bytes, tuple] = {}
    ret = np.empty_like(array, dtype=object)

    for i, ar in enumerate(array):
        byte = ar.tobytes()
        try:
            tup = cache[byte]
        except KeyError:
            cache[byte] = tup = tuple(ar)
        ret[i] = tup
    return ret


def _set_index(cls: Type[PDBContainer], group: h5py.Group,
               dtype: DtypeLike, length: int, **kwargs: Any) -> h5py.Dataset:
    scale = group.create_dataset('index', shape=(length,), maxshape=(None,), dtype=dtype, **kwargs)
    scale.make_scale('index')

    iterator = (group[k] for k in cls.keys() if k != 'scale')
    for dset in iterator:
        dset.dims[0].label = 'index'
        dset.dims[0].attach_scale(scale)

    group['atoms'].dims[1].label = 'atoms'
    group['bonds'].dims[1].label = 'bonds'
    return scale


def if_exception(func: Callable[[Any, Any], None]) -> Callable[[FT], FT]:
    """A decorator which executes **func** if the decorated instance-/class-method raises an exception."""  # noqa: E501
    def decorator(meth: FT) -> FT:
        @wraps(meth)
        def wrapper(self, obj, *args, **kwargs):
            try:
                return meth(self, obj, *args, **kwargs)
            except Exception as ex:
                func(self, obj)
                raise ex
        return cast(FT, wrapper)
    return decorator
