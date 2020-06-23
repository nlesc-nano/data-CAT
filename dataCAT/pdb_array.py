"""A module for constructing array-representations of .pdb files.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    PDBContainer
    DTYPE_ATOM
    DTYPE_BOND

API
---
.. autoclass:: PDBContainer
    :members: atoms, bonds, atom_count, bond_count, __getitem__, __len__, keys, values, items, from_molecules, to_molecules, create_hdf5_group, from_hdf5, to_hdf5

.. data:: DTYPE_ATOM
    :type: Mapping[str, np.dtype]
    :value: ...

    A mapping representing the dtype of :attr:`PDBContainer.atoms`.

    Most field names are based on to their, identically named, counterpart as produced by
    :func:`readpdb()<scm.plams.interfaces.molecule.rdkit.readpdb>`,
    the data in question being stored in the
    :class:`Atom.properties.pdb_info<scm.plams.mol.atom.Atom>` block.

    There are six exception to this general rule:

    * ``x``, ``y`` & ``z``: Based on :class:`Atom.x<scm.plams.mol.atom.Atom>`,
      :class:`Atom.y<scm.plams.mol.atom.Atom>` and :class:`Atom.z<scm.plams.mol.atom.Atom>`.
    * ``symbol``: Based on :class:`Atom.symbol<scm.plams.mol.atom.Atom>`.
    * ``charge``: Based on :class:`Atom.properties.charge<scm.plams.mol.atom.Atom>`.
    * ``charge_float``: Based on :class:`Atom.properties.charge_float<scm.plams.mol.atom.Atom>`.

    .. code:: python

        mappingproxy({
            'IsHeteroAtom':  dtype('bool'),
            'SerialNumber':  dtype('int16'),
            'Name':          dtype('S4'),
            'ResidueName':   dtype('S3'),
            'ChainId':       dtype('S1'),
            'ResidueNumber': dtype('int16'),
            'x':             dtype('float32'),
            'y':             dtype('float32'),
            'z':             dtype('float32'),
            'Occupancy':     dtype('float32'),
            'TempFactor':    dtype('float32'),
            'symbol':        dtype('S4'),
            'charge':        dtype('int8'),
            'charge_float':  dtype('float64')
        })


.. data:: DTYPE_BOND
    :type: Mapping[str, np.dtype]
    :value: ...

    A mapping representing the dtype of :attr:`PDBContainer.bonds`.

    Field names are based on to their, identically named,
    counterpart in :class:`~scm.plams.mol.bond.Bond`.

    .. code:: python

        mappingproxy({
            'atom1': dtype('int32'),
            'atom2': dtype('int32'),
            'order': dtype('int8')
        })

"""  # noqa: E501

import textwrap
from types import MappingProxyType
from itertools import repeat
from collections import abc
from typing import (
    List, Collection, Iterable, Union, Type, TypeVar, Optional, Dict, Any,
    overload, Sequence, Mapping, Tuple, Generator, ClassVar, TYPE_CHECKING
)

import h5py
import numpy as np
from scm.plams import Molecule, Atom, Bond
from nanoutils import SupportsIndex, TypedDict, Literal
from assertionlib import assertion

from .functions import update_pdb_values, append_pdb_values, int_to_slice

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

else:
    ArrayLike = 'numpy.typing.ArrayLike'

__all__ = ['DTYPE_ATOM', 'DTYPE_BOND', 'PDBContainer']

ST = TypeVar('ST', bound='PDBContainer')

_DTYPE_ATOM = {
    'IsHeteroAtom': 'bool',
    'SerialNumber': 'int16',
    'Name': 'S4',
    'ResidueName': 'S3',
    'ChainId': 'S1',
    'ResidueNumber': 'int16',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32',
    'Occupancy': 'float32',
    'TempFactor': 'float32',
    'symbol': 'S4',
    'charge': 'int8',
    'charge_float': 'float64'
}
DTYPE_ATOM: Mapping[str, np.dtype] = MappingProxyType(
    {k: np.dtype(v) for k, v in _DTYPE_ATOM.items()}
)
del _DTYPE_ATOM

_DTYPE_BOND = {
    'atom1': 'int32',
    'atom2': 'int32',
    'order': 'int8'
}
DTYPE_BOND: Mapping[str, np.dtype] = MappingProxyType({
    k: np.dtype(v) for k, v in _DTYPE_BOND.items()
})
del _DTYPE_BOND

_AtomTuple = Tuple[
    bool,  # IsHeteroAtom
    int,  # SerialNumber
    str,  # Name
    str,  # ResidueName
    str,  # ChainId
    int,  # ResidueNumber
    float,  # x
    float,  # y
    float,  # z
    float,  # Occupancy
    float,  # TempFactor
    str,  # symbol
    int,  # charge
    float  # charge_float
]

_BondTuple = Tuple[
    int,  # atom1
    int,  # atom2
    int,  # order
]

ReduceTuple = Tuple[np.recarray, np.recarray, np.ndarray, np.ndarray, Literal[False]]
Coords = Tuple[float, float, float]


class _PDBInfo(TypedDict):
    IsHeteroAtom: bool
    SerialNumber: int
    Name: str
    ResidueName: str
    ChainId: str
    ResidueNumber: int
    Occupancy: float
    TempFactor: float


class _Properties(TypedDict):
    charge: int
    charge_float: float
    pdb_info: _PDBInfo


def _get_atom_info(at: Atom, i: int) -> _AtomTuple:
    """Helper function for :meth:`PDBContainer.from_molecules`: create a tuple representing a single :attr:`PDBContainer.atoms` row."""  # noqa: E501
    prop = at.properties
    symbol = at.symbol
    charge = prop.get('charge', 0)

    pdb = prop.get('pdb_info', {})
    return (
        pdb.get('IsHeteroAtom', False),  # type: ignore
        pdb.get('SerialNumber', i),
        pdb.get('Name', symbol),
        pdb.get('ResidueName', 'LIG'),
        pdb.get('ChainId', 'A'),
        pdb.get('ResidueNumber', 1),
        *at.coords,
        pdb.get('Occupancy', 1.0),
        pdb.get('TempFactor', 0.0),
        symbol,
        charge,
        prop.get('charge_float', charge)
    )


def _get_bond_info(mol: Molecule) -> List[_BondTuple]:
    """Helper function for :meth:`PDBContainer.from_molecules`: create a tuple representing a single :attr:`PDBContainer.bonds` row.

    Note that the atomic indices are 1-based.
    """  # noqa: E501
    mol.set_atoms_id(start=1)
    ret = [(b.atom1.id, b.atom2.id, b.order) for b in mol.bonds]
    mol.unset_atoms_id()
    return ret


def _iter_rec(atom_array: np.recarray) -> Generator[Tuple[_Properties, Coords, str], None, None]:
    """Helper function for :func:`_rec_to_mol`: create an iterator yielding atom properties and attributes."""  # noqa: E501
    for IsHeteroAtom, SerialNumber, Name, ResidueName, ChainId, ResidueNumber, x, y, z, Occupancy, TempFactor, symbol, charge, charge_float in atom_array:  # noqa: E501
        _pdb_info = {
            'IsHeteroAtom': IsHeteroAtom,
            'SerialNumber': SerialNumber,
            'Name': Name.decode(),
            'ResidueName': ResidueName.decode(),
            'ChainId': ChainId.decode(),
            'ResidueNumber': ResidueNumber,
            'Occupancy': Occupancy,
            'TempFactor': TempFactor
        }

        properties = {
            'charge': charge,
            'charge_float': charge_float,
            'pdb_info': _pdb_info
        }
        yield properties, (x, y, z), symbol.decode()  # type: ignore


def _rec_to_mol(atom_array: np.recarray, bond_array: np.recarray,
                atom_len: Optional[int] = None,
                bond_len: Optional[int] = None,
                mol: Optional[Molecule] = None) -> Molecule:
    """Helper function for :meth:`PDBContainer.from_molecules`: update/create a single molecule from the passed **atom_array** and **bond_array**."""  # noqa: E501
    if mol is None:
        ret = Molecule()
        for _ in range(len(atom_array[:atom_len])):
            ret.add_atom(Atom(mol=ret))
    else:
        ret = mol

    iterator = _iter_rec(atom_array[:atom_len])
    for atom, (properties, coords, symbol) in zip(ret, iterator):
        atom.coords = coords
        atom.symbol = symbol
        atom.properties.update(properties)

    if ret.bonds:
        ret.delete_all_bonds()
    for i, j, order in bond_array[:bond_len]:
        bond = Bond(atom1=ret[i], atom2=ret[j], order=order, mol=ret)
        ret.add_bond(bond)
    return ret


Hdf5Mode = Literal['append', 'update']
IndexLike = Union[SupportsIndex, Sequence[int], slice, np.ndarray]
AttrName = Literal['atoms', 'bonds', 'atom_count', 'bond_count']


class PDBContainer:
    """An (immutable) class for holding array-like representions of a set of .pdb files.

    The :class:`PDBContainer` class serves as an (intermediate) container
    for storing .pdb files in the hdf5 format,
    thus facilitating the storage and interconversion
    between PLAMS molecules and the :mod:`h5py` interface.

    The methods implemented in this class can roughly be divided into three categories:

    * Molecule-interconversion: :meth:`~PDBContainer.to_molecules` &
      :meth:`~PDBContainer.from_molecules`.
    * hdf5-interconversion: :meth:`~PDBContainer.create_hdf5_group`,
      :meth:`~PDBContainer.validate_hdf5`,
      :meth:`~PDBContainer.to_hdf5` &  :meth:`~PDBContainer.from_hdf5`.
    * Miscellaneous: :meth:`~PDBContainer.keys`, :meth:`~PDBContainer.values`,
      :meth:`~PDBContainer.items`, :meth:`~PDBContainer.__getitem__` &
      :meth:`~PDBContainer.__len__`.

    Examples
    --------
    .. testsetup:: python

        >>> import os
        >>> from pathlib import Path

        >>> from dataCAT.testing_utils import (
        ...     MOL_TUPLE as mol_list,
        ...     PDB as pdb
        ... )

        >>> hdf5_file = Path('tests') / 'test_files' / 'tmp_file.hdf5'
        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    .. code:: python

        >>> import h5py
        >>> from scm.plams import readpdb
        >>> from dataCAT import PDBContainer

        >>> mol_list [readpdb(...), ...]  # doctest: +SKIP
        >>> pdb = PDBContainer.from_molecules(mol_list)
        >>> print(pdb)
        PDBContainer(
            atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
            bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
            atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
            bond_count = numpy.ndarray(..., shape=(23,), dtype=int32)
        )

        >>> hdf5_file = str(...)  # doctest: +SKIP
        >>> with h5py.File(hdf5_file, 'a') as f:
        ...     group = pdb.create_hdf5_group(f, name='ligand')
        ...     pdb.to_hdf5(group, mode='append')
        ...
        ...     print('group', '=', group)
        ...     for name, dset in group.items():
        ...         print(f'group[{name!r}]', '=', dset)
        group = <HDF5 group "/ligand" (4 members)>
        group['atoms'] = <HDF5 dataset "atoms": shape (23, 76), type "|V46">
        group['bonds'] = <HDF5 dataset "bonds": shape (23, 75), type "|V9">
        group['atom_count'] = <HDF5 dataset "atom_count": shape (23,), type "<i4">
        group['bond_count'] = <HDF5 dataset "bond_count": shape (23,), type "<i4">

    .. testcleanup:: python

        >>> if os.path.isfile(hdf5_file):
        ...     os.remove(hdf5_file)

    """

    __slots__ = ('__weakref__', '_hash', '_atoms', '_bonds', '_atom_count', '_bond_count')

    #: A mapping holding the dimensionality of each array embedded within this class.
    NDIM: ClassVar[Mapping[AttrName, int]] = MappingProxyType({
        'atoms': 2,
        'bonds': 2,
        'atom_count': 1,
        'bond_count': 1,
    })

    #: A mapping holding the dtype of each array embedded within this class.
    DTYPE: ClassVar[Mapping[AttrName, np.dtype]] = MappingProxyType({
        'atoms': np.dtype(list(DTYPE_ATOM.items())),
        'bonds': np.dtype(list(DTYPE_BOND.items())),
        'atom_count': np.dtype('int32'),
        'bond_count': np.dtype('int32'),
    })

    @property
    def atoms(self) -> np.recarray:
        """:class:`numpy.recarray`, shape :math:`(n, m)`: Get a read-only padded recarray for keeping track of all atom-related information.

        See :data:`dataCAT.DTYPE_ATOM` for a comprehensive overview of
        all field names and dtypes.

        """  # noqa: E501
        return self._atoms

    @property
    def bonds(self) -> np.recarray:
        """:class:`numpy.recarray`, shape :math:`(n, k)` : Get a read-only padded recarray for keeping track of all bond-related information.

        Note that all atomic indices are 1-based.

        See :data:`dataCAT.DTYPE_BOND` for a comprehensive overview of
        all field names and dtypes.

        """  # noqa: E501
        return self._bonds

    @property
    def atom_count(self) -> np.ndarray:
        """:class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.atoms`."""  # noqa: E501
        return self._atom_count

    @property
    def bond_count(self) -> np.ndarray:
        """:class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.bonds`."""  # noqa: E501
        return self._bond_count

    @overload
    def __init__(self, atoms: np.recarray, bonds: np.recarray,
                 atom_count: np.ndarray, bond_count: np.ndarray,
                 validate: Literal[False]) -> None:
        ...
    @overload  # noqa: E301
    def __init__(self, atoms: ArrayLike, bonds: ArrayLike,
                 atom_count: ArrayLike, bond_count: ArrayLike,
                 validate: Literal[True] = ..., copy: bool = ...) -> None:
        ...
    def __init__(self, atoms, bonds, atom_count, bond_count, validate=True, copy=True):  # noqa: E501,E301
        """Initialize an instance.

        Parameters
        ----------
        atoms : :class:`numpy.recarray`, shape :math:`(n, m)`
            A padded recarray for keeping track of all atom-related information.
            See :attr:`PDBContainer.atoms`.
        bonds : :class:`numpy.recarray`, shape :math:`(n, k)`
            A padded recarray for keeping track of all bond-related information.
            See :attr:`PDBContainer.bonds`.
        atom_count : :class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)`
            An ndarray for keeping track of the number of atoms in each molecule in **atoms**.
            See :attr:`PDBContainer.atom_count`.
        bond_count : :class:`numpy.ndarray[int32]<numpy.ndarray>`, shape :math:`(n,)`
            An ndarray for keeping track of the number of bonds in each molecule in **bonds**.
            See :attr:`PDBContainer.bond_count`.

        Keyword Arguments
        -----------------
        validate : :class:`bool`
            If :data:`True` perform more thorough validation of the input arrays.
            Note that this also allows the parameters to-be passed as array-like objects
            in addition to aforementioned :class:`~numpy.ndarray` or
            :class:`~numpy.recarray` instances.
        copy : :class:`bool`
            If :data:`True`, set the passed arrays as copies.
            Only relevant if :data:`validate = True<True>`.


        :rtype: :data:`None`

        """
        if validate:
            cls = type(self)
            rec_set = {'atoms', 'bonds'}
            items = [
                ('atoms', atoms),
                ('bonds', bonds),
                ('atom_count', atom_count),
                ('bond_count', bond_count)
            ]

            for name, _array in items:
                ndmin = cls.NDIM[name]
                dtype = cls.DTYPE[name]

                array = np.array(_array, dtype=dtype, ndmin=ndmin, copy=copy)
                if name in rec_set:
                    array = array.view(np.recarray)
                setattr(self, f'_{name}', array)

            len_set = {len(ar) for ar in self.values()}
            if len(len_set) != 1:
                raise ValueError("All passed arrays should be of the same length")

        # Assume the input does not have to be parsed
        else:
            self._atoms: np.recarray = atoms
            self._bonds: np.recarray = bonds
            self._atom_count: np.ndarray = atom_count
            self._bond_count: np.ndarray = bond_count

        for ar in self.values():
            ar.setflags(write=False)

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        wdith = max(len(k) for k in self.keys())

        def _str(k, v):
            if isinstance(v, np.recarray):
                dtype = '...'
            else:
                dtype = str(v.dtype)
            return (f'{k:{wdith}} = {v.__class__.__module__}.{v.__class__.__name__}'
                    f'(..., shape={v.shape}, dtype={dtype})')

        ret = ',\n'.join(_str(k, v) for k, v in self.items())
        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    def __reduce__(self: ST) -> Tuple[Type[ST], ReduceTuple]:
        """Helper for :mod:`pickle`."""
        cls = type(self)
        return cls, (*self.values(), False)  # type: ignore

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self  # self is immutable

    def __deepcopy__(self: ST, memo: Optional[Dict[int, Any]] = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self  # self is immutable

    def __len__(self) -> int:
        """Implement :func:`len(self)<len>`.

        Returns
        -------
        :class:`int`
            Returns the length of the arrays embedded within this instance
            (which are all of the same length).

        """
        return len(self.atom_count)

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False
        elif hash(self) != hash(value):
            return False

        iterator = ((v, getattr(value, k)) for k, v in self.items())
        return all(np.all(ar1 == ar2) for ar1, ar2 in iterator)

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            args = []

            # The hash of each individual array consists of its shape appended
            # with the array's first and last element along axis 0
            for ar in self.values():
                if not len(ar):
                    first_and_last: Tuple[Any, ...] = ()
                else:
                    first = ar[0] if ar.ndim == 1 else ar[0, 0]
                    last = ar[-1] if ar.ndim == 1 else ar[-1, 0]
                    first_and_last = (first, last)
                args.append(ar.shape + first_and_last)

            cls = type(self)
            self._hash: int = hash((cls, tuple(args)))
            return self._hash

    def __getitem__(self: ST, key: IndexLike) -> ST:
        """Implement :meth:`self[key]<object.__getitem__>`.

        Constructs a new :class:`PDBContainer` instance by slicing all arrays with **key**.
        Follows the standard NumPy broadcasting rules:
        if an integer or slice is passed then a shallow copy is returned;
        otherwise a deep copy will be created.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> print(pdb)
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(23,), dtype=int32)
            )

            >>> pdb[0]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(1, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(1, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(1,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(1,), dtype=int32)
            )

            >>> pdb[:10]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(10, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(10, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(10,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(10,), dtype=int32)
            )

            >>> pdb[[0, 5, 7, 9, 10]]
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(5, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(5, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(5,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(5,), dtype=int32)
            )

        Parameters
        ----------
        idx : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`
            An object for slicing arrays along :code:`axis=0`.

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A shallow or deep copy of a slice of this instance.

        """
        cls = type(self)
        try:
            index = int_to_slice(key, len(self))  # type: ignore
        except (AttributeError, TypeError):
            index = np.asarray(key) if not isinstance(key, slice) else key
            assert getattr(index, 'ndim', 1) == 1

        iterator = (ar[index] for ar in self.values())
        return cls(*iterator, validate=False)  # type: ignore

    @classmethod
    def keys(cls) -> Generator[AttrName, None, None]:
        """Yield the (public) attribute names in this class.

        Examples
        --------
        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> for name in PDBContainer.keys():
            ...     print(name)
            atoms
            bonds
            atom_count
            bond_count

        Yields
        ------
        :class:`str`
            The names of all attributes in this class.

        """
        return (name.strip('_') for name in cls.__slots__[2:])  # type: ignore

    def values(self) -> Generator[Union[np.ndarray, np.recarray], None, None]:
        """Yield the (public) attributes in this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> for value in pdb.values():
            ...     print(object.__repr__(value))  # doctest: +ELLIPSIS
            <numpy.recarray object at ...>
            <numpy.recarray object at ...>
            <numpy.ndarray object at ...>
            <numpy.ndarray object at ...>

        Yields
        ------
        :class:`str`
            The values of all attributes in this instance.

        """
        cls = type(self)
        return (getattr(self, name) for name in cls.__slots__[2:])

    def items(self) -> Generator[Tuple[AttrName, Union[np.ndarray, np.recarray]], None, None]:
        """Yield the (public) attribute name/value pairs in this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import PDB as pdb

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP
            >>> for name, value in pdb.items():
            ...     print(name, '=', object.__repr__(value))  # doctest: +ELLIPSIS
            atoms = <numpy.recarray object at ...>
            bonds = <numpy.recarray object at ...>
            atom_count = <numpy.ndarray object at ...>
            bond_count = <numpy.ndarray object at ...>

        Yields
        ------
        :class:`str` and :class:`numpy.ndarray` / :class:`numpy.recarray`
            The names and values of all attributes in this instance.

        """
        return ((n, getattr(self, n)) for n in self.keys())

    @classmethod
    def from_molecules(cls: Type[ST], mol_list: Iterable[Molecule],
                       min_atom: int = 0,
                       min_bond: int = 0) -> ST:
        """Convert an iterable or sequence of molecules into a new :class:`PDBContainer` instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import (
            ...     PDB as pdb,
            ...     MOL_TUPLE as mol_list
            ... )

        .. code:: python

            >>> from typing import List
            >>> from dataCAT import PDBContainer
            >>> from scm.plams import readpdb, Molecule

            >>> mol_list: List[Molecule] = [readpdb(...), ...]  # doctest: +SKIP
            >>> PDBContainer.from_molecules(mol_list)
            PDBContainer(
                atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(23,), dtype=int32),
                bond_count = numpy.ndarray(..., shape=(23,), dtype=int32)
            )

        Parameters
        ----------
        mol_list : :class:`Iterable[Molecule]<typing.Iterable>`
            An iterable consisting of PLAMS molecules.
        min_atom : :class:`int`
            The minimum number of atoms which :attr:`PDBContainer.atoms` should accomodate.
        min_bond : :class:`int`
            The minimum number of bonds which :attr:`PDBContainer.bonds` should accomodate.

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A pdb container.

        """
        if isinstance(mol_list, abc.Iterator):
            mol_list_: Collection[Molecule] = list(mol_list)
        else:
            mol_list_ = mol_list  # type: ignore
        mol_count = len(mol_list_)

        # Gather the shape of the to-be created atom (pdb-file) array
        _atom_count = max((len(mol.atoms) for mol in mol_list_), default=0)
        atom_count = max(_atom_count, min_atom)
        atom_shape = mol_count, atom_count

        # Gather the shape of the to-be created bond array
        _bond_count = max((len(mol.bonds) for mol in mol_list_), default=0)
        bond_count = max(_bond_count, min_bond)
        bond_shape = mol_count, bond_count

        # Construct the to-be returned (padded) arrays
        DTYPE = cls.DTYPE
        atom_array = np.rec.array(None, shape=atom_shape, dtype=DTYPE['atoms'])
        bond_array = np.rec.array(None, shape=bond_shape, dtype=DTYPE['bonds'])
        atom_counter = np.empty(mol_count, dtype=DTYPE['atom_count'])
        bond_counter = np.empty(mol_count, dtype=DTYPE['bond_count'])

        # Fill the to-be returned arrays
        for i, mol in enumerate(mol_list_):
            j_atom = len(mol.atoms)
            j_bond = len(mol.bonds)

            atom_array[i, :j_atom] = [_get_atom_info(at, k) for k, at in enumerate(mol, 1)]
            bond_array[i, :j_bond] = _get_bond_info(mol)
            atom_counter[i] = j_atom
            bond_counter[i] = j_bond

        return cls(
            atoms=atom_array, bonds=bond_array,
            atom_count=atom_counter, bond_count=bond_counter,
            validate=False
        )

    @overload
    def to_molecules(self, idx: Union[None, Sequence[int], slice, np.ndarray] = ...,
                     mol: Optional[Iterable[Optional[Molecule]]] = ...) -> List[Molecule]:
        ...
    @overload  # noqa: E301
    def to_molecules(self, idx: SupportsIndex = ..., mol: Optional[Molecule] = ...) -> Molecule:
        ...
    def to_molecules(self, idx=None, mol=None):  # noqa: E301
        """Create a molecule or list of molecules from this instance.

        Examples
        --------
        .. testsetup:: python

            >>> from dataCAT.testing_utils import (
            ...     PDB as pdb,
            ...     MOL_TUPLE as mol_list,
            ...     MOL as mol
            ... )

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBContainer
            >>> from scm.plams import Molecule

            >>> pdb = PDBContainer(...)  # doctest: +SKIP

            # Create a single new molecule from `pdb`
            >>> pdb.to_molecules(idx=0)  # doctest: +ELLIPSIS
            <scm.plams.mol.molecule.Molecule object at ...>

            # Create three new molecules from `pdb`
            >>> pdb.to_molecules(idx=[0, 1])  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
            [<scm.plams.mol.molecule.Molecule object at ...>,
             <scm.plams.mol.molecule.Molecule object at ...>]

        An example where one or more existing molecules are updated in-place.

        .. code:: python

            # Update `mol` with the info from `pdb`
            >>> mol = Molecule(...)  # doctest: +SKIP
            >>> mol_new = pdb.to_molecules(idx=2, mol=mol)
            >>> mol is mol_new
            True

            # Update all molecules in `mol_list` with info from `pdb`
            >>> mol_list = [Molecule(...), Molecule(...), Molecule(...)]  # doctest: +SKIP
            >>> mol_list_new = pdb.to_molecules(idx=range(3), mol=mol_list)
            >>> for m, m_new in zip(mol_list, mol_list_new):
            ...     print(m is m_new)
            True
            True
            True

        Parameters
        ----------
        idx : :class:`int` or :class:`Sequence[int]<typing.Sequence>`, optional
            An object for slicing the arrays embedded within this instance.
            Follows the standard numpy broadcasting rules (*e.g.* :code:`self.atoms[idx]`).
            If a scalar is provided (*e.g.* an integer) then a single molecule will be returned.
            If a sequence, range, slice, *etc.* is provided then
            a list of molecules will be returned.
        mol : :class:`~scm.plams.mol.molecule.Molecule` or :class:`Iterable[Molecule]<typing.Iterable>`, optional
            A molecule or list of molecules.
            If one or molecules are provided here then they will be updated in-place.

        Returns
        -------
        :class:`~scm.plams.mol.molecule.Molecule` or :class:`List[Molecule]<typing.List>`
            A molecule or list of molecules,
            depending on whether or not **idx** is a scalar or sequence / slice.
            Note that if :data:`mol is not None<None>`, then the-be returned molecules won't be copies.

        """  # noqa: E501
        if idx is None:
            i = slice(None)
            is_seq = True
        else:
            try:
                i = idx.__index__()
                is_seq = False
            except (AttributeError, TypeError):
                i = idx
                is_seq = True

        atoms = self.atoms[i]
        bonds = self.bonds[i]
        atom_count = self.atom_count[i]
        bond_count = self.bond_count[i]

        if not is_seq:
            return _rec_to_mol(atoms, bonds, atom_count, bond_count, mol)

        if mol is None:
            mol_list = repeat(None)
        elif isinstance(mol, Molecule):
            raise TypeError
        else:
            mol_list = mol

        iterator = zip(atoms, bonds, atom_count, bond_count, mol_list)
        return [_rec_to_mol(*args) for args in iterator]

    @classmethod
    def create_hdf5_group(cls, file: Union[h5py.File, h5py.Group],
                          name: str, **kwargs: Any) -> h5py.Group:
        r"""Create a h5py Group for storing :class:`dataCAT.PDBContainer` instances.

        Parameters
        ----------
        file : :class:`h5py.File` or :class:`h5py.Group`
            The h5py File or Group where the new Group will be created.
        name : :class:`str`
            The name of the to-be created Group.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for the creation of each dataset.
            The arguments already specified by default are:
            ``name``, ``shape``, ``maxshape`` and ``dtype``.

        Returns
        -------
        :class:`h5py.Group`
            The newly created Group.

        """
        cls_name = cls.__name__

        grp = file.create_group(name, track_order=True)
        grp.attrs['__doc__'] = f"A group of datasets representing `{cls_name}`.".encode()

        NDIM = cls.NDIM
        DTYPE = cls.DTYPE
        for key in cls.keys():
            maxshape = NDIM[key] * (None,)
            shape = NDIM[key] * (0,)
            dtype = DTYPE[key]

            dset = grp.create_dataset(key, shape=shape, maxshape=maxshape, dtype=dtype, **kwargs)
            dset.attrs['__doc__'] = f"A dataset representing `{cls_name}.atoms`.".encode()
        return grp

    @classmethod
    def validate_hdf5(cls, group: h5py.Group) -> None:
        """Validate the passed hdf5 **group**, ensuring it is compatible with :meth:`~PDBContainer.to_hdf5` and :meth:`~PDBContainer.from_hdf5`.

        An :exc:`AssertionError` will be raise if **group** does not validate.

        This method is called automatically when an exception is raised by
        :meth:`~PDBContainer.to_hdf5` or :meth:`~PDBContainer.from_hdf5`.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be validated hdf5 Group.

        Raises
        ------
        :exc:`AssertionError`
            Raised if the validation process fails.

        """  # noqa: E501
        if not isinstance(group, h5py.Group):
            raise TypeError("'group' expected a h5py.Group; "
                            f"observed type: {group.__class__.__name__}")

        # Check if **group** has all required keys
        keys = set(cls.keys())
        difference = keys - group.keys()
        if difference:
            missing_keys = ', '.join(repr(i) for i in difference)
            raise AssertionError(f"Missing keys in {group}: {missing_keys}")

        # Check the dimensionality and dtype of all datasets
        len_dict = {}
        iterator = ((k, group[k]) for k in cls.keys())
        for key, dset in iterator:
            len_dict[key] = len(dset)
            assertion.eq(dset.ndim, cls.NDIM[key], message=f"{key} ndim mismatch")
            assertion.eq(dset.dtype, cls.DTYPE[key], message=f"{key} dtype mismatch")

        # Check that all datasets are of the same length
        if len(set(len_dict.values())) != 1:
            raise AssertionError(
                f"All datasets in {group} should be of the same length.\n"
                f"Observed lengths: {len_dict!r}"
            )

    def to_hdf5(self, group: h5py.Group, mode: Hdf5Mode = 'append',
                idx: Optional[IndexLike] = None) -> None:
        """Export this instance to the specified hdf5 **group**.

        Important
        ---------
        If **idx** is passed as a sequence of integers then, contrary to NumPy,
        they *will* have to be sorted.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be updated/appended h5py group.
        mode : :class:`str`
            Whether to append or update the passed **group**.
            Accepted values are ``"append"`` and ``"update"``.
        idx : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`, optional
            An object for slicing all datasets in **group**.


        :rtype: :data:`None`

        """
        if idx is None:
            index: Union[slice, np.ndarray] = slice(None)
        else:
            try:
                index = int_to_slice(idx, len(self))  # type: ignore
            except (AttributeError, TypeError):
                index = np.asarray(idx) if not isinstance(idx, slice) else idx
                assert getattr(index, 'ndim', 1) == 1

        try:
            if mode == 'append':
                return append_pdb_values(group, self)
            elif mode == 'update':
                return update_pdb_values(group, self, index)
        except Exception as ex:
            cls = type(self)
            cls.validate_hdf5(group)
            raise ex
        raise ValueError(repr(mode))

    @classmethod
    def from_hdf5(cls: Type[ST], group: h5py.Group, idx: Optional[IndexLike] = None) -> ST:
        """Construct a new PDBContainer from the passed hdf5 **group**.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be read h5py group.
        idx : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`, optional
            An object for slicing all datasets in **group**.

        Returns
        -------
        :class:`dataCAT.PDBContainer`
            A new PDBContainer constructed from **group**.

        """
        if idx is None:
            index: Union[slice, np.ndarray] = slice(None)
        else:
            try:
                index = int_to_slice(idx, len(group['atom_count']))  # type: ignore
            except (AttributeError, TypeError):
                index = np.asarray(idx) if not isinstance(idx, slice) else idx
                assert getattr(index, 'ndim', 1) == 1

        try:
            return cls(
                atoms=group['atoms'][index].view(np.recarray),
                bonds=group['bonds'][index].view(np.recarray),
                atom_count=group['atom_count'][index],
                bond_count=group['bond_count'][index],
                validate=False
            )
        except Exception as ex:
            cls.validate_hdf5(group)
            raise ex
