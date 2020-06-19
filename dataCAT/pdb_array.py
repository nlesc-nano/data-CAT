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
    :members: atoms, bonds, atom_count, bond_count, __getitem__, __len__, items, from_molecules, to_molecules, from_hdf5, to_hdf5

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
from collections import abc
from itertools import repeat
from typing import (
    List, Collection, Iterable, Union, Type, TypeVar, Optional, Dict, Any,
    overload, Sequence, Mapping, Tuple, Generator, TYPE_CHECKING
)

import numpy as np
from scm.plams import Molecule, Atom, Bond
from nanoutils import SupportsIndex, TypedDict, Literal

from .functions import update_pdb_values, append_pdb_values, int_to_slice

if TYPE_CHECKING:
    from h5py import Group
else:
    Group = 'h5py.Group'

__all__ = ['DTYPE_ATOM', 'DTYPE_BOND', 'PDBContainer']

ST = TypeVar('ST', bound='PDBContainer')

DTYPE_ATOM: Mapping[str, np.dtype] = {
    'IsHeteroAtom': bool,
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
    'charge_float': float
}
DTYPE_ATOM = MappingProxyType({k: np.dtype(v) for k, v in DTYPE_ATOM.items()})
_DTYPE_ATOM = np.dtype(list(DTYPE_ATOM.items()))

DTYPE_BOND: Mapping[str, np.dtype] = {
    'atom1': 'int32',
    'atom2': 'int32',
    'order': 'int8'
}
DTYPE_BOND = MappingProxyType({k: np.dtype(v) for k, v in DTYPE_BOND.items()})
_DTYPE_BOND = np.dtype(list(DTYPE_BOND.items()))

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

PDBTuple = Tuple[np.recarray, np.recarray, np.ndarray, np.ndarray]
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


class PDBContainer:
    """An immutable class for holding array-like representions of a set of .pdb files."""

    __slots__ = ('__weakref__', '_hash', '_atoms', '_bonds', '_atom_count', '_bond_count')

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
        """:class:`numpy.ndarray[int]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.atoms`."""  # noqa: E501
        return self._atom_count

    @property
    def bond_count(self) -> np.ndarray:
        """:class:`numpy.ndarray[int]<numpy.ndarray>`, shape :math:`(n,)` : Get a read-only ndarray for keeping track of the number of atoms in each molecule in :attr:`~PDBContainer.bonds`."""  # noqa: E501
        return self._bond_count

    def __init__(self, atoms: np.recarray, bonds: np.recarray,
                 atom_count: np.ndarray, bond_count: np.ndarray) -> None:
        """Initialize an instance."""
        self._atoms = atoms
        self._bonds = bonds
        self._atom_count = atom_count
        self._bond_count = bond_count
        for _, ar in self.items():
            ar.setflags(write=False)

        for _, ar in self.items():
            ar.setflags(write=False)

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        wdith = max(len(k) for k, _ in self.items())

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

    def __reduce__(self: ST) -> Tuple[Type[ST], PDBTuple]:
        """Helper for :mod:`pickle`.

        Examples
        --------
        .. testsetup:: python

            >>> import os
            >>> from pathlib import Path
            >>> from scm.plams import readpdb

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'
            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)[:3]]
            >>> pdb = PDBContainer.from_molecules(mol_list)

        .. code:: python

            >>> import pickle
            >>> from dataCAT import PDBContainer

            >>> pdb = PDBContainer(...)  # doctest: +SKIP

            >>> pdb_bytes = pickle.dumps(pdb)
            >>> pdb_copy = pickle.loads(pdb_bytes)
            >>> pdb == pdb_copy
            True

        """
        cls = type(self)
        return cls, tuple(ar for _, ar in self.items())  # type: ignore

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
        return all([np.all(ar1 == ar2) for ar1, ar2 in iterator])

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            args = []

            # The hash of each individual array consists of its shape appended
            # with the array's first and last element along axis 0
            for _, ar in self.items():
                if not len(ar):
                    first_and_last: Tuple[Any, ...] = ()
                elif len(ar) == 1:
                    _first_and_last = ar[0] if ar.ndim == 1 else ar[0, 0]
                    first_and_last = (_first_and_last, _first_and_last)
                else:
                    i = len(ar) - 1
                    _first_and_last = ar[0::i] if ar.ndim == 1 else ar[0::i, 0]
                    first_and_last = tuple(_first_and_last)
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

            >>> import os
            >>> from pathlib import Path
            >>> from scm.plams import readpdb

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'
            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)]
            >>> pdb = PDBContainer.from_molecules(mol_list)

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

        iterator = (ar[index] for _, ar in self.items())
        return cls(*iterator)

    def items(self) -> Generator[Tuple[str, Union[np.ndarray, np.recarray]], None, None]:
        """Iterator over this intances attribute name/value pairs.

        Examples
        --------
        .. testsetup:: python

            >>> import os
            >>> from pathlib import Path
            >>> from scm.plams import readpdb
            >>> from dataCAT import PDBContainer

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'
            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)]
            >>> pdb_container = PDBContainer.from_molecules(mol_list)

        .. code:: python

            >>> from dataCAT import PDBContainer

            >>> pdb_container = PDBContainer(...)  # doctest: +SKIP
            >>> for name, value in pdb_container.items():
            ...     print(name, '=', object.__repr__(value))  # doctest: +ELLIPSIS
            atoms = <numpy.recarray object at ...>
            bonds = <numpy.recarray object at ...>
            atom_count = <numpy.ndarray object at ...>
            bond_count = <numpy.ndarray object at ...>

        Yields
        ------
        :class:`str` and :class:`numpy.ndarray` / :class:`numpy.recarray`
            Attribute names and values.

        """
        cls = type(self)
        return ((name.strip('_'), getattr(self, name)) for name in cls.__slots__[2:])

    @classmethod
    def from_molecules(cls: Type[ST], mol_list: Iterable[Molecule],
                       min_atom: int = 0,
                       min_bond: int = 0) -> ST:
        """Convert an iterable or sequence of molecules into a new :class:`PDBContainer` instance.

        Examples
        --------
        .. testsetup:: python

            >>> import os
            >>> from pathlib import Path
            >>> from scm.plams import readpdb

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'
            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)]

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
            The minimum number of atoms which :attr:`PDBContainer.bonds` should accomodate.

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
        try:
            _atom_count = max(len(mol.atoms) for mol in mol_list_)
        except ValueError:  # if mol_list is empty
            _atom_count = 0
        atom_count = max(_atom_count, min_atom)
        atom_shape = mol_count, atom_count

        # Gather the shape of the to-be created bond array
        try:
            _bond_count = max(len(mol.bonds) for mol in mol_list_)
        except ValueError:  # if mol_list is empty
            _bond_count = 0
        bond_count = max(_bond_count, min_bond)
        bond_shape = mol_count, bond_count

        # Construct the to-be returned (padded) arrays
        atom_array = np.rec.array(None, shape=atom_shape, dtype=_DTYPE_ATOM)
        bond_array = np.rec.array(None, shape=bond_shape, dtype=_DTYPE_BOND)
        atom_counter = np.empty(mol_count, dtype='int32')
        bond_counter = np.empty(mol_count, dtype='int32')

        # Fill the to-be returned arrays
        for i, mol in enumerate(mol_list_):
            j_atom = len(mol.atoms)
            j_bond = len(mol.bonds)

            atom_array[i, :j_atom] = [_get_atom_info(at, k) for k, at in enumerate(mol, 1)]
            bond_array[i, :j_bond] = _get_bond_info(mol)
            atom_counter[i] = j_atom
            bond_counter[i] = j_bond

        return cls(atoms=atom_array, atom_count=atom_counter,
                   bonds=bond_array, bond_count=bond_counter)

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

            >>> import os
            >>> from pathlib import Path
            >>> from scm.plams import readpdb
            >>> from dataCAT import PDBContainer

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'

            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)]
            >>> mol1 = mol_list[2]
            >>> mol_list1 = mol_list[:3]
            >>> pdb_container = PDBContainer.from_molecules(mol_list)

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBContainer
            >>> from scm.plams import Molecule

            >>> pdb_container = PDBContainer(...)  # doctest: +SKIP

            # Create a single new molecule from `pdb_container`
            >>> pdb_container.to_molecules(idx=0)  # doctest: +ELLIPSIS
            <scm.plams.mol.molecule.Molecule object at ...>

            # Create three new molecules from `pdb_container`
            >>> pdb_container.to_molecules(idx=[0, 1])  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
            [<scm.plams.mol.molecule.Molecule object at ...>,
             <scm.plams.mol.molecule.Molecule object at ...>]

        An example where one or more existing molecules are updated inplace.

        .. code:: python

            # Update `mol` with the info from `pdb_container`
            >>> mol1 = Molecule(...)  # doctest: +SKIP
            >>> mol2 = pdb_container.to_molecules(idx=2, mol=mol1)
            >>> mol1 is mol2
            True

            # Update all molecules in `mol_list` with info from `pdb_container`
            >>> mol_list1 = [Molecule(...), Molecule(...), Molecule(...)]  # doctest: +SKIP
            >>> mol_list2 = pdb_container.to_molecules(idx=range(3), mol=mol_list)
            >>> for m1, m2 in zip(mol_list1, mol_list2):
            ...     print(m1 is m2)
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
            If one or molecules are provided here then they will be updated inplace.

        Returns
        -------
        :class:`~scm.plams.mol.molecule.Molecule` or :class:`List[Molecule]<typing.List>`
            A molecule or list of molecules,
            depending on whether or not **idx** is a scalar or sequence / slice.
            Note that if :code:`mol is not None`, then the-be returned molecules won't be copies.

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

    def to_hdf5(self, group: Group, mode: Hdf5Mode = 'append',
                idx: Optional[IndexLike] = None) -> None:
        """Export this instance to the specified hdf5 **group**.

        Parameters
        ----------
        group : :class:`h5py.Group`
            The to-be updated/appended h5py group.
        mode : :class:`str`
            Whether to append or update the passed **group**.
            Accepted values are ``"append"`` and ``"update"``.
        idx : :class:`int`, :class:`Sequence[int]<typing.Sequence>` or :class:`slice`, optional
            An object for slicing all datasets in **group**.
            Note that, contrary to numpy, if a sequence of integers is provided
            then they *will* have to be ordered.


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

        if mode == 'append':
            append_pdb_values(group, self)
        elif mode == 'update':
            update_pdb_values(group, self, index)
        else:
            raise ValueError(repr(mode))

    @classmethod
    def from_hdf5(cls: Type[ST], group: Group, idx: Optional[IndexLike] = None) -> ST:
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

        return cls(
            atoms=group['atoms'][index].view(np.recarray),
            bonds=group['bonds'][index].view(np.recarray),
            atom_count=group['atom_count'][index],
            bond_count=group['bond_count'][index]
        )
