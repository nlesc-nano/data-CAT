"""A module for constructing array-representations of .pdb files.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    PDBTuple
    PDBTuple.from_molecules
    PDBTuple.to_molecules
    DTYPE_ATOM
    DTYPE_BOND

API
---
.. autoclass:: PDBTuple
.. automethod:: PDBTuple.from_molecules
.. automethod:: PDBTuple.to_molecules
.. data:: DTYPE_ATOM
    :value: pandas.DataFrame(...)

    A :class:`pandas.DataFrame` representing the dtype of :attr:`PDBTuple.atoms`.

    Most field names are based on to their, identically named, counterpart as produced by
    :func:`readpdb()<scm.plams.interfaces.molecule.rdkit.readpdb>`,
    the data in question being stored in the
    :class:`Atom.properties.pdb_info<scm.plams.mol.atom.Atom>` block.

    There are four exception to this general rule:

    * ``coords``: Based on :class:`Atom.coords<scm.plams.mol.atom.Atom>`.
    * ``symbol``: Based on :class:`Atom.symbol<scm.plams.mol.atom.Atom>`.
    * ``charge``: Based on :class:`Atom.properties.charge<scm.plams.mol.atom.Atom>`.
    * ``charge_float``: Based on :class:`Atom.properties.charge_float<scm.plams.mol.atom.Atom>`.

    .. code:: python

        >>> from dataCAT import DTYPE_ATOM
        >>> print(DTYPE_ATOM)  # doctest: +SKIP
{dtype_atom}


.. data:: DTYPE_BOND
    :value: pandas.DataFrame(...)

    A :class:`pandas.DataFrame` representing the dtype of :attr:`PDBTuple.bonds`.

    .. code:: python

        >>> from dataCAT import DTYPE_BOND
        >>> print(DTYPE_BOND)
{dtype_bond}

"""

import textwrap
from collections import abc
from itertools import repeat
from typing import (
    List, Collection, NamedTuple, Iterable, Union, Type, TypeVar, Optional, overload, Sequence
)

import numpy as np
import pandas as pd
from scm.plams import Molecule, Atom, Bond
from nanoutils import SupportsIndex

__all__ = ['DTYPE_ATOM', 'DTYPE_BOND', 'PDBTuple']

ST = TypeVar('ST', bound='PDBTuple')

DTYPE_ATOM = pd.Series({
    'IsHeteroAtom': bool,
    'SerialNumber': 'int16',
    'Name': 'S4',
    'ResidueName': 'S3',
    'ChainId': 'S1',
    'ResidueNumber': 'int16',
    'coords': 'float32',
    'Occupancy': 'float32',
    'TempFactor': 'float32',
    'symbol': 'S4',
    'charge': 'int8',
    'charge_float': float
}, name='dtype', dtype=object).to_frame()
DTYPE_ATOM.index.name = 'name'
DTYPE_ATOM['dtype'] = [np.dtype(i) for i in DTYPE_ATOM['dtype']]
DTYPE_ATOM['shape'] = 0
DTYPE_ATOM.at['coords', 'shape'] = 3

#: A list representing the dtype of :attr:`PDBTuple.atoms`.
_DTYPE_ATOM = [(i, dtype, shape or ()) for i, (dtype, shape) in DTYPE_ATOM.iterrows()]

DTYPE_BOND = pd.Series({
    'atoms': 'int32',
    'order': 'int8'
}, name='dtype', dtype=object).to_frame()
DTYPE_BOND.index.name = 'name'
DTYPE_BOND['dtype'] = [np.dtype(i) for i in DTYPE_BOND['dtype']]
DTYPE_BOND['shape'] = 0
DTYPE_BOND.at['atoms', 'shape'] = 2

#: A list representing the dtype of :attr:`PDBTuple.bonds`.
_DTYPE_BOND = [(i, dtype, shape or ()) for i, (dtype, shape) in DTYPE_BOND.iterrows()]


def _get_atom_info(at: Atom, i: int):
    """Helper function for :meth:`PDBTuple.from_molecules`: create a tuple representing a single :attr:`PDBTuple.atoms` row."""  # noqa: E501
    prop = at.properties
    symbol = at.symbol
    charge = prop.get('charge', 0)

    pdb = prop.get('pdb_info', {})
    return (
        pdb.get('IsHeteroAtom', False),
        pdb.get('SerialNumber', i),
        pdb.get('Name', symbol),
        pdb.get('ResidueName', 'LIG'),
        pdb.get('ChainId', 'A'),
        pdb.get('ResidueNumber', 1),
        at.coords,
        pdb.get('Occupancy', 1.0),
        pdb.get('TempFactor', 0.0),
        symbol,
        charge,
        prop.get('charge_float', charge)
    )


def _get_bond_info(mol: Molecule):
    """Helper function for :meth:`PDBTuple.from_molecules`: create a tuple representing a single :attr:`PDBTuple.bonds` row."""  # noqa: E501
    mol.set_atoms_id(start=1)
    ret = [((b.atom1.id, b.atom2.id), b.order) for b in mol.bonds]
    mol.unset_atoms_id()
    return ret


def _iter_rec(atom_array: np.recarray):
    """Helper function for :func:`_rec_to_mol`: create an iterator yielding atom properties and attributes."""  # noqa: E501
    for IsHeteroAtom, SerialNumber, Name, ResidueName, ChainId, ResidueNumber, coords, Occupancy, TempFactor, symbol, charge, charge_float in atom_array:  # noqa: E501
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
        yield properties, tuple(coords), symbol.decode()


def _rec_to_mol(atom_array: np.recarray, bond_array: np.recarray,
                atom_len: Optional[int] = None,
                bond_len: Optional[int] = None,
                mol: Optional[Molecule] = None) -> Molecule:
    """Helper function for :meth:`PDBTuple.from_molecules`: update/create a single molecule from the passed **atom_array** and **bond_array**."""  # noqa: E501
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
    for (i, j), order in bond_array[:bond_len]:
        bond = Bond(atom1=ret[i], atom2=ret[j], order=order, mol=ret)
        ret.add_bond(bond)
    return ret


class PDBTuple(NamedTuple):
    """A :func:`~collections.namedtuple` for holding an array-like represention of a set of .pdb files."""  # noqa: E501

    #: :class:`numpy.recarray`, shape :math:`(n, m)` : Get a padded
    #: recarray for keeping track of all atom-related information.
    #: See :data:`dataCAT.DTYPE_ATOM` for a comprehensive overview of
    #: all field names, dtypes and shapes.
    atoms: np.recarray

    #: :class:`numpy.recarray`, shape :math:`(n, k)` : Get a padded
    #: recarray for keeping track of all bond-related information.
    #: See :data:`dataCAT.DTYPE_BOND` for a comprehensive overview of
    #: all field names, dtypes and shapes.
    bonds: np.recarray

    #: :class:`numpy.ndarray[int]<numpy.ndarray>`, shape :math:`(n,)` : Get an ndarray
    #: for keeping track of the number of atoms in each molecule in :attr:`~PDBTuple.atoms`.
    atom_count: np.ndarray

    #: :class:`numpy.ndarray[int]<numpy.ndarray>`, shape :math:`(n,)` : Get an ndarray
    #: for keeping track of the number of atoms in each molecule in :attr:`~PDBTuple.bonds`.
    bond_count: np.ndarray

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        try:
            wdith = max(len(k) for k in self._fields)
        except ValueError:
            return f'{self.__class__.__name__}()'

        def _str(k, v):
            if isinstance(v, np.recarray):
                dtype = '...'
            else:
                dtype = str(v.dtype)
            return (f'{k:{wdith}} = {v.__class__.__module__}.{v.__class__.__name__}'
                    f'(..., shape={v.shape}, dtype={dtype})')

        ret = ',\n'.join(_str(k, v) for k, v in zip(self._fields, self))
        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    @classmethod
    def from_molecules(cls: Type[ST], mol_list: Iterable[Molecule],
                       min_atom: int = 0,
                       min_bond: int = 0) -> ST:
        """Convert an iterable or sequence of molecules into a new :class:`PDBTuple` instance.

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
            >>> from dataCAT import PDBTuple
            >>> from scm.plams import readpdb, Molecule

            >>> mol_list: List[Molecule] = [readpdb(...), ...]  # doctest: +SKIP
            >>> PDBTuple.from_molecules(mol_list)
            PDBTuple(
                atoms      = numpy.recarray(..., shape=(23, 76), dtype=...),
                bonds      = numpy.recarray(..., shape=(23, 75), dtype=...),
                atom_count = numpy.ndarray(..., shape=(23,), dtype=int64),
                bond_count = numpy.ndarray(..., shape=(23,), dtype=int64)
            )

        Parameters
        ----------
        mol_list : :class:`Iterable[Molecule]<typing.Iterable>`
            An iterable consisting of PLAMS molecules.
        min_atom : :class:`int`
            The minimum number of atoms which :attr:`PDBTuple.atoms` should accomodate.
        min_bond : :class:`int`
            The minimum number of atoms which :attr:`PDBTuple.bonds` should accomodate.

        Returns
        -------
        :class:`PDBTuple`
            A new namedtuple.

        """
        if isinstance(mol_list, abc.Iterator):
            mol_list_: Collection[Molecule] = list(mol_list)
        else:
            mol_list_ = mol_list  # type: ignore
        mol_count = len(mol_list_)

        # Gather the shape of the to-be created atom (pdb-file) array
        _atom_count = max(len(mol.atoms) for mol in mol_list_)
        atom_count = max(_atom_count, min_atom)
        atom_shape = mol_count, atom_count

        # Gather the shape of the to-be created bond array
        _bond_count = max(len(mol.bonds) for mol in mol_list_)
        bond_count = max(_bond_count, min_bond)
        bond_shape = mol_count, bond_count

        # Construct the to-be returned (padded) arrays
        atom_array = np.rec.array(None, shape=atom_shape, dtype=_DTYPE_ATOM)
        bond_array = np.rec.array(None, shape=bond_shape, dtype=_DTYPE_BOND)
        atom_counter = np.empty(mol_count, dtype=int)
        bond_counter = np.empty(mol_count, dtype=int)

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
            >>> from dataCAT import PDBTuple

            >>> path = Path('tests') / 'test_files' / 'ligand_pdb'

            >>> mol_list = [readpdb(str(path / f)) for f in os.listdir(path)]
            >>> mol1 = mol_list[2]
            >>> mol_list1 = mol_list[:3]
            >>> pdb_tup = PDBTuple.from_molecules(mol_list)

        An example where one or more new molecules are created.

        .. code:: python

            >>> from dataCAT import PDBTuple
            >>> from scm.plams import Molecule

            >>> pdb_tup = PDBTuple(...)  # doctest: +SKIP

            # Create a single new molecule from `pdb_tup`
            >>> pdb_tup.to_molecules(idx=0)  # doctest: +ELLIPSIS
            <scm.plams.mol.molecule.Molecule object at ...>

            # Create three new molecules from `pdb_tup`
            >>> pdb_tup.to_molecules(idx=[0, 1])  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
            [<scm.plams.mol.molecule.Molecule object at ...>,
             <scm.plams.mol.molecule.Molecule object at ...>]

        An example where one or more existing molecules are updated inplace.

        .. code:: python

            # Update `mol` with the info from `pdb_tup`
            >>> mol1 = Molecule(...)  # doctest: +SKIP
            >>> mol2 = pdb_tup.to_molecules(idx=2, mol=mol1)
            >>> mol1 is mol2
            True

            # Update all molecules in `mol_list` with info from `pdb_tup`
            >>> mol_list1 = [Molecule(...), Molecule(...), Molecule(...)]  # doctest: +SKIP
            >>> mol_list2 = pdb_tup.to_molecules(idx=range(3), mol=mol_list)
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


__doc__ = __doc__.format(
    dtype_atom=textwrap.indent(repr(DTYPE_ATOM), 8 * ' '),
    dtype_bond=textwrap.indent(repr(DTYPE_BOND), 8 * ' ')
)
