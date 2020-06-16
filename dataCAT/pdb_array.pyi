import sys
from typing import Any, Iterable, List, NamedTuple, Optional, Sequence, Type, Tuple, TypeVar, Union, Generator, overload

import numpy as np
import pandas as pd
from scm.plams import Molecule, Atom

if sys.version_info[1] > 8:
    from typing import SupportsIndex, TypedDict
else:
    from typing_extensions import TypedDict, Protocol

    class SupportsIndex(Protocol):
        def __index__(self) -> int: ...

__all__: List[str] = ...

ST = TypeVar('ST', bound=PDBTuple)
_DType = List[Tuple[
    str,   # Name
    np.dtype,  # Data type
    Union[int, Tuple[int, ...]]  # Shape
]]

DTYPE_ATOM: pd.DataFrame = ...
DTYPE_BOND: pd.DataFrame = ...
_DTYPE_ATOM: _DType = ...
_DTYPE_BOND: _DType = ...

_AtomTuple = Tuple[
    bool,  # IsHeteroAtom
    int,  # SerialNumber
    str,  # Name
    str,  # ResidueName
    str,  # ChainId
    int,  # ResidueNumber
    Tuple[float, float, float],  # coords
    float,  # Occupancy
    float,  # TempFactor
    str,  # symbol
    int,  # charge
    float  # charge_float
]

_BondTuple = Tuple[
    Tuple[int, int],  # atom1, atom2
    int,  # order
]

def _get_atom_info(at: Atom, i: int) -> _AtomTuple: ...
def _get_bond_info(mol: Molecule) -> List[_BondTuple]: ...

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

Coords = Tuple[float, float, float]

def _iter_rec(atom_array: np.recarray) -> Generator[Tuple[_Properties, Coords, str], None, None]: ...
def _rec_to_mol(atom_array: np.recarray, bond_array: np.recarray, atom_len: Optional[int] = ..., bond_len: Optional[int] = ..., mol: Optional[Molecule] = ...) -> Molecule: ...

class PDBTuple(NamedTuple):
    atoms: np.recarray
    bonds: np.recarray
    atom_count: np.ndarray
    bond_count: np.ndarray
    @classmethod
    def from_molecules(cls: Type[ST], mol_list: Iterable[Molecule], min_atom: int = ..., min_bond: int = ...) -> ST: ...
    @overload
    def to_molecules(self, idx: Union[None, Sequence[int], slice, np.ndarray] = ..., mol: Optional[Iterable[Optional[Molecule]]] = ...) -> List[Molecule]: ...
    @overload
    def to_molecules(self, idx: SupportsIndex = ..., mol: Optional[Molecule] = ...) -> Molecule: ...
