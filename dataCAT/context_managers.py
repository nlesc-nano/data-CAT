"""A module which holds context managers for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.context_managers
.. autosummary::
    OpenYaml
    OpenLig
    OpenQD

API
---
.. autoclass:: OpenYaml
.. autoclass:: OpenLig
.. autoclass:: OpenQD

"""

from os.path import abspath
from abc import abstractmethod, ABCMeta
from types import TracebackType
from typing import (
    Optional, Any, Generic, TypeVar, Tuple, Type, AnyStr, Dict, TYPE_CHECKING, Union
)

import sys
import yaml
import pandas as pd

from scm.plams import Settings
from nanoutils import final

from .df_collection import DFProxy

if TYPE_CHECKING:
    from os import PathLike  # noqa: F401

__all__ = ['OpenYaml', 'OpenLig', 'OpenQD']

T = TypeVar('T')
ST = TypeVar('ST', bound='FileManagerABC')


class FileManagerABC(Generic[AnyStr, T], metaclass=ABCMeta):
    """An abstract baseclass for opening and closing the various database components."""

    if sys.version_info < (3, 7):
        __slots__ = ('_filename', '_write', '_db', '_hash')
    else:
        __slots__ = ('__weakref__', '_filename', '_write', '_db', '_hash')

    _db: T

    @property
    def filename(self) -> AnyStr:
        """:data:`~typing.AnyStr`: Get the name of the to-be opened file."""
        return self._filename

    @property
    def write(self) -> bool:
        """:class:`bool`: Get whether or not :attr:`~FileManagerABC.filename` should be written to when closing the context manager."""  # noqa: E501
        return self._write

    @final
    def __init__(self, filename: Union[AnyStr, 'PathLike[AnyStr]'], write: bool = True) -> None:
        """Initialize the file context manager.

        Parameters
        ----------
        filename : :class:`str`, :class:`bytes` or :class:`os.PathLike`
            A path-like object.
        write : :class:`bool`
            Whether or not the database file should be updated after closing this instance.


        :rtype: :data:`None`

        """
        self._filename: AnyStr = abspath(filename)
        self._write: bool = write

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False
        return self.filename == value.filename and self.write is value.write  # type: ignore

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        return f'{self.__class__.__name__}(filename={self.filename!r}, write={self.write!r})'

    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[AnyStr, bool]]:
        """A helper function for :mod:`pickle`."""
        cls = type(self)
        return cls, (self.filename, self.write)

    def __copy__(self: ST) -> ST:
        """Implement :func:`copy.copy(self)<copy.copy>`."""
        return self

    def __deepcopy__(self: ST, memo: Optional[Dict[int, Any]] = None) -> ST:
        """Implement :func:`copy.deepcopy(self, memo=memo)<copy.deepcopy>`."""
        return self

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        try:
            return self._hash
        except AttributeError:
            self._hash: int = hash(self.__reduce__())
            return self._hash

    @abstractmethod
    def __enter__(self) -> T:
        """Enter the context manager; open, store and return the database."""
        raise NotImplementedError("Trying to call an abstract method")
        self._db: T = ...
        return self._db

    @abstractmethod
    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exit the context manager; close and, optionally, write the database."""
        raise NotImplementedError("Trying to call an abstract method")
        del self._db


class OpenYaml(FileManagerABC[AnyStr, Settings]):
    """Context manager for opening and closing job settings (:attr:`.Database.yaml`).

    Attributes
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    """

    def __enter__(self):
        """Open the :class:`.OpenYaml` context manager, importing :attr:`.settings`."""
        with open(self.filename, 'r', encoding='utf-8') as f:
            self._db = Settings(yaml.load(f, Loader=yaml.FullLoader))
        return self._db

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the :class:`.OpenYaml` context manager, exporting :attr:`.settings`."""
        if self.write:
            yml_dict = self._db.as_dict()
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.write(yaml.dump(yml_dict, default_flow_style=False, indent=4))
        del self._db


class OpenLig(FileManagerABC[AnyStr, DFProxy]):
    """Context manager for opening and closing the ligand database (:attr:`.Database.csv_lig`)."""

    def __enter__(self):
        """Open the :class:`.OpenLig` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
        self._db = df = DFProxy(
            pd.read_csv(self.filename, index_col=[0, 1], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return df

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the :class:`.OpenLig` context manager, exporting :attr:`.df`."""
        if self.write:
            self._db.to_csv(self.filename)
        del self._db


class OpenQD(FileManagerABC[AnyStr, DFProxy]):
    """Context manager for opening and closing the QD database (:attr:`.Database.csv_qd`)."""

    def __enter__(self):
        """Open the :class:`.OpenQD` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'settings': str, 'opt': bool}
        self._db = df = DFProxy(
            pd.read_csv(self.filename, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return df

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the :class:`.OpenQD` context manager, exporting :attr:`.df`."""
        if self.write:
            self._db.to_csv(self.filename)
        del self._db
