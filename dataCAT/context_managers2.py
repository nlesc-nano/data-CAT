"""A module which holds context managers for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT
.. autosummary::
    GroupManager

API
---
.. autoclass:: GroupManager

"""

import sys
from os.path import abspath
from abc import abstractmethod, ABCMeta
from types import TracebackType, MappingProxyType
from typing import (
    Optional, Any, Generic, TypeVar, Tuple, Type, AnyStr, Dict, TYPE_CHECKING, Union, Mapping
)

import h5py
import pandas as pd

if TYPE_CHECKING:
    from os import PathLike  # noqa: F401

__all__ = ['Hdf5Manager']

ST = TypeVar('ST', bound='GroupManager')


class GroupManager(Generic[AnyStr]):
    """An abstract baseclass for opening and closing the various database components."""

    if sys.version_info < (3, 7):
        __slots__ = ('_hash', '_filename', '_group', '_file', '_args', '_kwargs')
    else:
        __slots__ = ('__weakref__', '_hash', '_filename', '_group', '_file', '_args', '_kwargs')

    @property
    def filename(self) -> AnyStr:
        """:data:`~typing.AnyStr`: Get the name of the to-be opened file."""
        return self._filename

    @property
    def group(self) -> str:
        """:class:`str`: Get the name of the to-be opened file."""
        return self._group

    @property
    def args(self) -> Tuple[Any, ...]:
        """:data:`Tuple[Any, ...]<typing.Tuple>`: Get a tuple with all positional arguments for :class:`h5py.File`."""
        return self._args

    @property
    def kwargs(self) -> Mapping[str, Any]:
        """:data:`Mapping[str, Any]<typing.Mapping>`: Get a mapping with all keyword arguments for :class:`h5py.File`."""
        return self._kwargs

    def __init__(self, filename: Union[AnyStr, 'PathLike[AnyStr]'],
                 group: str, *args: Any, **kwargs: Any) -> None:
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
        self._group: str = group
        self._args: Tuple[Any, ...] = args
        self._kwargs: Mapping[str, Any] = MappingProxyType(kwargs)

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False
        return self.filename == value.filename and self.group == value.group  # type: ignore

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        return f'{self.__class__.__name__}(filename={self.filename!r}, group={self.group!r})'

    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[AnyStr, str]]:
        """A helper function for :mod:`pickle`."""
        cls = type(self)
        return cls, (self.filename, self.group)

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
    def __enter__(self) -> h5py.Group:
        """Enter the context manager; open, store and return the database."""
        try:
            self._file.open()
        except AttributeError:
            self._file: h5py.File = h5py.File(self.filename, *self.args, **self.kwargs)
        return self._file[self.group]

    def __exit__(self, exc_type: Optional[Type[BaseException]],
                 exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        """Exit the context manager; close and, optionally, write the database."""
        self._file.close()
