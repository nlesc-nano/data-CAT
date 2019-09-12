"""
dataCAT.context_managers
========================

A module which holds context managers for the :class:`.Database` class.

Index
-----
.. currentmodule:: dataCAT.context_managers
.. autosummary::
    MetaManager
    OpenYaml
    OpenLig
    OpenQD

API
---
.. autoclass:: MetaManager
    :members:
.. autoclass:: OpenYaml
.. autoclass:: OpenLig
.. autoclass:: OpenQD

"""

from os import getcwd, sep
from os.path import basename
from typing import (Callable, Optional, Any)
from collections.abc import Container
from contextlib import AbstractContextManager

import yaml
import pandas as pd

from scm.plams import Settings

from .df_collection import get_df_collection

__all__ = ['MetaManager', 'OpenYaml', 'OpenLig', 'OpenQD']


class MetaManager(Container):
    """A wrapper for context managers.

    Has a single important method, :meth:`MetaManager.open`,
    which calls and returns the context manager stored in :attr:`MetaManager.manager`.

    Note
    ----
    :attr:`MetaManager.filename` will be the first positional argument provided
    to :attr:`MetaManager.manager`.

    Paramaters
    ----------
    filename : str
        The path+filename of a database component
        See :attr:`MetaManager.filename`.

    manager : |type|_ [|AbstractContextManager|_]
        A type object of a context manager.
        TThe first positional argument of the context manager should be the filename.
        See :attr:`MetaManager.manager`.

    Attributes
    ----------
    filename : str
        The path+filename of a database component.

    manager : |type|_ [|AbstractContextManager|_]
        A type object of a context manager.
        The first positional argument of the context manager should be the filename.

    """

    def __init__(self, filename: str,
                 manager: Callable[..., AbstractContextManager]) -> None:
        """Initialize a :class:`MetaManager` instance."""
        self.filename = filename
        self.manager = manager

    def __repr__(self) -> str:
        """Return the canonical string representation of this instance."""
        filename = repr(f'...{sep}{basename(self.filename)}')
        return f'{self.__class__.__name__}(filename={filename}, manager={repr(self.manager)})'

    def __str__(self) -> str:
        """Create a new string object from this instance."""
        args = self.__class__.__name__, repr(self.filename), repr(self.manager)
        return '{}(\n    filename = {},\n    manager  = {}\n)'.format(*args)

    def __contains__(self, value: Any) -> bool:
        """Return if **value** is in :code:`dir(self)`."""
        return value in dir(self)

    def __eq__(self, value: Any) -> bool:
        """Return if this instance is equivalent to **value**."""
        return dir(self) == dir(value)

    def open(self, *args: Any, **kwargs: Any) -> AbstractContextManager:
        """Call and return :attr:`MetaManager.manager`."""
        return self.manager(self.filename, *args, **kwargs)


class OpenYaml(AbstractContextManager):
    """Context manager for opening and closing job settings (:attr:`.Database.yaml`).

    Parameters
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    Attributes
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    settings : |None|_ or |plams.Settings|_
        An attribute for (temporary) storing the opened .yaml file
        (:attr:`OpenYaml.filename`) as :class:`.Settings` instance.

    """

    def __init__(self, filename: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.OpenYaml` context manager."""
        self.filename: str = filename or getcwd()
        self.write: bool = write
        self.settings = None

    def __enter__(self) -> Settings:
        """Open the :class:`.OpenYaml` context manager, importing :attr:`.settings`."""
        with open(self.filename, 'r') as f:
            self.settings = Settings(yaml.load(f, Loader=yaml.FullLoader))
        return self.settings

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.OpenYaml` context manager, exporting :attr:`.settings`."""
        if self.write:
            yml_dict = self.settings.as_dict()

            # A fix for Settings.as_dict() not functioning when containg a lists of Settings
            for key in yml_dict:
                for i, value in enumerate(yml_dict[key]):
                    if isinstance(value, Settings):
                        yml_dict[key][i] = value.as_dict()

            # Write to the .yaml file
            with open(self.filename, 'w') as f:
                f.write(yaml.dump(yml_dict, default_flow_style=False, indent=4))
        self.settings = None
        assert self.settings is None


class OpenLig(AbstractContextManager):
    """Context manager for opening and closing the ligand database (:attr:`.Database.csv_lig`).

    Parameters
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    Attributes
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    df : |None|_ or |pd.DataFrame|_
        An attribute for (temporary) storing the opened .csv file
        (see :attr:`OpenLig.filename`) as a :class:`.DataFrame` instance.

    """

    def __init__(self, filename: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.OpenLig` context manager."""
        self.filename: str = filename or getcwd()
        self.write: bool = write
        self.df: Optional['DFCollection'] = None

    def __enter__(self) -> 'DFCollection':
        """Open the :class:`.OpenLig` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
        self.df = df = get_df_collection(
            pd.read_csv(self.filename, index_col=[0, 1], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return df

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.OpenLig` context manager, exporting :attr:`.df`."""
        if self.write:
            self.df.to_csv(self.filename)
        self.df = None


class OpenQD(AbstractContextManager):
    """Context manager for opening and closing the QD database (:attr:`.Database.csv_qd`).

    Parameters
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    Attributes
    ----------
    filename : str
        The path+filename to the database component.

    write : bool
        Whether or not the database file should be updated after closing this instance.

    df : |None|_ or |pd.DataFrame|_
        An attribute for (temporary) storing the opened .csv file
        (:attr:`OpenQD.filename`) as :class:`.DataFrame` instance.

    """

    def __init__(self, filename: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.OpenQD` context manager."""
        self.filename: str = filename or getcwd()
        self.write: bool = write
        self.df: Optional['DFCollection'] = None

    def __enter__(self) -> 'DFCollection':
        """Open the :class:`.OpenQD` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'settings': str, 'opt': bool}
        self.df = df = get_df_collection(
            pd.read_csv(self.filename, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return df

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.OpenQD` context manager, exporting :attr:`.df`."""
        if self.write:
            self.df.to_csv(self.filename)
        self.df = None
