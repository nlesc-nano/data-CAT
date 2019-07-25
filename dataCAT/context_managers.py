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

from os import getcwd
from typing import (Callable, Optional)
from contextlib import AbstractContextManager
from dataclasses import dataclass

import yaml
import pandas as pd

from scm.plams import Settings

from .df_collection import DFCollection

__all__ = ['MetaManager', 'OpenYaml', 'OpenLig', 'OpenQD']


@dataclass(frozen=True)
class MetaManager:
    """A wrapper for context managers.

    Has a single important method, :meth:`MetaManager.open`,
    which calls and returns the context manager stored in :attr:`MetaManager.manager`.

    Note
    ----
    :attr:`MetaManager.filename` will be the first argument provided to :attr:`MetaManager.manager`.

    Paramaters
    ----------
    filename : str
        The path+filename of a database component (see :attr:`MetaManager.filename`).

    Attributes
    ----------
    filename : str
        The path+filename of a database component.

    manager : |type|_ [|AbstractContextManager|_]
        A type object of a context manager.
        The provided context manager should have access to the **filename** argument.

    """

    filename: str
    manager: Callable[..., AbstractContextManager]

    def open(self, *args, **kwargs) -> AbstractContextManager:
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

    def __init__(self, path: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.open_yaml` context manager."""
        self.path: str = path or getcwd()
        self.write: bool = write
        self.settings: Optional[Settings] = None

    def __enter__(self) -> Settings:
        """Open the :class:`.open_yaml` context manager, importing :attr:`.settings`."""
        with open(self.filename, 'r') as f:
            self.settings = Settings(yaml.load(f, Loader=yaml.FullLoader))
        return self.settings

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.open_yaml` context manager, exporting :attr:`.settings`."""
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
        (see :attr:`OpenCsvLig.filename`) as a :class:`.DataFrame` instance.

    """

    def __init__(self, path: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.OpenCsvLig` context manager."""
        self.path: str = path or getcwd()
        self.write: bool = write
        self.df: Optional[DFCollection] = None

    def __enter__(self) -> DFCollection:
        """Open the :class:`.OpenCsvLig` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'formula': str, 'settings': str, 'opt': bool}
        self.df = df = DFCollection(
            pd.read_csv(self.path, index_col=[0, 1], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return self.df

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.OpenCsvLig` context manager, exporting :attr:`.df`."""
        if self.write:
            self.df.to_csv(self.path)
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
        (:attr:`OpenCsvQd.filename`) as :class:`.DataFrame` instance.

    """

    def __init__(self, path: Optional[str] = None,
                 write: bool = True) -> None:
        """Initialize the :class:`.OpenCsvQd` context manager."""
        self.path: str = path or getcwd()
        self.write: bool = write
        self.df: Optional[DFCollection] = None

    def __enter__(self) -> DFCollection:
        """Open the :class:`.OpenCsvQd` context manager, importing :attr:`.df`."""
        # Open the .csv file
        dtype = {'hdf5 index': int, 'settings': str, 'opt': bool}
        self.df = df = DFCollection(
            pd.read_csv(self.path, index_col=[0, 1, 2, 3], header=[0, 1], dtype=dtype)
        )

        # Fix the columns
        idx_tups = [(i, '') if 'Unnamed' in j else (i, j) for i, j in df.columns]
        df.columns = pd.MultiIndex.from_tuples(idx_tups, names=df.columns.names)
        return DFCollection(self.df)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the :class:`.OpenCsvQD` context manager, exporting :attr:`.df`."""
        if self.write:
            self.df.to_csv(self.path)
        self.df = None
