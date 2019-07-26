"""
dataCAT.df_container
====================

A module which holds the :class:`DFCollection` class, a mutable collection for holding DataFrames.

Index
-----
.. currentmodule:: dataCAT.df_container
.. autosummary::
    DFCollection

API
---
.. autoclass:: DFCollection
    :members:
    :private-members:
    :special-members:

"""

from typing import (Any, Iterator)
from textwrap import indent
from collections.abc import Collection

import pandas as pd

__all__ = ['get_df_collection']

_MAGIC_METHODS = (
    '__abs__', '__add__', '__and__', '__contains__', '__div__', '__eq__', '__floordiv__', '__ge__',
    '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__', '__imod__',
    '__imul__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__',
    '__len__', '__lt__', '__matmul__', '__mod__', '__mul__', '__or__', '__pow__', '__setitem__',
    '__sub__', '__truediv__', '__xor__', '__contains__', '__iter__', '__len__'
)


def _is_magic(key: str) -> bool:
    """Check if **key** belongs to a magic method."""
    return key.startswith('__') and key.endswith('__')


def get_df_collection(df: pd.DataFrame):
    """Return a mutable collection for holding dataframes.

    Paramaters
    ----------
    df : |pd.DataFrame|_
        A Pandas DataFrame.

    Returns
    -------
    A :class:`DFCollection` instance.

    Note
    ----
    As the :class:`DFCollection` class is defined within the scope of this function,
    two instances of :class:`DFCollection` will *not* belong to the same class (see example below).
    In more technical terms: The class bound to a particular :class:`DFCollection` instance is a
    unique instance of :class:`type`.

    .. code:: python

        >>> import numpy as np
        >>> import pandas as pd

        >>> df = pd.DataFrame(np.random.rand(5, 5))

        >>> collection1 = get_df_collection(df)
        >>> collection2 = get_df_collection(df)
        >>> print(collection1.__class__.__name__ == collection2.__class__.__name__)
        True

        >>> print(collection1.__class__ == collection2.__class__)
        False

    """
    class DFCollection:
        """A mutable collection for holding dataframes.

        Paramaters
        ----------
        df : |pd.DataFrame|_
            A Pandas DataFrame (see :attr:`DFCollection.df`).

        Attributes
        ----------
        df : |pd.DataFrame|_
            A Pandas DataFrame (see :attr:`DFCollection.df`).

        """

        def __init__(self, df: pd.DataFrame) -> None:
            """Initialize the :class:`DFCollection` instance."""
            super().__setattr__('df', df)

        def __getattribute__(self, key: str) -> Any:
            """Call :meth:`pd.DataFrame.__getattribute__` unless a magic method is provided."""
            if key == 'df' or _is_magic(key):
                return super().__getattribute__(key)
            else:
                return self.df.__getattribute__(key)

        def __setattr__(self, key: str, value: Any) -> None:
            """Call :meth:`pd.DataFrame.__setattr__` unless a ``"df"`` is provided as **key**."""
            if key == 'df':
                super().__setattr__(key, value)
                for k in _MAGIC_METHODS:  # Populate this instance with **df** magic methods
                    method = getattr(value, k)
                    setattr(self.__class__, k, method)
            else:
                self.df.__setattr__(key, value)

        def __repr__(self) -> str:
            """Return a machine string representation of this instance."""
            df = self.df
            args = (
                self.__class__.__name__, df.__class__.__module__, df.__class__.__name__, hex(id(df))
            )
            return '{}(df=<{}.{} at {}>)'.format(*args)

        def __str__(self) -> str:
            """Return a human string representation of this instance."""
            df = indent(str(self.df), '    ', bool)
            return f'{self.__class__.__name__}(\n{df}\n)'

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df expects an instance of 'pandas.DataFrame'; "
                        "observed type: '{df.__class__.__name__}'")

    ret = DFCollection
    for key in _MAGIC_METHODS:
        method = getattr(df, key)
        setattr(ret, key, method)
    Collection.register(ret)
    return ret(df)
