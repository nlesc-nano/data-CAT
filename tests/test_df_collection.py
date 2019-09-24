"""Tests for the :class:`dataCAT.df_collection.DFCollection` class."""

import numpy as np
import pandas as pd

from CAT.assertion.assertion_manager import assertion
from dataCAT.df_collection import get_df_collection

_DF = pd.DataFrame(np.random.rand(3, 10))
DF = get_df_collection(_DF)

REF_CAT = ('__init__', '__getattribute__', '__setattr__', '__str__', '__repr__', 'df')
REF_PANDAS = (
    '__abs__', '__add__', '__and__', '__contains__', '__div__', '__eq__', '__floordiv__',
    '__ge__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__ifloordiv__',
    '__imod__', '__imul__', '__ior__', '__ipow__', '__isub__', '__iter__', '__itruediv__',
    '__ixor__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__', '__mul__', '__or__',
    '__pow__', '__setitem__', '__sub__', '__truediv__', '__xor__',
)


def test_init() -> None:
    """Test :meth:`.DFCollection.__init__`."""
    for key in REF_PANDAS:
        assertion.hasattr(DF, key)

    for key in REF_CAT:
        assertion.hasattr(DF, key)


def test_getattribute() -> None:
    """Test :meth:`.DFCollection.__getattribute__`."""
    for key in REF_PANDAS:
        method = getattr(DF, key)
        assertion.is_(method.__self__, _DF)

    for key in REF_CAT[:-1]:
        method = getattr(DF, key)
        assertion.is_(method.__self__, DF)


def test_setattr() -> None:
    """Test :meth:`.DFCollection.__setattr__`."""
    df1 = get_df_collection(pd.DataFrame(np.random.rand(3, 10)))
    idx = pd.RangeIndex(10, 20)
    df1.columns = idx
    assertion.is_(df1.columns, idx)

    _df = pd.DataFrame(np.ones([3, 10]))
    df2 = get_df_collection(_df)
    assertion.is_(df2.df, _df)


def test_repr() -> None:
    """Test :meth:`.DFCollection.__repr__`."""
    out = repr(DF)
    assertion.eq(out[:48], 'DFCollection(df=<pandas.core.frame.DataFrame at ')
    assertion.eq(out[-2:], '>)')


def test_str() -> None:
    """Test :meth:`.DFCollection.__str__`."""
    df = get_df_collection(pd.DataFrame(np.ones([3, 10])))
    out = str(df)
    ref = (
        'DFCollection(\n         0    1    2    3    4    5    6    7    8    9\n'
        '    0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0\n'
        '    1  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0\n'
        '    2  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0  1.0\n)'
    )
    assertion.eq(out, ref)
