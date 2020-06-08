"""Tests for the :class:`dataCAT.DFProxy` class."""

from itertools import chain

import numpy as np
import pandas as pd

from assertionlib import assertion
from dataCAT import DFProxy
from dataCAT.df_collection import _DFMeta

_DF = pd.DataFrame(np.random.rand(3, 10))
DF = DFProxy(_DF)


def test_dfproxy() -> None:
    """Test :class:`dataCAT.DFProxy`."""
    name_iterator = (i for i in dir(DFProxy.NDTYPE) if not i.startswith('_'))
    iterator = ((name, getattr(DF, name), getattr(_DF, name)) for
                name in chain(_DFMeta.MAGIC, name_iterator))

    for name, func1, func2 in iterator:
        assertion.is_(func1, func2, message=name)
