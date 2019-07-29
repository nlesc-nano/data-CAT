"""Tests for :mod:`dataCAT.database_functions`."""

import numpy as np
import pandas as pd

from CAT.assertion_functions import assert_hasattr, assert_eq, assert_id
from dataCAT.database_functions import (
    get_nan_row, as_pdb_array, from_pdb_array, sanitize_yaml_settings, even_index
)


def test_get_nan_row() -> None:
    """Test :func:`dataCAT.database_functions.get_nan_row`."""
    pass
