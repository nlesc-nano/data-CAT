"""Tests for :mod:`dataCAT.context_managers`."""

from os.path import join
from functools import partial

import numpy as np
import pandas as pd

from assertionlib import assertion

from dataCAT import OpenYaml, OpenLig, OpenQD

PATH = join('tests', 'test_files')
PATH1 = join(PATH, 'qd.csv')


def test_openyaml() -> None:
    """Test :class:`dataCAT.context_managers.OpenYaml`."""
    path = join(PATH, 'settings.yaml')
    manager = partial(OpenYaml, path)

    ref = {None: [None], 'RDKit_2019.03.2': ['UFF'], 'type': [{'ams': {'GeometryOptimization': {'MaxIterations': 500}, 'Task': 'GeometryOptimization'}, 'input': {'ams': {'constraints': {'atom': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250]}, 'system': {'bondorders': {'_1': None}}}, 'uff': {'library': 'uff'}}}]}  # noqa
    with manager(write=False) as s:
        assertion.eq(s, ref)


def test_openlig() -> None:
    """Test :class:`dataCAT.context_managers.OpenLig`."""
    path = join(PATH, 'ligand.csv')
    manager = partial(OpenLig, path)

    idx = pd.MultiIndex.from_tuples(
        [('-', '-'), ('C[O-]', 'O2'), ('CC[O-]', 'O3'), ('CCC[O-]', 'O4')],
        names=['smiles', 'anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('formula', ''), ('hdf5 index', ''), ('opt', ''), ('settings', '1')],
        names=['index', 'sub-index']
    )

    data = np.array(
        [['str', -1, False, 'str'],
         ['C1H3O1', 2, True, 'RDKit_2019.03.2 0'],
         ['C2H5O1', 1, True, 'RDKit_2019.03.2 0'],
         ['C3H7O1', 0, True, 'RDKit_2019.03.2 0']], dtype=object
    )

    ref = pd.DataFrame(data, idx, columns)
    ref['opt'] = ref['opt'].astype(bool)
    ref['hdf5 index'] = ref['hdf5 index'].astype(int)

    with manager(write=False) as df:
        for k in ref:
            np.testing.assert_array_equal(df[k], ref[k])
        np.testing.assert_array_equal(df.index, ref.index)
        np.testing.assert_array_equal(df.columns, ref.columns)


def test_openqd() -> None:
    """Test :class:`dataCAT.context_managers.OpenQD`."""
    path = join(PATH, 'qd.csv')
    manager = partial(OpenQD, path)

    idx = pd.MultiIndex.from_tuples(
        [('-', '-', '-', '-'),
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'CCC[O-]', 'O4'),  # noqa
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'CC[O-]', 'O3'),  # noqa
         ('Cd68Cl26Se55', '124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149', 'C[O-]', 'O2')],  # noqa
        names=['core', 'core anchor', 'ligand smiles', 'ligand anchor']
    )

    columns = pd.MultiIndex.from_tuples(
        [('hdf5 index', ''), ('ligand count', ''), ('opt', ''), ('settings', '1'),
         ('settings', '2')],
        names=['index', 'sub index']
    )

    data = np.array(
        [[-1, -1, False, 'str', 'str'],
         [0, -1, True, 'type 0', 'type 0'],
         [1, -1, True, 'type 0', 'type 0'],
         [2, -1, True, 'type 0', 'type 0']], dtype=object
    )

    ref = pd.DataFrame(data, idx, columns)
    ref['opt'] = ref['opt'].astype(bool)
    ref['hdf5 index'] = ref['hdf5 index'].astype(int)
    ref['ligand count'] = ref['ligand count'].astype(int)

    with manager(write=False) as df:
        for k in ref:
            np.testing.assert_array_equal(df[k], ref[k])
        np.testing.assert_array_equal(df.index, ref.index)
        np.testing.assert_array_equal(df.columns, ref.columns)
