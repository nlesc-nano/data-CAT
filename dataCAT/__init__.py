"""A databasing framework for the Compound Attachment Tools package (CAT)."""

# flake8: noqa: F401,E402,N812

from nanoutils import VersionInfo

from .__version__ import __version__

from CAT import version_info as CAT_VERSION
try:
    from nanoCAT import version_info as NANOCAT_VERSION
except ImportError:
    NANOCAT_VERSION = VersionInfo(-1, -1, -1)

version_info = DATACAT_VERSION = VersionInfo.from_str(__version__)
del VersionInfo

from .df_proxy import DFProxy
from .hdf5_log import (
    create_hdf5_log, update_hdf5_log, reset_hdf5_log, log_to_dataframe,
    DT_MAPPING, VERSION_MAPPING
)
from .pdb_array import PDBContainer, ATOM_MAPPING, BOND_MAPPING
from .context_managers import OpenYaml, OpenLig, OpenQD
from .database import Database
from . import functions, testing_utils

__author__ = 'B. F. van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'CAT_VERSION', 'NANOCAT_VERSION', 'DATACAT_VERSION',
    'functions', 'testing_utils',
    'create_hdf5_log', 'update_hdf5_log', 'reset_hdf5_log', 'DT_MAPPING', 'VERSION_MAPPING',
    'DFProxy',
    'PDBContainer', 'ATOM_MAPPING', 'BOND_MAPPING',
    'OpenYaml', 'OpenLig', 'OpenQD',
    'Database',
]
