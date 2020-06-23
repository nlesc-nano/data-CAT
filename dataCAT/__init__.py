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
from .pdb_array import DTYPE_ATOM, DTYPE_BOND, PDBContainer
from .context_managers import OpenYaml, OpenLig, OpenQD
from .database import Database
from . import functions, testing_utils

__author__ = 'B. F. van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'CAT_VERSION', 'NANOCAT_VERSION', 'DATACAT_VERSION',
    'functions', 'testing_utils',
    'DFProxy',
    'DTYPE_ATOM', 'DTYPE_BOND', 'PDBContainer',
    'OpenYaml', 'OpenLig', 'OpenQD',
    'Database',
]
