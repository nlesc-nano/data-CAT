"""A databasing framework for the Compound Attachment Tools package (CAT)."""

# flake8: noqa: F401,E402

from nanoutils import VersionInfo

from .__version__ import __version__

version_info = VersionInfo.from_str(__version__)
del VersionInfo

from .functions import df_to_mongo_dict
from .df_proxy import DFProxy
from .pdb_array import DTYPE_ATOM, DTYPE_BOND, PDBContainer
from .context_managers import OpenYaml, OpenLig, OpenQD
from .database import Database

__author__ = 'B. F. van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'df_to_mongo_dict',
    'DFProxy',
    'DTYPE_ATOM', 'DTYPE_BOND', 'PDBContainer',
    'OpenYaml', 'OpenLig', 'OpenQD',
    'Database',
]
