"""A databasing framework for the Compound Attachment Tools package (CAT)."""

# noqa: F401

from nanoutils import VersionInfo

from .__version__ import __version__

from .df_collection import DFProxy
from .context_managers import OpenYaml, OpenLig, OpenQD
from .database_functions import df_to_mongo_dict
from .database import Database

version_info = VersionInfo.from_str(__version__)
del VersionInfo

__author__ = 'Bas van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'DFProxy',
    'OpenYaml', 'OpenLig', 'OpenQD',
    'df_to_mongo_dict',
    'Database',
]
