"""A databasing framework for the Compound Attachment Tools package (CAT)."""

from .__version__ import __version__

from .df_collection import DFProxy
from .database import Database
from .database_functions import df_to_mongo_dict

__author__ = 'Bas van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'DFProxy',
    'Database',
    'df_to_mongo_dict'
]
