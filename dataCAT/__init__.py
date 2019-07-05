"""
Data-CAT
========

A databasing framework for the Compound Attachment Tools package (CAT).

"""

from .__version__ import __version__

from .database import Database
from .database_functions import (mol_to_file, df_to_mongo_dict)


__version__ = __version__
__author__ = 'Bas van Beek'
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'Database',
    'mol_to_file', 'df_to_mongo_dict'
]
