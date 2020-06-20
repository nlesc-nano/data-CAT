"""A pytest ``conftest.py`` file."""

import logging
from typing import Any, TYPE_CHECKING

from scm.plams import add_to_instance
from assertionlib import assertion

if TYPE_CHECKING:
    import dataCAT


@add_to_instance(assertion)
def repr_PDBContainer(self, obj: 'dataCAT.PDBContainer', level: int) -> str:
    """Return a string-representation of **obj**."""
    return repr(obj)


@add_to_instance(assertion)
def repr_Database(self, obj: 'dataCAT.Database', level: int) -> str:
    """Return a string-representation of **obj**."""
    return repr(obj)


def pytest_configure(config: Any) -> None:
    """Flake8 is very verbose by default. Silence it.

    See Also
    --------
    https://github.com/eisensheng/pytest-catchlog/issues/59

    """
    logging.getLogger("flake8").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.WARNING)
