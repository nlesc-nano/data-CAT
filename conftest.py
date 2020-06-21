"""A pytest ``conftest.py`` file."""

import logging
from itertools import chain
from typing import Any

from scm.plams import add_to_instance
from assertionlib import assertion

MAX_N: int = 8


@add_to_instance(assertion)
def repr_instance(self, obj: Any, level: int) -> str:
    """Return a string-representation of **obj**."""
    if level <= 0:
        return f'{obj.__class__.__name__}(...)'

    ret = repr(obj)
    if ret.count('\n') < MAX_N:
        return ret

    # Split the to-be returned string in a top and bottom half
    i = MAX_N // 2
    ret_list = ret.split('\n')
    top, bottom = ret_list[:i], ret_list[-i:]

    # Determine the indentation of the ellipsis
    _top = top[-1]
    indent = len(_top) - len(_top.lstrip(' '))
    mid = indent * " " + '...'

    return '\n'.join(i for i in chain(top, mid, bottom))


def pytest_configure(config: Any) -> None:
    """Flake8 is very verbose by default. Silence it.

    See Also
    --------
    https://github.com/eisensheng/pytest-catchlog/issues/59

    """
    logging.getLogger("flake8").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.WARNING)
