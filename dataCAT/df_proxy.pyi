import sys
import pandas as pd
from pandas.core.generic import NDFrame
from typing import Any, ClassVar, Dict, FrozenSet, NoReturn, Tuple, Type, TypeVar, List

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final

__all__: List[str] = ...

TT = TypeVar('TT', bound=_DFMeta)

class _DFMeta(type):
    MAGIC: FrozenSet[str] = ...
    SETTERS: FrozenSet[str] = ...
    NDTYPE: Type[NDFrame] = ...
    @staticmethod
    def _construct_getter(cls_name: str, func_name: str) -> property: ...
    @staticmethod
    def _construct_setter(prop: property, cls_name: str, func_name: str) -> property: ...
    def __new__(mcls: Type[TT], name: str, bases: Tuple[type, ...], namespace: Dict[str, Any]) -> TT: ...

@final
class DFProxy(pd.DataFrame, metaclass=_DFMeta):
    NDTYPE: ClassVar[Type[NDFrame]] = ...
    ndframe: pd.DataFrame
    def __init__(self, ndframe: pd.DataFrame) -> None: ...
    def __reduce__(self) -> NoReturn: ...
