import abc
from typing import IO, Union, NamedTuple, Iterable, List, Tuple, Dict, Any, Callable, NewType


class IPathSelector:
    def __str__(self) -> str:
        pass


SPath = Union[str, IPathSelector]


StorageItem = NamedTuple('StorageItem', [('name', str), ('is_file', bool)])


class IStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, path: SPath) -> bytes:
        pass

    @abc.abstractmethod
    def __setitem__(self, path: SPath, val: bytes):
        pass

    @abc.abstractmethod
    def __delitem__(self, path: SPath):
        pass

    @abc.abstractmethod
    def __contains__(self, path: SPath) -> bool:
        pass

    @abc.abstractmethod
    def getfd(self, path: SPath, mode: str) -> IO:
        pass

    @abc.abstractmethod
    def substorage(self, path: SPath) -> 'IStorage':
        pass

    @abc.abstractmethod
    def list(self, path: SPath) -> Iterable[StorageItem]:
        pass

    @abc.abstractmethod
    def walk(self, path: SPath) -> Iterable[List[StorageItem]]:
        pass


class _AttredLocator(metaclass=abc.ABCMeta):
    def __init__(self, selector: 'ISelector', attr_map: Any) -> None:
        self.selector = selector
        self.attr_map = attr_map

    def __getattr__(self, name: str) -> Callable:
        path = getattr(self.attr_map, name)
        def closure(storage: IStorage, **params: str) -> Iterable[Tuple[Dict[str, str], StorageItem]]:
            return self.selector.find(storage, path, **params)
        return closure


class ISelector(metaclass=abc.ABCMeta):
    def __init__(self, attrs_obj: Any) -> None:
        self.attrs_obj = attrs_obj

    def set_mapping(self, name: str, ext: List[Dict[str, str]]):
        pass

    def find(self, storage: IStorage, name: str, **attrs: str) -> Iterable[Tuple[Dict[str, str], StorageItem]]:
        pass

    def loc(self) -> _AttredLocator:
        return _AttredLocator(self, self.attrs_obj)


NumArray = NewType('NumArray', Any)


class IArrayStorage(metaclass=abc.ABCMeta):
    storage = None  # type: IStorage

    @abc.abstractmethod
    def put_array(self, path: SPath, data: NumArray, header: List[str], header2: NumArray = None,
                  append_on_exists: bool = False) -> None:
        pass

    @abc.abstractmethod
    def get_array(self, path: str) -> Tuple[NumArray, List[str], Any]:
        pass

