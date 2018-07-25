import abc
import itertools
from typing import Any, IO, Tuple, List, Dict, Iterator, Union, Optional, Type, TypeVar, Iterable
from .types import DataSource, NumVector


FieldsDct = Dict[str, Union[str, List[str]]]
SensorIDS = Iterable[Dict[str, str]]


class PathSelector:
    def __init__(self) -> None:
        self.extra_mappings = {}  # type: Dict[str, Dict[str, List[FieldsDct]]]
        # new_field_name:
        #     new_field_value:
        #         - {old_field1: old_value1(s), ...}
        #         - {old_field2: old_value2(s), ...}
        #         ....

    def add_mapping(self, param: str, val: str, **fields: Union[str, List[str]]) -> None:
        self.extra_mappings.setdefault(param, {}).setdefault(val, []).append(fields)

    def __call__(self, **mapping: Union[str, List[str]]) -> Iterator[Dict[str, str]]:
        # final cache may be created: mapping => result
        if not mapping:
            yield {}
        else:
            to_lst = lambda x: x if isinstance(x, list) else [x]
            extra = {}  # type: Dict[str, List[str]]
            mapping_lst = {key: to_lst(val) for key, val in mapping.items()}
            cleared_mapping = mapping_lst.copy()
            for name, vals in mapping_lst.items():
                if name in self.extra_mappings:
                    extra[name] = to_lst(cleared_mapping.pop(name))

            if not extra:
                assert cleared_mapping == mapping_lst
                keys, vals = zip(*mapping_lst.items())
                for combination in itertools.product(*vals):
                    yield dict(zip(keys, combination))
            else:
                for extra_name, extra_vals in extra.items():
                    for extra_val in extra_vals:
                        for pdict in self.extra_mappings[extra_name][extra_val]:
                            params = cleared_mapping.copy()  # type: Dict[str, List[str]]
                            for pkey, pval in pdict.items():
                                assert pkey not in params
                                params[pkey] = pval
                            yield from self(**params)


class IStorable(metaclass=abc.ABCMeta):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self) -> Dict[str, Any]:
        pass

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        pass


class Storable(IStorable):
    """Default implementation"""

    __ignore_fields__ = []  # type: List[str]

    def raw(self) -> Dict[str, Any]:
        return {name: val
                for name, val in self.__dict__.items()
                if not name.startswith("_") and name not in self.__ignore_fields__}

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'IStorable':
        obj = cls.__new__(cls)
        if cls.__ignore_fields__:
            data = data.copy()
            data.update(dict.fromkeys(cls.__ignore_fields__))
        obj.__dict__.update(data)
        return obj


Basic = Union[int, str, bytes, bool, None]
StorableType = Union[IStorable, Dict[str, Any], List[Any], int, str, bytes, bool, None]


class ISerializer(metaclass=abc.ABCMeta):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value: StorableType) -> bytes:
        pass

    @abc.abstractmethod
    def unpack(self, data: bytes) -> Any:
        pass


class _Raise:
    pass


class ISimpleStorage(metaclass=abc.ABCMeta):
    """interface for low-level storage, which doesn't support serialization
    and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def rm(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "rb+") -> IO:
        pass

    @abc.abstractmethod
    def get_fname(self, path: str) -> str:
        pass

    @abc.abstractmethod
    def sub_storage(self, path: str) -> 'ISimpleStorage':
        pass

    @abc.abstractmethod
    def list(self, path: str) -> Iterator[Tuple[bool, str]]:
        pass


ObjClass = TypeVar('ObjClass', bound=IStorable)


class IStorageNNP(metaclass=abc.ABCMeta):
    other_caches = None  # type: Dict[str, Dict]

    @abc.abstractmethod
    def __init__(self, sstorage: ISimpleStorage, serializer: ISerializer) -> None:
        pass

    @abc.abstractmethod
    def sub_storage(self, *path: str) -> 'IStorageNNP':
        pass

    @abc.abstractmethod
    def put(self, value: Any, *path: str) -> None:
        pass

    @abc.abstractmethod
    def get(self, path: str, default: Any = _Raise) -> Any:
        pass

    @abc.abstractmethod
    def rm(self, *path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, obj_class: Type[ObjClass], *path: str) -> ObjClass:
        pass

    # ---------------  List of values ----------------------------------------------------------------------------------

    @abc.abstractmethod
    def put_list(self, value: Iterable[IStorable], *path: str) -> None:
        pass

    @abc.abstractmethod
    def load_list(self, obj_class: Type[ObjClass], *path: str) -> List[ObjClass]:
        pass

    @abc.abstractmethod
    def __contains__(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def put_raw(self, val: bytes, *path: str) -> str:
        pass

    @abc.abstractmethod
    def get_fname(self, fpath: str) -> str:
        pass

    @abc.abstractmethod
    def get_raw(self, *path: str) -> bytes:
        pass

    @abc.abstractmethod
    def append_raw(self, value: bytes, *path: str) -> None:
        pass

    @abc.abstractmethod
    def get_fd(self, path: str, mode: str = "r") -> IO:
        pass

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> 'IStorageNNP':
        pass

    @abc.abstractmethod
    def __exit__(self, x: Any, y: Any, z: Any) -> None:
        pass

    @abc.abstractmethod
    def list(self, *path: str) -> Iterator[Tuple[bool, str]]:
        pass

    @abc.abstractmethod
    def iter_paths(self, root: str, path_parts: List[str],
                   already_found_groups: Dict[str, str]) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        pass

    # --------------  Arrays -------------------------------------------------------------------------------------------

    @abc.abstractmethod
    def put_array(self, path: str,
                  data: NumVector,
                  header: List[str],
                  header2: NumVector = None,
                  append_on_exists: bool = False) -> None:
        pass



SensorsIter = Iterable[Dict[str, str]]


class ISensorStorageNNP(metaclass=abc.ABCMeta):
    storage = None  # type: IStorageNNP
    locator = None  # type: PathSelector

    ts_arr_tag = 'csv'
    csv_file_encoding = 'utf8'

    @abc.abstractmethod
    def sync(self) -> None:
        pass

    @abc.abstractmethod
    def add_mapping(self, param: str, val: str, **fields: Union[str, List[str]]) -> None:
        pass

    @abc.abstractmethod
    def append_sensor(self, data: NumVector, ds: DataSource, units: str) -> None:
        pass

    @abc.abstractmethod
    def iter_paths(self, path_templ: str) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        pass

    @abc.abstractmethod
    def iter_sensors(self, **vals) -> Iterator[DataSource]:
        pass


class IImagesStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check_plot_file(self, source: DataSource) -> Optional[str]:
        pass

    @abc.abstractmethod
    def put_plot_file(self, data: bytes, source: DataSource) -> str:
        pass
