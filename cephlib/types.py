import re
import array
from typing import TypeVar, List, Union, Optional, Tuple, cast, Any

try:
    import numpy
    NumVector = Union[numpy.ndarray, array.array, List[int], List[float]]
except ImportError:
    NumVector = Union[array.array, List[int], List[float]]


TNumber = TypeVar('TNumber', int, float)
Number = Union[int, float]

array1d = NumVector
array2d = NumVector


class DataStorageTags:
    node_id = r'\d+.\d+.\d+.\d+:\d+'
    job_id = r'[-a-zA-Z0-9_]+_\d+'
    suite_id = r'[a-z_]+_\d+'
    sensor = r'[-a-z_]+'
    dev = r'[-a-zA-Z0-9_]+'
    metric = r'[-a-z_]+'
    tag = r'[a-z_.]+'


DataStorageTagsDct = {name: r"(?P<{}>{})".format(name, rr)
                      for name, rr in DataStorageTags.__dict__.items()
                      if not name.startswith("__")}


class DataSource:
    def __init__(self, suite_id: str = None, job_id: str = None, node_id: str = None,
                 sensor: str = None, dev: str = None, metric: str = None, tag: str = None) -> None:
        self.suite_id = suite_id
        self.job_id = job_id
        self.node_id = node_id
        self.sensor = sensor
        self.dev = dev
        self.metric = metric
        self.tag = tag

    @property
    def metric_fqdn(self) -> str:
        return "{0.sensor}.{0.dev}.{0.metric}".format(self)

    def verify(self):
        for attr_name, attr_val in self.__dict__.items():
            if '__' not in attr_name and attr_val is not None:
                assert re.match(getattr(DataStorageTags, attr_name) + "$", attr_val), \
                    "Wrong field in DataSource - {}=={!r}".format(attr_name, attr_val)

    def __call__(self, **kwargs) -> 'DataSource':
        dct = self.__dict__.copy()
        dct.update(kwargs)
        return self.__class__(**dct)

    def __str__(self) -> str:
        return ("suite={0.suite_id},job={0.job_id},node={0.node_id}," +
                "path={0.sensor}.{0.dev}.{0.metric},tag={0.tag}").format(self)

    def __repr__(self) -> str:
        return str(self)

    @property
    def tpl(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str],
                           Optional[str], Optional[str], Optional[str]]:
        return self.suite_id, self.job_id, self.node_id, self.sensor, self.dev, self.metric, self.tag

    def __eq__(self, o: object) -> bool:
        return self.tpl == cast(DataSource, o).tpl

    def __hash__(self) -> int:
        return hash(self.tpl)


def get_arr_info(obj: Any) -> Tuple[str, List[int]]:
    if numpy is not None:
        if isinstance(obj, numpy.ndarray):
            return obj.dtype.name, list(obj.shape)

    if isinstance(obj, array.array):
        return ({'f': 'float32', 'd': 'float64', 'b': 'int8', 'B': 'uint8',
                 'h': 'int16', 'H': 'uint16', 'i': 'int16', 'I': 'uint16',
                 'l': 'int32', 'L': 'uint32', 'q': 'int64', 'Q': 'uint64'}[obj.typecode], [len(obj)])

    shape = []
    while isinstance(obj, (list, tuple)):
        shape.append(len(obj))
        obj = obj[0]

    if isinstance(obj, int):
        return 'int64', shape

    assert isinstance(obj, float)
    return 'float64', shape