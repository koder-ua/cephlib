import re
import copy
from typing import Tuple, Optional, cast, List, NamedTuple

import numpy


ndarray1d = numpy.ndarray
ndarray2d = numpy.ndarray


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


ArrayData = NamedTuple("ArrayData",
                       [('header', List[str]),
                        ('histo_bins', Optional[numpy.ndarray]),
                        ('data', Optional[numpy.ndarray])])


class TimeSeries:
    """Data series from sensor - either system sensor or from load generator tool (e.g. fio)"""

    def __init__(self, data: numpy.ndarray, times: ndarray1d, units: str, time_units: str, source: DataSource,
                 histo_bins: ndarray1d = None) -> None:
        self.units = units
        self.time_units = time_units

        self.times = times
        self.data = data

        self.source = source
        self.histo_bins = histo_bins

    def __str__(self) -> str:
        return "TS(src={}, time_size={}, dshape={}):\n".format(self.source, len(self.times), *self.data.shape)

    def __repr__(self) -> str:
        return str(self)

    def copy(self, no_data: bool = False) -> 'TimeSeries':
        cp = copy.copy(self)

        if not no_data:
            cp.times = self.times.copy()
            cp.data = self.data.copy()

        cp.source = self.source()
        return cp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSeries):
            return False

        o = cast(TimeSeries, other)

        return o.units == self.units and \
               o.time_units == self.time_units and \
               numpy.array_equal(o.data, self.data) and \
               numpy.array_equal(o.times, self.times) and \
               o.source == self.source and \
               ((self.histo_bins is None and o.histo_bins is None) or numpy.array_equal(self.histo_bins, o.histo_bins))


