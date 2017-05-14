import copy
from typing import Optional, cast, List, NamedTuple

import numpy

from .types import DataSource

ndarray1d = numpy.ndarray
ndarray2d = numpy.ndarray


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


