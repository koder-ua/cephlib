import os
import ctypes
import logging
from fractions import Fraction
from typing import List, Tuple, Optional, Dict, Union, Iterator

import numpy

import cephlib
from .numeric_types import TimeSeries
from .istorage import ISensorStorage, FiltersType
from .units import unit_conversion_coef


logger = logging.getLogger("cephlib")


qd_metrics = {'io_queue'}
summ_sensors_cache = {}  # type: Dict[Tuple, Optional[TimeSeries]]

interpolated_cache = {}

c_interp_func_agg = None
c_interp_func_qd = None
c_interp_func_fio = None


def c_interpolate_ts_on_seconds_border(ts: TimeSeries,
                                       tp: str = 'agg',
                                       allow_broken_step: bool = False) -> TimeSeries:
    "Interpolate time series to values on seconds borders"
    key = (ts.source.tpl, tp)
    if key in interpolated_cache:
        return interpolated_cache[key].copy()

    if tp in ('qd', 'agg'):
        # both data and times must be 1d compact arrays
        assert len(ts.data.strides) == 1, "ts.data.strides must be 1D, not " + repr(ts.data.strides)
        assert ts.data.dtype.itemsize == ts.data.strides[0], "ts.data array must be compact"

    assert len(ts.times.strides) == 1, "ts.times.strides must be 1D, not " + repr(ts.times.strides)
    assert ts.times.dtype.itemsize == ts.times.strides[0], "ts.times array must be compact"

    assert len(ts.times) == len(ts.data), "len(times)={} != len(data)={} for {!s}"\
            .format(len(ts.times), len(ts.data), ts.source)

    rcoef = 1 / unit_conversion_coef(ts.time_units, 's')  # type: Union[int, Fraction]

    if isinstance(rcoef, Fraction):
        assert rcoef.denominator == 1, "Incorrect conversion coef {!r}".format(rcoef)
        rcoef = rcoef.numerator

    assert rcoef >= 1 and isinstance(rcoef, int), "Incorrect conversion coef {!r}".format(rcoef)
    coef = int(rcoef)   # make typechecker happy

    global c_interp_func_agg
    global c_interp_func_qd
    global c_interp_func_fio

    uint64_p = ctypes.POINTER(ctypes.c_uint64)

    if c_interp_func_agg is None:
        dirname = os.path.dirname(cephlib.__file__)
        path = os.path.join(dirname, 'clib', 'libwally.so')
        cdll = ctypes.CDLL(path)

        c_interp_func_agg = cdll.interpolate_ts_on_seconds_border
        c_interp_func_qd = cdll.interpolate_ts_on_seconds_border_qd

        for func in (c_interp_func_agg, c_interp_func_qd):
            func.argtypes = [
                ctypes.c_uint,  # input_size
                ctypes.c_uint,  # output_size
                uint64_p,  # times
                uint64_p,  # values
                ctypes.c_uint,  # time_scale_coef
                uint64_p,  # output
            ]
            func.restype = ctypes.c_uint  # output array used size

        c_interp_func_fio = cdll.interpolate_ts_on_seconds_border_fio
        c_interp_func_fio.restype = ctypes.c_int
        c_interp_func_fio.argtypes = [
                ctypes.c_uint,  # input_size
                ctypes.c_uint,  # output_size
                uint64_p,  # times
                ctypes.c_uint,  # time_scale_coef
                uint64_p,  # output indexes
                ctypes.c_uint64,  # empty placeholder
                ctypes.c_bool  # allow broken steps
            ]

    assert ts.data.dtype.name == 'uint64', "Data dtype for {}=={} != uint64".format(ts.source, ts.data.dtype.name)
    assert ts.times.dtype.name == 'uint64', "Time dtype for {}=={} != uint64".format(ts.source, ts.times.dtype.name)

    output_sz = int(ts.times[-1]) // coef - int(ts.times[0]) // coef + 2
    result = numpy.zeros(output_sz, dtype=ts.data.dtype.name)

    if tp in ('qd', 'agg'):
        assert not allow_broken_step, "Broken steps aren't supported for non-fio arrays"
        func = c_interp_func_qd if tp == 'qd' else c_interp_func_agg
        sz = func(ts.data.size,
                  output_sz,
                  ts.times.ctypes.data_as(uint64_p),
                  ts.data.ctypes.data_as(uint64_p),
                  coef,
                  result.ctypes.data_as(uint64_p))

        result = result[:sz]
        output_sz = sz

        rtimes = int(ts.times[0] // coef) + numpy.arange(output_sz, dtype=ts.times.dtype)
    else:
        assert tp == 'fio'
        ridx = numpy.zeros(output_sz, dtype=ts.times.dtype)
        no_data = (output_sz + 1)
        sz_or_err = c_interp_func_fio(ts.times.size,
                                      output_sz,
                                      ts.times.ctypes.data_as(uint64_p),
                                      coef,
                                      ridx.ctypes.data_as(uint64_p),
                                      no_data,
                                      allow_broken_step)
        if sz_or_err <= 0:
            raise ValueError("Error in input array at index {}. {}".format(-sz_or_err, ts.source))

        rtimes = int(ts.times[0] // coef) + numpy.arange(sz_or_err, dtype=ts.times.dtype)

        empty = numpy.zeros(len(ts.histo_bins), dtype=ts.data.dtype) if ts.source.metric == 'lat' else 0
        res = []
        for idx in ridx[:sz_or_err]:
            if idx == no_data:
                res.append(empty)
            else:
                res.append(ts.data[idx])
        result = numpy.array(res, dtype=ts.data.dtype)

    res_ts = TimeSeries(result,
                        times=rtimes,
                        units=ts.units,
                        time_units='s',
                        source=ts.source(),
                        histo_bins=ts.histo_bins)

    interpolated_cache[ts.source.tpl] = res_ts.copy()
    return res_ts


def get_ts_for_time_range(ts: TimeSeries, time_range: Tuple[int, int]) -> TimeSeries:
    """Return sensor values for given node for given period. Return per second estimated values array
    Raise an error if required range is not full covered by data in storage"""

    assert ts.time_units == 's', "{} != s for {!s}".format(ts.time_units, ts.source)
    assert len(ts.times) == len(ts.data), "Time(={}) and data(={}) sizes doesn't equal for {!s}"\
            .format(len(ts.times), len(ts.data), ts.source)

    if time_range[0] < ts.times[0] or time_range[1] > ts.times[-1]:
        raise AssertionError(("Incorrect data for get_sensor - time_range={!r}, collected_at=[{}, ..., {}]," +
                              "sensor = {}_{}.{}.{}").format(time_range, ts.times[0], ts.times[-1],
                                                             ts.source.node_id, ts.source.sensor, ts.source.dev,
                                                             ts.source.metric))
    idx1, idx2 = numpy.searchsorted(ts.times, time_range)
    return TimeSeries(ts.data[idx1:idx2],
                      times=ts.times[idx1:idx2],
                      units=ts.units,
                      time_units=ts.time_units,
                      source=ts.source,
                      histo_bins=ts.histo_bins)


def iter_interpolated_sensors(sstorage: ISensorStorage, time_range: Tuple[int, int],
                              filters: Dict[str, FiltersType]) -> Iterator[TimeSeries]:

    for ds in sstorage.iter_sensors(**filters):
        data = sstorage.get_sensor(ds)
        data = c_interpolate_ts_on_seconds_border(data, 'qd' if ds.metric in qd_metrics else 'agg')
        yield get_ts_for_time_range(data, time_range)


def summ_sensors(sstorage: ISensorStorage, time_range: Tuple[int, int], **filters: FiltersType) -> Optional[TimeSeries]:

    key = dict(time_range=time_range, stor_id=id(sstorage))
    for name, val in filters.items():
        key[name] = val if isinstance(val, str) else tuple(val)

    ckey = tuple(sorted(key.items()))

    if ckey in summ_sensors_cache:
        return summ_sensors_cache[ckey].copy()

    res = None  # type: Optional[TimeSeries]
    for ts in iter_interpolated_sensors(sstorage, time_range, filters):
        if res is None:
            res = ts
            res.data = res.data.copy()
        else:
            res.data += ts.data

    summ_sensors_cache[ckey] = res
    if len(summ_sensors_cache) > 1024:
        logger.warning("summ_sensors_cache cache too large %s > 1024", len(summ_sensors_cache))

    return res if res is None else res.copy()


def find_sensors_to_2d(sstorage: ISensorStorage, time_range: Tuple[int, int], **filters: FiltersType) -> numpy.ndarray:

    res = []  # type: List[TimeSeries]
    for ts in iter_interpolated_sensors(sstorage, time_range, filters):
        res.append(ts.data)
    res2d = numpy.concatenate(res)
    res2d.shape = (len(res), -1)
    return res2d
