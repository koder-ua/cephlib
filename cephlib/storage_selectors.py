import os
import re
import ctypes
import logging
import functools
from fractions import Fraction
from typing import List, Tuple, Optional, Dict, Union, Iterator, Callable, Any, Set

import numpy

import cephlib
from .numeric_types import TimeSeries
from .istorage import ISensorStorage
from .units import unit_conversion_coef
from .node import NodeInfo, NodeRole

from .sensors_rpc_plugin import SENSOR2DEV_TYPE


logger = logging.getLogger("cephlib")
qd_metrics = {'io_queue'}


c_interp_func_agg = None
c_interp_func_qd = None
c_interp_func_fio = None


def c_interpolate_ts_on_seconds_border(ts: TimeSeries,
                                       tp: str = 'agg',
                                       allow_broken_step: bool = False) -> TimeSeries:
    "Interpolate time series to values on seconds borders"
    if tp in ('qd', 'agg'):
        # both data and times must be 1d compact arrays
        assert len(ts.data.strides) == 1, "ts.data.strides must be 1D, not " + repr(ts.data.strides)
        assert ts.data.dtype.itemsize == ts.data.strides[0], "ts.data array of {} must be compact".format(ts.source)

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

    output_sz = int(ts.times[-1] / coef + 0.5) - int(ts.times[0] / coef + 0.5) + 2
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
                                      True)
        if sz_or_err <= 0:
            raise ValueError("Error in input array at index {}. {}".format(-sz_or_err, ts.source))
        output_sz = sz_or_err

        empty = numpy.zeros(len(ts.histo_bins), dtype=ts.data.dtype) if ts.source.metric == 'lat' else 0
        res = []
        for idx in ridx[:sz_or_err]:
            if idx == no_data:
                res.append(empty)
            else:
                res.append(ts.data[idx])
        result = numpy.array(res, dtype=ts.data.dtype)

    rtimes = int(ts.times[0] / coef + 0.5) + numpy.arange(output_sz, dtype=ts.times.dtype)
    res_ts = TimeSeries(result,
                        times=rtimes,
                        units=ts.units,
                        time_units='s',
                        source=ts.source(),
                        histo_bins=ts.histo_bins)

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
                              **params: Union[str, List[str]]) -> Iterator[TimeSeries]:

    for ds in sstorage.iter_sensors(**params):
        data = sstorage.get_sensor(ds, time_range)
        data = c_interpolate_ts_on_seconds_border(data, 'qd' if ds.metric in qd_metrics else 'agg')
        yield get_ts_for_time_range(data, time_range)


def sum_sensors(sstorage: ISensorStorage, time_range: Tuple[int, int],
                **params: Union[str, List[str]]) -> Optional[TimeSeries]:
    # TODO: need to cache somehow
    # key = dict(time_range=time_range, stor_id=id(sstorage))
    # for name, val in sensors.items():
    #     key[name] = val if isinstance(val, str) else tuple(val)
    # ckey = tuple(sorted(key.items()))
    # summ_sensors_cache = cast(Dict[Tuple, Optional[TimeSeries]], sstorage.storage.other_caches['summ_sensors'])
    # if ckey in summ_sensors_cache:
    #     return summ_sensors_cache[ckey].copy()

    res = None  # type: Optional[TimeSeries]
    for ts in iter_interpolated_sensors(sstorage, time_range, **params):
        if res is None:
            res = ts
        else:
            res.data += ts.data

    # summ_sensors_cache[ckey] = res
    # if len(summ_sensors_cache) > 1024:
    #     logger.warning("sum_sensors_cache cache too large %s > 1024", len(summ_sensors_cache))
    # return res if res is None else res.copy()
    return res


# maybe need cache for this too? Check if cache improve anything
def find_sensors_to_2d(sstorage: ISensorStorage, time_range: Tuple[int, int],
                       **params: Union[str, List[str]]) -> numpy.ndarray:
    res = [ts.data for ts in iter_interpolated_sensors(sstorage, time_range, **params)]
    res2d = numpy.concatenate(res)
    res2d.shape = (len(res), -1)
    return res2d


def get_node_checker(selector: str) -> Callable[[NodeInfo], bool]:
    def node_name_checker(rr: Any, node: NodeInfo) -> bool:
        return (node.hostname and rr.match(node.hostname)) or rr.match(node.node_id)

    def node_role_checker(rr: Any, node: NodeInfo) -> bool:
        return any(map(rr.match, node.roles))

    if selector.startswith('role='):
        sel = selector[len("role="):]
        node_checker = functools.partial(node_role_checker, re.compile(sel))
        node_checker.flt = "has role like {!r}".format(sel)
    else:
        node_checker = functools.partial(node_name_checker, re.compile(selector))
        node_checker.flt = "hostname like {!r}".format(selector)

    return node_checker


def get_dev_checker(selector: str) -> Callable[[str, str], bool]:
    def dev_checker(expected_tp: Optional[str], name_rr: Any, tp: str, dev: str) -> bool:
        return (name_rr is None or name_rr.match(dev)) and (expected_tp is None or expected_tp == tp)

    if selector.startswith('type='):
        tp_sel = selector[len("type="):]
        flt = 'dev.type == {!r}'.format(tp_sel)
        if ',' in tp_sel:
            tp_sel, name_sel_s = tp_sel.split(',')
            flt += ' and dev like {!r}'.format(name_sel_s)
        else:
            name_sel_s = None
    else:
        tp_sel = None
        name_sel_s = selector
        flt = 'dev like {!r}'.format(name_sel_s)

    name_sel = None if not name_sel_s else re.compile(name_sel_s)

    dev_checker = functools.partial(dev_checker, tp_sel, name_sel)
    dev_checker.flt = flt
    return dev_checker


class DevType:
    block = 'block'
    eth = 'eth'
    weth = 'weth'
    cpu = 'cpu'


class DevRoles:
    client_block = 'client_block'
    client_net = 'client_net'
    client_cpu = 'client_cpu'

    storage_block = 'storage_block'
    storage_cpu = 'storage_cpu'
    storage_net = 'storage_net'
    storage_cluster_net = 'storage_cluster_net'
    storage_client_net = 'storage_client_net'

    osd_cpu = 'osd_cpu'
    osd_storage = 'osd_storage'
    osd_journal = 'osd_journal'

    compute_cpu = 'compute_cpu'
    compute_net = 'compute_net'
    compute_block = 'compute_block'


DevRolesConfig = List[Dict[str,  # node selector
                         List[Dict[str,   # dev selector
                                   Union[str, List[str]]]]]]  # dev roles

Role2Devs = Dict[str, List[Tuple[str, str]]]
AllSensorDevs = Dict[str, Dict[str, List[str]]]


DEFAULT_ROLES = {
    NodeRole.client: {
        DevType.cpu: DevRoles.client_cpu,
        DevType.block: DevRoles.client_block,
        DevType.eth: DevRoles.client_net,
        DevType.weth: DevRoles.client_net
    },
    NodeRole.storage: {
        DevType.cpu: DevRoles.storage_cpu,
        DevType.block: DevRoles.storage_block,
        DevType.eth: DevRoles.storage_net,
        DevType.weth: DevRoles.storage_net
    },
    NodeRole.compute: {
        DevType.cpu: DevRoles.compute_cpu,
        DevType.block: DevRoles.compute_block,
        DevType.eth: DevRoles.compute_net,
        DevType.weth: DevRoles.compute_net
    }
}


def update_storage_selector(storage: ISensorStorage, rc: DevRolesConfig, nodes: List[NodeInfo]) -> None:
    """
    Classify all devices from all_devs using rc config
    :param rc: Config, which maps devices to roles in format
            [
                {node_selector: [
                    {dev_selector: [dev_role, ...]},
                    ...]
                }
            ]
                
    :param nodes: All available nodes, required to get role node and hostname
    """
    rc = rc[:]
    for node_role, dev_def_types in DEFAULT_ROLES.items():
        per_node = []
        for dev_tp, dev_role in dev_def_types.items():
            per_node.append({"type=" + dev_tp: dev_role})
        rc.append({"role=" + node_role: per_node})

    all_devs = {}  # type: Dict[str, Dict[str, Set[str]]]
    # All devices, available in sensor storage in format
    #       {node_id: {dev_type: {dev_name, ...}}}
    for ds in storage.iter_sensors():
        assert ds.tag == ISensorStorage.ts_arr_tag
        node_sens = all_devs.setdefault(ds.node_id, {})
        node_sens.setdefault(SENSOR2DEV_TYPE.get(ds.sensor), set()).add(ds.dev)

    # rules hold next structure:
    #    - node_checker:
    #       - dev_checker: [dev_role, ...]
    #       ...
    #    ...
    rules = []  # type: List[Tuple[Callable[[NodeInfo], bool], List[Tuple[Callable[[str, str], bool], List[str]]]]]
    for selectors_tree in rc:
        assert len(selectors_tree) == 1
        (node_selector, devs_selectors), = selectors_tree.items()
        # print("node_selector =", node_selector)
        node_checker = get_node_checker(node_selector)

        dev2roles_mappers = []
        for devs_selector_dct in devs_selectors:
            assert len(devs_selector_dct) == 1
            (dev_selector, roles), = devs_selector_dct.items()
            dev2roles_mappers.append((get_dev_checker(dev_selector),
                                      roles if isinstance(roles, list) else [roles]))

        rules.append((node_checker, dev2roles_mappers))

    node_id2node = {node.node_id: node for node in nodes}

    for node_id, types2devs in all_devs.items():
        # first prefilter rules, using node
        curr_rules = []  # type: List[Tuple[Callable[[str, str], bool], List[str]]]
        node = node_id2node[node_id]
        for node_checker, dev_checkers in rules:
            if node_checker(node):
                curr_rules.extend(dev_checkers)

        if curr_rules:
            for dev_type, devs in types2devs.items():
                for dev in devs:
                    for dev_checker, roles in curr_rules:
                        if dev_checker(dev_type, dev):
                            for role in roles:
                                # print('dev_role={} => node_id={}, dev={}'.format(role, node_id, dev))
                                storage.add_mapping('dev_role', role, node_id=node_id, dev=dev)
                            break
