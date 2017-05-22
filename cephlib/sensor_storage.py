from typing import Dict, Tuple, Any, Iterator, Optional, List, Union, Set

import numpy

from .units import unit_conversion_coef
from .types import NumVector, DataSource, DataStorageTagsDct, get_arr_info
from .numeric_types import TimeSeries
from .istorage import ISensorStorage, IStorage, SensorsIter, PathSelector


def partial_format(path: str, params: Dict[str, Optional[str]]) -> str:
    for name, val in params.items():
        if val is not None:
            path = path.replace("{" + name + "}", val)
    return path


def append_sensor(storage: IStorage,
                  data: NumVector,
                  ds: DataSource,
                  units: str,
                  sensor_time_path: str,
                  sensor_data_path: str,
                  expected_arr_tag: str = 'csv') -> None:

    assert ds.tag == expected_arr_tag, "Incorrect source tag == {!r}, must be {!r}".format(ds.tag, expected_arr_tag)

    dtype, shape = get_arr_info(data)

    if ds.metric == 'collected_at':
        assert len(shape) == 1, "collected_at data must be 1D array"
        path = sensor_time_path
    else:
        path = sensor_data_path

    path = path.format_map(ds.__dict__)
    storage.put_array(path, data, header=[units], append_on_exists=True)


class SensorStorage(ISensorStorage):
    def __init__(self, storage: IStorage, db_paths: Any) -> None:
        self.storage = storage
        self.db_paths = db_paths
        self.locator = PathSelector()
        self.sensor2nodedev = {}  # type: Dict[str, Set[Tuple[str, str]]]

    def sync(self) -> None:
        self.storage.sync()

    def flush(self) -> None:
        self.storage.flush()

    #  -----------------  TS  ------------------------------------------------------------------------------------------
    # Sensors are different from TS as several sensors has the same collect time and share one storage array
    # for it. Also sensors allows to append data.
    # time_range applied on this stage to match HTTP storages
    # No caching in this function, as sensor is not supposed to be used directly
    # instead high-level functions should be called, which can do it's own caching
    def get_sensor(self, ds: DataSource, time_range: Tuple[float, float] = None) -> TimeSeries:
        # sensors has no shape
        path = self.db_paths.sensor_time.format_map(ds.__dict__)
        (time_units,), must_be_none, collected_at = self.storage.get_array(path)

        # force to remove gaps from array, this is required for c_interpolate
        collected_at = collected_at[::2].copy()

        if time_range is not None:
            coef = float(unit_conversion_coef('s', time_units))
            time_range = [time_range[0] * coef, time_range[1] * coef]
            line1, line2 = numpy.searchsorted(collected_at, time_range)
            slc = slice(max(0, line1 - 1), line2 + 1)
        else:
            slc = slice(0, None)

        # there must be no histogram for collected_at
        assert must_be_none is None, "Extra header2 {!r} in collect_at file at {!r}".format(must_be_none, path)
        assert len(collected_at.shape) == 1, "collected_at must be 1D at {!r}".format(path)

        data_path = self.db_paths.sensor_data.format_map(ds.__dict__)
        (data_units,), must_be_none, data  = self.storage.get_array(data_path)

        # there must be no histogram for any sensors
        assert must_be_none is None, "Extra header2 {!r} in sensor data file {!r}".format(must_be_none, data_path)
        assert len(data.shape) == 1, "Sensor data must be 1D at {!r}".format(data_path)

        return TimeSeries(data[slc], times=collected_at[slc], source=ds, units=data_units, time_units=time_units)

    def append_sensor(self, data: NumVector, ds: DataSource, units: str) -> None:
        append_sensor(self.storage, data, ds, units,
                      sensor_time_path=self.db_paths.sensor_time,
                      sensor_data_path=self.db_paths.sensor_data,
                      expected_arr_tag=self.ts_arr_tag)

    # -------------   ITER OVER STORAGE   ------------------------------------------------------------------------------

    def add_mapping(self, param: str, val: str, **fields: Union[str, List[str]]) -> None:
        self.locator.add_mapping(param, val, **fields)

    def iter_paths(self, path_templ: str) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        path_parts = path_templ.format_map(DataStorageTagsDct).split("/")
        yield from self.storage.iter_paths("", path_parts, {})

    _all_keys = {'sensor', 'metric', 'node_id', 'dev'}

    def iter_objs(self, path_templ: str, **path_parts: Union[str, List[str]]) -> Iterator[DataSource]:
        all_locs = list(self.locator(**path_parts))

        if len(all_locs) != 0:
            if path_templ == self.db_paths.sensor_data_r and self._all_keys.issubset(all_locs[0].keys()):
                for loc in all_locs:
                    yield DataSource(tag=self.ts_arr_tag, **loc)
            else:
                for ds_parts in all_locs:
                    path = partial_format(path_templ, ds_parts)
                    for is_node, ipath, path_chunks in self.iter_paths(path):
                        assert is_node
                        expected_path = path.replace(r'\.', '.').format_map(path_chunks)
                        assert expected_path == ipath, "{} != {}".format(expected_path, ipath)
                        yield DataSource(**{**path_chunks, **ds_parts})

    def iter_sensors(self, **path_parts: Union[str, List[str]]) -> Iterator[DataSource]:
        yield from self.iter_objs(self.db_paths.sensor_data_r, **path_parts)
