import itertools
from typing import Dict, Tuple, Any, Iterator, Optional

import numpy

from .units import unit_conversion_coef
from .numeric_types import ArrayData, DataSource, TimeSeries, DataStorageTagsDct, ndarray1d
from .storage import Storage
from .istorage import ISensorStorage


def partial_format(path: str, params: Dict[str, Optional[str]]) -> str:
    for name, val in params.items():
        if val is not None:
            path = path.replace("{" + name + "}", val)
    return path


class SensorStorageBase(ISensorStorage):
    ts_arr_tag = 'csv'
    csv_file_encoding = 'utf8'

    def __init__(self, storage: Storage, db_paths: Any) -> None:
        self.storage = storage
        self.cache = {}  # type: Dict[str, Tuple[int, int, ArrayData]]
        self.db_paths = db_paths

    def sync(self) -> None:
        self.storage.sync()

    #  -----------------  TS  ------------------------------------------------------------------------------------------
    # Sensors are different from TS as several sensors has the same collect time and share one storage array
    # for it. Also sensors allows to append data.
    def get_sensor(self, ds: DataSource, time_range: Tuple[float, float] = None) -> TimeSeries:
        # sensors has no shape
        path = self.db_paths.sensor_time.format_map(ds.__dict__)
        (time_units,), must_be_none, collected_at = self.storage.get_array(path)
        collected_at = collected_at[::2].copy()

        if time_range is not None:
            coef = float(unit_conversion_coef('s', time_units))
            time_range = [time_range[0] * coef, time_range[1] * coef]
            line1, line2 = numpy.searchsorted(collected_at, time_range)
            slc = slice(line1, line2)
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

    def append_sensor(self, data: ndarray1d, ds: DataSource, units: str) -> None:
        assert ds.tag is None or ds.tag == self.ts_arr_tag, \
            "Incorrect source tag == {!r}, must be {!r}".format(ds.tag, self.ts_arr_tag)

        if ds.metric == 'collected_at':
            path = self.db_paths.sensor_time
            assert len(data.shape) == 1, "collected_at data must be 1D array"
        else:
            path = self.db_paths.sensor_data

        path = path.format_map(ds.__dict__)
        self.storage.put_array(path, data, header=[units], append_on_exists=True)

    # -------------   ITER OVER STORAGE   ------------------------------------------------------------------------------

    def iter_paths(self, path_templ) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        path_parts = path_templ.format_map(DataStorageTagsDct).split("/")
        yield from self.storage.iter_paths("", path_parts, {})

    def iter_paths2(self, path_templ, **ds_parts) -> Iterator[Tuple[bool, str, Dict[str, str]]]:
        keys, raw_vals = zip(*ds_parts.items())
        vals = [[val] if isinstance(val, str) else val for val in raw_vals]
        for combination in itertools.product(*vals):
            path = partial_format(path_templ, dict(zip(keys, combination)))
            yield from self.iter_paths(path)

    def iter_objs(self, path_templ, **ds_parts) -> Iterator[DataSource]:
        if ds_parts:
            keys, raw_vals = zip(*ds_parts.items())
            vals = [[val] if isinstance(val, str) else val for val in raw_vals]
            for combination in itertools.product(*vals):
                path = partial_format(path_templ, dict(zip(keys, combination)))
                for ds in self.iter_objs(path):
                    yield ds(**ds_parts)
        else:
            for is_node, path, path_chunks in self.iter_paths(path_templ):
                assert is_node
                expected_path = path_templ.replace(r'\.', '.').format_map(path_chunks)
                assert expected_path == path, "{} != {}".format(expected_path, path)
                yield DataSource(**path_chunks)

    def iter_sensors(self, **ds_parts) -> Iterator[DataSource]:
        return self.iter_objs(self.db_paths.sensor_data_r, **ds_parts)
