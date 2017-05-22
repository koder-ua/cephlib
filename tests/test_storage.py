import os
import json
import shutil
import tempfile
import contextlib

try:
    import numpy
except ImportError:
    pass

import pytest


from cephlib.istorage import IStorable
from cephlib.storage import make_storage, FSStorage, make_attr_storage
from cephlib.numeric_types import TimeSeries
from cephlib.types import DataSource
from cephlib.wally_storage import WallyDB
from cephlib.sensor_storage import SensorStorage

@contextlib.contextmanager
def in_temp_dir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


# noinspection PyStatementEffect
def test_filestorage():
    with in_temp_dir() as root:
        fs = FSStorage(root, existing=True)
        path = "a/b/t.txt"
        d1 = "a"
        d2 = "b"

        v1 = "test".encode('utf8')
        v2 = "1234".encode('utf8')

        fs.put(v1, path)
        assert fs.get(path) == v1
        assert fs.get_fd(path, "rb+").read() == v1
        fs.get_fd(path, "wb+").write(v2)
        assert fs.get_fd(path, "rb+").read() == v2
        assert fs.get(path) == v2
        assert fs.get_fname(path) == os.path.join(root, path)
        assert open(fs.get_fname(path), 'rb').read() == v2

        f1 = fs.sub_storage(d1)
        f2 = f1.sub_storage(d2)
        f21 = fs.sub_storage(os.path.join(d1, d2))
        assert f2.get("t.txt") == v2
        assert f21.get("t.txt") == v2
        assert f1.get(d2 + "/t.txt") == v2


# noinspection PyStatementEffect
def test_typed_attrstorage():
    with in_temp_dir() as root:
        fsstorage = FSStorage(root, existing=True)
        txt = make_attr_storage(fsstorage, 'txt')
        js = make_attr_storage(fsstorage, 'json')
        fsstorage.put(b"test", "a/b/c.txt")

        assert txt.a.b.c == b"test"

        assert isinstance(txt.a, txt.__class__)
        assert isinstance(js.a, js.__class__)

        with pytest.raises(AttributeError):
            assert js.a.b.c == b"test"

        assert js.get('a/b/c') is None
        assert js.get('a/b/c', 1) == 1

        a = txt.a
        assert a.b.c == b"test"

        data = {"a": 1}
        fsstorage.put(json.dumps(data).encode("utf8"), "d/e.json")
        assert js.d.e == data
        assert js['d/e'] == data
        assert js.d['e'] == data
        assert isinstance(js['d'], js.__class__)
        assert js['d']['e'] == data
        assert js['d'].e == data


# noinspection PyStatementEffect
def test_storage2():
    with in_temp_dir() as root:
        values = {
            "int": 1,
            "str/1": "test",
            "bytes/2": b"test",
            "none/s/1": None,
            "bool/xx/1/2/1": None,
            "float/s/1": 1.234,
            "list": [1, 2, "3"],
            "dict": {1: 3, "2": "4", "1.2": 1.3}
        }

        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage.put(val, path)

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                assert storage.get(path) == val


# noinspection PyStatementEffect
def test_overwrite():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("1", "some_path")
            storage.put([1, 2, 3], "some_path")

        with make_storage(root, existing=True) as storage:
            assert storage.get("some_path") == [1, 2, 3]


# noinspection PyStatementEffect
def test_multy_level():
    with in_temp_dir() as root:
        values = {
            "dict1": {1: {3: 4, 6: [12, {123, 3}, {4: 3}]}, "2": "4", "1.2": 1.3}
        }

        with make_storage(root, existing=False) as storage:
            for path, val in values.items():
                storage.put(val, path)

        with make_storage(root, existing=True) as storage:
            for path, val in values.items():
                assert storage.get(path) == val


# noinspection PyStatementEffect
def test_arrays():
    sz = 10000
    rep = 10
    with in_temp_dir() as root:
        val_l = list(range(sz)) * rep
        val_i = numpy.array(val_l, numpy.int32)
        val_f = numpy.array(val_l, numpy.float32)
        val_2f = numpy.array(val_l + val_l, numpy.float32)

        with make_storage(root, existing=False) as storage:
            storage.put_array("array_i", val_i, ["array_i"])
            storage.put_array("array_f", val_f, ["array_f"])
            storage.put_array("array_x2", val_f, ["array_x2"])
            storage.put_array("array_x2", val_f, ["array_x2"], append_on_exists=True)

        with make_storage(root, existing=True) as storage:
            header, header2, arr = storage.get_array("array_i")
            assert header2 is None
            assert (arr == val_i).all()
            assert ["array_i"] == header

            header, header2, arr = storage.get_array("array_f")
            assert (arr == val_f).all()
            assert header2 is None
            assert ["array_f"] == header

            header, header2, arr = storage.get_array("array_x2")
            assert header2 is None
            assert (arr == val_2f).all()
            assert ["array_x2"] == header



class LoadMe(IStorable):
    def __init__(self, **vals):
        self.__dict__.update(vals)

    def raw(self):
        return dict(self.__dict__.items())

    @classmethod
    def fromraw(cls, data):
        return cls(**data)


# noinspection PyStatementEffect
def test_load_user_obj():
    obj = LoadMe(x=1, y=12, z=[1, 2, 3], t="asdad", gg={"a": 1, "g": [["x"]]})
    obj2 = LoadMe(x=2, y=None, z=[1, "2", 3], t=123, gg={"a": 1, "g": [["x"]]})
    obj3 = LoadMe(x="3", y=[1, 2], z=["1", 2, 3], t=None, gg={"a": 1, "g": [["x"]]})
    objs = [obj, obj2, obj3]

    with in_temp_dir() as root:
        with make_storage(root, existing=False, serializer='yaml') as storage:
            storage.put(obj, "obj")
            storage.put_list(objs, "objs")

        with make_storage(root, existing=True, serializer='yaml') as storage:
            obj2 = storage.load(LoadMe, "obj")
            assert isinstance(obj2, LoadMe)
            assert obj2.__dict__ == obj.__dict__

            objs2 = storage.load_list(LoadMe, "objs")
            assert isinstance(objs2, list)
            assert len(objs2) == len(objs)
            for o1, o2 in zip(objs, objs2):
                assert o1.__dict__ == o2.__dict__


def test_path_not_exists():
    with in_temp_dir() as root:
        pass

    with make_storage(root, existing=False) as storage:
        with pytest.raises(KeyError):
            storage.get("x")


# noinspection PyStatementEffect
def test_substorage():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("data", "x/y")
            storage.sub_storage("t").put("sub_data", "r")

        with make_storage(root, existing=True) as storage:
            assert storage.get("t/r") == "sub_data"
            assert storage.sub_storage("x").get("y") == "data"


# # noinspection PyStatementEffect
# def test_hlstorage_ts():
#     with in_temp_dir() as root:
#         with make_storage(root, existing=False) as storage:
#             sstorage = SensorStorage(storage, WallyDB)
#
#             ds = DataSource(suite_id='suite_1',
#                             job_id='job_11',
#                             node_id="node1",
#                             sensor='io_sensor',
#                             metric='io',
#                             tag='csv')
#
#             with pytest.raises(AssertionError):
#                 ds.verify()
#
#             ds = DataSource(suite_id='suite_1',
#                             job_id='job_11',
#                             node_id="1.1.2.3:23",
#                             sensor='sensor',
#                             metric='io',
#                             tag='csv')
#             ds.verify()
#
#             data = numpy.arange(100, dtype='uint64')
#             data.shape = [10, 10]
#             ts = TimeSeries(data,
#                             times=numpy.arange(10, 20, dtype='uint64'),
#                             units='x',
#                             time_units='s',
#                             source=ds,
#                             histo_bins=None)
#
#             # should allows non-lat 2d ts
#             sstorage.append_sensor(ts.data, ts.source, ts.units)
#
#             assert ts == sstorage.get_sensor(ts.source)
#
#             ts.data = numpy.arange(10, dtype='uint64')
#             hlstorage.put_ts(ts)
#             assert ts == hlstorage.get_ts(ds)
#
#             ts_ds = list(hlstorage.iter_ts())
#             assert len(ts_ds) == 1
#             assert ts_ds[0] == ds
#
#             ts_ds = list(hlstorage.iter_ts(suite_id=ds.suite_id))
#             assert len(ts_ds) == 1
#             assert ts_ds[0] == ds
#
#             ts_ds = list(hlstorage.iter_ts(job_id=ds.job_id))
#             assert len(ts_ds) == 1
#             assert ts_ds[0] == ds
#
#             ts_ds = list(hlstorage.iter_ts(suite_id=ds.suite_id, job_id=ds.job_id))
#             assert len(ts_ds) == 1
#             assert ts_ds[0] == ds
#
#             assert ts == hlstorage.get_ts(ts.source)
#
#             ts2 = TimeSeries(numpy.arange(20, dtype='uint64'),
#                              times=numpy.arange(30, 50, dtype='uint64'),
#                              units='Kx',
#                              time_units='us',
#                              source=ds,
#                              histo_bins=None)
#
#             assert ts2 != ts
#             assert ts2.source == ts.source
#
#             hlstorage.put_ts(ts2)
#             assert ts2 == hlstorage.get_ts(ts2.source)
#             assert ts2 == hlstorage.get_ts(ts.source)
#
#             ts.source = ts.source(node_id='3.3.3.3:333')
#             assert ts2.source != ts.source
#             hlstorage.put_ts(ts)
#
#             assert ts2 == hlstorage.get_ts(ts2.source)
#             assert ts == hlstorage.get_ts(ts.source)
#
#
# # noinspection PyStatementEffect
# def test_hlstorage_hist():
#     with in_temp_dir() as root:
#         with make_storage(root, existing=False) as storage:
#             hlstorage = HLStorageBase(storage, WallyDB)
#             ds = DataSource(suite_id='suite_1',
#                             job_id='job_11',
#                             node_id="1.1.2.3:23",
#                             sensor='sensor',
#                             metric='lat',
#                             tag='csv')
#             ds.verify()
#
#             histo_bins = numpy.arange(1024, 1034)
#             data = numpy.arange(10 * 10, dtype='uint64')
#             data.shape = [10, 10]
#
#             ts = TimeSeries(data,
#                             times=numpy.arange(10, 20, dtype='uint64'),
#                             units='x',
#                             time_units='s',
#                             source=ds,
#                             histo_bins=None)
#
#             with pytest.raises(AssertionError):
#                 hlstorage.put_ts(ts)
#
#             ts.histo_bins = histo_bins
#             hlstorage.put_ts(ts)
#             assert ts == hlstorage.get_ts(ds)
#
#             ts_ds = list(hlstorage.iter_ts())
#             assert len(ts_ds) == 1
#             assert ts_ds[0] == ds
#
#             assert ts == hlstorage.get_ts(ts.source)


# noinspection PyStatementEffect
def test_hlstorage_sensor():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            hlstorage = SensorStorage(storage, WallyDB)
            time_ds = DataSource(node_id="1.1.2.3:23", metric='collected_at')
            sensor_times = numpy.arange(20)
            sensor_data = numpy.arange(10)

            hlstorage.append_sensor(sensor_times, time_ds, 's')

            data_ds = DataSource(node_id="1.1.2.3:23", sensor='io', dev='sda', metric='iops', tag='csv')
            hlstorage.append_sensor(sensor_data, data_ds, 'Kx')

            ts_ds = list(hlstorage.iter_sensors())
            assert len(ts_ds) == 1
            assert ts_ds[0] == data_ds

            ts = hlstorage.get_sensor(data_ds)
            assert numpy.array_equal(ts.times, sensor_times[::2])
            assert numpy.array_equal(ts.data, sensor_data)
            assert ts.time_units == 's'
            assert ts.units == 'Kx'
            assert ts.source == data_ds

            sensor_data2 = numpy.arange(10, 20)
            data_ds2 = DataSource(node_id="1.1.2.3:23", sensor='io', dev='sdb', metric='iops', tag='csv')
            hlstorage.append_sensor(sensor_data2, data_ds2, 'Mx')

            ts_ds = list(hlstorage.iter_sensors())
            assert len(ts_ds) == 2
            ts_ds.sort(key=str)

            assert ts_ds[0] == data_ds
            assert ts_ds[1] == data_ds2
            ts2 = hlstorage.get_sensor(data_ds2)
            assert ts2.units == 'Mx'
            assert ts2.source == data_ds2
            assert numpy.array_equal(ts2.times, sensor_times[::2])
            assert numpy.array_equal(ts2.data, sensor_data2)
