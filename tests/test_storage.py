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
from oktest import ok


from cephlib.storage import make_storage, IStorable, FSStorage, TxtResultStorage
from cephlib.storage import JsonResultStorage, XMLResultStorage


@contextlib.contextmanager
def in_temp_dir():
    dname = tempfile.mkdtemp()
    try:
        yield dname
    finally:
        shutil.rmtree(dname)


def test_filestorage():
    with in_temp_dir() as root:
        fs = FSStorage(root, existing=True)
        path = "a/b/t.txt"
        d1 = "a"
        d2 = "b"

        v1 = "test"
        v2 = "1234"

        fs.put(v1, path)
        ok(fs.get(path)) == v1
        ok(fs.get_fd(path, "rb+").read()) == v1
        fs.get_fd(path, "wb+").write(v2)
        ok(fs.get_fd(path, "rb+").read()) == v2
        ok(fs.get(path)) == v2
        ok(open(fs._get_fname(path)) == os.path.join(root, path))
        ok(open(fs._get_fname(path)).read()) == v2

        f1 = fs.sub_storage(d1)
        f2 = f1.sub_storage(d2)
        f21 = fs.sub_storage(os.path.join(d1, d2))
        ok(f2.get("t.txt")) == v2
        ok(f21.get("t.txt")) == v2
        ok(f1.get(d2 + "/t.txt")) == v2


def test_typed_attrstorage():
    with in_temp_dir() as root:
        fsstorage = FSStorage(root, existing=True)
        txt = TxtResultStorage(fsstorage)
        js = JsonResultStorage(fsstorage)
        fsstorage.put("test", "a/b/c.txt")

        ok(txt.a.b.c) == "test"

        assert isinstance(txt.a, TxtResultStorage)
        assert isinstance(js.a, JsonResultStorage)

        with pytest.raises(AttributeError):
            ok(js.a.b.c) == "test"
        ok(js.get('a/b/c')).is_(None)
        ok(js.get('a/b/c', 1)) == 1

        a = txt.a
        ok(a.b.c) == "test"

        data = {"a": 1}
        fsstorage.put(json.dumps(data), "d/e.json")
        ok(js.d.e) == data
        ok(js['d/e']) == data
        ok(js.d['e']) == data
        assert isinstance(js['d'], JsonResultStorage)
        ok(js['d']['e']) == data
        ok(js['d'].e) == data

        data2 = {"b": 2}
        fsstorage.put(json.dumps(data2), "d/f/t.json")


def test_hlstorage():
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
                ok(storage.get(path)) == val


def test_overwrite():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("1", "some_path")
            storage.put([1, 2, 3], "some_path")

        with make_storage(root, existing=True) as storage:
            assert storage.get("some_path") == [1, 2, 3]


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
                ok(storage.get(path)) == val


if numpy:
    def test_arrays():
        with in_temp_dir() as root:
            val_l = list(range(10000)) * 10
            val_i = numpy.array(val_l, numpy.int32)
            val_f = numpy.array(val_l, numpy.float32)
            val_2f = numpy.array(val_l + val_l, numpy.float32)

            with make_storage(root, existing=False) as storage:
                storage.put_array("array_i", val_i, ["array_i"])
                storage.put_array("array_f", val_f, ["array_f"])
                storage.put_array("array_x2", val_f, ["array_x2"])
                storage.put_array("array_x2", val_f, ["array_x2"], append_on_exists=True)

            with make_storage(root, existing=True) as storage:
                arr, header = storage.get_array("array_i")
                assert numpy.all(arr == val_i)
                ok(["array_i"]) == header

                arr, header = storage.get_array("array_f")
                assert numpy.all(arr == val_f)
                ok(["array_f"]) == header

                arr, header = storage.get_array("array_x2")
                assert numpy.all(arr == val_2f)
                ok(["array_x2"]) == header



class LoadMe(IStorable):
    def __init__(self, **vals):
        self.__dict__.update(vals)

    def raw(self):
        return dict(self.__dict__.items())

    @classmethod
    def fromraw(cls, data):
        return cls(**data)


def test_load_user_obj():
    obj = LoadMe(x=1, y=12, z=[1,2,3], t="asdad", gg={"a": 1, "g": [["x"]]})

    with in_temp_dir() as root:
        with make_storage(root, existing=False, serializer='yaml') as storage:
            storage.put(obj, "obj")

        with make_storage(root, existing=True, serializer='yaml') as storage:
            obj2 = storage.load(LoadMe, "obj")
            assert isinstance(obj2, LoadMe)
            ok(obj2.__dict__) == obj.__dict__


def test_path_not_exists():
    with in_temp_dir() as root:
        pass

    with make_storage(root, existing=False) as storage:
        with pytest.raises(KeyError):
            storage.get("x")


def test_substorage():
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            storage.put("data", "x/y")
            storage.sub_storage("t").put("sub_data", "r")

        with make_storage(root, existing=True) as storage:
            ok(storage.get("t/r")) == "sub_data"
            ok(storage.sub_storage("x").get("y")) == "data"

