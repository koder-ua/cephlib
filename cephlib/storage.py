"""
This module contains interfaces for storage classes
"""

import os
import re
import abc
import array
import shutil
import logging

import json
from .pyver import tostr

try:
    import yaml

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

except ImportError:
    yaml = None


logger = logging.getLogger("storage")


try:
    import numpy
except ImportError:
    numpy = None


class ISimpleStorage(object):
    """interface for low-level storage, which doesn't support serialization and can operate only on bytes"""

    @abc.abstractmethod
    def put(self, value, path):
        pass

    @abc.abstractmethod
    def get(self, path):
        pass

    @abc.abstractmethod
    def rm(self, path):
        pass

    @abc.abstractmethod
    def sync(self):
        pass

    @abc.abstractmethod
    def __contains__(self, path):
        pass

    @abc.abstractmethod
    def get_fd(self, path, mode="rb+"):
        pass

    @abc.abstractmethod
    def sub_storage(self, path):
        pass

    @abc.abstractmethod
    def list(self, path):
        pass


class IArrayStorage(object):
    """Interface for put/get/append multi-dimensional arrays(numpy.ndarray) of simple types"""

    @abc.abstractmethod
    def put(self, path, value, header, append_on_exists=False):
        pass

    @abc.abstractmethod
    def get(self, path):
        pass

    @abc.abstractmethod
    def rm(self, path):
        pass

    @abc.abstractmethod
    def sync(self):
        pass

    @abc.abstractmethod
    def __contains__(self, path):
        pass

    @abc.abstractmethod
    def list(self, path):
        pass

    @abc.abstractmethod
    def sub_storage(self, path):
        pass


class ISerializer(object):
    """Interface for serialization class"""
    @abc.abstractmethod
    def pack(self, value):
        pass

    @abc.abstractmethod
    def unpack(self, data):
        pass


class IStorable(object):
    """Interface for type, which can be stored"""

    @abc.abstractmethod
    def raw(self):
        pass

    @classmethod
    def fromraw(cls, data):
        raise NotImplementedError()


class FSStorage(ISimpleStorage):
    """Store all data in files on FS"""

    def __init__(self, root_path, existing):
        self.root_path = root_path
        self.existing = existing
        self.ignored = {'.', '..'}

    def _get_fname(self, path):
        return os.path.join(self.root_path, path)

    def isdir(self, path):
        return os.path.isdir(os.path.join(self.root_path, path))

    def put(self, value, path):
        jpath = self._get_fname(path)

        dpath = os.path.dirname(jpath)
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        with open(jpath, "wb") as fd:
            fd.write(value)

    def get(self, path):
        try:
            with open(self._get_fname(path), "rb") as fd:
                return fd.read()
        except IOError:
            raise KeyError(path)

    def rm(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)

    def __contains__(self, path):
        return os.path.exists(self._get_fname(path))

    def get_fd(self, path, mode="rb+"):
        jpath = self._get_fname(path)

        if mode in ('cb', 'ct'):
            create_on_fail = True
            mode = "rb+" if mode == 'cb' else 'rt+'

            dpath = os.path.dirname(jpath)
            if not os.path.exists(dpath):
                os.makedirs(dpath)

        else:
            create_on_fail = False

        try:
            fd = open(jpath, mode)
        except IOError:
            if not create_on_fail:
                raise

            if 't' in mode:
                fd = open(jpath, "wt")
            else:
                fd = open(jpath, "wb")

        return fd

    def sub_storage(self, path):
        return self.__class__(self._get_fname(path), self.existing)

    def sync(self):
        pass

    def list(self, path):
        path = self._get_fname(path)

        if not os.path.exists(path):
            return

        if not os.path.isdir(path):
            raise OSError("{!r} is not a directory".format(path))

        if hasattr(os, 'scandir'):
            for fobj in os.scandir(path):
                if fobj.path not in self.ignored:
                    yield not fobj.is_dir(), fobj.name
        else:
            for fname in os.listdir(path):
                if fname not in self.ignored:
                    fpath = os.path.join(path, fname)
                    yield not os.path.isdir(fpath), fname

    def put_array(self, data, path):
        with self.get_fd(path, "cb") as fd:
            fd.write(data.typecode)
            fd.write(data)

    def get_array(self, path):
        with self.get_fd(path, "rb+") as fd:
            data = array.array(fd.read(1))
            data.fromstring(fd.read())
        return data


class RawSerializer(ISerializer):
    """Serialize data to json"""
    def pack(self, value):
        value = tostr(value)

        if not isinstance(value, str):
            raise ValueError("Can't serialize object {!r}".format(type(value)))

        return value

    def unpack(self, data):
        return data


class JsonSerializer(ISerializer):
    """Serialize data to json"""
    def pack(self, value):
        try:
            return json.dumps(value)
        except Exception as exc:
            raise ValueError("Can't pickle object {!r} to json. Message: {}".format(type(value), exc))

    def unpack(self, data):
        return json.loads(data)


if yaml:
    class YAMLSerializer(ISerializer):
        """Serialize data to yaml"""
        def pack(self, value):
            try:
                return yaml.dump(value, Dumper=Dumper, encoding="utf8")
            except Exception as exc:
                raise ValueError("Can't pickle object {!r} to yaml. Message: {}".format(type(value), exc))

        def unpack(self, data):
            return yaml.load(data, Loader=Loader)


    class SAFEYAMLSerializer(ISerializer):
        """Serialize data to yaml"""
        def pack(self, value):
            try:
                return yaml.safe_dump(value, encoding="utf8")
            except Exception as exc:
                raise ValueError("Can't pickle object {!r} to yaml. Message: {}".format(type(value), exc))

        def unpack(self, data):
            return yaml.safe_load(data)
else:
    YAMLSerializer = SAFEYAMLSerializer = None


class _Raise:
    pass


if not numpy:
    ArrayStorage = None
else:
    class ArrayStorage(IArrayStorage):
        csv_file_encoding = 'ascii'

        def __init__(self, storage):
            self.storage = storage
            self.cache = {}

        def get(self, path):
            with self.storage.get_fd(path, "rb") as fd:
                stats = os.fstat(fd.fileno())
                curr_atime = stats.st_atime_ns if hasattr(stats, "st_atime_ns") else stats.st_atime
                if path in self.cache:
                    size, atime, obj, header = self.cache[path]
                    if size == stats.st_size and atime == curr_atime:
                        return obj, header

                header = fd.readline().decode(self.csv_file_encoding).strip().split(",")

                dt = fd.read().decode("utf-8").strip()

                arr = numpy.fromstring(dt.replace("\n", ','), sep=',', dtype=header[0])

                if len(dt) != 0:
                    lines = dt.count("\n") + 1
                    columns = dt.split("\n", 1)[0].count(",") + 1
                    assert lines * columns == len(arr)
                    if columns == 1:
                        arr.shape = (lines,)
                    else:
                        arr.shape = (lines, columns)

            self.cache[path] = (stats.st_size, curr_atime, arr, header[1:])
            return arr, header[1:]

        def put(self, path, data, header, append_on_exists=False):
            header = [data.dtype.name] + header

            if len(data.shape) == 1:
                # make array vertical to simplify reading
                vw = data.view().reshape((data.shape[0], 1))
            else:
                vw = data

            exists = (path in self) and append_on_exists
            with self.storage.get_fd(path, "cb" if exists else "wb") as fd:
                if exists:
                    curr_header = fd.readline().decode(self.csv_file_encoding).rstrip().split(",")
                    assert header == curr_header, \
                        "Path {!r}. Expected header ({!r}) and current header ({!r}) don't match"\
                            .format(path, header, curr_header)
                    fd.seek(0, os.SEEK_END)
                else:
                    assert not any(',' in vl or "\n" in vl for vl in header)
                    fd.write((",".join(header) + "\n").encode(self.csv_file_encoding))

                numpy.savetxt(fd, vw, delimiter=',', newline="\n", fmt="%lu")

        def rm(self, path):
            self.storage.rm(path)

        def sync(self):
            self.storage.sync()

        def __contains__(self, path):
            return path in self.storage

        def list(self, path):
            return self.storage.list(path)

        def sub_storage(self, path):
            return self.__class__(self.storage.sub_storage(path))


class Storage(object):
    def __init__(self, sstorage, serializer, array_storage):
        self.sstorage = sstorage
        self.serializer = serializer
        self.cache = {}
        self.array_storage = array_storage

    def sub_storage(self, *path):
        fpath = "/".join(path)
        return self.__class__(self.sstorage.sub_storage(fpath),
                              self.serializer,
                              None if not self.array_storage else self.array_storage.sub_storage(fpath))

    def put(self, value, *path):
        dct_value = value.raw() if isinstance(value, IStorable) else value
        serialized = self.serializer.pack(dct_value)
        self.sstorage.put(serialized, "/".join(path))

    def get(self, path, default=_Raise):
        try:
            vl = self.sstorage.get(path)
        except:
            if default is _Raise:
                raise
            return default
        return self.serializer.unpack(vl)

    def rm(self, *path):
        self.sstorage.rm("/".join(path))

    def load(self, obj_class, *path):
        return obj_class.fromraw(self.get("/".join(path)))

    # ---------------  List of values ----------------------------------------------------------------------------------

    def put_list(self, value, *path):
        serialized = self.serializer.pack([obj.raw() for obj in value])
        self.sstorage.put(serialized, "/".join(path))

    def load_list(self, obj_class, *path):
        raw_val = self.get("/".join(path))
        assert isinstance(raw_val, list)
        return [obj_class.fromraw(val) for val in raw_val]

    def __contains__(self, path):
        return path in self.sstorage

    # -------------  Raw data ------------------------------------------------------------------------------------------

    def put_raw(self, val, *path):
        fpath = "/".join(path)
        self.sstorage.put(val, fpath)
        return self.resolve_raw(fpath)

    def resolve_raw(self, fpath):
        # TODO: dirty hack
        return self.sstorage._get_fname(fpath)

    def get_raw(self, *path):
        return self.sstorage.get("/".join(path))

    def append_raw(self, value, *path):
        with self.sstorage.get_fd("/".join(path), "rb+") as fd:
            fd.seek(0, os.SEEK_END)
            fd.write(value)

    def get_fd(self, path, mode="r"):
        return self.sstorage.get_fd(path, mode)

    def sync(self):
        self.sstorage.sync()

    def __enter__(self):
        return self

    def __exit__(self, x, y, z):
        self.sync()

    def list(self, *path):
        return self.sstorage.list("/".join(path))

    def _iter_paths(self, root, path_parts, groups):

        curr = path_parts[0]
        rest = path_parts[1:]

        for is_file, name in self.list(root):
            if rest and is_file:
                continue

            rr = re.match(pattern=curr + "$", string=name)
            if rr:
                if root:
                    path = root + "/" + name
                else:
                    path = name

                new_groups = rr.groupdict().copy()
                new_groups.update(groups)

                if rest:
                    for val in self._iter_paths(path, rest, new_groups):
                        yield val
                else:
                    yield is_file, path, new_groups

    # --------------  Arrays -------------------------------------------------------------------------------------------

    def get_array(self, path):
        if not self.array_storage:
            raise ValueError("No ArrayStorage provided(probably no numpy module installed)")
        return self.array_storage.get(path)

    def put_array(self, path, data, header, append_on_exists=False):
        if not self.array_storage:
            raise ValueError("No ArrayStorage provided(probably no numpy module installed)")
        self.array_storage.put(path, data, header, append_on_exists=append_on_exists)

    def get_array_simple(self, path):
        return self.sstorage.get_array(path)

    def put_array_simple(self, data, path):
        return self.sstorage.put_array(data, path)


serializer_map = {
    'safe': SAFEYAMLSerializer,
    'yaml': YAMLSerializer,
    'json': JsonSerializer,
    'raw': RawSerializer
}


class _Def(object):
    pass


class AttredStorage(object):
    def __init__(self, storage, serializer, ext):
        self.__dict__.update({
            "_AttredStorage__storage": storage,
            "_AttredStorage__serializer" : serializer,
            "_AttredStorage__ext": ext,
            "_AttredStorage__r": storage.root_path
        })

    def __load(self, spath, ext, dir_allowed=True):
        # print("__load({!r}, ext={!r}, dir_allowed={!r})".format(spath, ext, dir_allowed))
        path = spath.split("/")
        curr = self

        if ext is not None:
            last = path[-1]
            path = path[:-1]
        else:
            last = None

        for step in path:
            if not curr.__storage.isdir(step):
                raise KeyError("Path {0!r} expected to be a dir, but it's a file at {1!r}".format(step, curr.__r))
            curr = curr.__class__(curr.__storage.sub_storage(step), curr.__serializer, curr.__ext)

        if not last:
            return True, curr

        if curr.__storage.isdir(last):
            if dir_allowed:
                return True, curr.__class__(curr.__storage.sub_storage(last), curr.__serializer, curr.__ext)
            else:
                raise KeyError("Path {0!r} expected to be a dir, but it's a file at {1!r}".format(last, curr.__r))

        return False, curr.__storage.get(last + ("." + ext if ext != '' else ''))

    def __getitem__(self, path):
        isdir, val = self.__load(path, self.__ext)
        return val if isdir else self.__serializer.unpack(val)

    def __setitem__(self, path, val):
        self.__storage.put(self.__serializer.pack(val), path + '.' + self.__ext)

    def __setattr__(self, name, val):
        self.__storage.put(self.__serializer.pack(val), name + '.' + self.__ext)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError("Can't found file '{0}.{1}' or dir '{0}' at {2!r}. {3!s}"
                                 .format(name, self.__ext, self.__r, exc))

    def __str__(self):
        return "{0.__class__.__name__}({0.__r})".format(self)

    def __iter__(self):
        return iter(self.__storage.list("."))

    def get(self, path, default=None, ext=_Def):
        try:
            return self.__load(path, self.__ext if ext is _Def else ext, dir_allowed=False)[1]
        except (AttributeError, KeyError):
            return default

    def put(self, path, val, ext=_Def):
        if ext is _Def:
            ext = self.__ext
        return self.__storage.put(val, path + ("." + ext if ext != '' else ''))

    def __len__(self):
        return len(self.__storage.list("."))



class JsonResultStorage(AttredStorage):
    def __init__(self, storage, serializer=None, ext=None):
        assert serializer is None or isinstance(serializer, JsonSerializer)
        AttredStorage.__init__(self, storage, JsonSerializer() if serializer is None else serializer, 'json')


class TxtResultStorage(AttredStorage):
    def __init__(self, storage, serializer=None, ext=None):
        assert serializer is None or isinstance(serializer, RawSerializer)
        AttredStorage.__init__(self, storage, RawSerializer() if serializer is None else serializer, 'txt')


class XMLResultStorage(AttredStorage):
    def __init__(self, storage, serializer=None, ext=None):
        assert serializer is None or isinstance(serializer, RawSerializer)
        AttredStorage.__init__(self, storage, RawSerializer() if serializer is None else serializer, 'xml')


def make_storage(url, existing=False, serializer='safe'):
    fstor = FSStorage(url, existing)
    return Storage(fstor,
                   serializer_map[serializer](),
                   ArrayStorage(fstor) if ArrayStorage else None)
