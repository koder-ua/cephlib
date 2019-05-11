from __future__ import annotations

import array
import bz2
import json
import logging
import collections
import os
import zlib
from enum import Enum, IntEnum, Flag
from io import BytesIO
from struct import Struct
from typing import (Any, Dict, Iterator, List, cast, Union, Tuple, Optional, BinaryIO, Callable, Set)

import msgpack
from koder_utils import get_discretizer

from . import CephOp, OpType


logger = logging.getLogger("cephlib")


MAX_POOL_VAL = 63
HEADER_V12 = b"OSD OPS LOG v1.2\0"
HEADER_LAST = HEADER_V12
HEADER_LAST_NAME = cast(bytes, HEADER_LAST[:-1]).decode("ascii")
ALL_SUPPORTED_HEADERS = [HEADER_V12]
assert HEADER_LAST in ALL_SUPPORTED_HEADERS
HEADER_LEN = len(HEADER_LAST)
assert all(len(hdr) == HEADER_LEN for hdr in ALL_SUPPORTED_HEADERS), "All headers must have the same size"


discretize, undiscretize = get_discretizer(255, 1.0372259)


class UnexpectedEOF(ValueError):
    pass


class RecId(IntEnum):
    ops = 1
    pools = 2
    cluster_info = 3
    params = 4
    packed = 5
    pgdump = 6


class HistoricFields(Flag):
    compact = 1
    raw = 2
    with_names = 4


class OpField(Enum):
    obj_name = 0
    flags_and_pool = 1
    pg = 2

    # compact
    duration = 3
    wait_for_pg = 4
    download = 5
    local_io = 6
    wait_for_replica = 7

    # raw
    queued_for_pg = 8
    reached_pg = 9
    started = 10
    wait_for_subop = 11
    op_commit = 12
    op_applied = 13
    sub_op_commit_rec = 14
    commit_sent = 15
    done = 16

    # calculated during unpack
    tp = 17
    pack_pool_id = 18


def prepare_historic(ops: List[CephOp], fields: HistoricFields) \
        -> Dict[OpType, Dict[OpField, Union[List[int], List[str]]]]:

    result: Dict[OpType, Dict[OpField, Union[List[int], List[str]]]] = {}

    for op in ops:
        dct = result.setdefault(op.tp, collections.defaultdict(list))

        assert op.pack_pool_id is not None
        assert 0 <= op.pack_pool_id <= MAX_POOL_VAL
        assert op.tp is not None

        flags_and_pool = (cast(int, op.tp.value) << 6) + op.pack_pool_id
        assert flags_and_pool < 256
        dct[OpField.flags_and_pool].append(flags_and_pool)
        dct[OpField.pg].append(op.pg)
        dct[OpField.duration].append(discretize(op.duration))

        if fields & HistoricFields.compact:
            timings = op.get_hl_timings()
            dct[OpField.wait_for_pg].append(discretize(timings.wait_for_pg))
            dct[OpField.download].append(discretize(timings.download))

            if op.tp in (OpType.write_primary, OpType.write_secondary):
                dct[OpField.local_io].append(discretize(timings.local_io))

                if op.tp == OpType.write_primary:
                    dct[OpField.wait_for_replica].append(discretize(timings.wait_for_replica))
            else:
                assert op.tp == OpType.read, f"Unknown op type {op.tp}"

        elif fields & HistoricFields.raw:
            assert op.tp in (OpType.write_primary, OpType.write_secondary, OpType.read), f"Unknown op type {op.tp}"

            queued_for_pg = -1
            reached_pg = -1
            sub_op_commit_rec = -1
            wait_for_subop = -1

            for evt, tm in op.events:
                if evt == 'queued_for_pg' and queued_for_pg == -1:
                    queued_for_pg = tm
                elif evt == 'reached_pg':
                    reached_pg = tm
                elif evt.startswith("sub_op_commit_rec from "):
                    sub_op_commit_rec = tm
                elif evt.startswith("waiting for subops from "):
                    wait_for_subop = tm

            assert reached_pg != -1
            assert queued_for_pg != -1

            if op.tp is OpType.write_primary:
                assert sub_op_commit_rec != -1
                assert wait_for_subop != -1

            dct[OpField.queued_for_pg].append(discretize(queued_for_pg))
            dct[OpField.reached_pg].append(discretize(reached_pg))
            dct[OpField.started].append(discretize(op.evt_map['started']))
            dct[OpField.wait_for_subop].append(discretize(wait_for_subop))
            dct[OpField.op_commit].append(discretize(op.evt_map['op_commit']))
            dct[OpField.op_applied].append(discretize(op.evt_map['op_applied']))
            dct[OpField.sub_op_commit_rec].append(discretize(sub_op_commit_rec))
            dct[OpField.commit_sent].append(discretize(op.evt_map['commit_sent']))
            dct[OpField.done].append(discretize(op.evt_map['done']))

        else:
            raise ValueError(f"Unknown field type {fields}")

        if fields & HistoricFields.with_names:
            dct[OpField.obj_name].append(op.obj_name)

    return result


NAMES_KEY = "0"
EXTRA_KEY = "1"


def pack_historic(ops: List[CephOp], fields: HistoricFields, extra: Any = None) -> bytes:
    res = {}
    prepared = prepare_historic(ops, fields)

    all_names_set: Set[str] = set()
    for op_tp, ops in prepared.items():
        for key, data in ops.items():
            if key is OpField.obj_name:
                assert all(isinstance(vl, str) for vl in data)
                all_names_set.update(data)

    all_names_lst = list(all_names_set)
    name2idx_map = {name: idx for idx, name in enumerate(all_names_lst)}

    for op_tp, ops in prepared.items():
        for key, data in ops.items():
            key_v = op_tp.value * 255 + key.value
            if key is OpField.obj_name:
                res[key_v] = [name2idx_map[name] for name in data]
            elif key is OpField.pg:
                res[key_v] = data
            else:
                try:
                    res[key_v] = array.array("B", data).tobytes()
                except OverflowError:
                    print("\n\n>>>>>>>>>>>>>>>>>>>>", key)
                    logger.error(f"Too large value in array {key}")
                    raise

    if all_names_set:
        assert NAMES_KEY not in res
        res[NAMES_KEY] = all_names_lst

    assert EXTRA_KEY not in res
    res[EXTRA_KEY] = extra

    return msgpack.packb(res, use_bin_type=True)


def unpack_historic(data: bytes) -> Tuple[Any, Iterator[Dict[str, Any]]]:
    data_dct = msgpack.unpackb(data, raw=False)

    if NAMES_KEY in data_dct:
        names_map = dict(enumerate(data_dct.pop(NAMES_KEY)))
    else:
        names_map = None

    def unpack_iter() -> Iterator[Dict[str, Any]]:
        unpacked = collections.defaultdict(dict)
        for key_vb, data in data_dct.items():
            key = OpField(key_vb % 255)
            op_type = OpType(key_vb // 255)

            if key is OpField.obj_name:
                assert names_map is not None
                res = [names_map[idx] for idx in data]
            elif key is OpField.pg:
                res = data
            else:
                arr = array.array("B")
                arr.frombytes(data)
                res = list(arr)

            if key is OpField.flags_and_pool:
                tps = []
                pools = []
                for v in res:
                    tps.append(v >> 6)
                    pools.append(v & 0x3F)
                unpacked[op_type][OpField.tp] = tps
                unpacked[op_type][OpField.pack_pool_id] = pools
            else:
                unpacked[op_type][key] = res

        for op_type, fields_v in unpacked.items():
            # all arrays have the same len

            assert len(set(map(len, fields_v.values()))) == 1
            items = list(fields_v.items())
            names_keys, arrays = zip(*items)
            names = [key.name for key in names_keys]
            for vals in zip(*arrays):
                obj = dict(zip(names, vals))
                obj['tp'] = op_type
                yield obj

    return data_dct.pop(EXTRA_KEY), unpack_iter()


def extract_pg_dump_info(pg_dump_s: str) -> Dict[str, Union[List[str], List[int]]]:
    res: Dict[str, Union[List[str], List[int]]] = collections.defaultdict(list)
    for pg_info in json.loads(pg_dump_s)['pg_stats']:
        info = {'pgid': pg_info['pgid'], 'acting': pg_info['acting']}
        info.update(pg_info['stat_sum'])
        for key, val in info.items():
            res[key].append(val)
    return {key: list(vals) for key, vals in res.items()}


def pack_record(rec_tp: RecId, data: Any) -> Optional[Tuple[RecId, bytes]]:

    if rec_tp in (RecId.pools, RecId.cluster_info, RecId.params):
        assert isinstance(data, dict), str(data)[:100]
        return rec_tp, msgpack.packb(data, use_bin_type=True)
    elif rec_tp == RecId.pgdump:
        assert isinstance(data, str), str(data)[:100]
        return rec_tp, msgpack.packb(extract_pg_dump_info(data), use_bin_type=True)
    elif rec_tp == RecId.ops:
        osd_id, ctime, ops, fields = data
        assert isinstance(osd_id, int), str(osd_id)
        assert isinstance(ctime, int), str(ctime)
        assert isinstance(ops, list), str(ops)
        assert all(isinstance(rec, CephOp) for rec in ops), str([op for op in ops if not isinstance(op, CephOp)])
        return RecId.ops, pack_historic(ops, fields=fields, extra=[osd_id, ctime])
    else:
        raise AssertionError(f"Unknown record type {rec_tp}")


def unpack_record(rec_tp: RecId, data: bytes) -> Any:
    if rec_tp in (RecId.pools, RecId.params, RecId.cluster_info, RecId.pgdump):
        return msgpack.unpackb(data, raw=False)
    elif rec_tp == RecId.ops:
        (osd_id, ctime), iter = unpack_historic(data)
        ops: List[Dict[str, Any]] = []
        extra = {"osd_id": osd_id, "time": ctime}
        for op in iter:
            op.update(extra)
            ops.append(op)
        return ops
    else:
        raise AssertionError(f"Unknown record type {rec_tp}")


def format_op_compact(op: Dict[str, Any]) -> str:
    assert 'wait_for_pg' in op
    assert 'local_io' in op
    sep = "  "

    if op['tp'] in (OpType.write_primary, OpType.write_secondary):
        assert 'download' in op
        if op['tp'] is OpType.write_secondary:
            assert 'wait_for_replica' not in op
            name = "WRITE_SECONDARY"
            remote_io = ""
        else:
            name = "WRITE_PRIMARY"
            remote_io = f"{sep}remote_io={int(op['remote_io']):>5d}"
        dload = f"dload={int(op['download']):>5d}{sep}"
    elif op['tp'] == OpType.read:
        assert 'download' not in op
        assert 'wait_for_replica' not in op
        name = "READ"
        dload = " " * len(f"dload={1:>5d}")
        remote_io = ""
    else:
        assert False, f"Unknown op {op['tp']}"

    return f"{name:<12s}{sep}{op['pool_name']:>25s}:{op['pg']:<5x}{sep}" + \
           f"osd_id={op['osd_id']:>4d}{sep}" + \
           f"duration={int(op['duration']):>5d}{sep}" + \
           f"{dload}" + \
           f"wait_pg={int(op['wait_for_pg']):>5d}{sep}" + \
           f"local_io={int(op['local_io']):>5d}{remote_io}"


class RecordFile:
    rec_header = Struct("!II")

    def __init__(self, fd: BinaryIO, pack_each: int = 2 ** 20,
                 compress: Callable[[bytes], bytes] = bz2.compress,
                 decompress: Callable[[bytes], bytes] = bz2.decompress) -> None:

        if pack_each != 0:
            assert fd.seekable()

        self.fd = fd
        self.pack_each = pack_each
        self.compress = compress
        self.decompress = decompress
        self.cached = []
        self.cache_size = 0
        self.unpacked_offset = 0

    def close(self) -> None:
        self.fd = None  # typing: ignore

    def seek_to_last_valid_record(self) -> None:
        for _ in self.iter_records():
            pass

    def prepare_for_append(self, truncate_invalid: bool = False) -> bool:
        """returns true if any data truncated alread"""
        header = self.read_file_header()

        if header is None:
            self.fd.seek(0, os.SEEK_SET)
            self.fd.write(HEADER_LAST)
            return False
        else:
            assert header == HEADER_LAST, f"Can only append to file with {HEADER_LAST_NAME} version"
            if truncate_invalid:
                try:
                    self.seek_to_last_valid_record()
                    return False
                except (UnexpectedEOF, AssertionError, ValueError):
                    self.fd.truncate()
                    return True
            else:
                self.seek_to_last_valid_record()
                return False

    def tell(self) -> int:
        return self.fd.tell()

    def read_file_header(self) -> Optional[bytes]:
        """
        read header from file, return fd positioned to first byte after the header
        check that header is in supported headers, fail otherwise
        must be called from offset 0
        """
        assert self.fd.seekable()
        assert self.fd.tell() == 0, "read_file_header must be called from beginning of the file"
        self.fd.seek(0, os.SEEK_END)
        size = self.fd.tell()
        if size == 0:
            return None

        assert self.fd.readable()
        self.fd.seek(0, os.SEEK_SET)
        assert size >= HEADER_LEN, "Incorrect header"
        hdr = self.fd.read(HEADER_LEN)
        assert hdr in ALL_SUPPORTED_HEADERS, f"Unknown header {hdr!r}"
        return hdr

    def make_header_for_rec(self, rec_type: RecId, data: bytes) -> bytes:
        id_bt = bytes((rec_type.value,))
        checksum = zlib.adler32(data, zlib.adler32(id_bt))
        return self.rec_header.pack(checksum, len(data) + 1) + id_bt

    def flush(self) -> None:
        self.fd.flush()

    def write_record(self, rec_type: RecId, data: bytes, flush: bool = True) -> None:
        header = self.make_header_for_rec(rec_type, data)
        truncate = False

        if self.pack_each != 0:
            if self.cache_size == 0:
                self.unpacked_offset = self.fd.tell()

            self.cached.extend([header, data])
            self.cache_size += len(header) + len(data)

            if self.cache_size >= self.pack_each:
                data = self.compress(b"".join(self.cached))
                header = self.make_header_for_rec(RecId.packed, data)
                logger.debug(f"Repack data orig size={self.cache_size // 1024}KiB  " +
                             f"new_size={(len(header) + len(data)) // 1024}KiB")
                self.cached = []
                self.cache_size = 0
                self.fd.seek(self.unpacked_offset)
                truncate = True
                self.unpacked_offset = self.fd.tell()

        self.fd.write(header)
        self.fd.write(data)
        if truncate:
            self.fd.truncate()

        if flush:
            self.fd.flush()

    def iter_records(self) -> Iterator[Tuple[RecId, bytes]]:
        """
        iterate over records in output file, written with write_record function
        """

        rec_size = self.rec_header.size
        unpack = self.rec_header.unpack

        offset = self.fd.tell()
        self.fd.seek(0, os.SEEK_END)
        size = self.fd.tell()
        self.fd.seek(offset, os.SEEK_SET)

        try:
            while offset < size:
                data = self.fd.read(rec_size)
                if len(data) != rec_size:
                    raise UnexpectedEOF()
                checksum, data_size = unpack(data)
                data = self.fd.read(data_size)
                if len(data) != data_size:
                    raise UnexpectedEOF()
                assert checksum == zlib.adler32(data), f"record corrupted at offset {offset}"

                rec_id = RecId(data[0])
                data = data[1:]

                if rec_id == RecId.packed:
                    yield from RecordFile(cast(BinaryIO, BytesIO(self.decompress(data)))).iter_records()
                else:
                    yield rec_id, data

                offset += rec_size + data_size
        except Exception:
            self.fd.seek(offset, os.SEEK_SET)
            raise


def parse_historic_file(os_fd: BinaryIO) -> Iterator[Tuple[RecId, Any]]:
    fd = RecordFile(os_fd)
    header = fd.read_file_header()
    if header is None:
        return

    riter = fd.iter_records()
    pools_map: Optional[Dict[int, Tuple[str, int]]] = None

    for rec_type, data in riter:
        if rec_type in (RecId.ops, RecId.cluster_info, RecId.pools, RecId.pgdump):
            res = unpack_record(rec_type, data)
            if rec_type == RecId.ops:
                assert pools_map is not None, "No 'pools' record found in file"
                for op in res:
                    op['pool_name'], op['pool'] = pools_map[op['pack_pool_id']]
            elif rec_type == RecId.pools:
                # int(...) is a workaround for json issue - it only allows string to be keys
                pools_map = {pack_id: (name, int(real_id)) for real_id, (name, pack_id) in res.items()}
            yield rec_type, res
        elif rec_type == RecId.params:
            params = json.loads(data.decode())
            yield rec_type, params
        else:
            raise AssertionError(f"Unknown rec type {rec_type} at offset {fd.tell()}")
