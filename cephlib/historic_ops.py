from __future__ import annotations

import datetime
import os
import random
import re
import time
import abc
import bisect
import bz2
import json
import logging
import pprint
import zlib
from enum import Enum, IntEnum
from io import BytesIO
from struct import Struct
from dataclasses import dataclass, field
from typing import (Any, Dict, Tuple, Iterator, List, Iterable, cast, Type, Callable,
                    Optional, NamedTuple, NewType, Match, BinaryIO, TypeVar, Generic)


logger = logging.getLogger("cephlib.ops")

compress = bz2.compress
decompress = bz2.decompress
DEFAULT_DURATION = 600
DEFAULT_SIZE = 20

MAX_PG_VAL = (2 ** 16 - 1)
MAX_POOL_VAL = 63
MKS_TO_MS = 1000
S_TO_MS = 1000


OpRec = NewType('OpRec', Dict[str, Any])


class OpType(Enum):
    read = 0
    write_primary = 1
    write_secondary = 2


class ParseResult(Enum):
    ok = 0
    failed = 1
    ignored = 2
    unknown = 3


class OpDescription(NamedTuple):
    type: OpType
    client:  str
    pool_id: int
    pg: int
    obj_name: str
    op_size: Optional[int]


class HLTimings(NamedTuple):
    download: int
    wait_for_pg: int
    local_io: int
    wait_for_replica: int


@dataclass
class CephOp:
    raw_data: OpRec
    description: str
    initiated_at: int
    obj_name: str
    client: str
    op_size: Optional[int]
    tp: Optional[OpType]
    duration: int
    pool_id: int
    pg: int
    events: List[Tuple[str, int]]
    pack_pool_id: Optional[int] = None

    def __post_init__(self):
        self.evt_map = dict(self.events)

    @classmethod
    def parse_op(cls, op: OpRec) -> Tuple[ParseResult, Optional['CephOp']]:
        try:
            res, descr = parse_description(op['description'])
        except AssertionError:
            descr = None
            res = ParseResult.failed

        if descr is None:
            return res, None

        initiated_at = to_unix_ms(op['initiated_at'])
        return res, cls(
            raw_data=op,
            description=op['description'],
            initiated_at=initiated_at,
            tp=descr.type,
            duration=int(op['duration'] * 1000),
            pool_id=descr.pool_id,
            pg=descr.pg,
            obj_name=descr.obj_name,
            client=descr.client,
            op_size=descr.op_size,
            events=parse_events(op, initiated_at))

    def get_hl_timings(self) -> HLTimings:
        assert self.tp is not None
        return get_hl_timings(self.tp, self.evt_map)

    def short_descr(self) -> str:
        return f"{self.tp.name} pg={self.pool_id}.{self.pg:x} size={self.op_size} duration={self.duration}"

    def __str__(self) -> str:
        return f"{self.description}\n    initiated_at: {self.initiated_at}\n" + \
               f"    tp: {self.tp}\n    duration: {self.duration}\n    " + \
               f"pool_id: {self.pool_id}\n    pack_pool_id: {self.pack_pool_id}\n" + \
               f"    pg: {self.pg}\n    events:\n        " + \
               "\n        ".join(f"{name}: {tm}" for name, tm in sorted(self.evt_map.items()))


client_re = r"(?P<client>[^ ]*)"
pool_pg_re = r"(?P<pool>\d+)\.(?P<pg>[0-9abcdef]+)"
obj_name_re = r"(?P<obj_name>(?P=pool):[^ ]*)"

osd_op_re = re.compile(rf"osd_op\({client_re} {pool_pg_re} {obj_name_re} \[.*?\b(?P<op>(read|write|writefull)) " +
                       rf"(?P<start_offset>\d+)~(?P<end_offset>\d+).*?\] ")

osd_repop_re = re.compile(rf"osd_repop\({client_re} {pool_pg_re} [^ ]* {obj_name_re} ")


ignored_tps = {'delete', 'call', 'watch', 'stat', 'notify-ack'}
ignored_types = {"osd_repop_reply", "pg_info", "replica scrub", "rep_scrubmap", "pg_update_log_missing",
                 "MOSDScrubReserve"}


def get_pool_pg(rr: Match) -> Tuple[int, int]:
    """
    returns pool and pg for op
    """
    return int(rr.group('pool')), int(rr.group('pg'), 16)


def to_unix_ms(dtm: str) -> int:
    # "2019-02-03 20:53:47.429996"
    date, tm = dtm.split()
    y, m, d = date.split("-")
    h, minute, smks = tm.split(':')
    s, mks = smks.split(".")
    dt = datetime.datetime(int(y), int(m), int(d), int(h), int(minute), int(s))
    return int(time.mktime(dt.timetuple()) * S_TO_MS) + int(mks) // MKS_TO_MS


def parse_events(op: OpRec, initiated_at: int) -> List[Tuple[str, int]]:
    return [(evt["event"], to_unix_ms(evt["time"]) - initiated_at)
            for evt in op["type_data"]["events"] if evt["event"] != "initiated"]


def parse_description(descr: str) -> Tuple[ParseResult, Optional[OpDescription]]:
    """
    Get type for operation
    """
    rr = osd_op_re.match(descr)
    if rr:
        pool, pg = get_pool_pg(rr)
        if rr.group('op') == 'read':
            tp = OpType.read
        else:
            assert rr.group('op') in ('write', 'writefull')
            tp = OpType.write_primary
        client = rr.group('client')
        object_name = rr.group('obj_name')
        size = int(rr.group('end_offset')) - int(rr.group('start_offset'))
        return ParseResult.ok, OpDescription(tp, client, pool, pg, object_name, size)

    rr = osd_repop_re.match(descr)
    if rr:
        pool, pg = get_pool_pg(rr)
        return ParseResult.ok, \
            OpDescription(OpType.write_secondary, rr.group('client'), pool, pg, rr.group('obj_name'), None)

    raw_tp = descr.split("(", 1)[0]
    if raw_tp in ignored_types:
        return ParseResult.ignored, None
    elif raw_tp == 'osd_op':
        for info in descr.split("[", 1)[1].split("]", 1)[0].split(","):
            if info.split()[0] in ignored_tps.intersection():
                return ParseResult.ignored, None

    return ParseResult.unknown, None


def get_hl_timings(tp: OpType, evt_map: Dict[str, int]) -> HLTimings:
    qpg_at = evt_map.get("queued_for_pg", -1)
    started = evt_map.get("started", evt_map.get("reached_pg", -1))

    try:
        if tp == OpType.write_secondary:
            # workaround for jewel ceph
            local_done = evt_map["sub_op_applied"] if "sub_op_applied" in evt_map else evt_map['done']
        elif tp == OpType.write_primary:
            local_done = evt_map["op_applied"] if "op_applied" in evt_map else evt_map['done']
        else:
            local_done = evt_map["done"]
    except KeyError:
        local_done = -1

    last_replica_done = -1
    subop = -1
    for evt, tm in evt_map.items():
        if evt.startswith("waiting for subops from") or evt == "wait_for_subop":
            last_replica_done = tm
        elif evt.startswith("sub_op_commit_rec from") or evt == "sub_op_commit_rec":
            subop = max(tm, subop)

    wait_for_pg = started - qpg_at
    assert wait_for_pg >= 0
    local_io = local_done - started
    assert local_io >= 0, f"local_io = {local_done} - {started} < 0"

    wait_for_replica = -1
    if tp in (OpType.write_primary, OpType.write_secondary):
        download = qpg_at
        if tp == OpType.write_primary:
            assert subop != -1
            assert last_replica_done != -1
            wait_for_replica = subop - last_replica_done
    else:
        download = -1

    return HLTimings(download=download, wait_for_pg=wait_for_pg, local_io=local_io,
                     wait_for_replica=wait_for_replica)


class RecId(IntEnum):
    ops = 1
    pools = 2
    cluster_info = 3
    params = 4
    packed = 5
    pgdump = 6


class DiscretizerExt:
    # discretization constants
    overfloat = 255
    step_coef = 1.0372259

    val = 0
    table = [val]
    for _ in range(254):
        val = max(round(step_coef * val), val + 1)
        table.append(val)

    @classmethod
    def discretize(cls, vl: float) -> int:
        return min(cls.overfloat, bisect.bisect_left(cls.table, round(vl)))

    @classmethod
    def undiscretize(cls, vl: int) -> int:
        return cls.table[vl]


class IPackerBase(metaclass=abc.ABCMeta):
    """
    Abstract base class to back ceph operations to bytes
    """
    name = None  # type: str

    @classmethod
    @abc.abstractmethod
    def pack_op(cls, op: CephOp) -> bytes:
        pass

    @classmethod
    @abc.abstractmethod
    def unpack_op(cls, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
        pass

    @staticmethod
    @abc.abstractmethod
    def format_op(op: Dict[str, Any]) -> str:
        pass

    op_header = Struct("!HI")

    @classmethod
    def unpack(cls, rec_tp: RecId, data: bytes) -> Any:
        if rec_tp in (RecId.pools, RecId.params, RecId.cluster_info):
            return json.loads(data.decode())
        elif rec_tp == RecId.ops:
            osd_id, ctime = cls.op_header.unpack(data[:cls.op_header.size])
            offset = cls.op_header.size
            ops: List[Dict[str, Any]] = []
            while offset < len(data):
                params, offset = cls.unpack_op(data, offset)
                params.update({"osd_id": osd_id, "time": ctime})
                ops.append(params)
            return ops
        else:
            raise AssertionError(f"Unknown record type {rec_tp}")

    @classmethod
    def pack_record(cls, rec_tp: RecId, data: Any) -> Optional[Tuple[RecId, bytes]]:
        if rec_tp in (RecId.pools, RecId.cluster_info, RecId.params):
            assert isinstance(data, dict), str(data)
            return rec_tp, json.dumps(data).encode()
        elif rec_tp == RecId.pgdump:
            assert isinstance(data, str), str(data)
            return rec_tp, data.encode()
        elif rec_tp == RecId.ops:
            osd_id, ctime, ops = data
            assert isinstance(osd_id, int), str(osd_id)
            assert isinstance(ctime, int), str(ctime)
            assert isinstance(ops, list), str(ops)
            assert all(isinstance(rec, CephOp) for rec in ops), str([op for op in ops if not isinstance(op, CephOp)])
            packed: List[bytes] = []
            for op in ops:
                try:
                    packed.append(cls.pack_op(op))
                except Exception:
                    logger.exception(f"Failed to pack op:\n{pprint.pformat(op.raw_data)}")
            packed_b = b"".join(packed)
            if packed:
                return RecId.ops, cls.op_header.pack(osd_id, ctime) + packed_b
        else:
            raise AssertionError(f"Unknown record type {rec_tp}")

    @classmethod
    def pack_iter(cls, data_iter: Iterable[Tuple[RecId, Any]]) -> Iterator[Tuple[RecId, bytes]]:
        for rec_tp, data in data_iter:
            rec = cls.pack_record(rec_tp, data)
            if rec:
                yield rec


class CompactPacker(IPackerBase):
    """
    Compact packer - pack op to 6-8 bytes with timings for high-level stages - downlaod, wait pg, local io, remote io
    """

    name = 'compact'

    OPRecortWP = Struct("!BHBBBBB")
    OPRecortWS = Struct("!BHBBBB")
    OPRecortR = Struct("!BHBBB")

    @classmethod
    def pack_op(cls, op: CephOp) -> bytes:
        assert op.pack_pool_id is not None
        assert 0 <= op.pack_pool_id <= MAX_POOL_VAL
        assert op.tp is not None

        # overflow pg
        if op.pg > MAX_PG_VAL:
            logger.debug("Too large pg = %d", op.pg)

        pg = min(MAX_PG_VAL, op.pg)

        timings = op.get_hl_timings()
        flags_and_pool = (cast(int, op.tp.value) << 6) + op.pack_pool_id
        assert flags_and_pool < 256

        if op.tp == OpType.write_primary:
            return cls.OPRecortWP.pack(flags_and_pool, pg,
                                       DiscretizerExt.discretize(op.duration),
                                       DiscretizerExt.discretize(timings.wait_for_pg),
                                       DiscretizerExt.discretize(timings.download),
                                       DiscretizerExt.discretize(timings.local_io),
                                       DiscretizerExt.discretize(timings.wait_for_replica))

        if op.tp == OpType.write_secondary:
            return cls.OPRecortWS.pack(flags_and_pool, pg,
                                       DiscretizerExt.discretize(op.duration),
                                       DiscretizerExt.discretize(timings.wait_for_pg),
                                       DiscretizerExt.discretize(timings.download),
                                       DiscretizerExt.discretize(timings.local_io))

        assert op.tp == OpType.read, "Unknown op type {}".format(op.tp)
        return cls.OPRecortR.pack(flags_and_pool, pg,
                                  DiscretizerExt.discretize(op.duration),
                                  DiscretizerExt.discretize(timings.wait_for_pg),
                                  DiscretizerExt.discretize(timings.download))

    @classmethod
    def unpack_op(cls, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:

        undiscretize_l: Callable[[int], float] = DiscretizerExt.table.__getitem__
        flags_and_pool = data[offset]
        op_type = OpType(flags_and_pool >> 6)
        pool = flags_and_pool & 0x3F

        if op_type == OpType.write_primary:
            _, pg, duration, wait_for_pg, download, local_io, wait_for_replica = \
                cls.OPRecortWP.unpack(data[offset: offset + cls.OPRecortWP.size])

            return {'tp': op_type,
                    'pack_pool_id': pool,
                    'pg': pg,
                    'duration': undiscretize_l(duration),
                    'wait_for_pg': undiscretize_l(wait_for_pg),
                    'local_io': undiscretize_l(local_io),
                    'wait_for_replica': undiscretize_l(wait_for_replica),
                    'download': undiscretize_l(download),
                    'packer': cls.name}, offset + cls.OPRecortWP.size

        if op_type == OpType.write_secondary:
            _, pg, duration, wait_for_pg, download, local_io = \
                cls.OPRecortWS.unpack(data[offset: offset + cls.OPRecortWS.size])
            return {'tp': op_type,
                    'pack_pool_id': pool,
                    'pg': pg,
                    'duration': undiscretize_l(duration),
                    'wait_for_pg': undiscretize_l(wait_for_pg),
                    'local_io': undiscretize_l(local_io),
                    'download': undiscretize_l(download),
                    'packer': cls.name}, offset + cls.OPRecortWS.size

        assert op_type == OpType.read, f"Unknown op type {op_type}"
        _, pg, duration, wait_for_pg, local_io = cls.OPRecortR.unpack(data[offset: offset + cls.OPRecortR.size])
        return {'tp': op_type,
                'pack_pool_id': pool,
                'pg': pg,
                'duration': undiscretize_l(duration),
                'wait_for_pg': undiscretize_l(wait_for_pg),
                'local_io': undiscretize_l(local_io),
                'packer': cls.name}, offset + cls.OPRecortR.size

    @staticmethod
    def format_op(op: Dict[str, Any]) -> str:
        if op['tp'] == OpType.write_primary:
            assert 'wait_for_pg' in op
            assert 'download' in op
            assert 'local_io' in op
            assert 'wait_for_replica' in op
            return (("WRITE_PRIMARY     {:>25s}:{:<5x} osd_id={:>4d}   duration={:>5d}   dload={:>5d}" +
                     "   wait_pg={:>5d}   local_io={:>5d}   remote_io={:>5d}").
                    format(op['pool_name'], op['pg'], op['osd_id'], int(op['duration']), int(op['download']),
                           int(op['wait_for_pg']), int(op['local_io']), int(op['wait_for_replica'])))
        elif op['tp'] == OpType.write_secondary:
            assert 'wait_for_pg' in op
            assert 'download' in op
            assert 'local_io' in op
            assert 'wait_for_replica' not in op
            return (("WRITE_SECONDARY   {:>25s}:{:<5x} osd_id={:>4d}   duration={:>5d}   " +
                     "dload={:>5d}   wait_pg={:>5d}   local_io={:>5d}").
                    format(op['pool_name'], op['pg'], op['osd_id'], int(op['duration']),
                           int(op['download']), int(op['wait_for_pg']), int(op['local_io'])))
        elif op['tp'] == OpType.read:
            assert 'wait_for_pg' in op
            assert 'download' not in op
            assert 'local_io' in op
            assert 'wait_for_replica' not in op
            return (("READ              {:>25s}:{:<5x} osd_id={:>4d}" +
                     "   duration={:>5d}                 wait_pg={:>5d}   local_io={:>5d}").
                    format(op['pool_name'], op['pg'], op['osd_id'], int(op['duration']), int(op['wait_for_pg']),
                           int(op['local_io'])))
        else:
            assert False, f"Unknown op {op['tp']}"


class RawPacker(IPackerBase):
    """
    Compact packer - pack op to 6-8 bytes with timings for high-level stages - downlaod, wait pg, local io, remote io
    """

    name = 'raw'

    OPRecortWP = Struct("!BH" + 'B' * 10)
    OPRecortWS = Struct("!BH" + 'B' * 7)

    @classmethod
    def pack_op(cls, op: CephOp) -> bytes:
        assert op.pack_pool_id is not None
        assert 0 <= op.pack_pool_id <= MAX_POOL_VAL
        assert op.tp is not None

        # overflow pg
        if op.pg > MAX_PG_VAL:
            logger.debug("Too large pg = %d", op.pg)

        pg = min(MAX_PG_VAL, op.pg)

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

        if op.tp == OpType.write_primary:
            assert sub_op_commit_rec != -1
            assert wait_for_subop != -1

        flags_and_pool = (cast(int, op.tp.value) << 6) + op.pack_pool_id
        assert flags_and_pool < 256

        try:
            if op.tp == OpType.write_primary:
                # first queued_for_pg
                # last reached_pg
                # started
                # wait_for_subop
                # op_commit
                # op_applied
                # last sub_op_commit_rec
                # commit_sent
                # done
                return cls.OPRecortWP.pack(flags_and_pool, pg,
                                           DiscretizerExt.discretize(op.duration),
                                           DiscretizerExt.discretize(queued_for_pg),
                                           DiscretizerExt.discretize(reached_pg),
                                           DiscretizerExt.discretize(op.evt_map['started']),
                                           DiscretizerExt.discretize(wait_for_subop),
                                           DiscretizerExt.discretize(op.evt_map['op_commit']),
                                           DiscretizerExt.discretize(op.evt_map['op_applied']),
                                           DiscretizerExt.discretize(sub_op_commit_rec),
                                           DiscretizerExt.discretize(op.evt_map['commit_sent']),
                                           DiscretizerExt.discretize(op.evt_map['done']))

            if op.tp == OpType.write_secondary:
                # first queued_for_pg
                # last reached_pg
                # started
                # commit_send
                # sub_op_applied
                # done
                return cls.OPRecortWS.pack(flags_and_pool, pg,
                                           DiscretizerExt.discretize(op.duration),
                                           DiscretizerExt.discretize(queued_for_pg),
                                           DiscretizerExt.discretize(reached_pg),
                                           DiscretizerExt.discretize(op.evt_map['started']),
                                           DiscretizerExt.discretize(op.evt_map['commit_sent']),
                                           DiscretizerExt.discretize(op.evt_map['sub_op_applied']),
                                           DiscretizerExt.discretize(op.evt_map['done']))
        except KeyError:
            logger.error(pprint.pprint(op.evt_map))
            raise
        assert op.tp == OpType.read, f"Unknown op type {op.tp}"
        return b""

    @classmethod
    def unpack_op(cls, data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:

        undiscretize_l = DiscretizerExt.table.__getitem__  # type: Callable[[int], float]
        flags_and_pool = data[offset]
        op_type = OpType(flags_and_pool >> 6)
        pool = flags_and_pool & 0x3F

        if op_type == OpType.write_primary:
            _, pg, duration, queued_for_pg, reached_pg, started, wait_for_subop, \
                op_commit, op_applied, sub_op_commit_rec, commit_sent, done = \
                cls.OPRecortWP.unpack(data[offset: offset + cls.OPRecortWP.size])

            return {'tp': op_type,
                    'pack_pool_id': pool,
                    'pg': pg,
                    'duration': undiscretize_l(duration),
                    'queued_for_pg': undiscretize_l(queued_for_pg),
                    'reached_pg': undiscretize_l(reached_pg),
                    'started': undiscretize_l(started),
                    'wait_for_subop': undiscretize_l(wait_for_subop),
                    'op_commit': undiscretize_l(op_commit),
                    'op_applied': undiscretize_l(op_applied),
                    'sub_op_commit_rec': undiscretize_l(sub_op_commit_rec),
                    'commit_sent': undiscretize_l(commit_sent),
                    'done': undiscretize_l(done),
                    'packer': cls.name}, offset + cls.OPRecortWP.size

        if op_type == OpType.write_secondary:
            _, pg, duration, queued_for_pg, reached_pg, started, commit_send, sub_op_applied, done = \
                cls.OPRecortWS.unpack(data[offset: offset + cls.OPRecortWS.size])
            return {'tp': op_type,
                    'pack_pool_id': pool,
                    'pg': pg,
                    'duration': undiscretize_l(duration),
                    'queued_for_pg': undiscretize_l(queued_for_pg),
                    'reached_pg': undiscretize_l(reached_pg),
                    'started': undiscretize_l(started),
                    'commit_send': undiscretize_l(commit_send),
                    'sub_op_applied': undiscretize_l(sub_op_applied),
                    'done': undiscretize_l(done),
                    'packer': cls.name}, offset + cls.OPRecortWS.size

        assert False, f"Unknown op type {op_type}"

    @staticmethod
    def format_op(op: Dict[str, Any]) -> str:
        if op['tp'] == OpType.write_primary:
            return (("WRITE_PRIMARY     {:>25s}:{:<5x} osd_id={:>4d}   q_for_pg={:>5d}  reached_pg={:>5d}" +
                     "   started={:>5d}   subop={:>5d}   commit={:>5d}   applied={:>5d} subop_ready={:>5d}" +
                     "  com_send={:>5d} done={:>5d}").
                    format(op['pool_name'], op['pg'], op['osd_id'], int(op['queued_for_pg']),
                           int(op['reached_pg']),
                           int(op['started']),
                           int(op['wait_for_subop']),
                           int(op['op_commit']),
                           int(op['op_applied']),
                           int(op['sub_op_commit_rec']),
                           int(op['commit_sent']),
                           int(op['done']),
                           ))
        elif op['tp'] == OpType.write_secondary:
            return (("WRITE_SECONDARY   {:>25s}:{:<5x} osd_id={:>4d}   q_for_pg={:>5d}  reached_pg={:>5d}" +
                     "   started={:>5d}                                applied={:>5d}" +
                     "                    com_send={:>5d} done={:>5d}").
                    format(op['pool_name'], op['pg'], op['osd_id'], int(op['queued_for_pg']),
                           int(op['reached_pg']),
                           int(op['started']),
                           int(op['commit_send']),
                           int(op['sub_op_applied']),
                           int(op['done']),
                           ))
        else:
            assert False, f"Unknown op {op['tp']}"


ALL_PACKERS = [CompactPacker, RawPacker]

IPacker = Type[IPackerBase]


def get_historic_packer(name: str) -> IPacker:
    for packer_cls in ALL_PACKERS:
        if packer_cls.name == name:
            return packer_cls
    raise AssertionError(f"Unknown packer {name}")


class UnexpectedEOF(ValueError):
    pass


HEADER_V12 = b"OSD OPS LOG v1.2\0"
HEADER_LAST = HEADER_V12
HEADER_LAST_NAME = cast(bytes, HEADER_LAST[:-1]).decode("ascii")
ALL_SUPPORTED_HEADERS = [HEADER_V12]
assert HEADER_LAST in ALL_SUPPORTED_HEADERS
HEADER_LEN = len(HEADER_LAST)
assert all(len(hdr) == HEADER_LEN for hdr in ALL_SUPPORTED_HEADERS), "All headers must have the same size"


class RecordFile:
    rec_header = Struct("!II")

    def __init__(self, fd: BinaryIO, pack_each: int = 2 ** 20,
                 packer: Tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]] = (compress, decompress)) -> None:
        if pack_each != 0:
            assert fd.seekable()

        self.fd = fd
        self.pack_each = pack_each
        self.compress, self.decompress = packer
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
            assert header == HEADER_LAST, "Can only append to file with {} version".format(HEADER_LAST_NAME)
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
                logger.debug("Repack data orig size=%sKiB new_size=%sKiB",
                             self.cache_size // 1024, (len(header) + len(data)) // 1024)
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

    packer: Optional[Type[IPackerBase]] = None

    riter = fd.iter_records()
    pools_map: Optional[Dict[int, Tuple[str, int]]] = None

    for rec_type, data in riter:
        if rec_type in (RecId.ops, RecId.cluster_info, RecId.pools):
            assert packer is not None, "No 'params' record found in file"
            res = packer.unpack(rec_type, data)
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
            packer = get_historic_packer(params['packer'])
            yield rec_type, params
        else:
            raise AssertionError(f"Unknown rec type {rec_type} at offset {fd.tell()}")


def print_records_from_file(file: str, limit: Optional[int]) -> None:
    with open(file, "rb") as fd:
        idx = 0
        for tp, val in parse_historic_file(fd):
            if tp == RecId.ops:
                for op in val:
                    idx += 1
                    if op['packer'] == 'compact':
                        print(CompactPacker.format_op(op))
                    if op['packer'] == 'raw':
                        print(RawPacker.format_op(op))
                    if limit is not None and idx == limit:
                        return


@dataclass
class HistoricBin:
    total_count: int
    bin_levels: List[int]

    # stage => [counts]
    bins: Dict[str, List[int]] = field(init=False, default=None)

    def __post_init__(self) -> None:
        assert sorted(self.bin_levels) == self.bin_levels
        assert len(set(self.bin_levels)) == len(self.bin_levels)
        self.bins = {name: [0] * len(self.bin_levels) for name in HLTimings.__annotations__}

    def add_op(self, op: HLTimings) -> None:
        self.total_count += 1
        for name, val in op.__dict__.items():
            self.bins[name][bisect.bisect_left(self.bin_levels, val)] += 1


T = TypeVar('T')


@dataclass
class SamplingList(Generic[T]):
    count: int
    total_processed_count: int = field(init=False, default=0)
    skip: int = field(init=False, default=0)
    curr_counter: int = field(init=False, default=0)
    samples: List[T] = field(init=False, default_factory=list)
    pretenders: List[T] = field(init=False, default_factory=list)

    def add(self, obj: T) -> None:
        self.total_processed_count += 1
        if self.curr_counter == self.skip:
            self.pretenders.append(obj)
            self.curr_counter = 0
            if len(self.pretenders) == self.count:
                self.samples = random.sample(self.samples + self.pretenders, self.count)
        else:
            self.curr_counter += 1


class HistStorage:
    def __init__(self) -> None:
        self.storage: Dict[int, HistoricBin] = {}  # key is max bin time value

    def add_op(self, op: HLTimings) -> None:
        pass



#
# import sys
# import pathlib
# import collections
# from typing import Set, Tuple, Dict, List, Iterator, IO
#
# from cephlib.sensors_rpc_plugin import CephOp
#
#
# def find_closest_pars(pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
#     # find pairs, which never breaks by other event (possible direct deps)
#     close_pairs = set()  # type: Set[Tuple[str, str]]
#
#     # need all events, that appears on first and second positions, as only such events can be in the middle of any other
#     e1s, e2s = zip(*pairs)
#     all_events = set(e1s).union(e2s)
#
#     for e1, e2 in pairs:
#         # for each pair iterate over all other event and check that pairs with this new env inside
#         # are missing in allowed_pairs
#         for e3 in all_events:
#             if (e1, e3) in pairs and (e3, e2) in pairs:
#                 break
#         else:
#             close_pairs.add((e1, e2))
#
#     return close_pairs
#
#
# ALL_STAGES = ['queued_for_pg', 'reached_pg', 'started', 'op_commit', 'sub_op_commit_rec', 'op_applied',
#               'commit_sent', 'done']
#
# prev_op = {'commit_sent': 'sub_op_commit_rec',
#            'done': 'commit_sent',
#            'op_applied': 'op_commit',
#            'op_commit': 'started',
#            'reached_pg': 'queued_for_pg',
#            'started': 'reached_pg',
#            'sub_op_commit_rec': 'started'}
#
#
# def detect_order(pairs: Set[Tuple[str, str]]) -> Iterator[Tuple[str, List[str]]]:
#     next_evts = {e1 for e1, _ in pairs}.difference({e2 for _, e2 in pairs})
#     processed: Set[str] = set()
#     while next_evts:
#         new_next = set()
#         for evt in next_evts:
#             if evt not in processed:
#                 next_to_evt = [evt2 for evt1, evt2 in pairs if evt1 == evt]
#                 yield evt, next_to_evt
#                 new_next.update(next_to_evt)
#                 processed.add(evt)
#         next_evts = new_next
#
#
# def get_events_for_op(op_events: List[Dict[str, str]]) -> List[str]:
#     name_map = {"waiting for rw locks": "waiting_for_rw_locks"}
#     names = []  # type: List[str]
#     for step in op_events:
#         ename = step['event']
#         if ename.startswith("waiting for subops from"):
#             id_first, id_second = ename[len("waiting for subops from "):].split(",")
#             name_map = {
#                 "sub_op_applied_rec from {}".format(id_first): "sub_op_applied_rec_first",
#                 "sub_op_applied_rec from {}".format(id_second): "sub_op_applied_rec_second",
#                 "sub_op_commit_rec from {}".format(id_first): "sub_op_commit_rec_first",
#                 "sub_op_commit_rec from {}".format(id_second): "sub_op_commit_rec_second"
#             }
#             ename = "waiting_for_subops"
#         else:
#             ename = name_map.get(ename, ename)
#         names.append(ename)
#
#     return names
#
#
# def get_uniq_evt_sets_counts(evts: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], int]:
#     uniq: Dict[Tuple[str, ...], int] = collections.defaultdict(int)
#     for names in evts:
#         uniq[tuple(sorted(names))] += 1
#     return uniq
#
#
# def find_deps(ops: List[CephOp]):
#     # j depend on i,  i always appears before j
#     dependents = {(i, j) for i in CephOp.all_stages for j in CephOp.all_stages if i != j and 'initiated' not in (i, j)}
#     all_found_stages: Set[str] = set()
#
#     for op in ops:
#         prev_evts: List[str] = []
#         for curr_evt, _ in op.iter_events():
#             for prev_evt in prev_evts:
#                 try:
#                     # prev_evt appears before curr_evt, so prev_evt can't depend on curr_evt
#                     dependents.remove((curr_evt, prev_evt))
#                 except KeyError:
#                     pass
#             prev_evts.append(curr_evt)
#             all_found_stages.add(curr_evt)
#
#     filtered_deps = set()
#     for i, j in dependents:
#         if i in all_found_stages and j in all_found_stages:
#             filtered_deps.add((i, j))
#
#     all_deps: Dict[str, List] = {op: [] for op in all_found_stages}
#     for i, j in filtered_deps:
#         all_deps[i].append(j)
#
#     closest_pairs = list(find_closest_pars(filtered_deps))
#     prev_stage = dict((j, i) for (i, j) in closest_pairs)
#
#     # import pprint
#     # pprint.pprint(prev_stage)
#     # pprint.pprint(all_found_stages)
#
#     closest_pairs.sort(key=lambda x: -len(all_deps[x[0]]))
#     for p1, p2 in closest_pairs:
#         print("{} => {}".format(p1, p2))
#
#
# def iter_ceph_ops(fd: IO):
#     data = fd.read()
#     offset = 0
#     while offset < len(data):
#         op, offset = CephOp.unpack(data, offset)
#         yield op
#
#
# def calc_stages_time(op: CephOp) -> Dict[str, int]:
#     all_ops = dict(op.iter_events())
#     res = {}
#     for name, vl in all_ops.items():
#         if vl != 0:
#             curr = name
#             dtime = vl
#             while curr in prev_op:
#                 if prev_op[curr] in all_ops:
#                     dtime = vl - all_ops[prev_op[curr]]
#                     break
#                 curr = prev_op[curr]
#             assert dtime >= 0, all_ops
#             res[name] = dtime
#     return res
#
#
# def main():
#     all_ops = []
#     for name in pathlib.Path(sys.argv[1]).glob("perf_monitoring/*/ceph.osd*.historic.bin"):
#         all_ops.extend(iter_ceph_ops(name.open("rb")))
#
#     # find_deps(all_ops)
#     stimes = {}
#     for op in all_ops:
#         for name, vl in calc_stages_time(op).items():
#             stimes.setdefault(name, []).append(vl)
#
#
#     for name, vals in sorted(stimes.items()):
#         print(f"{name:20s} {sum(vals) // len(vals):>10d} {len(vals):>10d}")
#
#     import seaborn
#     from matplotlib import pyplot
#
#     names, vals = zip(*stimes.items())
#     ax = seaborn.boxplot(data=vals)
#     ax.set_xticklabels(names, size=14, rotation=20)
#     ax.set_yscale("log")
#     pyplot.show()
#
#     return 0
#
#
# if __name__ == "__main__":
#     exit(main())