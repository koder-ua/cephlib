from __future__ import annotations

import datetime
import re
import time
import bisect
import bz2
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional, NamedTuple, NewType, Match

logger = logging.getLogger("cephlib")

compress = bz2.compress
decompress = bz2.decompress
DEFAULT_DURATION = 600
DEFAULT_SIZE = 20

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
    def parse_op(cls, op: OpRec) -> Tuple[ParseResult, Optional[CephOp]]:
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
            assert subop != -1, evt_map
            assert last_replica_done != -1
            wait_for_replica = subop - last_replica_done
    else:
        download = -1

    return HLTimings(download=download, wait_for_pg=wait_for_pg, local_io=local_io,
                     wait_for_replica=wait_for_replica)


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



class HistStorage:
    def __init__(self) -> None:
        self.storage: Dict[int, HistoricBin] = {}  # key is max bin time value

    def add_op(self, op: HLTimings) -> None:
        pass
