from __future__ import annotations

import re
import datetime
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Dict, Any, List, Optional, Union, Set, TypeVar, Callable, Type, Tuple, cast

from koder_utils import DiskType, ConvBase, field, IntArithmeticMixin, register_converter, ToInt, ToFloat

T = TypeVar("T")


# ------------- ENUMS --------------------------------------------------------------------------------------------------


class CephHealth(Enum):
    HEALTH_OK = 1
    SCRUB_MISSMATCH = 2
    CLOCK_SKEW = 3
    OSD_DOWN = 4
    REDUCED_AVAIL = 5
    DEGRADED = 6
    NO_ACTIVE_MGR = 7
    SLOW_REQUESTS = 8
    MON_ELECTION = 9


class CephRole(Enum):
    mon = 1
    osd = 2
    rgw = 3
    mgr = 4
    mds = 5


class OSDStatus(Enum):
    up = 0
    down = 1
    out = 2


class OSDState(Enum):
    up = 0
    exists = 1
    down = 2


class CephStatusCode(Enum):
    ok = 0
    warn = 1
    err = 2

    @classmethod
    def convert(cls, val: str) -> CephStatusCode:
        val_l = val.lower()
        if val_l in ('error', 'err', 'health_err'):
            return cls.err
        if val_l in ('warning', 'warn', 'health_warn'):
            return cls.warn
        assert val_l in ('ok', 'health_ok'), val_l
        return cls.ok


class MonRole(Enum):
    master = 0
    follower = 1
    unknown = 2


class CephRelease(Enum):
    jewel = 10
    kraken = 11
    luminous = 12
    mimic = 13
    nautilus = 14

    def __lt__(self, other: CephRelease) -> bool:
        return self.value < other.value

    def __gt__(self, other: CephRelease) -> bool:
        return self.value > other.value

    def __ge__(self, other: CephRelease) -> bool:
        return self.value >= other.value


class OSDStoreType(Enum):
    filestore = 0
    bluestore = 1
    unknown = 2


class PGState(Enum):
    activating = 1
    active = 2
    backfill_toofull = 3
    backfill_unfound = 4
    backfill_wait = 5
    backfilling = 6
    clean = 7
    creating = 8
    deep = 9
    degraded = 10
    down = 11
    forced_backfill = 12
    forced_recovery = 13
    incomplete = 14
    inconsistent = 15
    peered = 16
    peering = 17
    recovering = 18
    recovery_toofull = 19
    recovery_unfound = 20
    recovery_wait = 21
    remapped = 22
    repair = 23
    scrubbing = 24
    snaptrim_error = 25
    stale = 26
    undersized = 27


class CrushAlg(Enum):
    tree = 1
    straw = 2
    straw2 = 3


class HashAlg(Enum):
    rjenkins1 = 1


# Common classes -------------------------------------------------------------------------------------------------------


@dataclass(order=True, unsafe_hash=True)
class CephVersion:
    major: int
    minor: int
    bugfix: int
    extra: str
    commit_hash: str

    def __str__(self) -> str:
        res = f"{self.major}.{self.minor}.{self.bugfix}{self.extra}"
        if self.commit_hash:
            res = f"{res} [{self.commit_hash[:8]}]"
        return res

    @property
    def release(self) -> CephRelease:
        return CephRelease(self.major)


version_rr = re.compile(r'ceph version\s+(?P<version>\d+\.\d+\.\d+)(?P<extra>[^ ]*)\s+' +
                        r'[([](?P<hash>[^)\]]*?)[)\]]')


@register_converter(CephVersion)
def parse_ceph_version(version_str: str) -> CephVersion:
    rr = version_rr.match(version_str)
    if not rr:
        raise ValueError(f"Can't parse ceph version {version_str!r}")
    major, minor, bugfix = map(int, rr.group("version").split("."))
    return CephVersion(major, minor, bugfix, extra=rr.group("extra"), commit_hash=rr.group("hash"))


def parse_ceph_version_simple(v: str) -> CephVersion:
    return CephVersion(*[int(i) for i in v.split(".")], extra="", commit_hash="")


@dataclass
class EndpointAddr:
    ip: IPv4Address
    port: int
    pid: int

    @classmethod
    def convert(cls: Type[T], data: str) -> T:
        ip, port_pid = data.split(":")
        port, pid = map(int, port_pid.split("/"))
        return cls(IPv4Address(ip), port, pid)


class DateTime(datetime.datetime):
    @classmethod
    def convert(cls: Type[T], vl: str) -> T:
        datetm, mks = vl.split(".")
        return cls.strptime(datetm, '%Y-%m-%d %H:%M:%S').replace(microsecond=int(mks))


# CLI cmd classes ------------------------------------------------------------------------------------------------------


_CLASSES_MAPPING: Dict[Tuple[str, CephRelease], ConvBase] = {}


def from_cmd(cmd: str, *releases: CephRelease) -> Callable[[Type[T]], Type[T]]:
    def closure(cls: Type[T]) -> Type[T]:
        cls.__ceph_cmd__ = cmd
        cls.__ceph_releases__ = releases
        for release in releases:
            _CLASSES_MAPPING[(cmd, release)] = cast(ConvBase, cls)
        return cls
    return closure


def parse_cmd_output(cmd: str, release: CephRelease, data: Any) -> Any:
    return _CLASSES_MAPPING[(cmd, release)].convert(data)


@from_cmd("rados df", CephRelease.luminous)
@dataclass
class RadosDF(ConvBase):
    @dataclass
    class RadosDFPoolInfo(ConvBase, IntArithmeticMixin):
        name: str
        id: int
        size_bytes: int
        size_kb: int
        num_objects: int
        num_object_clones: int
        num_object_copies: int
        num_objects_missing_on_primary: int
        num_objects_unfound: int
        num_objects_degraded: int
        read_ops: int
        read_bytes: int
        write_ops: int
        write_bytes: int

    total_used: int
    total_objects: int
    total_space: int
    total_avail: int
    pools: List[RadosDFPoolInfo]


@from_cmd("ceph df", CephRelease.luminous)
@dataclass
class CephDF(ConvBase):
    @dataclass
    class Stats(ConvBase):
        total_used_bytes: int
        total_bytes: int
        total_avail_bytes: int

    @dataclass
    class Pools(ConvBase):
        @dataclass
        class PoolsStats(ConvBase):
            bytes_used: int
            kb_used: int
            max_avail: int
            objects: int
            percent_used: float

        stats: PoolsStats
        id: int
        name: str

    stats: Stats
    pools: List[Pools]


@from_cmd("ceph osd perf", CephRelease.luminous)
@dataclass
class OSDPerf(ConvBase):
    @dataclass
    class OSDPerfItem(ConvBase):
        @dataclass
        class PerfStat(ConvBase):
            apply_latency_ms: int
            commit_latency_ms: int

        id: int
        perf_stats: PerfStat

    osd_perf_infos: List[OSDPerfItem]


@from_cmd("ceph osd df", CephRelease.luminous)
@dataclass
class OSDDf(ConvBase):
    @dataclass
    class Node(ConvBase):
        crush_weight: float
        depth: int
        device_class: str
        id: int
        kb: int
        kb_avail: int
        kb_used: int
        name: str
        pgs: int
        pool_weights: Dict[str, float]
        reweight: float
        type: str
        type_id: int
        utilization: float
        var: float

    @dataclass
    class Summary(ConvBase):
        dev: float
        max_var: float
        min_var: float
        average_utilization: float

        total_kb_used: int
        total_kb: int
        total_kb_avail: int

    stray: List[Any]
    summary: Summary
    nodes: List[Node]


@from_cmd("ceph status", CephRelease.luminous)
@dataclass
class CephStatus(ConvBase):

    @dataclass
    class Health(ConvBase):
        @dataclass
        class Check(ConvBase):
            severity: CephStatusCode
            summary: str

        checks: Dict[str, Any]
        status: CephStatusCode
        summary: List[Check] = field(default_factory=list)
        overall_status: Optional[CephStatusCode] = field(default=None)

        def __post_init__(self) -> None:
            if self.overall_status is None:
                self.overall_status = self.status

    @dataclass
    class MonMap(ConvBase):
        @dataclass
        class MonBasicInfo(ConvBase):
            addr: EndpointAddr
            name: str
            public_addr: EndpointAddr
            rank: int

        created: DateTime
        epoch: int
        features: Dict[str, List[str]]
        fsid: str
        modified: DateTime
        mons: List[MonBasicInfo]

    @dataclass
    class PgMap(ConvBase):
        bytes_avail: int
        bytes_total: int
        bytes_used: int
        data_bytes: int
        num_objects: int
        num_pgs: int
        num_pools: int
        pgs_by_state: List[Dict[str, Any]]
        read_op_per_sec: int = field(default=0)
        read_bytes_per_sec: int = field(default=0)
        write_op_per_sec: int = field(default=0)
        write_bytes_sec: int = field(default=0)
        recovering_bytes_per_sec: int = field(default=0)
        recovering_objects_per_sec: int = field(default=0)

    election_epoch: int
    fsid: str
    health: Health
    monmap: MonMap
    pgmap: PgMap
    quorum: List[int]
    quorum_names: List[str]

    fsmap: Dict[str, Any]
    mgrmap: Dict[str, Any]
    osdmap: Dict[str, Any]
    servicemap: Dict[str, Any]


@dataclass
class CephIOStats(ConvBase, IntArithmeticMixin):
    num_bytes: int
    num_bytes_hit_set_archive: int
    num_bytes_recovered: int
    num_deep_scrub_errors: int
    num_evict: int
    num_evict_kb: int
    num_evict_mode_full: int
    num_evict_mode_some: int
    num_flush: int
    num_flush_kb: int
    num_flush_mode_high: int
    num_flush_mode_low: int
    num_keys_recovered: int
    num_object_clones: int
    num_object_copies: int
    num_objects: int
    num_objects_degraded: int
    num_objects_dirty: int
    num_objects_hit_set_archive: int
    num_objects_misplaced: int
    num_objects_missing: int
    num_objects_missing_on_primary: int
    num_objects_omap: int
    num_objects_pinned: int
    num_objects_recovered: int
    num_objects_unfound: int
    num_promote: int
    num_read: int
    num_read_kb: int
    num_scrub_errors: int
    num_shallow_scrub_errors: int
    num_whiteouts: int
    num_write: int
    num_write_kb: int
    num_legacy_snapsets: Optional[int]
    num_large_omap_objects: Optional[int]


class CephMGR:
    pass


class RadosGW:
    pass


@dataclass
class MonMetadata(ConvBase):
    name: str
    addr: EndpointAddr
    arch: str
    ceph_version: CephVersion
    cpu: str
    distro: str
    distro_description: str
    distro_version: str
    hostname: str
    kernel_description: str
    kernel_version: str
    os: str
    mem_swap_kb: ToInt
    mem_total_kb: ToInt


@from_cmd("ceph mon metadata", CephRelease.luminous)
@dataclass
class MonsMetadata(ConvBase):
    mons: List[MonMetadata]

    @classmethod
    def convert(cls, dt: List[Dict[str, Any]]) -> MonsMetadata:
        return MonsMetadata(mons=[MonMetadata.convert(dct) for dct in dt])


def from_disk_short_type(vl: str) -> DiskType:
    if vl == 'hdd':
        return DiskType.sata_hdd
    if vl == 'ssd':
        return DiskType.sata_ssd
    return DiskType.unknown


@dataclass
class BlueFSDev(ConvBase):
    access_mode: str
    dev_node: str
    driver: str
    model: str
    partition_path: Path
    size: ToInt
    block_size: ToInt
    # size: int = field(converter=int)
    # block_size: int = field(converter=int)
    type: DiskType = field(converter=from_disk_short_type)
    dev: Tuple[int, int] = field(converter=lambda v: tuple(map(int, v.split(":"))))
    rotational: bool = field(converter=lambda v: v == '1')


@dataclass
class PGId(ConvBase):
    pool: int
    num: int
    id: str

    @classmethod
    def convert(cls: T, data: str) -> T:
        pool, num = data.split(".")
        return cls(pool=int(pool), num=int(num, 16), id=data)


def convert_state(v: str) -> Set[PGState]:
    try:
        return {getattr(PGState, status.replace("-", "_")) for status in v.split("+")}
    except AttributeError:
        raise ValueError(f"Unknown status {v}")


@dataclass
class PGStat(ConvBase):
    acting: List[int]
    acting_primary: int
    blocked_by: List[int]
    created: int
    dirty_stats_invalid: bool
    hitset_bytes_stats_invalid: bool
    hitset_stats_invalid: bool
    last_active: DateTime
    last_became_active: DateTime
    last_became_peered: DateTime
    last_change: DateTime
    last_clean: DateTime
    last_clean_scrub_stamp: DateTime
    last_deep_scrub: str
    last_deep_scrub_stamp: DateTime
    last_epoch_clean: int
    last_fresh: DateTime
    last_fullsized: DateTime
    last_peered: DateTime
    last_scrub: str
    last_scrub_stamp: DateTime
    last_undegraded: DateTime
    last_unstale: DateTime
    log_size: int
    log_start: str
    mapping_epoch: int
    omap_stats_invalid: bool
    ondisk_log_size: int
    ondisk_log_start: str
    parent: str
    parent_split_bits: int
    pgid: PGId
    pin_stats_invalid: bool
    reported_epoch: ToInt
    reported_seq: ToInt
    stats_invalid: bool
    up: List[int]
    up_primary: int
    version: str
    stat_sum: CephIOStats
    state: Set[PGState] = field(converter=convert_state)


@dataclass
class PGStatSum(ConvBase):
    ondisk_log_size: int
    log_size: int
    acting: int
    stat_sum: CephIOStats
    up: int


@dataclass
class PoolStatSum(ConvBase):
    stat_sum: CephIOStats
    acting: int
    log_size: int
    num_pg: int
    ondisk_log_size: int
    poolid: int
    up: int


@dataclass
class OSDStat(ConvBase):
    up_from: int
    seq: int
    num_pgs: int
    kb: int
    kb_used: int
    kb_avail: int
    hb_peers: List[int]
    snap_trim_queue_len: int
    num_snap_trimming: int
    op_queue_age_hist: Dict[str, Any]
    perf_stat: Dict[str, Any]
    osd: Optional[int] = field(default=None)


@from_cmd("ceph pg dump", CephRelease.luminous)
@dataclass
class PGDump(ConvBase):
    version: int
    stamp: DateTime
    pg_stats: List[PGStat]
    osd_stats_sum: Dict[str, Any]
    osd_epochs: List[Dict[str, Any]]
    pool_stats: List[PoolStatSum]
    min_last_epoch_clean: int
    full_ratio: float
    pg_stats_sum: PGStatSum
    osd_stats: List[OSDStat]
    last_osdmap_epoch: int
    pg_stats_delta: Dict[str, Any]
    near_full_ratio: float
    last_pg_scan: int


@dataclass
class CrushRuleStepTake(ConvBase):
    item: int
    item_name: str


@dataclass
class CrushRuleStepChooseLeafFirstN(ConvBase):
    num: int
    type: str


@dataclass
class CrushRuleStepChooseLeafIndepth(ConvBase):
    num: int
    type: str


class CrushRuleStepEmit:
    pass


CrushRuleStep = Union[CrushRuleStepTake, CrushRuleStepChooseLeafFirstN, CrushRuleStepChooseLeafIndepth,
                      CrushRuleStepEmit]

STEP_MAP: Dict[str, Callable[[Any], CrushRuleStep]] = {
    'take': CrushRuleStepTake.convert,
    'chooseleaf_firstn': CrushRuleStepChooseLeafFirstN.convert,
    'chooseleaf_indep': CrushRuleStepChooseLeafIndepth.convert,
    'emit': lambda _: CrushRuleStepEmit()
}


def crush_rule_step(v: List[Dict[str, Any]]) -> List[CrushRuleStep]:
    res: List[CrushRuleStep] = []
    for step in v:
        try:
            res.append(STEP_MAP[step['op']](step))
        except KeyError:
            raise ValueError(f"Can't parse crush step {step}") from None
    return res


@dataclass
class CrushMap(ConvBase):
    @dataclass
    class Device(ConvBase):
        id: int
        name: str
        class_name: Optional[str] = field(key='class')

    @dataclass
    class Bucket(ConvBase):
        @dataclass
        class Item(ConvBase):
            id: int
            pos: int
            weight: float = field(converter=lambda v: float(v) / 65536.)

        id: int
        name: str
        type_id: int
        type_name: str
        alg: Optional[CrushAlg]
        hash: Optional[HashAlg]
        items: List[Item]
        class_name: Optional[str] = field(default=None)
        weight: float = field(converter=lambda v: float(v) / 65536.)

        @property
        def is_osd(self) -> bool:
            return self.id >= 0

    @dataclass
    class Rule(ConvBase):
        rule_id: int
        rule_name: str
        ruleset: int
        type: int
        min_size: int
        max_size: int
        steps: List[CrushRuleStep] = field(converter=crush_rule_step)

    devices: List[Device]
    buckets: List[Bucket]
    rules: List[Rule]
    tunables: Dict[str, Any]
    choose_args: Dict[str, Any]
    types: Dict[str, int] = field(converter=lambda v: {itm["name"]: itm["type_id"] for itm in v})


class BlueStoreDevices(ConvBase):
    db: BlueFSDev = field(noauto=True)
    wal: BlueFSDev = field(noauto=True)
    data: BlueFSDev = field(noauto=True)

    @classmethod
    def convert(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = cls()
        attrs = {}
        for name in BlueFSDev.__annotations__:
            attrs[name] = v[f"bluefs_db_{name}"]
        obj.db = BlueFSDev.convert(attrs)
        obj.wal = obj.db

        attrs = {}
        for name in BlueFSDev.__annotations__:
            attrs[name] = v[f"bluestore_bdev_{name}"]
        obj.data = BlueFSDev.convert(attrs)
        return obj


@dataclass
class OSDMetadata(ConvBase):
    id: int
    arch: str
    back_addr: EndpointAddr
    back_iface: str
    ceph_version: CephVersion
    cpu: str
    default_device_class: str
    distro: str
    distro_description: str
    distro_version: str
    front_addr: EndpointAddr
    front_iface: str
    hb_back_addr: EndpointAddr
    hb_front_addr: EndpointAddr
    hostname: str
    kernel_description: str
    kernel_version: str
    os: str
    osd_data: Path
    osd_objectstore: OSDStoreType

    bluefs_single_shared_device: bool = field(converter=lambda v: v == '1')
    bluefs: bool = field(converter=lambda v: v == '1')
    rotational: bool = field(converter=lambda v: v == '1')
    journal_rotational: bool = field(converter=lambda v: v == '1')
    mem_swap_kb: int = field(converter=int)
    mem_total_kb: int = field(converter=int)
    bs_info: Optional[BlueStoreDevices] = field(noauto=True)

    @classmethod
    def convert(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = cast(OSDMetadata, super().convert(v))

        if obj.bluefs:
            obj.bs_info = BlueStoreDevices.convert(v)

        return obj


@dataclass
class OSDMapPool(ConvBase):
    pool: int
    pool_name: str
    flags: int
    flags_names: str
    type: int
    size: int
    min_size: int
    crush_rule: int
    object_hash: int
    pg_num: int
    pg_placement_num: int
    crash_replay_interval: int
    auid: int
    snap_mode: str
    snap_seq: int
    snap_epoch: int
    pool_snaps: List[Any]
    removed_snaps: str
    quota_max_bytes: int
    quota_max_objects: int
    tiers: List[Any]
    tier_of: int
    read_tier: int
    write_tier: int
    cache_mode: str
    target_max_bytes: int
    target_max_objects: int
    cache_target_dirty_ratio_micro: int
    cache_target_dirty_high_ratio_micro: int
    cache_target_full_ratio_micro: int
    cache_min_flush_age: int
    cache_min_evict_age: int
    erasure_code_profile: str
    hit_set_params: Dict[str, Any]
    hit_set_period: int
    hit_set_count: int
    use_gmt_hitset: bool
    min_read_recency_for_promote: int
    min_write_recency_for_promote: int
    hit_set_grade_decay_rate: int
    hit_set_search_last_n: int
    grade_table: List[Any]
    stripe_width: int
    expected_num_objects: int
    fast_read: bool
    options: Dict[str, Any]
    application_metadata: Dict[str, Any]
    last_change: int = field(converter=int)
    last_force_op_resend: int = field(converter=int)
    last_force_op_resend_preluminous: int = field(converter=int)


@dataclass
class OSDMap(ConvBase):
    @dataclass
    class OSD(ConvBase):
        osd: int
        uuid: str
        weight: float
        primary_affinity: float
        last_clean_begin: int
        last_clean_end: int
        up_from: int
        up_thru: int
        down_at: int
        lost_at: int
        public_addr: EndpointAddr
        cluster_addr: EndpointAddr
        heartbeat_back_addr: EndpointAddr
        heartbeat_front_addr: EndpointAddr
        state: Set[OSDState] = field(converter=lambda v: {OSDState[name] for name in v})
        in_: bool = field(key='in', converter=lambda x: x == 1)
        up: bool = field(converter=lambda x: x == 1)

    @dataclass
    class OSDXInfo(ConvBase):
        osd: int
        down_stamp: Union[DateTime, ToFloat]
        laggy_probability: float
        laggy_interval: int
        features: int
        old_weight: float

    epoch: int
    fsid: str
    created: DateTime
    modified: DateTime
    crush_version: int
    full_ratio: float
    backfillfull_ratio: float
    nearfull_ratio: float
    cluster_snapshot: str
    pool_max: int
    max_osd: int
    require_min_compat_client: CephRelease
    min_compat_client: CephRelease
    require_osd_release: CephRelease
    pools: List[OSDMapPool]
    osds: List[OSD]
    osd_xinfo: List[OSDXInfo]
    pg_upmap: List[Any]
    pg_upmap_items: List[Any]
    pg_temp: List[Any]
    primary_temp: List[Any]
    blacklist: Dict[str, Any]
    erasure_code_profiles: Dict[str, Any]
    flags: List[str] = field(converter=lambda v: v.split(","))


@from_cmd("ceph report", CephRelease.luminous)
@dataclass
class CephReport(ConvBase):
    @dataclass
    class PoolsSum(ConvBase):
        stat_sum: CephIOStats
        acting: int
        log_size: int
        ondisk_log_size: int
        up: int

    @dataclass
    class PGOnOSD(ConvBase):
        osd: int
        num_primary_pg: int
        num_acting_pg: int
        num_up_pg: int

    quorum: List[int]
    monmap_first_committed: int
    monmap_last_committed: int
    cluster_fingerprint: str
    commit: str
    timestamp: DateTime
    tag: str
    health: CephStatus.Health
    osd_metadata: List[OSDMetadata]
    monmap: CephStatus.MonMap
    crushmap: CrushMap
    osdmap_first_committed: int
    osdmap_last_committed: int
    mdsmap_first_committed: int
    mdsmap_last_committed: int
    num_pg: int
    num_pg_active: int
    num_pg_unknown: int
    num_osd: int
    pool_sum: PoolsSum
    pool_stats: List[PoolStatSum]
    num_pg_by_osd: List[PGOnOSD]
    osdmap: OSDMap
    osd_sum: OSDStat

    auth: Dict[str, Any]
    fsmap: Dict[str, Any]
    osd_stats: List[Dict[str, Any]]
    paxos: Dict[str, Any]

    version: CephVersion = field(converter=parse_ceph_version_simple)
    num_pg_by_state: List[Tuple[Set[PGState], int]] = field(noauto=True)

    @classmethod
    def convert(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = super().convert(v)
        num_pg_by_state = []
        for st in v['num_pg_by_state']:
            states = {PGState[name] for name in st['state'].split("+")}
            num_pg_by_state.append((states, int(st["num"])))
        obj.num_pg_by_state = num_pg_by_state
        return obj


def from_ceph_str_size(v: str) -> int:
    assert len(v) >= 1
    mp = {'k': 2 ** 10, 'm': 2 ** 20, 'g': 2 ** 30, 't': 2 ** 40}
    if v[-1] in mp:
        return int(float(v[:-1]) * mp[v[-1]])
    return int(v)


def lvtags_parser(v: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    for item in v.split(","):
        k, v = item.split("=")
        assert k not in res
        res[k] = v
    return res


@dataclass
class LVMListDevice(ConvBase):
    devices: List[Path]
    lv_name: str
    lv_path: Path
    lv_uuid: str
    name: str
    path: Path
    tags: Dict[str, Any]
    type: str
    vg_name: str
    lv_size: int = field(converter=from_ceph_str_size)
    lv_tags: Dict[str, str] = field(converter=lvtags_parser)


@dataclass
class VolumeLVMList(ConvBase):
    osds: Dict[int, List[LVMListDevice]]

    @classmethod
    def convert(cls: Type[T], v: Dict[str, Any]) -> T:
        osds = {}
        for key, items in v.items():
            osds[int(key)] = [LVMListDevice.convert(item) for item in items]
        return cls(osds=osds)
