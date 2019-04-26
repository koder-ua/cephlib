import re
import datetime
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from ipaddress import IPv4Address
from typing import Dict, Any, List, Optional, Union, Set, TypeVar, Callable, Type, Tuple, cast

from koder_utils import Host, DiskType, JsonBase, js, register_from_json

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
    def from_json(cls, val: str) -> 'CephStatusCode':
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

    def __lt__(self, other: 'CephRelease') -> bool:
        return self.value < other.value

    def __gt__(self, other: 'CephRelease') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'CephRelease') -> bool:
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


@register_from_json(CephVersion)
def parse_ceph_version(version_str: str) -> CephVersion:
    rr = version_rr.match(version_str)
    if not rr:
        raise ValueError(f"Can't parse ceph version {version_str!r}")
    major, minor, bugfix = map(int, rr.group("version").split("."))
    return CephVersion(major, minor, bugfix, extra=rr.group("extra"), commit_hash=rr.group("hash"))


@dataclass
class EndpointAddr:
    ip: IPv4Address
    port: int
    pid: int

    @classmethod
    def from_json(cls: Type[T], data: str) -> T:
        ip, port_pid = data.split(":")
        port, pid = map(int, port_pid.split("/"))
        return cls(IPv4Address(ip), port, pid)


class DateTime(datetime.datetime):
    @classmethod
    def from_json(cls: Type[T], vl: str) -> T:
        datetm, mks = vl.split(".")
        return cls.strptime(datetm, '%Y-%m-%d %H:%M:%S').replace(microsecond=int(mks))


# CLI cmd classes ------------------------------------------------------------------------------------------------------


_CLASSES_MAPPING: Dict[Tuple[str, CephRelease], JsonBase] = {}


def from_cmd(cmd: str, *releases: CephRelease) -> Callable[[Type[T]], Type[T]]:
    def closure(cls: Type[T]) -> Type[T]:
        cls.__ceph_cmd__ = cmd
        cls.__ceph_releases__ = releases
        for release in releases:
            _CLASSES_MAPPING[(cmd, release)] = cast(JsonBase, cls)
        return cls
    return closure


@from_cmd("rados df", CephRelease.luminous)
@dataclass
class RadosDF(JsonBase):
    @dataclass
    class RadosDFPoolInfo(JsonBase):
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
class CephDF(JsonBase):
    @dataclass
    class Stats(JsonBase):
        total_used_bytes: int
        total_bytes: int
        total_avail_bytes: int

    @dataclass
    class Pools(JsonBase):
        @dataclass
        class PoolsStats(JsonBase):
            percent_used: int
            bytes_used: int
            kb_used: int
            max_avail: int
            objects: int

        stats: PoolsStats
        id: int
        name: str

    stats: Stats
    pools: List[Pools]


@from_cmd("ceph osd df", CephRelease.luminous)
@dataclass
class OSDDf(JsonBase):
    @dataclass
    class Node(JsonBase):
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
        var: int

    @dataclass
    class Summary(JsonBase):
        dev: int
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
class Status(JsonBase):

    @dataclass
    class Health(JsonBase):
        @dataclass
        class Check(JsonBase):
            severity: CephStatusCode
            summary: str

        checks: Dict[str, Any]
        overall_status: CephStatusCode
        status: CephStatusCode
        summary: List[Check]

    @dataclass
    class MonMap(JsonBase):
        @dataclass
        class MonBasicInfo(JsonBase):
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
    class PgMap(JsonBase):
        bytes_avail: int
        bytes_total: int
        bytes_used: int
        data_bytes: int
        num_objects: int
        num_pgs: int
        num_pools: int
        pgs_by_state: List[Dict[str, Any]]

    election_epoch: int
    fsid: str
    fsmap: Dict[str, Any]
    health: Health
    mgrmap: Dict[str, Any]
    osdmap: Dict[str, Any]
    pgmap: PgMap
    quorum: List[int]
    quorum_names: List[str]
    servicemap: Dict[str, Any]


@dataclass
class StatSum(JsonBase):
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
class MonMetadata(JsonBase):
    ip: EndpointAddr
    name: str
    metadata: Dict[str, Any]


def from_disk_short_type(vl: str) -> DiskType:
    if vl == 'hdd':
        return DiskType.sata_hdd
    if vl == 'ssd':
        return DiskType.sata_ssd
    return DiskType.unknown


@dataclass
class BlueFSDevInfo(JsonBase):
    access_mode: str
    block_size: int
    dev_node: str
    driver: str
    model: str
    partition_path: Path
    size: int
    type: DiskType = js(converter=from_disk_short_type)
    dev: Tuple[int, int] = js(converter=lambda v: tuple(map(int, v.split(":"))))
    rotational: bool = js(converter=lambda v: v == '1')


@dataclass
class Pool:
    id: int
    name: str
    size: int
    min_size: int
    pg: int
    pgp: int
    crush_rule: int
    extra: Dict[str, Any]
    df: RadosDF.RadosDFPoolInfo
    apps: List[str]


@dataclass
class CephMonitorStorage:
    database_size: int
    b_avail: int
    avail_percent: int


@dataclass
class CephMonitor:
    name: str
    status: Optional[str]
    host: Host
    role: MonRole
    version: Optional[CephVersion]
    storage: Optional[CephMonitorStorage] = None


@dataclass
class CephStatus:
    status: CephStatusCode
    health_summary: Any
    num_pgs: int

    bytes_used: int
    bytes_total: int
    bytes_avail: int

    data_bytes: int
    pgmap_stat: Any
    monmap_stat: Dict[str, Any]

    write_bytes_sec: int
    read_bytes_sec: int
    write_op_per_sec: int
    read_op_per_sec: int


@dataclass
class PGId(JsonBase):
    pool: int
    num: int
    id: str

    @classmethod
    def from_json(cls: Type[T], dt: str) -> T:
        pool, num = dt.split(".")
        return cls(pool=int(pool), num=int(num, 16), id=dt)


@dataclass
class PGStat(JsonBase):
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
    reported_epoch: int
    reported_seq: int
    state: Set[PGState]
    stats_invalid: bool
    up: List[int]
    up_primary: int
    version: str
    stat_sum: StatSum

    @classmethod
    def __convert_state__(cls, v: str) -> Set[PGState]:
        try:
            return {getattr(PGState, status.replace("-", "_")) for status in v.split("+")}
        except AttributeError:
            raise ValueError(f"Unknown status {v}")


@dataclass
class PGStatSum(JsonBase):
    ondisk_log_size: int
    log_size: int
    acting: int
    stat_sum: StatSum
    up: int


@dataclass
class PoolStatSum(JsonBase):
    stat_sum: StatSum
    acting: int
    log_size: int
    num_pg: int
    ondisk_log_size: int
    poolid: int
    up: int


@dataclass
class OSDStat(JsonBase):
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


@from_cmd("ceph pg dump", CephRelease.luminous)
@dataclass
class PGDump(JsonBase):
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
class CrushRuleStepTake(JsonBase):
    item: int
    item_name: str


@dataclass
class CrushRuleStepChooseLeafFirstN(JsonBase):
    num: int
    type: str


@dataclass
class CrushRuleStepChooseLeafIndepth(JsonBase):
    num: int
    type: str


class CrushRuleStepEmit:
    pass


CrushRuleStep = Union[CrushRuleStepTake, CrushRuleStepChooseLeafFirstN, CrushRuleStepChooseLeafIndepth,
                      CrushRuleStepEmit]


def crush_rule_step(v: List[Dict[str, Any]]) -> List[CrushRuleStep]:
    res: List[CrushRuleStep] = []
    for step in v:
        if step['op'] == 'take':
            res.append(CrushRuleStepTake.from_json(step))
        elif step['op'] == 'chooseleaf_firstn':
            res.append(CrushRuleStepChooseLeafFirstN.from_json(step))
        elif step['op'] == 'chooseleaf_indep':
            res.append(CrushRuleStepChooseLeafIndepth.from_json(step))
        elif step['op'] == 'emit':
            res.append(CrushRuleStepEmit())
        else:
            raise ValueError(f"Can't parse crush step {step}")
    return res


@dataclass
class CrushMap(JsonBase):
    @dataclass
    class Device(JsonBase):
        id: int
        name: str
        class_: Optional[str] = js(key='class')

    @dataclass
    class Bucket(JsonBase):
        @dataclass
        class Item(JsonBase):
            id: int
            weight: int
            pos: int

        id: int
        name: str
        type_id: int
        type_name: str
        weight: float
        alg: CrushAlg
        hash: HashAlg
        items: List[Item]

    @dataclass
    class Rule(JsonBase):
        rule_id: int
        rule_name: str
        ruleset: int
        type: int
        min_size: int
        max_size: int
        steps: List[CrushRuleStep] = js(converter=crush_rule_step)

    devices: List[Device]
    buckets: List[Bucket]
    rules: List[Rule]
    tunables: Dict[str, Any]
    choose_args: Dict[str, Any]
    types: Dict[str, int] = js(converter=lambda v: {itm["name"]: itm["type_id"] for itm in v})


class ReportBlueStoreInfo(JsonBase):
    db: BlueFSDevInfo = js(noauto=True)
    wal: BlueFSDevInfo = js(noauto=True)
    data: BlueFSDevInfo = js(noauto=True)

    @classmethod
    def from_json(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = cls()
        attrs = {}
        for name in BlueFSDevInfo.__annotations__:
            attrs[name] = v[f"bluefs_db_{name}"]
        obj.db = BlueFSDevInfo.from_json(attrs)
        obj.wal = obj.db

        attrs = {}
        for name in BlueFSDevInfo.__annotations__:
            attrs[name] = v[f"bluestore_bdev_{name}"]
        obj.data = BlueFSDevInfo.from_json(attrs)
        return obj


@dataclass
class OSDMetadata(JsonBase):
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
    mem_swap_kb: int
    mem_total_kb: int
    os: str
    osd_data: Path
    osd_objectstore: OSDStoreType

    bluefs_single_shared_device: bool = js(converter=lambda v: v == '1')
    bluefs: bool = js(converter=lambda v: v == '1')
    rotational: bool = js(converter=lambda v: v == '1')
    journal_rotational: bool = js(converter=lambda v: v == '1')
    bs_info: Optional[ReportBlueStoreInfo] = js(noauto=True)

    @classmethod
    def from_json(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = cast(OSDMetadata, super().from_json(v))

        if obj.bluefs:
            obj.bs_info = ReportBlueStoreInfo.from_json(v)

        return obj


@dataclass
class OSDMapPool(JsonBase):
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
    last_change: int
    last_force_op_resend: int
    last_force_op_resend_preluminous: int
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


@dataclass
class OSDMap(JsonBase):
    @dataclass
    class OSD(JsonBase):
        osd: int
        uuid: str
        up: int
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
        state: Set[OSDState] = js(converter=lambda v: {OSDState[name] for name in v})
        in_: int = js(key='in')

    @dataclass
    class OSDXInfo(JsonBase):
        osd: int
        down_stamp: DateTime
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
    flags: List[str] = js(converter=lambda v: v.split(","))


def parse_ceph_version_simple(v: str) -> CephVersion:
    return CephVersion(*[int(i) for i in v.split(".")], extra="", commit_hash="")


@from_cmd("ceph report", CephRelease.luminous)
@dataclass
class CephReport(JsonBase):
    @dataclass
    class PoolsSum(JsonBase):
        stat_sum: StatSum
        acting: int
        log_size: int
        ondisk_log_size: int
        up: int

    @dataclass
    class PGOnOSD(JsonBase):
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
    health: Status.Health
    osd_metadata: List[OSDMetadata]
    monmap: Status.MonMap
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

    version: CephVersion = js(converter=parse_ceph_version_simple)
    num_pg_by_state: List[Tuple[Set[PGState], int]] = js(noauto=True)

    @classmethod
    def from_json(cls: Type[T], v: Dict[str, Any]) -> T:
        obj = super().from_json(v)
        num_pg_by_state = []
        for st in v['num_pg_by_state']:
            states = {PGState[name] for name in st['state'].split("+")}
            num_pg_by_state.append((states, int(st["num"])))
        obj.num_pg_by_state = num_pg_by_state
        return obj
