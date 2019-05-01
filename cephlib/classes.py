import copy
from pathlib import Path
from dataclasses import dataclass, field
from ipaddress import IPv4Network, IPv4Address
from typing import Dict, Any, List, Optional, Union, Set, TypeVar, Iterable, Iterator, Tuple, cast

from koder_utils import AttredDict, Array, Host, Disk, LogicBlockDev, JsonBase, js

from . import (PGStat, CephVersion, OSDStatus, CephMGR, RadosGW, CephStatus, CephReport, PGDump, OSDStoreType,
               CephIOStats, CrushMap, CrushRuleStepTake, CrushRuleStepEmit, CrushRuleStepChooseLeafFirstN,
               OSDMapPool, RadosDF, MonRole)


T = TypeVar("T")


@dataclass
class StatusRegion:
    healty: bool
    begin: int
    end: int


@dataclass
class OSDProcessInfo:
    procinfo: Dict[str, Any]
    opened_socks: int
    fd_count: int
    th_count: int
    cpu_usage: float
    vm_rss: int
    vm_size: int
    cmdline: Optional[List[str]] = None


@dataclass
class NodePGStats:
    name: str
    pgs: List[PGStat]
    pg_stats: CephIOStats
    d_pg_stats: Optional[CephIOStats] = None


@dataclass
class CephDevInfo:
    hostname: str
    dev_info: Disk
    partition_info: LogicBlockDev

    @property
    def name(self) -> str:
        return self.dev_info.name

    @property
    def path(self) -> Path:
        return self.dev_info.dev_path

    @property
    def partition_name(self) -> str:
        return self.partition_info.name

    @property
    def partition_path(self) -> Path:
        return self.partition_info.dev_path


@dataclass
class BlueStoreInfo:
    data: CephDevInfo
    db: CephDevInfo
    wal: CephDevInfo


@dataclass
class FileStoreInfo:
    data: CephDevInfo
    journal: CephDevInfo


@dataclass
class OSDSpace:
    free_perc: int
    used: int
    free: int
    total: int


@dataclass
class CephOSD:
    id: int
    host: Host
    version: Optional[CephVersion]
    status: OSDStatus
    config: Optional[Dict[str, str]]

    cluster_ip: IPv4Address
    public_ip: IPv4Address
    pg_count: Optional[int]
    reweight: float
    class_name: Optional[str]
    space: Optional[OSDSpace]

    pgs: Optional[List[PGStat]]
    expected_weight: Optional[float]
    crush_rules_weights: Dict[int, float]
    # load from the very beginning for all owned PG's & other pg stats
    pg_stats: Optional[CephIOStats]
    storage_info: Union[BlueStoreInfo, FileStoreInfo, None]
    osd_perf_counters: Optional[Dict[str, Union[Array[float], float]]]
    osd_perf_dump: Dict[str, int]

    run_info: Optional[OSDProcessInfo]
    d_pg_stats: Optional[CephIOStats] = None

    @property
    def daemon_runs(self) -> bool:
        return self.run_info is not None

    def __str__(self) -> str:
        return f"OSD(id={self.id}, host={self.host.name})"


@dataclass
class Crush:
    def __init__(self, crushmap: CrushMap) -> None:
        self.crushmap = crushmap
        self.id2bucket: Dict[int, CrushMap.Bucket] = {bkt.id: bkt for bkt in crushmap.buckets}
        self.name2bucket: Dict[str, CrushMap.Bucket] = {bkt.name: bkt for bkt in crushmap.buckets}
        self.id2rule: Dict[int, CrushMap.Rule] = {rule.rule_id: rule for rule in crushmap.rules}

        # osd's not in crush as buckets, so simulate them
        weights = {}
        for bucket in crushmap.buckets:
            for itm in bucket.items:
                if itm.id >= 0:
                    weights[itm.id] = itm.weight

        osd_type_id = self.crushmap.types['osd']
        classes = {dev.id: dev.class_name for dev in crushmap.devices}
        for osd_id, w in weights.items():
             bkt = CrushMap.Bucket(id=osd_id,
                                   name=f"osd.{osd_id}",
                                   type_id=osd_type_id,
                                   type_name="osd",
                                   weight=w,
                                   alg=None,
                                   hash=None,
                                   items=[],
                                   class_name=classes[osd_id])
             self.id2bucket[osd_id] = bkt
             self.name2bucket[bkt.name] = bkt

    @property
    def rules(self) -> Iterable[CrushMap.Rule]:
        return self.crushmap.rules

    def rule_by_id(self, rid: int) -> CrushMap.Rule:
        return self.id2rule[rid]

    def bucket_by_name(self, name: str) -> CrushMap.Bucket:
        return self.name2bucket[name]

    def bucket_by_id(self, bid: int) -> CrushMap.Bucket:
        return self.id2bucket[bid]

    def get_root_bucket_for_rule(self, rule: CrushMap.Rule) -> CrushMap.Bucket:
        assert len(rule.steps) == 3, len(rule.steps)
        assert isinstance(rule.steps[0], CrushRuleStepTake), rule.steps[0]
        assert isinstance(rule.steps[1], CrushRuleStepChooseLeafFirstN), rule.steps[1]
        assert isinstance(rule.steps[2], CrushRuleStepEmit), rule.steps[2]
        return self.name2bucket[rule.steps[0].item_name]

    def iter_osds_for_rule(self, rule: CrushMap.Rule) -> Iterator[Tuple[int, float]]:
        return self.iter_osds_for_bucket(self.get_root_bucket_for_rule(rule))

    def iter_osds_for_bucket(self, bucket: CrushMap.Bucket) -> Iterator[Tuple[int, float]]:
        yield from ((osd_id, w)
                    for osd_id, w in self.iter_childs(bucket)
                    if osd_id > 0)

    def iter_childs(self, bucket: CrushMap.Bucket) -> Iterator[Tuple[int, float]]:
        for child in bucket.items:
            yield child.id, child.weight
            yield from self.iter_childs(self.id2bucket[child.id])


def get_rule_osd_class(rule: CrushMap.Rule) -> Optional[str]:
    assert isinstance(rule.steps[0], CrushRuleStepTake), rule.steps[0]
    step = cast(CrushRuleStepTake, rule.steps[0])
    if '~' in step.item_name:
        _, class_name = step.item_name.split('~')
        return class_name
    return None


def get_rule_replication_level(rule: CrushMap.Rule) -> str:
    assert isinstance(rule.steps[1], CrushRuleStepChooseLeafFirstN), rule.steps[1]
    step = cast(CrushRuleStepChooseLeafFirstN, rule.steps[1])
    return step.type


@dataclass
class CephMonitorStorage:
    database_size: int
    b_avail: int
    avail_percent: int


@dataclass
class Pool:
    id: int
    name: str
    size: int
    min_size: int
    pg: int
    pgp: int
    crush_rule: int
    extra: OSDMapPool
    df: RadosDF.RadosDFPoolInfo
    apps: List[str]
    d_df: Optional[RadosDF.RadosDFPoolInfo] = None


@dataclass
class CephMonitor:
    name: str
    host: Host
    role: MonRole
    version: Optional[CephVersion]
    storage: Optional[CephMonitorStorage] = None


@dataclass
class CephInfo:
    osds: Dict[int, CephOSD]
    mons: Dict[str, CephMonitor]
    mgrs: Dict[str, CephMGR]
    radosgw: Dict[str, RadosGW]
    pools: Dict[str, Pool]

    version: CephVersion  # largest monitor version
    status: CephStatus
    report: CephReport

    # pg distribution
    osd_pool_pg_2d: Dict[int, Dict[str, int]]
    sum_per_pool: Dict[str, int]
    sum_per_osd: Dict[int, int]
    osds4rule: Dict[int, List[CephOSD]]

    crush: Crush
    cluster_net: IPv4Network
    public_net: IPv4Network
    settings: AttredDict

    pgs: Optional[PGDump]

    errors_count: Optional[Dict[str, int]]
    status_regions: Optional[List[StatusRegion]]
    log_err_warn: List[str]

    has_fs: bool = field(init=False)
    has_bs: bool = field(init=False)

    hidden_nodes_pg_info: Optional[Dict[str, NodePGStats]] = field(init=False, default=None)  # type: ignore

    def __post_init__(self):
        self.has_fs = any(isinstance(osd.storage_info, FileStoreInfo) for osd in self.osds.values())
        self.has_bs = any(isinstance(osd.storage_info, BlueStoreInfo) for osd in self.osds.values())
        assert self.has_fs or self.has_bs

    @property
    def is_luminous(self) -> bool:
        return self.version.major >= 12

    @property
    def fs_types(self) -> Set[OSDStoreType]:
        if self.has_bs and self.has_fs:
            return {OSDStoreType.bluestore, OSDStoreType.filestore}
        elif self.has_bs:
            return {OSDStoreType.bluestore}
        else:
            return {OSDStoreType.filestore}

    @property
    def sorted_osds(self) -> List[CephOSD]:
        return [osd for _, osd in sorted(self.osds.items())]

    @property
    def nodes_pg_info(self) -> Dict[str, NodePGStats]:
        if not self.hidden_nodes_pg_info:
            self.hidden_nodes_pg_info = {}
            for osd in self.osds.values():
                assert osd.pg_stats, osd
                assert osd.pgs is not None, osd
                if osd.host.name in self.hidden_nodes_pg_info:
                    info = self.hidden_nodes_pg_info[osd.host.name]
                    info.pg_stats += osd.pg_stats
                    info.pgs.extend(osd.pgs)
                else:
                    self.hidden_nodes_pg_info[osd.host.name] = NodePGStats(name=osd.host.name,
                                                                           pg_stats=copy.copy(osd.pg_stats),
                                                                           pgs=osd.pgs[:])
        return self.hidden_nodes_pg_info


@dataclass
class OSDDevCfg(JsonBase):
    type: str
    data: Optional[Path] = js(default=None)
    r_data: Optional[Path] = js(default=None)
    db: Optional[Path] = js(default=None)
    r_db: Optional[Path] = js(default=None)
    wal: Optional[Path] = js(default=None)
    r_wal: Optional[Path] = js(default=None)
    r_journal: Optional[Path] = js(default=None)
    journal: Optional[Path] = js(default=None)
