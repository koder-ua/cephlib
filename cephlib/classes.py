import copy
from pathlib import Path
from dataclasses import dataclass, field
from ipaddress import IPv4Network
from typing import Dict, Any, List, Optional,  Union, Set, TypeVar

from cephlib import CrushMap, OSDMapPool, RadosDF, MonRole
from koder_utils import AttredDict, Array, Host, Disk, LogicBlockDev

from . import (PGStat, CephVersion, OSDStatus, CephMGR, RadosGW, CephStatus, CephReport, PGDump,
               OSDStoreType, CephIOStats)

T = TypeVar("T")


@dataclass
class StatusRegion:
    healty: bool
    begin: int
    end: int


@dataclass
class OSDProcessInfo:
    procinfo: Dict[str, Any]
    cmdline: List[str]
    opened_socks: int
    fd_count: int
    th_count: int
    cpu_usage: float
    vm_rss: int
    vm_size: int


@dataclass
class NodePGStats:
    name: str
    pgs: List[PGStat]
    pg_stats: CephIOStats


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

    cluster_ip: str
    public_ip: str
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
    @property
    def daemon_runs(self) -> bool:
        return self.run_info is not None

    def __str__(self) -> str:
        return f"OSD(id={self.id}, host={self.host.name})"


@dataclass
class Crush:
    def __init__(self, crushmap: CrushMap) -> None:
        self.crushmap = crushmap
        self.name2bucket: Dict[str, CrushMap.Bucket] = {bkt.id: bkt for bkt in crushmap.buckets}
        self.id2bucket: Dict[str, CrushMap.Bucket] = {bkt.name: bkt for bkt in crushmap.buckets}


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
                assert osd.pg_stats
                assert osd.pgs is not None
                if osd.host.name in self.hidden_nodes_pg_info:
                    info = self.hidden_nodes_pg_info[osd.host.name]
                    info.pg_stats.num_shallow_scrub_errors += osd.pg_stats.num_shallow_scrub_errors
                    info.pg_stats.num_scrub_errors += osd.pg_stats.num_scrub_errors
                    info.pg_stats.num_deep_scrub_errors += osd.pg_stats.num_deep_scrub_errors
                    info.pg_stats.num_write_kb += osd.pg_stats.num_write_kb
                    info.pg_stats.num_write += osd.pg_stats.num_write
                    info.pg_stats.num_read_kb += osd.pg_stats.num_read_kb
                    info.pg_stats.num_read += osd.pg_stats.num_read
                    info.pg_stats.num_bytes += osd.pg_stats.num_bytes
                    info.pgs.extend(osd.pgs)
                else:
                    self.hidden_nodes_pg_info[osd.host.name] = NodePGStats(name=osd.host.name,
                                                                           pg_stats=copy.copy(osd.pg_stats),
                                                                           pgs=osd.pgs[:])
        return self.hidden_nodes_pg_info

