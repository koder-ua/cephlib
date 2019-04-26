import copy
from pathlib import Path
from dataclasses import dataclass, field
from ipaddress import IPv4Network
from typing import Dict, Any, List, Optional, Iterator, Union, Set, TypeVar

from koder_utils import AttredDict, Array, Host, Disk, LogicBlockDev

from . import (PGStat, CephVersion, OSDStatus, CephMonitor, CephMGR, RadosGW, Pool, CephStatus, CephReport, PGDump,
               OSDStoreType)

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
class OSDPGStats:
    bytes: int
    reads: int
    read_b: int
    writes: int
    write_b: int
    scrub_errors: int
    deep_scrub_errors: int
    shallow_scrub_errors: int


@dataclass
class NodePGStats:
    name: str
    pgs: List[PGStat]
    pg_stats: OSDPGStats


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
    pg_stats: Optional[OSDPGStats]
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
class CrushNode:
    id: int
    name: str
    type: str
    weight: Optional[float]
    childs: List['CrushNode']
    class_name: Optional[str] = None
    full_path: Optional[str] = None
    reweight: Optional[float] = None

    def str_path(self) -> Optional[str]:
        if self.full_path:
            return "/".join(f"{tp}={name}" for tp, name in self.full_path)
        return None

    def __str__(self) -> str:
        w = f", w={self.weight}" if self.weight is not None else ""
        fp = (", " + self.str_path()) if self.full_path else ""
        return f"{self.type}(name={self.name!r}, id={self.id}{w}{fp})"

    def __repr__(self) -> str:
        return str(self)

    def tree(self, tabs: int = 0, tabstep: str = " " * 4) -> Iterator[str]:
        w = f", w={self.weight}" if self.weight is not None else ""
        yield tabstep * tabs + f"{self.type}(name={self.name!r}, id={self.id}{w})"
        for cr_node in self.childs:
            yield from cr_node.tree(tabs=tabs + 1, tabstep=tabstep)

    def copy(self) -> 'CrushNode':
        res = self.__class__(id=self.id, name=self.name, type=self.type, weight=self.weight, childs=self.childs,
                             class_name=self.class_name, reweight=self.reweight)
        return res

    def iter_nodes(self, node_type: str, class_name: str = None) -> Iterator['CrushNode']:
        if self.type == node_type and (class_name in (None, "") or class_name == self.class_name):
            yield self
        for node in self.childs:
            yield from node.iter_nodes(node_type, class_name=class_name)


@dataclass
class Rule:
    name: str
    id: int
    root: str
    replicated_on: str
    class_name: Optional[str] = None

    def __str__(self) -> str:
        return f"Rule({self.name}, {self.id}, root={self.root}, class={self.class_name!r}, repl={self.replicated_on})"


@dataclass
class Crush:
    nodes_map: Dict[str, CrushNode]
    roots: List[CrushNode]
    rules: Dict[int, Rule]
    search_cache: Optional[List] = None

    def get_root(self, name: str) -> CrushNode:
        for root in self.roots:
            if root.name == name:
                return root
        else:
            raise KeyError(f"Can't found crush root {name}")

    def __str__(self):
        return "\n".join("\n".join(root.tree()) for root in self.roots)

    def iter_nodes_for_rule(self, rule_id: int, tp: str) -> Iterator[CrushNode]:
        root = self.get_root(self.rules[rule_id].root)
        return root.iter_nodes(tp, class_name=self.rules[rule_id].class_name)

    def iter_osds_for_rule(self, rule_id: int) -> Iterator[CrushNode]:
        return self.iter_nodes_for_rule(rule_id, 'osd')

    def iter_nodes(self, node_type, class_name: str = None):
        for node in self.roots:
            yield from node.iter_nodes(node_type, class_name=class_name)

    def build_search_idx(self) -> None:
        if self.search_cache is None:
            not_done = []
            for node in self.roots:
                node.full_path = [(node.type, node.name)]
                not_done.append(node)

            done = []
            while not_done:
                new_not_done = []
                for node in not_done:
                    if node.childs:
                        for chnode in node.childs:
                            chnode.full_path = node.full_path[:] + [(chnode.type, chnode.name)]
                            new_not_done.append(chnode)
                    else:
                        done.append(node)
                not_done = new_not_done
            self.search_cache = done

    def find_nodes(self, path) -> List[CrushNode]:
        if not path:
            return self.roots

        res: List[CrushNode] = []
        for node in self.search_cache:
            nfilter = dict(path)
            for tp, val in node.full_path:
                if tp in nfilter:
                    if nfilter[tp] == val:
                        del nfilter[tp]
                        if not nfilter:
                            break
                    else:
                        break

            if not nfilter:
                res.append(node)

        return res

    def find_node(self, path) -> CrushNode:
        nodes = self.find_nodes(path)
        if not nodes:
            raise IndexError(f"Can't found any node with path {path!r}")
        if len(nodes) > 1:
            raise IndexError(f"Found {len(nodes)} nodes  for path {path!r} (should be only 1)")
        return nodes[0]


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
                    info.pg_stats.shallow_scrub_errors += osd.pg_stats.shallow_scrub_errors
                    info.pg_stats.scrub_errors += osd.pg_stats.scrub_errors
                    info.pg_stats.deep_scrub_errors += osd.pg_stats.deep_scrub_errors
                    info.pg_stats.write_b += osd.pg_stats.write_b
                    info.pg_stats.writes += osd.pg_stats.writes
                    info.pg_stats.read_b += osd.pg_stats.read_b
                    info.pg_stats.reads += osd.pg_stats.reads
                    info.pg_stats.bytes += osd.pg_stats.bytes
                    info.pgs.extend(osd.pgs)
                else:
                    self.hidden_nodes_pg_info[osd.host.name] = NodePGStats(name=osd.host.name,
                                                                           pg_stats=copy.copy(osd.pg_stats),
                                                                           pgs=osd.pgs[:])
        return self.hidden_nodes_pg_info

