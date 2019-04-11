from enum import IntEnum, Enum

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterator


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


@dataclass(order=True, unsafe_hash=True)
class CephVersion:
    major: int
    minor: int
    bugfix: int
    extra: str
    commit_hash: str

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.bugfix}{self.extra}  [{self.commit_hash[:8]}]"

    @property
    def release(self) -> CephRelease:
        return CephRelease(self.major)


@dataclass
class OSDMetadata:
    osd_id: int
    hostname: str
    metadata: Dict[str, Any]
    public_ip: str
    cluster_ip: str
    version: CephVersion

    def __str__(self) -> str:
        return f"OSDInfo({self.osd_id})"


@dataclass
class MonMetadata:
    ip: int
    name: str
    metadata: Dict[str, Any]


@dataclass
class CephReport:
    version: CephVersion
    raw: Dict[str, Any]
    osds: List[OSDMetadata]
    mons: List[MonMetadata]


@dataclass
class CrushNode:
    id: int
    name: str
    type: str
    weight: Optional[float]
    childs: List['CrushNode']
    class_name: Optional[str] = None
    full_path: Optional[str] = None

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
        w = ", w={0.weight}".format(self) if self.weight is not None else ""
        yield tabstep * tabs + "{0.type}(name={0.name!r}, id={0.id}{1})".format(self, w)
        for cr_node in self.childs:
            yield from cr_node.tree(tabs=tabs + 1, tabstep=tabstep)

    def copy(self) -> 'CrushNode':
        res = self.__class__(id=self.id, name=self.name, type=self.type, weight=self.weight, childs=self.childs,
                             class_name=self.class_name)
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

    def build_search_idx(self):
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

    def find_nodes(self, path):
        if not path:
            return self.roots

        res = []
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

    def find_node(self, path):
        nodes = self.find_nodes(path)
        if not nodes:
            raise IndexError(f"Can't found any node with path {path!r}")
        if len(nodes) > 1:
            raise IndexError(f"Found {len(nodes)} nodes  for path {path!r} (should be only 1)")
        return nodes[0]

    # def copy_tree_for_rule(self, rule: Rule) -> Optional[CrushNode]:
    #     return copy_class_subtree(self.get_root(rule.root), rule.class_name)
