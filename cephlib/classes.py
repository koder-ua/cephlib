from enum import Enum

from dataclasses import dataclass
from typing import Dict, Any, List


class CephHealth(Enum):
    HEALTH_OK = 0
    SCRUB_MISSMATCH = 1
    CLOCK_SKEW = 2
    OSD_DOWN = 3
    REDUCED_AVAIL = 4
    DEGRADED = 5
    NO_ACTIVE_MGR = 6
    SLOW_REQUESTS = 7
    MON_ELECTION = 8


class CephReleases(Enum):
    jewel = 10
    kraken = 11
    luminous = 12
    mimic = 13
    nautilus = 14

    def __lt__(self, other: 'CephReleases') -> bool:
        return self.value < other.value

    def __gt__(self, other: 'CephReleases') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'CephReleases') -> bool:
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
    def release(self) -> CephReleases:
        return CephReleases(self.major)



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
