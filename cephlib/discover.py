""" Collect data about ceph nodes"""
import random
import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set

from koder_utils.rpc_node import IAsyncNode
from koder_utils.utils import async_map

from .classes import CephVersion, OSDMetadata, MonMetadata, CephReport


logger = logging.getLogger("cephlib")


async def get_osd_metadata(node: IAsyncNode, extra_args: str, osd_id: int) -> OSDMetadata:
    mdata = await node.run_json(f"ceph {extra_args} osd metadata {osd_id}")
    public_ip, _ = mdata['front_addr'].split(":")
    cluster_ip, _ = mdata['back_addr'].split(":")
    return OSDMetadata(osd_id, mdata['hostname'], mdata, public_ip=public_ip, cluster_ip=cluster_ip)


async def get_osds_metadata(node: IAsyncNode, extra_args: str = "", max_workers: int = 32) \
        -> Tuple[List[OSDMetadata], Set[int]]:

    """Get dict, which maps node ip to list of OSDInfo"""

    osd_ids = []
    for _, osds in (await node.run_json(f"ceph {extra_args} node ls osd")).items():
        osd_ids.extend(osds)

    async def get_osd_config_coro(osd_id: int) -> Optional[OSDMetadata]:
        try:
            return await get_osd_metadata(node, extra_args, osd_id)
        except Exception:
            return None

    # shuffle to run in parallel for osd's from different nodes
    random.shuffle(osd_ids)

    maybe_osd_configs_iter = await async_map(get_osd_config_coro, osd_ids, max_workers=max_workers)
    result = [maybe_osd_cfg for maybe_osd_cfg in maybe_osd_configs_iter if maybe_osd_cfg]
    return result, set(osd_ids).difference([cfg.osd_id for cfg in result])


async def get_osds_metadata_osdmap(node: IAsyncNode, extra_args: str = "") -> Tuple[List[OSDMetadata], Set[int]]:
    """Get dict, which maps node ip to list of OSDInfo"""

    osd_host: Dict[int, str] = {}
    for hostname, osd_ids in (await node.run_json(f"ceph {extra_args} node ls osd")).items():
        osd_host.update({osd_id: hostname for osd_id in osd_ids})

    result: List[OSDMetadata] = []
    for osd_info in (await node.run_json(f"ceph {extra_args} osd dump -f json"))['osds']:
        public_ip, _ = osd_info['public_addr'].split(":")
        cluster_ip, _ = osd_info['cluster_addr'].split(":")
        osd_id = osd_info['osd']
        result.append(OSDMetadata(osd_id=osd_id,
                                  hostname=osd_host[osd_id],
                                  metadata=osd_info,
                                  public_ip=public_ip,
                                  cluster_ip=cluster_ip))

    return result, set(osd_host).difference([cfg.osd_id for cfg in result])


async def get_mons_nodes(node: IAsyncNode, extra_args: str = "") -> Tuple[List[MonMetadata], List[str]]:
    """Return mapping mon_id => mon_ip"""
    data = await node.run_json(f"ceph {extra_args} --format json mon_status")

    result: List[MonMetadata] = []
    failed_mons: List[str] = []

    for mon_data in data["monmap"]["mons"]:
        if "addr" not in mon_data:
            failed_mons.append(mon_data.get("name", "<MON_NAME_MISSED>"))
        else:
            ip = mon_data["addr"].split(":")[0]
            result.append(MonMetadata(ip, mon_data["name"], mon_data))

    return result, failed_mons


async def discover_report(node: IAsyncNode, extra_args: str = "") -> CephReport:
    return parse_ceph_report(await node.run_json(f"ceph {extra_args} report"))


def parse_ceph_report(raw_report: Dict[str, Any]) -> CephReport:
    osds = []
    for osd_info in raw_report["osd_metadata"]:
        public_ip, _ = osd_info['front_addr'].split(":")
        cluster_ip, _ = osd_info['back_addr'].split(":")
        osd_id = osd_info['id']
        osds.append(OSDMetadata(osd_id=osd_id,
                                hostname=osd_info["hostname"],
                                metadata=osd_info,
                                public_ip=public_ip,
                                cluster_ip=cluster_ip,
                                version=parse_ceph_version("ceph_version")))

    mons = []
    for mon_info in raw_report["monmap"]["mons"]:
        ip = mon_info["public_addr"].split(":")[0]
        mons.append(MonMetadata(ip, mon_info["name"], mon_info))

    major, minor, bugfix = map(int, raw_report["version"].split("."))
    version = CephVersion(major, minor, bugfix, extra="", commit_hash=raw_report["commit"])
    return CephReport(osds=osds, mons=mons, raw=raw_report, version=version)


version_rr = re.compile(r'ceph version\s+(?P<version>\d+\.\d+\.\d+)(?P<extra>[^ ]*)\s+' +
                        r'[([](?P<hash>[^)\]]*?)[)\]]')


def parse_ceph_version(version_str: str) -> CephVersion:
    rr = version_rr.match(version_str)
    if not rr:
        raise ValueError(f"Can't parse ceph version {version_str!r}")
    major, minor, bugfix = map(int, rr.group("version").split("."))
    return CephVersion(major, minor, bugfix, extra=rr.group("extra"), commit_hash=rr.group("hash"))