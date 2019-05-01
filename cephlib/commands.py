import datetime
import glob
import gzip
import json
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterator, Set, Any, List, Optional, Union, Tuple, TextIO, Iterable

from koder_utils import run_stdout, IAsyncNode

from . import CephRelease, parse_ceph_version, CephHealth, CephReport, CephVersion, OSDMetadata, MonMetadata


class CephCmd(IntEnum):
    ceph = 1
    rados = 2
    rbd = 3
    radosgw_admin = 4


async def get_ceph_version(node: IAsyncNode, extra_args: Iterable[str] = tuple(), **kwargs) -> CephVersion:
    ver_s = await node.run_str(['ceph', *extra_args, '--version'], **kwargs)
    return parse_ceph_version(ver_s)


# for ceph osd tree
def get_all_child_osds(node: Dict, crush_nodes: Dict[int, Dict], target_class: str = None) -> Iterator[int]:
    # workaround for incorrect node classes on some prod clusters
    if node['type'] == 'osd' or re.match("osd\.\d+", node['name']):
        if target_class is None or node.get('device_class') == target_class:
            yield node['id']
        return

    for ch_id in node['children']:
        yield from get_all_child_osds(crush_nodes[ch_id], crush_nodes)


@dataclass
class CephCLI:
    node: Optional[IAsyncNode]
    extra_params: List[str]
    timeout: float
    release: CephRelease
    env: Optional[Dict[str, str]] = None

    binaries = {CephCmd.ceph: 'ceph',
                CephCmd.rados: 'rados',
                CephCmd.rbd: 'rbd',
                CephCmd.radosgw_admin: 'radosgw-admin'}

    async def run_no_ceph(self, cmd: Union[str, List[str]], **kwargs) -> str:
        kwargs.setdefault('timeout', self.timeout)
        if self.node is None:
            kwargs.setdefault('env', self.env)
            return await run_stdout(cmd, **kwargs)
        else:
            return await self.node.run_str(cmd, **kwargs)

    async def run(self, cmd: Union[str, List[str]], *, target: CephCmd = CephCmd.ceph, **kwargs) -> str:
        if isinstance(cmd, str):
            cmd = cmd.split()
        return await self.run_no_ceph([self.binaries[target], *self.extra_params, *cmd], **kwargs)

    async def run_json_raw(self, cmd: Union[str, List[str]], **kwargs) -> str:
        if isinstance(cmd, str):
            cmd = cmd.split()
        return await self.run(['--format', 'json'] + cmd, merge_err=False, **kwargs)

    async def run_json(self, cmd: Union[str, List[str]], **kwargs) -> Any:
        return json.loads(await self.run_json_raw(cmd, **kwargs))

    async def get_local_osds(self, target_class: str = None) -> Set[int]:
        """
        Get OSD id's for current node from ceph osd tree for selected osd class (all classes by default)
        Search by hostname, as returned from socket.gethostname
        In case if method above failed - search by osd cluster/public ip address
        """

        # find by node name
        hostnames = {socket.gethostname(), socket.getfqdn()}

        try:
            osd_nodes = await self.run_json("node ls osd")
        except subprocess.SubprocessError:
            osd_nodes = None

        all_osds_by_node_name = None
        if osd_nodes:
            for name in hostnames:
                if name in osd_nodes:
                    all_osds_by_node_name = osd_nodes[name]

        if all_osds_by_node_name is not None:
            return all_osds_by_node_name

        tree_js = await self.run_json("osd tree")
        nodes = {node['id']: node for node in tree_js['nodes']}

        for node in nodes.values():
            if node['type'] == 'host' and node['name'] in hostnames:
                assert all_osds_by_node_name is None, f"Current node with names {hostnames} found two times in osd tree"
                all_osds_by_node_name = set(get_all_child_osds(node, nodes, target_class))

        if all_osds_by_node_name is not None:
            return all_osds_by_node_name

        all_osds_by_node_ip = set()

        # find by node ips
        all_ips = (await self.run_no_ceph("hostname -I")).split()
        for osd in (await self.run_json("osd dump"))['osds']:
            public_ip = osd['public_addr'].split(":", 1)[0]
            cluster_ip = osd['cluster_addr'].split(":", 1)[0]
            if public_ip in all_ips or cluster_ip in all_ips:
                if target_class is None or target_class == nodes[osd['id']].get('device_class'):
                    all_osds_by_node_ip.add(osd['id'])

        return all_osds_by_node_ip

    async def set_history_size_duration(self, osd_id: int, size: int, duration: int) -> bool:
        """
        Set size and duration for historic_ops log
        """
        prefix = f"daemon osd.{osd_id} config set"
        try:
            assert "success" in await self.run_json(f"{prefix} osd_op_history_duration {duration}")
            assert "success" in await self.run_json(f"{prefix} osd_op_history_size {size}")
        except subprocess.SubprocessError:
            return False
        return True

    async def get_historic(self, osd_id: int) -> Any:
        """
        Get historic ops from osd
        """
        return await self.run_json(f"daemon osd.{osd_id} dump_historic_ops")

    async def get_pools(self) -> Dict[int, str]:
        data = await self.run_json("osd lspools")
        return {pdata["poolnum"]: pdata["poolname"] for pdata in data}

    async def get_osd_metadata(self, osd_id: int) -> OSDMetadata:
        return OSDMetadata.from_json(await self.run_json(f"osd metadata {osd_id}"))

    async def get_mons_nodes(self) -> Tuple[List[MonMetadata], List[str]]:
        """Return mapping mon_id => mon_ip"""
        data = await self.run_json(f"mon_status")

        result: List[MonMetadata] = []
        failed_mons: List[str] = []

        for mon_data in data["monmap"]["mons"]:
            if "addr" not in mon_data:
                failed_mons.append(mon_data.get("name", "<MON_NAME_MISSED>"))
            else:
                ip = mon_data["addr"].split(":")[0]
                result.append(MonMetadata(ip, mon_data["name"], mon_data))

        return result, failed_mons

    async def discover_report(self) -> Tuple[CephReport, Any]:
        report_dct = await self.run_json("report")
        return CephReport.from_json(report_dct), report_dct


def iter_ceph_logs_fd() -> Iterator[TextIO]:
    all_files = []
    for name in glob.glob("/var/log/ceph/ceph.log*"):
        if name == '/var/log/ceph/ceph.log':
            all_files.append((0, open(name, 'r')))
        else:
            rr = re.match(r"/var/log/ceph/ceph\.log\.(\d+)\.gz$", name)
            if rr:
                all_files.append((-int(rr.group(1)), gzip.open(name, mode='rt', encoding='utf8')))

    for _, fd in sorted(all_files):
        yield fd


def iter_log_messages(fd: TextIO) -> Iterator[Tuple[float, CephHealth]]:
    for ln in fd:
        msg = None
        dt, tm, service_name, service_id, addr, uid, _, src, level, message = ln.split(" ", 9)
        if 'overall HEALTH_OK' in message or 'Cluster is now healthy' in message:
            msg = CephHealth.HEALTH_OK
        elif message == 'scrub mismatch':
            msg = CephHealth.SCRUB_MISSMATCH
        elif 'clock skew' in message:
            msg = CephHealth.CLOCK_SKEW
        elif 'marked down' in message:
            msg = CephHealth.OSD_DOWN
        elif 'Reduced data availability' in message:
            msg = CephHealth.REDUCED_AVAIL
        elif 'Degraded data redundancy' in message:
            msg = CephHealth.DEGRADED
        elif 'no active mgr' in message:
            msg = CephHealth.NO_ACTIVE_MGR
        elif "slow requests" in message and "included below" in message:
            msg = CephHealth.SLOW_REQUESTS
        elif 'calling monitor election' in message:
            msg = CephHealth.MON_ELECTION

        if msg is not None:
            y_month_day = dt.split("-")
            h_m_s = tm.split('.')[0].split(":")
            date = datetime.datetime(*map(int, y_month_day + h_m_s))
            yield time.mktime(date.timetuple()), msg
