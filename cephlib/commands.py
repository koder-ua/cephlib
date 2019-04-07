import json
import re
import socket
import subprocess
from typing import Dict, Iterator, Set, Any

from koder_utils.cli import run


def get_all_child_osds(node: Dict, crush_nodes: Dict[int, Dict], target_class: str = None) -> Iterator[int]:
    # workaround for incorrect node classes on some prod clusters
    if node['type'] == 'osd' or re.match("osd\.\d+", node['name']):
        if target_class is None or node.get('device_class') == target_class:
            yield node['id']
        return

    for ch_id in node['children']:
        yield from get_all_child_osds(crush_nodes[ch_id], crush_nodes)


async def get_local_osds(target_class: str = None, timeout: int = 15) -> Set[int]:
    """
    Get OSD id's for current node from ceph osd tree for selected osd class (all classes by default)
    Search by hostname, as returned from socket.gethostname
    In case if method above failed - search by osd cluster/public ip address
    """

    # find by node name
    hostnames = {socket.gethostname(), socket.getfqdn()}

    try:
        osd_nodes_s = (await run("ceph node ls osd -f json", timeout=timeout)).stdout
    except subprocess.SubprocessError:
        osd_nodes_s = None

    all_osds_by_node_name = None
    if osd_nodes_s:
        osd_nodes = json.loads(osd_nodes_s)
        for name in hostnames:
            if name in osd_nodes:
                all_osds_by_node_name = osd_nodes[name]

    if all_osds_by_node_name is not None:
        return all_osds_by_node_name

    tree_js = (await run("ceph osd tree -f json", timeout=timeout)).stdout
    nodes = {node['id']: node for node in json.loads(tree_js)['nodes']}

    for node in nodes.values():
        if node['type'] == 'host' and node['name'] in hostnames:
            assert all_osds_by_node_name is None, \
                "Current node with names {} found two times in osd tree".format(hostnames)
            all_osds_by_node_name = set(get_all_child_osds(node, nodes, target_class))

    if all_osds_by_node_name is not None:
        return all_osds_by_node_name

    all_osds_by_node_ip = set()

    # find by node ips
    all_ips = (await run("hostname -I", timeout=timeout)).stdout.split()
    osds_js = (await run("ceph osd dump -f json", timeout=timeout)).stdout
    for osd in json.loads(osds_js)['osds']:
        public_ip = osd['public_addr'].split(":", 1)[0]
        cluster_ip = osd['cluster_addr'].split(":", 1)[0]
        if public_ip in all_ips or cluster_ip in all_ips:
            if target_class is None or target_class == nodes[osd['id']].get('device_class'):
                all_osds_by_node_ip.add(osd['id'])

    return all_osds_by_node_ip


async def set_size_duration(osd_ids: Set[int],
                            size: int,
                            duration: int,
                            timeout: int = 15):
    """
    Set size and duration for historic_ops log
    """
    not_inited_osd = set()
    for osd_id in osd_ids:
        try:
            for set_part in ["osd_op_history_duration {}".format(duration), "osd_op_history_size {}".format(size)]:
                cmd = "ceph daemon osd.{} config set {}"
                out = (await run(cmd.format(osd_id, set_part), timeout=timeout)).stdout
                assert "success" in out
        except subprocess.SubprocessError:
            not_inited_osd.add(osd_id)
    return not_inited_osd


async def get_historic(osd_id: int, timeout: int = 15) -> str:
    """
    Get historic ops from osd
    """
    return (await run("ceph daemon osd.{} dump_historic_ops".format(osd_id), timeout=timeout)).stdout


def parse_ceph_status(data: Dict[str, Any]) -> None:
    # status = json.loads(status.stdout)
    # health = status['health']
    # status_str = health['status'] if 'status' in health else health['overall_status']
    # ceph_health:  Dict[str, Union[str, int]] = {'status': status_str}
    # avail = status['pgmap']['bytes_avail']
    # total = status['pgmap']['bytes_total']
    # ceph_health['free_pc'] = int(avail * 100 / total + 0.5)
    #
    # active_clean = sum(pg_sum['count']
    #                    for pg_sum in status['pgmap']['pgs_by_state']
    #                    if pg_sum['state_name'] == "active+clean")
    # total_pg = status['pgmap']['num_pgs']
    # ceph_health["ac_perc"] = int(active_clean * 100 / total_pg + 0.5)
    # ceph_health["blocked"] = "unknown"
    #
    # ceph_health_js = json.dumps(ceph_health)
    # self.save("ceph_health_dict", "js", 0, ceph_health_js)
    pass


def parse_ceph_volumes_js(cephvollist_js: str) -> Dict[int, Dict[str, str]]:
    devs_for_osd: Dict[int, Dict[str, str]] = {}
    cephvolume_dct = json.loads(cephvollist_js)
    for osd_id_s, osd_data in cephvolume_dct.items():
        assert len(osd_data) == 1
        osd_data = osd_data[0]
        assert len(osd_data['devices']) == 1
        dev = osd_data['devices'][0]
        devs_for_osd[int(osd_id_s)] = {"block_dev": dev,
                                       "block.db_dev": dev,
                                       "block.wal_dev": dev,
                                       "store_type": "bluestore"}
        return devs_for_osd


def parse_ceph_disk_js(cephdisklist_js: str) -> Dict[int, Dict[str, str]]:
    devs_for_osd: Dict[int, Dict[str, str]] = {}
    cephdisk_dct = json.loads(cephdisklist_js)
    for dev_info in cephdisk_dct:
        for part_info in dev_info.get('partitions', []):
            if "cluster" in part_info and part_info.get('type') == 'data':
                osd_id = int(part_info['whoami'])
                devs_for_osd[osd_id] = {attr: part_info[attr]
                                        for attr in ("block_dev", "journal_dev", "path",
                                                     "block.db_dev", "block.wal_dev")
                                        if attr in part_info}
                devs_for_osd[osd_id]['store_type'] = 'filestore' if "journal_dev" in part_info else 'bluestore'
    return devs_for_osd
