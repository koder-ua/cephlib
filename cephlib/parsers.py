import json
import re
from typing import Dict, Any, Iterator, Optional

from cephlib.classes import Crush, Rule, CrushNode, CephReport, OSDMetadata, MonMetadata, CephVersion


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


def copy_class_subtree(src_node: CrushNode, classname: str = None) -> Optional[CrushNode]:
    childs = []
    for ch in src_node.childs:
        if ch.type == 'osd' and (ch.class_name != classname and classname is not None):
            continue
        ch_copy = copy_class_subtree(ch, classname)
        if ch_copy:
            childs.append(ch_copy)

    weight = sum((ch.weight for ch in childs), 0) if childs else src_node.weight
    if (src_node.type == 'osd' or childs) and weight != 0:
        return CrushNode(id=src_node.id,
                         name=src_node.name,
                         type=src_node.type,
                         class_name=src_node.class_name,
                         weight=weight,
                         childs=childs)


def load_crushmap_js(filename: str = None, crush: Dict = None) -> Crush:
    assert not filename or not crush, "filename and content should not be passed at the same time"

    if filename:
        crush = json.load(open(filename))

    assert crush
    rules = {}

    for rule in crush["rules"]:
        for step in rule['steps']:
            if step['op'] == 'take':
                root = step["item_name"]
                break
        else:
            continue

        if '~' in root:
            root, class_name = root.split("~")
        else:
            class_name = None

        replicated_on = None
        for step in rule['steps'][1:]:
            if step['op'] in ("chooseleaf_firstn", "chooseleaf_indep"):
                replicated_on = step["type"]

        rules[rule['rule_id']] = Rule(name=rule['rule_name'],
                                      id=rule['rule_id'],
                                      root=root,
                                      class_name=class_name,
                                      replicated_on=replicated_on)

    nodes_dct: Dict[int, Dict] = {}
    nodes: Dict[int, CrushNode] = {}
    osd_classes = {osd['id']: osd.get("class", "") for osd in crush['devices']}
    for bucket in crush["buckets"]:
        nodes_dct[bucket['id']] = bucket
        for child in bucket["items"]:
            cid = child['id']
            if cid >= 0:
                nodes[cid] = CrushNode(cid, f"osd.{cid}", "osd", child['weight'] / 65536, [],
                                       class_name=osd_classes.get(cid))
    roots = []
    while nodes_dct:
        update_one = False
        for node_id in list(nodes_dct):
            node = nodes_dct[node_id]
            for item in node['items']:
                if item['id'] not in nodes:
                    break
            else:
                update_one = True
                nodes[node_id] = CrushNode(node_id, node['name'], node['type_name'], node['weight'] / 65536,
                                           [nodes[cdict['id']] for cdict in node['items']],
                                           class_name=None)
                del nodes_dct[node_id]
                if node['type_name'] == 'root':
                    roots.append(nodes[node_id])

        assert update_one, "Failed to parse crush"

    nodes_map = {node.name: node for node in nodes.values()}

    crush = Crush(nodes_map, roots, rules)
    crush.build_search_idx()
    return crush


def get_replication_nodes(rule: Rule, crush: Crush) -> Iterator[CrushNode]:
    return crush.get_root(rule.root).iter_nodes(rule.replicated_on)


def calc_node_class_weight(node: CrushNode, class_name: str) -> float:
    return sum(ch.weight for ch in node.iter_nodes('osd', class_name))


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


def get_all_child_osds(node: Dict, crush_nodes: Dict[int, Dict], target_class: str = None) -> Iterator[int]:
    # workaround for incorrect node classes on some prod clusters
    if node['type'] == 'osd' or re.match("osd\.\d+", node['name']):
        if target_class is None or node.get('device_class') == target_class:
            yield node['id']
        return

    for ch_id in node['children']:
        yield from get_all_child_osds(crush_nodes[ch_id], crush_nodes)

