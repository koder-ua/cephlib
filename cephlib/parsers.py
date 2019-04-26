import collections
import dataclasses
import datetime
import json
import re
from typing import Dict, Any, Iterator, Optional, List, Set, Tuple

from . import Crush, CephVersion, PGDump, PGState, Pool


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


# def copy_class_subtree(src_node: CrushNode, classname: str = None) -> Optional[CrushNode]:
#     childs = []
#     for ch in src_node.childs:
#         if ch.type == 'osd' and (ch.class_name != classname and classname is not None):
#             continue
#         ch_copy = copy_class_subtree(ch, classname)
#         if ch_copy:
#             childs.append(ch_copy)
#
#     weight = sum((ch.weight for ch in childs), 0) if childs else src_node.weight
#     if (src_node.type == 'osd' or childs) and weight != 0:
#         return CrushNode(id=src_node.id,
#                          name=src_node.name,
#                          type=src_node.type,
#                          class_name=src_node.class_name,
#                          weight=weight,
#                          childs=childs)


def parse_crushmap_js(filename: str = None, crush: Dict = None) -> Crush:
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
#
#
# def get_replication_nodes(rule: Rule, crush: Crush) -> Iterator[CrushNode]:
#     return crush.get_root(rule.root).iter_nodes(rule.replicated_on)
#
#
# def calc_node_class_weight(node: CrushNode, class_name: str) -> float:
#     return sum(ch.weight for ch in node.iter_nodes('osd', class_name))


def get_all_child_osds(node: Dict, crush_nodes: Dict[int, Dict], target_class: str = None) -> Iterator[int]:
    # workaround for incorrect node classes on some prod clusters
    if node['type'] == 'osd' or re.match("osd\.\d+", node['name']):
        if target_class is None or node.get('device_class') == target_class:
            yield node['id']
        return

    for ch_id in node['children']:
        yield from get_all_child_osds(crush_nodes[ch_id], crush_nodes)


def parse_txt_ceph_config(data: str) -> Dict[str, str]:
    config = {}
    for line in data.strip().split("\n"):
        name, val = line.split("=", 1)
        config[name.strip()] = val.strip()
    return config


def parse_pg_dump(data: Dict[str, Any]) -> PGDump:
    pgs: List[PG] = []

    def pgid_conv(vl_any):
        pool_s, pg_s = vl_any.split(".")
        return PGId(pool=int(pool_s), num=int(pg_s, 16), id=vl_any)

    def datetime_conv(vl_any):
        # datetime.datetime.strptime is too slow
        vl_s, mks = vl_any.split(".")
        ymd, hms = vl_s.split()
        return datetime.datetime(*map(int, ymd.split("-")+ hms.split(":")), int(mks))

    def process_status(status: str) -> Set[PGState]:
        try:
            return {getattr(PGState, status.replace("-", "_")) for status in status.split("+")}
        except AttributeError:
            raise ValueError(f"Unknown status {status}")

    name_map = {
        "stat_sum": lambda x: PGStatSum(**x),
        "state": process_status,
        "pgid": pgid_conv
    }

    for field in dataclasses.fields(PG):
        if field.name not in name_map:
            if field.type is datetime.datetime:
                name_map[field.name] = datetime_conv

    class TabulaRasa:
        pass

    for pg_info in data['pg_stats']:
        # hack to optimize load speed
        dt = {k: (name_map[k](v) if k in name_map else v) for k, v in pg_info.items()}
        pgs.append(PG(**dt))

    datetm, mks = data['stamp'].split(".")
    collected_at = datetime.datetime.strptime(datetm, '%Y-%m-%d %H:%M:%S').replace(microsecond=int(mks))
    return PGDump(collected_at=collected_at,
                  pgs={pg.pgid.id: pg for pg in pgs},
                  version=data['version'])


def parse_pg_distribution(pools: Dict[str, Pool], pg_dump: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, int]],
                                                                                    Dict[str, int],
                                                                                    Dict[int, int]]:

    pool_id2name = {pool.id: pool.name for pool in pools.values()}
    osd_pool_pg_2d: Dict[int, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    sum_per_pool: Dict[str, int] = collections.Counter()
    sum_per_osd: Dict[int, int] = collections.Counter()

    if pg_dump:
        for pg in pg_dump['pg_stats']:
            pool_id = int(pg['pgid'].split('.', 1)[0])
            for osd_id in pg['acting']:
                pool_name = pool_id2name[pool_id]
                osd_pool_pg_2d[osd_id][pool_name] += 1
                sum_per_pool[pool_name] += 1
                sum_per_osd[osd_id] += 1

    return {osd_id: dict(per_pool.items()) for osd_id, per_pool in osd_pool_pg_2d.items()}, \
            dict(sum_per_pool.items()), dict(sum_per_osd.items())


def parse_ceph_versions(data: str) -> Dict[str, CephVersion]:
    osd_ver_rr = re.compile(r"(osd\.\d+|mon\.[a-z0-9A-Z-]+):\s+{")
    vers: Dict[str, CephVersion] = {}
    for line in data.split("\n"):
        line = line.strip()
        if osd_ver_rr.match(line):
            name, data_js = line.split(":", 1)
            rr = version_rr.match(json.loads(data_js)["version"])
            try:
                vers[name.strip()] = parse_ceph_version(json.loads(data_js)["version"])
            except Exception as exc:
                raise ValueError(f"Can't parse version {line}: {exc}")
    return vers


def cmd(cmd_name: str):
    def closure(f):
        return f
    return closure
