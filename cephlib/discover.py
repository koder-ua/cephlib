""" Collect data about ceph nodes"""
import json
import random
import logging
from typing import Callable, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger("cephlib")


class OSDInfo:
    def __init__(self, id: int, journal: str, storage: str, config: str, db: str = None,
                 bluestore: bool = None) -> None:
        self.id = id
        self.journal = journal
        self.storage = storage
        self.config = config
        self.db = db
        self.bluestore = bluestore

    def __str__(self) -> str:
        res = "OSDInfo({0.id!r}):\n    journal: {0.journal!r}\n    storage: {0.storage!r}".format(self)
        if self.db:
            res += "\n    db: {0.db}".format(self)
        return res


def get_osd_config(check_output: Callable[[str], str], extra_args: str, osd_id: str) -> str:
    return check_output("ceph {0} -n osd.{1} --show-config".format(extra_args, osd_id))


def pmap():
    pass


def get_osds_nodes(check_output: Callable[[str], str], extra_args: str = "",
                   thcount: int = 1) -> Dict[str, List[OSDInfo]]:
    """Get dict, which maps node ip to list of OSDInfo"""

    data = check_output("ceph {0} --format json osd dump".format(extra_args))
    jdata = json.loads(data)

    osd_infos = {}
    osd_ips = {}
    first_error = True

    for osd_data in jdata["osds"]:
        osd_id = int(osd_data["osd"])
        if "public_addr" not in osd_data:
            if first_error:
                logger.warning("No 'public_addr' field in 'ceph osd dump' output for osd %s" +
                               "(all subsequent errors omitted)", osd_id)
                first_error = False
        else:
            osd_ips[osd_id] = osd_data["public_addr"].split(":")[0]

    def worker(osd_id: str) -> Optional[str]:
        try:
            return get_osd_config(check_output, extra_args, osd_id)
        except:
            return None

    first_error = True
    ids = list(osd_ips)
    random.shuffle(ids)
    with ThreadPoolExecutor(thcount) as pool:
        for osd_id, osd_cfg in zip(ids, pool.map(worker, ids)):
            if osd_cfg is None:
                if first_error:
                    logger.warning("Failed to get config for OSD {0}".format(osd_id))
                    first_error = False
            else:
                if osd_cfg.count("osd_journal =") != 1 or osd_cfg.count("osd_data =") != 1:
                    logger.warning("Can't detect osd.{} journal or storage path. Use default values".format(osd_id))
                    osd_data_path = "/var/lib/ceph/osd/ceph-{0}".format(osd_id)
                    osd_journal_path = "/var/lib/ceph/osd/ceph-{0}/journal".format(osd_id)
                else:
                    osd_journal_path = osd_cfg.split("osd_journal =")[1].split("\n")[0].strip()
                    osd_data_path = osd_cfg.split("osd_data =")[1].split("\n")[0].strip()

                ip = osd_ips[osd_id]
                osd_infos.setdefault(ip, []).append(OSDInfo(osd_id,
                                                            journal=osd_journal_path,
                                                            storage=osd_data_path,
                                                            config=osd_cfg))
    return osd_infos


def get_mons_nodes(check_output: Callable[[str], str], extra_args: str = "") -> Dict[int, Tuple[str, str]]:
    """Return mapping mon_id => mon_ip"""
    data = check_output("ceph {0} --format json mon_status".format(extra_args))
    jdata = json.loads(data)
    ips = {}

    first_error = True
    for mon_data in jdata["monmap"]["mons"]:
        if "addr" not in mon_data:
            if first_error:
                mon_name = mon_data.get("name", "<MON_NAME_MISSED>")
                logger.warning("No 'addr' field in 'ceph mon_status' output for mon %s" +
                               "(all subsequent errors omitted)", mon_name)
                first_error = False
        else:
            ip = mon_data["addr"].split(":")[0]
            ips[mon_data["rank"]] = (ip, mon_data["name"])

    return ips

