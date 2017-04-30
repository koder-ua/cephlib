""" Collect data about ceph nodes"""
import json
import logging
from functools import partial

from .pyver import tostr
from .common import pmap


logger = logging.getLogger("cephlib")


class OSDInfo(object):
    def __init__(self, id, journal, storage, config):
        self.id = id
        self.journal = journal
        self.storage = storage
        self.config = config

    def __str__(self):
        return "OSDInfo({0.id!r}):\n    journal: {0.journal!r}\n    storage: {0.storage!r}".format(self)


def get_osd_config(check_output, extra_args, osd_id):
    return check_output("ceph {0} -n osd.{1} --show-config".format(extra_args, osd_id))


def get_osds_nodes(check_output, extra_args="", thcount=1):
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
            osd_ips[osd_id] = tostr(osd_data["public_addr"]).split(":")[0]

    worker = partial(get_osd_config, check_output, extra_args)
    osd_ids = list(osd_ips.keys())
    first_error = True
    for (is_ok, osd_cfg), osd_id in zip(pmap(worker, osd_ids, thcount=thcount), osd_ids):
        if not is_ok:
            if first_error:
                logger.warning("Failed to get config for OSD {0}".format(osd_id))
                first_error = False
            continue

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


def get_mons_nodes(check_output, extra_args=""):
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
            ip = tostr(mon_data["addr"]).split(":")[0]
            ips[mon_data["rank"]] = (ip, tostr(mon_data["name"]))

    return ips

