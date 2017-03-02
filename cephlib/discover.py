""" Collect data about ceph nodes"""
import json
import logging


logger = logging.getLogger("cephlib")


class OSDInfo:
    def __init__(self, id, journal, storage, config):
        self.id = id
        self.journal = journal
        self.storage = storage
        self.config = config


def get_osds_nodes(rpc_run, extra_args=""):
    """Get dict, which maps node ip to list of OSDInfo"""

    data = rpc_run("ceph {} --format json osd dump".format(extra_args))
    jdata = json.loads(data)

    ips = {}
    first_error = True
    for osd_data in jdata["osds"]:
        osd_id = int(osd_data["osd"])
        if "public_addr" not in osd_data:
            if first_error:
                logger.warning("No 'public_addr' field in 'ceph osd dump' output for osd %s" +
                               "(all subsequent errors omitted)", osd_id)
                first_error = False
        else:
            ip_port = osd_data["public_addr"]
            ip = ip_port.split(":")[0]
            osd_cfg = rpc_run("ceph {} -n osd.{} --show-config".format(extra_args, osd_id))

            if osd_cfg.count("osd_journal =") != 1 or osd_cfg.count("osd_data =") != 1:
                logger.warning("Can't detect osd.{} journal or storage path. Use default values".format(osd_id))
                osd_data_path = "/var/lib/ceph/osd/ceph-{0}".format(osd_id)
                osd_journal_path = "/var/lib/ceph/osd/ceph-{0}/journal".format(osd_id)
            else:
                osd_journal_path = osd_cfg.split("osd_journal =")[1].split("\n")[0].strip()
                osd_data_path = osd_cfg.split("osd_data =")[1].split("\n")[0].strip()

            ips.setdefault(ip, []).append(OSDInfo(osd_id,
                                                  journal=osd_journal_path,
                                                  storage=osd_data_path,
                                                  config=osd_cfg))
    return ips


def get_mons_nodes(rpc_run, extra_args=""):
    """Return mapping mon_id => mon_ip"""

    data = rpc_run("ceph {} --format json mon_status".format(extra_args))
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
            ips[mon_data["rank"]] = (mon_data["addr"].split(":")[0], mon_data["name"])

    return ips

