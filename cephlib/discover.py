""" Collect data about ceph nodes"""
import asyncio
import json
import random
import logging
from typing import Callable, Dict, List, Tuple, Optional


logger = logging.getLogger("cephlib")


class OSDConfig:
    def __init__(self, osd_id: int, config: Dict[str, str]) -> None:
        self.osd_id = osd_id
        self.config = config

    def __str__(self) -> str:
        return "OSDInfo({})".format(self.osd_id)


async def get_osd_config(check_output: Callable[[str], str], extra_args: str, osd_id: int) -> str:
    return check_output("ceph {0} -n osd.{1} --show-config".format(extra_args, osd_id))


async def get_osds_nodes(check_output: Callable,
                         extra_args: str = "",
                         get_config: bool = True,
                         max_workers: int = 16) -> Dict[str, List[OSDConfig]]:
    """Get dict, which maps node ip to list of OSDInfo"""

    jdata = json.loads(await check_output("ceph {0} --format json osd dump".format(extra_args)))

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

    semaphore = asyncio.Semaphore(max_workers)

    async def get_osd_config_coro(osd_id: int) -> Tuple[int, Optional[str]]:
        await semaphore.acquire()
        try:
            return osd_id, (await get_osd_config(check_output, extra_args, osd_id))
        except Exception:
            return osd_id, None

    if not get_config:
        for osd_id, ip in osd_ips.items():
            osd_infos.setdefault(ip, []).append(OSDConfig(osd_id, {}))
        return osd_infos

    first_error = True
    osd_ids = list(osd_ips)

    # shuffle to run in parallel for osd's from different nodes
    random.shuffle(osd_ids)

    for osd_id, osd_cfg in asyncio.gather(*map(get_osd_config_coro, osd_ids)):
        if osd_cfg is None:
            if first_error:
                logger.warning("Failed to get config for OSD {0}".format(osd_id))
                first_error = False
        else:
            osd_infos.setdefault(osd_ips[osd_id], []).append(OSDConfig(osd_id, config=osd_cfg))
    return osd_infos


async def get_mons_nodes(check_output, extra_args: str = "") -> Dict[int, Tuple[str, str]]:
    """Return mapping mon_id => mon_ip"""
    data = await check_output("ceph {0} --format json mon_status".format(extra_args))
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
