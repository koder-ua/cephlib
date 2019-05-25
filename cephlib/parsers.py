from __future__ import annotations

import json
import collections
import dataclasses
from pathlib import Path
from typing import Dict, Union

from . import VolumeLVMList, PGDump


@dataclasses.dataclass
class OSDBSDevices:
    db: Path
    wal: Path
    block: Path


@dataclasses.dataclass
class OSDFSDevices:
    journal: Path
    data: Path


OSDDevInfo = Union[OSDBSDevices, OSDFSDevices]


def parse_ceph_volumes_js(cephvollist_js: str) -> Dict[int, OSDDevInfo]:
    devs_for_osd: Dict[int, Dict[str, Path]] = {}

    for osd_id, devs in VolumeLVMList.convert(json.loads(cephvollist_js)).osds.items():
        assert len(devs) == 1, "FixME"
        for dev in devs:
            assert len(dev.devices) == 1, "FixME"
            devs_for_osd.setdefault(osd_id, {})[dev.type] = Path(dev.devices[0])

    result: Dict[int, OSDDevInfo] = {}
    for osd_id, attrs in devs_for_osd.items():
        if 'block' in attrs:
            block = attrs['block']
            db = attrs.get("db", block)
            wal = attrs.get("wal", db)
            result[osd_id] = OSDBSDevices(Path(db), Path(wal), Path(block))
        else:
            data = attrs['data']
            journal = attrs.get('journal', data)
            result[osd_id] = OSDFSDevices(Path(journal), Path(data))

    return result


def parse_ceph_disk_js(cephdisklist_js: str) -> Dict[int, OSDDevInfo]:
    devs_for_osd: Dict[int, OSDDevInfo] = {}
    cephdisk_dct = json.loads(cephdisklist_js)
    for dev_info in cephdisk_dct:
        for part_info in dev_info.get('partitions', []):
            if "cluster" in part_info and part_info.get('type') == 'data':
                osd_id = int(part_info['whoami'])
                if "journal_dev" in part_info:
                    data = part_info['path']
                    journal = part_info.get('journal_dev', data)
                    devs_for_osd[osd_id] = OSDFSDevices(Path(journal), Path(data))
                else:
                    block = part_info['block_dev']
                    db = part_info.get("block.db_dev", block)
                    wal = part_info.get("block.wal_dev", db)
                    devs_for_osd[osd_id] = OSDBSDevices(Path(db), Path(wal), Path(block))
    return devs_for_osd


def parse_txt_ceph_config(data: str) -> Dict[str, str]:
    config = {}
    for line in data.strip().split("\n"):
        name, val = line.split("=", 1)
        config[name.strip()] = val.strip()
    return config


def parse_pg_distribution(pg_dump: PGDump) -> Dict[int, Dict[int, int]]:
    """ returns: {osd_id: {pool_id: pg_count}}, {pool_id: pg_count} """
    osd_pool_pg_2d: Dict[int, Dict[str, int]] = collections.defaultdict(lambda: collections.Counter())

    for pg in pg_dump.pg_stats:
        for osd_id in pg.acting:
            osd_pool_pg_2d[osd_id][pg.pgid.pool] += 1

    return {osd_id: dict(per_pool.items()) for osd_id, per_pool in osd_pool_pg_2d.items()}
