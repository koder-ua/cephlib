import json
from pathlib import Path

import pytest

from cephlib import (CephHealth, CephRole, OSDStatus, CephStatusCode, MonRole, CephRelease, OSDStoreType, PGState,
                     EndpointAddr, DateTime, RadosDF, CephDF, OSDDf, CephStatus, CephIOStats, CephMGR, RadosGW,
                     PGStat, PGDump, CrushAlg, HashAlg, CrushMap, MonsMetadata, CrushRuleStepTake,
                     CrushRuleStepEmit, CrushRuleStepChooseLeafFirstN, OSDMapPool, OSDMap, CephReport, OSDState,
                     CephVersion, MonMetadata, OSDMetadata, parse_ceph_version, VolumeLVMList, LVMListDevice,
                     MonMetadata, parse_cmd_output, parse_ceph_version_simple, OSDPerf)
# from cephlib import (StatusRegion, Host, CephInfo, CephOSD, FileStoreInfo, BlueStoreInfo, OSDProcessInfo, CephDevInfo,
#                      Crush, OSDSpace, Pool, CephMonitor, OSDDevCfg, get_rule_osd_class, get_rule_replication_level)
# from cephlib import (OpType, ParseResult, OpDescription, HLTimings, CephOp, IPacker, get_historic_packer,
#                      UnexpectedEOF, parse_historic_file, print_records_from_file, RecId, RecordFile, OpRec)


def get_test_file(name: str) -> str:
    return (Path(__file__).parent / 'ceph_outputs' / name).open().read()


def test_load_not_fail():
    CephStatus.from_json(get_test_file("status.luminous.json"))
    CephDF.from_json(get_test_file("df.luminous.json"))
    OSDDf.from_json(get_test_file("osd_df.luminous.json"))
    CephReport.from_json(get_test_file("report.luminous.json"))
    RadosDF.from_json(get_test_file("rados_df.luminous.json"))


def test_ceph_health():
    content_js = get_test_file("status.luminous.json")
    st = CephStatus.from_json(content_js)
    st_js = json.loads(content_js)
    assert st.election_epoch == st_js['election_epoch']
