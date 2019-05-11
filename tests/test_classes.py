import json
from pathlib import Path

from cephlib import RadosDF, CephDF, OSDDf, CephStatus, PGDump, CephReport


def get_test_file(name: str) -> str:
    return (Path(__file__).parent / 'ceph_outputs' / name).open().read()


def test_load_not_fail():
    CephStatus.from_json(get_test_file("status.luminous.json"))
    CephDF.from_json(get_test_file("df.luminous.json"))
    OSDDf.from_json(get_test_file("osd_df.luminous.json"))
    CephReport.from_json(get_test_file("report.luminous.json"))
    RadosDF.from_json(get_test_file("rados_df.luminous.json"))
    PGDump.from_json(get_test_file("pg_dump.luminous.json"))


def test_ceph_health():
    content_js = get_test_file("status.luminous.json")
    st = CephStatus.from_json(content_js)
    st_js = json.loads(content_js)
    assert st.election_epoch == st_js['election_epoch']
