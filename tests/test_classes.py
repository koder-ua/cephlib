import json
import subprocess
from pathlib import Path

from cephlib import RadosDF, CephDF, OSDDf, CephStatus, PGDump, CephReport


def get_test_file_path(name: str) -> Path:
    return Path(__file__).parent / 'ceph_outputs' / name


def get_test_file(name: str) -> str:
    return get_test_file_path(name).open().read()


def test_load_not_fail():
    CephStatus.from_json(get_test_file("status.luminous.json"))
    CephDF.from_json(get_test_file("df.luminous.json"))
    OSDDf.from_json(get_test_file("osd_df.luminous.json"))
    CephReport.from_json(get_test_file("report.luminous.json"))
    RadosDF.from_json(get_test_file("rados_df.luminous.json"))
    PGDump.from_json(get_test_file("pg_dump.luminous.json"))


def test_load_not_fail2():
    CephDF.from_json(get_test_file("df.luminous.2.json"))
    OSDDf.from_json(get_test_file("osd_df.luminous.2.json"))
    RadosDF.from_json(get_test_file("rados_df.luminous.2.json"))

    cmd = f"gunzip --keep --stdout {get_test_file_path('report.luminous.2.json.gz')}"
    CephReport.from_json(subprocess.check_output(cmd, shell=True))

    cmd = f"xz --decompress --keep --stdout {get_test_file_path('pg_dump.luminous.2.json.xz')}"
    PGDump.from_json(subprocess.check_output(cmd, shell=True))


def test_ceph_health():
    content_js = get_test_file("status.luminous.json")
    st = CephStatus.from_json(content_js)
    st_js = json.loads(content_js)
    assert st.election_epoch == st_js['election_epoch']
