import json

from cephlib import CephOp, pack_historic, unpack_historic, HistoricFields, pack_record, RecId, unpack_record

from test_classes import get_test_file

fc = get_test_file("historic_dump.lumious.json")


def test_parsing():
    ops = [CephOp.parse_op(op) for op in json.loads(fc)['ops']]


def test_packing():
    ops = []
    for op_j in json.loads(fc)['ops']:
        _, op = CephOp.parse_op(op_j)
        if op:
            op.pack_pool_id = op.pool_id
            ops.append(op)

    data = pack_historic(ops, fields=HistoricFields.compact | HistoricFields.with_names, extra=(1, 2))
    extra, itr = unpack_historic(data)
    assert extra == [1, 2]
    uops = list(itr)

    assert len(ops) == len(uops)

    ops.sort(key=lambda x: x.obj_name)
    uops.sort(key=lambda x: x["obj_name"])

    for op, uop in zip(ops, uops):
        assert op.pg == uop["pg"]
        assert op.pack_pool_id == uop["pack_pool_id"]
        assert op.obj_name == uop["obj_name"]

    tpl = (1, 2, ops, HistoricFields.compact | HistoricFields.with_names)
    _, data2 = pack_record(RecId.ops, tpl)
    uops2 = unpack_record(RecId.ops, data2)

    assert len(ops) == len(uops2)

    uops2.sort(key=lambda x: x["obj_name"])

    for op, uop2 in zip(ops, uops2):
        assert op.pg == uop2["pg"]
        assert op.pack_pool_id == uop2["pack_pool_id"]
        assert op.obj_name == uop2["obj_name"]
        assert 1 == uop2["osd_id"]
        assert 2 == uop2["time"]
