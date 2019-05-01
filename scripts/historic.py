import sys
import json
from dataclasses import dataclass
from typing import Dict, Optional

from cephlib import CephOp, OpType
from cephlib.historic_ops import to_unix_ms, get_hl_timings



@dataclass
class OP:
    duration: float
    descr: str
    stages: Dict[str, float]

    def __str__(self) -> str:
        stages = "\n    ".join(f"{key}: {tm}" for key, tm in self.stages.items())
        return f"OP(duration={self.duration * 1000:.1f}, descr={self.get_type()}):\n    {stages}"

    def get_type(self) -> Optional[OpType]:
        tp = self.descr.split('(')[0]
        if tp == 'osd_op':
            if "+write+" in self.descr:
                if 'sub_op_commit_rec' in self.stages:
                    return OpType.write_primary
                return OpType.write_secondary
            elif "+read+" in self.descr:
                return OpType.read

        if tp == 'osd_repop_reply' or tp == 'osd_repop':
            return OpType.write_secondary

        if tp not in ('osd_sub_op', 'replica scrub', 'osd_sub_op_reply'):
            raise ValueError(self.descr)


io = 0
wpg = 0

fl = open(sys.argv[1]).read()
for chunk in fl.split("\n}\n"):
    ops = []

    if not chunk.strip():
        continue

    try:
        dt = json.loads(chunk + "}")
    except json.JSONDecodeError:
        print(chunk + "}")
        raise
    for op in dt["Ops"]:
        started_at = to_unix_ms(op['initiated_at'])
        stages = {}
        for stage in op['type_data'][-1]:
            stages[stage["event"]] = to_unix_ms(stage["time"]) - started_at

        pop = OP(op['duration'], op['description'], stages)
        if pop.get_type() is not None:
            ops.append(pop)

    for op in ops:
        try:
            tm = get_hl_timings(op.get_type(), op.stages)
        except:
            print(op.descr, op)
            raise
        if tm.wait_for_pg > 0:
            wpg += tm.wait_for_pg

        if tm.local_io > 0:
            io += tm.local_io

        # print(f"{int(op.duration * 1000):>10d} {tm.download:>10d} {tm.wait_for_pg:>10d} {tm.local_io:>10d} {tm.wait_for_replica:>10d}")

print(wpg, io)