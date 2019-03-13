import sys
import json
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Bucket:
    name: str
    id: int
    weight: int
    type: str
    child_ids: List[Tuple[int, int]]
    childs: List['Bucket'] = field(default_factory=list)
    class_name: Optional[str] = None

    def is_osd(self) -> bool:
        return self.id >= 0

    def to_str(self, offset: int = 0, step: str = ' ' * 4) -> str:
        if self.is_osd():
            if self.class_name:
                return step * offset + "osd.{0.id} {0.class_name} {1:.3f}\n".format(self, self.weight / 1000000.)
            else:
                return step * offset + "osd.{0.id} {1:.3f}\n".format(self, self.weight / 1000000.)
        else:
            res = step * offset + "{0.type} {0.name} {1:.3f}\n"
            return res + "".join(bucket.to_str(offset + 1, step) for bucket in self.childs)

    def copy_cls(self, class_name: str = None) -> 'Bucket':
        nch = []
        for ch in self.childs:
            if ch.is_osd() and class_name is not None and ch.class_name != class_name:
                continue
            nch.append(ch.copy_cls(class_name))
        return self.__class__(self.name,
                              self.id,
                              sum(ch.weight for ch in nch),
                              self.type,
                              child_ids=[],
                              childs=nch,
                              class_name=self.class_name)


class OP(Enum):
    take = 0
    chooseleaf_firstn = 1
    emit = 2


@dataclass
class Step:
    op: str
    params: Dict[str, Any]


@dataclass
class Rule:
    name: str
    min_size: int
    max_size: int
    ruleset: int
    type: int
    id: int
    steps: List[Step]

    def get_root(self) -> str:
        for step in self.steps:
            if step.op == OP.take:
                return step.params['item_name']
        raise ValueError("No root take step found")

    def get_selected_class(self) -> Optional[str]:
        pass


@dataclass
class Crush:
    buckets: Dict[int, Bucket] = field(default_factory=dict)
    bucket_names: Dict[str, Bucket] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    def get_tree_for_rule(self, rule_id: int) -> Bucket:
        rule = self.rules[rule_id]
        return self.bucket_names[rule.get_root()].copy_cls(rule.get_selected_class())


def load_crush(report: Any) -> Crush:
    osd_classes = {dev['id']: dev['class'] for dev in report['devices']}
    crush = report['crushmap']
    buckets = {}
    roots = []
    rules = []

    for bucket_dct in crush['buckets']:
        if '~' in bucket_dct['name']:
            name, cls_name = bucket_dct['name'].split('~')
        else:
            name, cls_name = bucket_dct['name'], ''
        bkt = Bucket(name=name,
                     id=bucket_dct['id'],
                     weight=bucket_dct['weight'],
                     type=bucket_dct['type_name'],
                     child_ids=[(ch['id'], ch['weight']) for ch in bucket_dct['items']])
        buckets[bucket_dct['id']] = bkt

    for bkt in buckets.values():
        for chid, chw in bkt.child_ids:
            if chid >= 0:
                bkt.childs.append(Bucket('osd.{}'.format(chid), chid, chw, 'osd', [], [], osd_classes.get(chid)))

    for rule in crush['rules']:
        rules.append(Rule(name=rule['rule_name'],
                          max_size=rule['max_size'],
                          min_size=rule['min_size'],
                          id=rule['rule_id'],
                          ruleset=rule['ruleset'],
                          steps=[Step(op=getattr(OP, step['op']), params=step) for step in rule['steps']],
                          type=rule['type']))

    return Crush(buckets=buckets, bucket_names={bkt.name: bkt for bkt in buckets.values()}, rules=rules)


def main() -> int:
    cmd = sys.argv[1]
    path = sys.argv[2]
    report = json.load(open(path))
    assert cmd == 'tree'

    osd_classes = {dev['id']: dev['class'] for dev in report['devices']}

    crush = report['crushmap']
    buckets = {}
    roots = []

    for bucket in crush['buckets']:
        bkt = Bucket(name=bucket['name'], id=bucket['id'],
                     weight=bucket['weight'], type=bucket['type_name'],
                     ch_ids=[(j['id'], j['weight']) for j in bucket['items']])
        if bkt.type == 'root':
            roots.append(bkt)
        buckets[bucket['id']] = bkt


    for root in roots:
        print(root.to_str(0, buckets), params=)
    return 0


if __name__ == "__main__":
    exit(main())


