import json
from typing import Iterator, List, Optional, Dict


class CrushNode:
    def __init__(self, id: int, name: str, type: str, weight: Optional[float], childs: List['CrushNode'],
                 class_name: str = None) -> None:
        self.id = id
        self.name = name
        self.type = type
        self.childs = childs
        self.full_path = None
        self.weight = weight
        self.class_name = class_name

    def str_path(self) -> Optional[str]:
        if self.full_path:
            return "/".join("{0}={1}".format(tp, name) for tp, name in self.full_path)
        return None

    def __str__(self) -> str:
        w = ", w={0.weight}".format(self) if self.weight is not None else ""
        fp = (", " + self.str_path()) if self.full_path else ""
        return "{0.type}(name={0.name!r}, id={0.id}{1}{2})".format(self, w, fp)

    def __repr__(self) -> str:
        return str(self)

    def tree(self, tabs: int = 0, tabstep: str = " " * 4) -> Iterator[str]:
        w = ", w={0.weight}".format(self) if self.weight is not None else ""
        yield tabstep * tabs + "{0.type}(name={0.name!r}, id={0.id}{1})".format(self, w)
        for cr_node in self.childs:
            yield from cr_node.tree(tabs=tabs + 1, tabstep=tabstep)

    def copy(self) -> 'CrushNode':
        res = self.__class__(id=self.id, name=self.name, type=self.type, weight=self.weight, childs=self.childs,
                             class_name=self.class_name)
        return res

    def iter_nodes(self, node_type: str, class_name: str = None) -> Iterator['CrushNode']:
        if self.type == node_type and (class_name in (None, "") or class_name == self.class_name):
            yield self
        for node in self.childs:
            yield from node.iter_nodes(node_type, class_name=class_name)


class Rule:
    def __init__(self, name: str, id: int, root: str, replicated_on: str, class_name: str = None) -> None:
        self.name = name
        self.id = id
        self.root = root
        self.class_name = class_name
        self.replicated_on = replicated_on

    def __str__(self) -> str:
        return "Rule({0.name}, {0.id}, root={0.root}, class={0.class_name!r}, repl={0.replicated_on})".format(self)


class Crush:
    def __init__(self, nodes_map: Dict[str, CrushNode], roots: List[CrushNode], rules: Dict[int, Rule]) -> None:
        self.nodes_map = nodes_map
        self.roots = roots
        self.rules = rules
        self.search_cache = None

    def get_root(self, name: str) -> CrushNode:
        for root in self.roots:
            if root.name == name:
                return root
        else:
            raise KeyError("Can't found crush root {}".format(name))

    def __str__(self):
        return "\n".join("\n".join(root.tree()) for root in self.roots)

    def iter_nodes_for_rule(self, rule_id: int, tp: str) -> Iterator[CrushNode]:
        root = self.get_root(self.rules[rule_id].root)
        return root.iter_nodes(tp, class_name=self.rules[rule_id].class_name)

    def iter_osds_for_rule(self, rule_id: int) -> Iterator[CrushNode]:
        return self.iter_nodes_for_rule(rule_id, 'osd')

    def iter_nodes(self, node_type, class_name: str = None):
        for node in self.roots:
            yield from node.iter_nodes(node_type, class_name=class_name)

    def build_search_idx(self):
        if self.search_cache is None:
            not_done = []
            for node in self.roots:
                node.full_path = [(node.type, node.name)]
                not_done.append(node)

            done = []
            while not_done:
                new_not_done = []
                for node in not_done:
                    if node.childs:
                        for chnode in node.childs:
                            chnode.full_path = node.full_path[:] + [(chnode.type, chnode.name)]
                            new_not_done.append(chnode)
                    else:
                        done.append(node)
                not_done = new_not_done
            self.search_cache = done

    def find_nodes(self, path):
        if not path:
            return self.roots

        res = []
        for node in self.search_cache:
            nfilter = dict(path)
            for tp, val in node.full_path:
                if tp in nfilter:
                    if nfilter[tp] == val:
                        del nfilter[tp]
                        if not nfilter:
                            break
                    else:
                        break

            if not nfilter:
                res.append(node)

        return res

    def find_node(self, path):
        nodes = self.find_nodes(path)
        if not nodes:
            raise IndexError("Can't found any node with path {0!r}".format(path))
        if len(nodes) > 1:
            raise IndexError("Found {0} nodes  for path {1!r} (should be only 1)".format(len(nodes), path))
        return nodes[0]

    def copy_tree_for_rule(self, rule: Rule) -> Optional[CrushNode]:
        return copy_class_subtree(self.get_root(rule.root), rule.class_name)


def copy_class_subtree(src_node: CrushNode, classname: str = None) -> Optional[CrushNode]:
    childs = []
    for ch in src_node.childs:
        if ch.type == 'osd' and (ch.class_name != classname and classname is not None):
            continue
        ch_copy = copy_class_subtree(ch, classname)
        if ch_copy:
            childs.append(ch_copy)

    weight = sum((ch.weight for ch in childs), 0) if childs else src_node.weight
    if (src_node.type == 'osd' or childs) and weight != 0:
        return CrushNode(id=src_node.id,
                         name=src_node.name,
                         type=src_node.type,
                         class_name=src_node.class_name,
                         weight=weight,
                         childs=childs)


def load_crushmap_js(filename: str = None, crush: Dict = None) -> Crush:
    assert not filename or not crush, "filename and content should not be passed at the same time"

    if filename:
        crush = json.load(open(filename))

    assert crush
    rules = {}

    for rule in crush["rules"]:
        for step in rule['steps']:
            if step['op'] == 'take':
                root = step["item_name"]
                break
        else:
            continue

        if '~' in root:
            root, class_name = root.split("~")
        else:
            class_name = None

        replicated_on = None
        for step in rule['steps'][1:]:
            if step['op'] in ("chooseleaf_firstn", "chooseleaf_indep"):
                replicated_on = step["type"]

        rules[rule['rule_id']] = Rule(name=rule['rule_name'],
                                      id=rule['rule_id'],
                                      root=root,
                                      class_name=class_name,
                                      replicated_on=replicated_on)

    nodes_dct: Dict[int, Dict] = {}
    nodes: Dict[int, CrushNode] = {}
    osd_classes = {osd['id']: osd.get("class", "") for osd in crush['devices']}
    for bucket in crush["buckets"]:
        nodes_dct[bucket['id']] = bucket
        for child in bucket["items"]:
            cid = child['id']
            if cid >= 0:
                nodes[cid] = CrushNode(cid, f"osd.{cid}", "osd", child['weight'] / 65536, [],
                                       class_name=osd_classes.get(cid))
    roots = []
    while nodes_dct:
        update_one = False
        for node_id in list(nodes_dct):
            node = nodes_dct[node_id]
            for item in node['items']:
                if item['id'] not in nodes:
                    break
            else:
                update_one = True
                nodes[node_id] = CrushNode(node_id, node['name'], node['type_name'], node['weight'] / 65536,
                                           [nodes[cdict['id']] for cdict in node['items']],
                                           class_name=None)
                del nodes_dct[node_id]
                if node['type_name'] == 'root':
                    roots.append(nodes[node_id])

        assert update_one, "Failed to parse crush"

    nodes_map = {node.name: node for node in nodes.values()}

    crush = Crush(nodes_map, roots, rules)
    crush.build_search_idx()
    return crush


def get_replication_nodes(rule: Rule, crush: Crush) -> Iterator[CrushNode]:
    return crush.get_root(rule.root).iter_nodes(rule.replicated_on)


def calc_node_class_weight(node: CrushNode, class_name: str) -> float:
    return sum(ch.weight for ch in node.iter_nodes('osd', class_name))

