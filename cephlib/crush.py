import re
from typing import Iterator, Tuple, List, Optional, Dict, Union, Type


class Node:
    def __init__(self, id: int, name: str, type: str, weight: Optional[float], childs: List['Node'],
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

    def copy(self) -> 'Node':
        res = self.__class__(id=self.id, name=self.name, type=self.type, weight=self.weight, childs=self.childs,
                             class_name=self.class_name)
        return res

    def iter_nodes(self, node_type: str, class_name: str = None) -> Iterator['Node']:
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
    def __init__(self, nodes_map: Dict[str, Node], roots: List[Node], rules: Dict[int, Rule]) -> None:
        self.nodes_map = nodes_map
        self.roots = roots
        self.rules = rules
        self.search_cache = None

    def get_root(self, name: str) -> Node:
        for root in self.roots:
            if root.name == name:
                return root
        else:
            raise KeyError("Can't found crush root {}".format(name))

    def __str__(self):
        return "\n".join("\n".join(root.tree()) for root in self.roots)

    def iter_nodes_for_rule(self, rule_id: int, tp: str) -> Iterator[Node]:
        root = self.get_root(self.rules[rule_id].root)
        return root.iter_nodes(tp, class_name=self.rules[rule_id].class_name)

    def iter_osds_for_rule(self, rule_id: int) -> Iterator[Node]:
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


def crush_prep_line(line):
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def iter_crush_lines(crushmap: str) -> Iterator[str]:
    for line in crushmap.split("\n"):
        line = crush_prep_line(line)
        if line:
            yield line


def iter_buckets(crushmap: str) -> Iterator[Tuple[str, str, str]]:
    bucket_lines = None
    bucket_start_re = re.compile(r"(?P<type>[^ ]+)\s+(?P<name>[^ ]+)\s*\{$")

    bucket_type = None
    bucket_name = None
    in_bucket = False

    for line in iter_crush_lines(crushmap):
        if in_bucket:
            if line == '}':
                in_bucket = False
                yield bucket_name, bucket_type, bucket_lines
                bucket_lines = None
            else:
                bucket_lines.append(line)
        else:
            rr = bucket_start_re.match(line)
            if rr:
                in_bucket = True
                bucket_type = rr.group('type')
                bucket_name = rr.group('name')
                bucket_lines = []


def find_device_classes(crushmap: str) -> Dict[int, str]:
    osd_class_re = re.compile(r"device\s+(?P<osd_id>\d+)\s+osd\.\d+\s+class\s+(?P<class_name>[^\s]+)$")
    res = {}  # type: Dict[int, str]
    for line in iter_crush_lines(crushmap):
        rr = osd_class_re.match(line)
        if rr:
            res[int(rr.group('osd_id'))] = rr.group('class_name')
    return res


def load_crushmap(filename: str = None, content: str = None) -> Crush:
    roots = []
    osd_re = re.compile(r"osd\.\d+$")
    id_line_re = re.compile(r"id\s+-?\d+$")
    item_line_re = re.compile(r"item\s+(?P<name>[^ ]+)\s+weight\s+(?P<weight>[0-9.]+)$")
    rule_re = re.compile(r"(?ms)\brule\s+(?P<name>[^\s]+)\s+{[^}]*?\b" +
                         r"id\s+(?P<id>\d+)\s+[^}]*\bstep\s+take\s+(?P<root>.+?)\s")
    class_re = re.compile(r"class\s+(?P<class_name>[^\s]+)")
    step_take = re.compile(r"step\s+chooseleaf\s+firstn\s+0\s+type\s+(?P<bucket_type_name>[^\s]+)")
    nodes_map = {}

    if filename:
        content = open(filename).read()

    assert content
    rules = {}
    for rr in rule_re.finditer(content):
        rule_id = int(rr.group('id'))
        _, endp = rr.span()
        rest_of_rule = content[endp: content.find("}", endp)]
        assert '{' not in rest_of_rule
        class_rr = class_re.match(rest_of_rule)
        class_name = None if class_rr is None else class_rr.group('class_name')

        step_take_rr = step_take.search(rest_of_rule)
        assert step_take_rr

        rules[rule_id] = Rule(name=rr.group('name'),
                              id=rule_id,
                              root=rr.group("root"),
                              class_name=class_name,
                              replicated_on=step_take_rr.group('bucket_type_name'))

    classes = find_device_classes(content)

    for name, type, lines in iter_buckets(content):
        node_id = None
        childs = []
        for line in lines:
            if id_line_re.match(line):
                node_id = int(line.split()[1])
            else:
                item_rr = item_line_re.match(line)
                if item_rr:
                    node_name = item_rr.group('name')
                    weight = float(item_rr.group('weight'))

                    # append OSD child
                    if osd_re.match(node_name):
                        node_id = int(node_name.split(".")[1])
                        ch_node = Node(node_id, node_name, "osd", weight, [],
                                       class_name=classes.get(node_id))
                        nodes_map[node_name] = ch_node
                    else:
                        # append child of other type (must be described already in file)
                        ch_node = nodes_map[node_name].copy()
                        ch_node.weight = weight

                    childs.append(ch_node)

        assert node_id is not None

        node = Node(node_id, name, type, None, childs)
        if type == 'root':
            roots.append(node)

        nodes_map[node.name] = node

    crush = Crush(nodes_map, roots, rules)
    crush.build_search_idx()
    return crush


def get_replication_nodes(rule: Rule, crush: Crush) -> Iterator[Node]:
    return crush.get_root(rule.root).iter_nodes(rule.replicated_on)


def calc_node_class_weight(node: Node, class_name: str) -> float:
    return sum(ch.weight for ch in node.iter_nodes('osd', class_name))

