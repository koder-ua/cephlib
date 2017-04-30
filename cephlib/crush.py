import re


class Node(object):
    def __init__(self, id, name, type, weight):
        self.id = id
        self.name = name
        self.type = type
        self.childs = []
        self.full_path = None
        self.weight = weight

    def str_path(self):
        if self.full_path:
            return "/".join("{0}={1}".format(tp, name) for tp, name in self.full_path)
        return None

    def __str__(self):
        w = ", w={0.weight}".format(self) if self.weight is not None else ""
        fp = (", " + self.str_path()) if self.full_path else ""
        return "{0.type}(name={0.name!r}, id={0.id}{1}{2})".format(self, w, fp)

    def __repr__(self):
        return str(self)

    def tree(self, tabs=0, tabstep=" " * 4):
        w = ", w={0.weight}".format(self) if self.weight is not None else ""
        yield tabstep * tabs + "{0.type}(name={0.name!r}, id={0.id}{1})".format(self, w)
        for cr_node in self.childs:
            for line in cr_node.tree(tabs=tabs + 1, tabstep=tabstep):
                yield line

    def copy(self):
        res = self.__class__(id=self.id, name=self.name, type=self.type, weight=self.weight)
        res.childs = self.childs
        return res

    def iter_nodes(self, node_type):
        if self.type == node_type:
            yield self
        for node in self.childs:
            for res in node.iter_nodes(node_type):
                yield res


class Crush(object):
    def __init__(self, nodes_map, roots):
        self.nodes_map = nodes_map
        self.roots = roots
        self.search_cache = None

    def __str__(self):
        return "\n".join("\n".join(root.tree()) for root in self.roots)

    def iter_nodes(self, node_type):
        for node in self.roots:
            for res in node.iter_nodes(node_type):
                yield res

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


def iter_buckets(crushmap):
    bucket_lines = None
    bucket_start_re = re.compile(r"(?P<type>[^ ]+)\s+(?P<name>[^ ]+)\s*\{$")

    bucket_type = None
    bucket_name = None
    in_bucket = False

    for line in crushmap.split("\n"):
        line = crush_prep_line(line)
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


def load_crushmap(filename=None, content=None):
    roots = []
    osd_re = re.compile(r"osd\.\d+$")
    id_line_re = re.compile(r"id\s+-?\d+$")
    item_line_re = re.compile(r"item\s+(?P<name>[^ ]+)\s+weight\s+(?P<weight>[0-9.]+)$")
    nodes_map = {}

    if filename:
        content = open(filename).read()

    for name, type, lines in iter_buckets(content):
        node = Node(None, name, type, None)

        for line in lines:
            if id_line_re.match(line):
                node.id = int(line.split()[1])
            else:
                item_rr = item_line_re.match(line)
                if item_rr:
                    node_name = item_rr.group('name')
                    weight = float(item_rr.group('weight'))

                    # append OSD child
                    if osd_re.match(node_name):
                        node_id = int(node_name.split(".")[1])
                        ch_node = Node(node_id, node_name, "osd", weight)
                        nodes_map[node_name] = ch_node
                    else:
                        # append child of other type (must be described already in file)
                        ch_node = nodes_map[node_name].copy()
                        ch_node.weight = weight

                    node.childs.append(ch_node)

        if type == 'root':
            roots.append(node)

        nodes_map[node.name] = node

    crush = Crush(nodes_map, roots)
    crush.build_search_idx()
    return crush
