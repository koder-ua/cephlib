import re
import abc
import logging
import collections
import xml.etree.ElementTree as ET
from typing import List, Tuple, cast, Any, Set, Dict, Optional, Sequence, NamedTuple

from .units import b2ssize
from .istorage import IStorable, Storable
from .ssh import ConnCreds


logger = logging.getLogger("cephlib")


OSRelease = NamedTuple("OSRelease",
                       [("distro", str),
                        ("release", str),
                        ("arch", str)])


class NodeInfo(IStorable):
    """Node information object, result of discovery process or config parsing"""
    def __init__(self, ssh_creds: ConnCreds, roles: Set[str], params: Dict[str, Any] = None) -> None:
        # ssh credentials
        self.ssh_creds = ssh_creds
        self.roles = roles
        self.os_vm_id = None  # type: Optional[int]
        self.params = {}  # type: Dict[str, Any]
        if params is not None:
            self.params = params

    @property
    def node_id(self) -> str:
        return "{0.host}:{0.port}".format(self.ssh_creds.addr)

    def __str__(self) -> str:
        return self.node_id

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        dct = self.__dict__.copy()
        dct['ssh_creds'] = self.ssh_creds.raw()
        dct['roles'] = list(self.roles)
        return dct

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'NodeInfo':
        data = data.copy()
        data['ssh_creds'] = ConnCreds.fromraw(data['ssh_creds'])
        data['roles'] = set(data['roles'])
        obj = cls.__new__(cls)  # type: ignore
        obj.__dict__.update(data)
        return obj


class ISSHHost(metaclass=abc.ABCMeta):
    """Minimal interface, required to setup RPC connection"""
    info = None  # type: NodeInfo

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: Optional[str], content: bytes) -> str:
        pass

    def __enter__(self) -> 'ISSHHost':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False

    @property
    def node_id(self) -> str:
        return self.info.node_id


class IRPCNode(metaclass=abc.ABCMeta):
    """Remote filesystem interface"""
    info = None  # type: NodeInfo
    conn = None  # type: Any
    rpc_log_file = None  # type: str

    @property
    def node_id(self) -> str:
        return self.info.node_id

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def run(self, cmd: str, timeout: int = 60, nolog: bool = False, check_timeout: float = 0.01) -> str:
        pass

    @abc.abstractmethod
    def copy_file(self, local_path: str, remote_path: str = None,
                  expanduser: bool = False, compress: bool = False) -> str:
        pass

    @abc.abstractmethod
    def get_file_content(self, path: str, expanduser: bool = False, compress: bool = False) -> bytes:
        pass

    @abc.abstractmethod
    def put_to_file(self, path: Optional[str], content: bytes, expanduser: bool = False, compress: bool = False) -> str:
        pass

    @abc.abstractmethod
    def stat_file(self, path:str, expanduser: bool = False) -> Any:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass

    @abc.abstractmethod
    def upload_plugin(self, name: str, code: bytes, version: str = None) -> None:
        pass

    def __enter__(self) -> 'IRPCNode':
        return self

    def __exit__(self, x, y, z) -> bool:
        self.disconnect()
        return False


def log_nodes_statistic(nodes: Sequence[IRPCNode]) -> None:
    logger.info("Found {0} nodes total".format(len(nodes)))

    per_role = collections.defaultdict(int)  # type: Dict[str, int]
    for node in nodes:
        for role in node.info.roles:
            per_role[role] += 1

    for role, count in sorted(per_role.items()):
        logger.debug("Found {0} nodes with role {1}".format(count, role))


def get_os(node: IRPCNode) -> OSRelease:
    """return os type, release and architecture for node.
    """
    arch = node.run("arch", nolog=True).strip()

    try:
        node.run("ls -l /etc/redhat-release", nolog=True)
        return OSRelease('redhat', None, arch)
    except:
        pass

    try:
        node.run("ls -l /etc/debian_version", nolog=True)

        release = None
        for line in node.run("lsb_release -a", nolog=True).split("\n"):
            if ':' not in line:
                continue
            opt, val = line.split(":", 1)

            if opt == 'Codename':
                release = val.strip()

        return OSRelease('ubuntu', release, arch)
    except:
        pass

    raise RuntimeError("Unknown os")


def get_data(rr: str, data: str) -> str:
    match_res = re.search("(?ims)" + rr, data)
    return match_res.group(0)


class HWInfo(Storable):
    __ignore_fields__ = ['raw_xml']

    def __init__(self) -> None:
        self.hostname = None  # type: str
        self.cores = []  # type: List[Tuple[str, int]]

        # /dev/... devices
        self.disks_info = {}  # type: Dict[str, Tuple[str, int]]

        # real disks on raid controller
        self.disks_raw_info = {}  # type: Dict[str, str]

        # name => (speed, is_full_diplex, ip_addresses)
        self.net_info = {}  # type: Dict[str, Tuple[Optional[int], Optional[bool], List[str]]]

        self.ram_size = 0  # type: int
        self.sys_name = None  # type: str
        self.mb = None  # type: str
        self.raw_xml = None  # type: Optional[str]

        self.storage_controllers = []  # type: List[str]

    def get_summary(self) -> Dict[str, int]:
        cores = sum(count for _, count in self.cores)
        disks = sum(size for _, size in self.disks_info.values())

        return {'cores': cores,
                'ram': self.ram_size,
                'storage': disks,
                'disk_count': len(self.disks_info)}

    def __str__(self):
        res = []

        summ = self.get_summary()
        summary = "Simmary: {cores} cores, {ram}B RAM, {disk}B storage"
        res.append(summary.format(cores=summ['cores'],
                                  ram=b2ssize(summ['ram']),
                                  disk=b2ssize(summ['storage'])))
        res.append(str(self.sys_name))
        if self.mb:
            res.append("Motherboard: " + self.mb)

        if not self.ram_size:
            res.append("RAM: Failed to get RAM size")
        else:
            res.append("RAM " + b2ssize(self.ram_size) + "B")

        if not self.cores:
            res.append("CPU cores: Failed to get CPU info")
        else:
            res.append("CPU cores:")
            for name, count in self.cores:
                if count > 1:
                    res.append("    {0} * {1}".format(count, name))
                else:
                    res.append("    " + name)

        if self.storage_controllers:
            res.append("Disk controllers:")
            for descr in self.storage_controllers:
                res.append("    " + descr)

        if self.disks_info:
            res.append("Storage devices:")
            for dev, (model, size) in sorted(self.disks_info.items()):
                ssize = b2ssize(size) + "B"
                res.append("    {0} {1} {2}".format(dev, ssize, model))
        else:
            res.append("Storage devices's: Failed to get info")

        if self.disks_raw_info:
            res.append("Disks devices:")
            for dev, descr in sorted(self.disks_raw_info.items()):
                res.append("    {0} {1}".format(dev, descr))
        else:
            res.append("Disks devices's: Failed to get info")

        if self.net_info:
            res.append("Net adapters:")
            for name, (speed, dtype, _) in self.net_info.items():
                res.append("    {0} {2} duplex={1}".format(name, dtype, speed))
        else:
            res.append("Net adapters: Failed to get net info")

        return str(self.hostname) + ":\n" + "\n".join("    " + i for i in res)


class SWInfo(Storable):
    def __init__(self) -> None:
        self.mtab = None  # type: str
        self.kernel_version = None  # type: str
        self.libvirt_version = None  # type: Optional[str]
        self.qemu_version = None  # type: Optional[str]
        self.os_version = None  # type: Tuple[str, ...]


def get_sw_info(node: IRPCNode) -> SWInfo:
    res = SWInfo()

    res.os_version = tuple(get_os(node))
    res.kernel_version = node.get_file_content('/proc/version').decode('utf8').strip()
    res.mtab = node.get_file_content('/etc/mtab').decode('utf8').strip()

    try:
        res.libvirt_version = node.run("virsh -v", nolog=True).strip()
    except OSError:
        res.libvirt_version = None

    # dpkg -l ??

    try:
        res.qemu_version = node.run("qemu-system-x86_64 --version", nolog=True).strip()
    except OSError:
        res.qemu_version = None

    return res


def get_hw_info(node: IRPCNode) -> Optional[HWInfo]:
    try:
        lshw_out = node.run('sudo lshw -xml 2>/dev/null')
    except Exception as exc:
        logger.warning("lshw failed on node %s: %s", node.node_id, exc)
        return None

    res = HWInfo()
    res.raw_xml = lshw_out
    lshw_et = ET.fromstring(lshw_out)

    try:
        res.hostname = cast(str, lshw_et.find("node").attrib['id'])
    except Exception:
        pass

    try:

        res.sys_name = cast(str, lshw_et.find("node/vendor").text) + " " + \
            cast(str, lshw_et.find("node/product").text)
        res.sys_name = res.sys_name.replace("(To be filled by O.E.M.)", "")
        res.sys_name = res.sys_name.replace("(To be Filled by O.E.M.)", "")
    except Exception:
        pass

    core = lshw_et.find("node/node[@id='core']")
    if core is None:
        return res

    try:
        res.mb = " ".join(cast(str, core.find(node).text)
                          for node in ['vendor', 'product', 'version'])
    except Exception:
        pass

    for cpu in core.findall("node[@class='processor']"):
        try:
            model = cast(str, cpu.find('product').text)
            threads_node = cpu.find("configuration/setting[@id='threads']")
            if threads_node is None:
                threads = 1
            else:
                threads = int(threads_node.attrib['value'])
            res.cores.append((model, threads))
        except Exception:
            pass

    res.ram_size = 0
    for mem_node in core.findall(".//node[@class='memory']"):
        descr = mem_node.find('description')
        try:
            if descr is not None and descr.text == 'System Memory':
                mem_sz = mem_node.find('size')
                if mem_sz is None:
                    for slot_node in mem_node.find("node[@class='memory']"):
                        slot_sz = slot_node.find('size')
                        if slot_sz is not None:
                            assert slot_sz.attrib['units'] == 'bytes'
                            res.ram_size += int(slot_sz.text)
                else:
                    assert mem_sz.attrib['units'] == 'bytes'
                    res.ram_size += int(mem_sz.text)
        except Exception:
            pass

    for net in core.findall(".//node[@class='network']"):
        try:
            link = net.find("configuration/setting[@id='link']")
            if link.attrib['value'] == 'yes':
                name = cast(str, net.find("logicalname").text)
                speed_node = net.find("configuration/setting[@id='speed']")

                if speed_node is None:
                    speed = None
                else:
                    speed = int(speed_node.attrib['value'])

                dup_node = net.find("configuration/setting[@id='duplex']")
                if dup_node is None:
                    dup = None
                else:
                    dup = cast(str, dup_node.attrib['value']).lower() == 'yes'

                ips = []  # type: List[str]
                res.net_info[name] = (speed, dup, ips)
        except Exception:
            pass

    for controller in core.findall(".//node[@class='storage']"):
        try:
            description = getattr(controller.find("description"), 'text', "")
            product = getattr(controller.find("product"), 'text', "")
            vendor = getattr(controller.find("vendor"), 'text', "")
            dev = getattr(controller.find("logicalname"), 'text', "")
            if dev != "":
                res.storage_controllers.append(
                    "{0}: {1} {2} {3}".format(dev, description,
                                              vendor, product))
            else:
                res.storage_controllers.append(
                    "{0} {1} {2}".format(description,
                                         vendor, product))
        except Exception:
            pass

    for disk in core.findall(".//node[@class='disk']"):
        try:
            lname_node = disk.find('logicalname')
            if lname_node is not None:
                dev = cast(str, lname_node.text).split('/')[-1]

                if dev == "" or dev[-1].isdigit():
                    continue

                sz_node = disk.find('size')
                assert sz_node.attrib['units'] == 'bytes'
                sz = int(sz_node.text)
                res.disks_info[dev] = ('', sz)
            else:
                description = disk.find('description').text
                product = disk.find('product').text
                vendor = disk.find('vendor').text
                version = disk.find('version').text
                serial = disk.find('serial').text

                full_descr = "{0} {1} {2} {3} {4}".format(
                    description, product, vendor, version, serial)

                businfo = cast(str, disk.find('businfo').text)
                res.disks_raw_info[businfo] = full_descr
        except Exception:
            pass

    return res
