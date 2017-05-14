import re
import time
import errno
import socket
import getpass
import logging
import os.path
import selectors
from io import StringIO
from typing import cast, Set, Optional, List, Any, Dict, NamedTuple

try:
    import paramiko
except ImportError:
    paramiko = None

from .common import to_ip, Timeout
from .storage import IStorable


logger = logging.getLogger("wally")


IP = str
IPAddr = NamedTuple("IPAddr", [("host", IP), ("port", int)])


class ConnCreds(IStorable):
    def __init__(self, host: str, user: str, passwd: str = None, port: str = '22',
                 key_file: str = None, key: bytes = None) -> None:
        self.user = user
        self.passwd = passwd
        self.addr = IPAddr(host, int(port))
        self.key_file = key_file
        self.key = key

    def __str__(self) -> str:
        return "{}@{}:{}".format(self.user, self.addr.host, self.addr.port)

    def __repr__(self) -> str:
        return str(self)

    def raw(self) -> Dict[str, Any]:
        return {
            'user': self.user,
            'host': self.addr.host,
            'port': self.addr.port,
            'passwd': self.passwd,
            'key_file': self.key_file
        }

    @classmethod
    def fromraw(cls, data: Dict[str, Any]) -> 'ConnCreds':
        return cls(**data)


class URIsNamespace:
    class ReParts:
        user_rr = "[^:]*?"
        host_rr = "[^:@]*?"
        port_rr = "\\d+"
        key_file_rr = "[^:@]*"
        passwd_rr = ".*?"

    re_dct = ReParts.__dict__

    for attr_name, val in re_dct.items():
        if attr_name.endswith('_rr'):
            new_rr = "(?P<{0}>{1})".format(attr_name[:-3], val)
            setattr(ReParts, attr_name, new_rr)

    re_dct = ReParts.__dict__

    templs = [
        "^{host_rr}$",
        "^{host_rr}:{port_rr}$",
        "^{host_rr}::{key_file_rr}$",
        "^{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}@{host_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}$",
        "^{user_rr}@{host_rr}::{key_file_rr}$",
        "^{user_rr}@{host_rr}:{port_rr}:{key_file_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}$",
        "^{user_rr}:{passwd_rr}@{host_rr}:{port_rr}$",
    ]

    uri_reg_exprs = []  # type: List[str]
    for templ in templs:
        uri_reg_exprs.append(templ.format(**re_dct))


def parse_ssh_uri(uri: str) -> ConnCreds:
    """Parse ssh connection URL from one of following form
        [ssh://]user:passwd@host[:port]
        [ssh://][user@]host[:port][:key_file]
    """

    if uri.startswith("ssh://"):
        uri = uri[len("ssh://"):]

    for rr in URIsNamespace.uri_reg_exprs:
        rrm = re.match(rr, uri)
        if rrm is not None:
            params = {"user": getpass.getuser()}  # type: Dict[str, str]
            params.update(rrm.groupdict())
            params['host'] = to_ip(params['host'])
            return ConnCreds(**params)  # type: ignore

    raise ValueError("Can't parse {0!r} as ssh uri value".format(uri))


NODE_KEYS = {}  # type: Dict[IPAddr, Any]
SSH_KEY_PASSWD = None  # type: Optional[str]


def set_ssh_key_passwd(passwd: str) -> None:
    global SSH_KEY_PASSWD
    SSH_KEY_PASSWD = passwd


def set_key_for_node(host_port: IPAddr, key: bytes) -> None:
    if paramiko is None:
        raise RuntimeError("paramiko module is not available")

    with StringIO(key.decode("utf8")) as sio:
        NODE_KEYS[host_port] = paramiko.RSAKey.from_private_key(sio)  # type: ignore


def connect(creds: ConnCreds,
            conn_timeout: int = 60,
            tcp_timeout: int = 15,
            default_banner_timeout: int = 30) -> Any:

    if paramiko is None:
        raise RuntimeError("paramiko module is not available")

    ssh = paramiko.SSHClient()
    ssh.load_host_keys('/dev/null')
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.known_hosts = None

    end_time = time.time() + conn_timeout  # type: float

    logger.debug("SSH connecting to %s", creds)

    while True:
        try:
            time_left = end_time - time.time()
            c_tcp_timeout = min(tcp_timeout, time_left)

            banner_timeout_arg = {}  # type: Dict[str, int]
            if paramiko.__version_info__ >= (1, 15, 2):
                banner_timeout_arg['banner_timeout'] = int(min(default_banner_timeout, time_left))

            if creds.passwd is not None:
                ssh.connect(creds.addr.host,
                            timeout=c_tcp_timeout,
                            username=creds.user,
                            password=cast(str, creds.passwd),
                            port=creds.addr.port,
                            allow_agent=False,
                            look_for_keys=False,
                            **banner_timeout_arg)
            elif creds.key_file is not None:
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            pkey=paramiko.RSAKey.from_private_key_file(creds.key_file, password=SSH_KEY_PASSWD),
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            elif creds.key is not None:
                with StringIO(creds.key.decode("utf8")) as sio:
                    ssh.connect(creds.addr.host,
                                username=creds.user,
                                timeout=c_tcp_timeout,
                                pkey=paramiko.RSAKey.from_private_key(sio, password=SSH_KEY_PASSWD),  # type: ignore
                                look_for_keys=False,
                                port=creds.addr.port,
                                **banner_timeout_arg)
            elif (creds.addr.host, creds.addr.port) in NODE_KEYS:
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            pkey=NODE_KEYS[creds.addr],
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            else:
                key_file = os.path.expanduser('~/.ssh/id_rsa')
                ssh.connect(creds.addr.host,
                            username=creds.user,
                            timeout=c_tcp_timeout,
                            key_filename=key_file,
                            look_for_keys=False,
                            port=creds.addr.port,
                            **banner_timeout_arg)
            return ssh
        except (socket.gaierror, paramiko.PasswordRequiredException):
            raise
        except socket.error:
            if time.time() > end_time:
                raise
            time.sleep(1)


def wait_ssh_available(addrs: List[IPAddr],
                       timeout: int = 300,
                       tcp_timeout: float = 1.0) -> None:

    addrs_set = set(addrs)  # type: Set[IPAddr]

    for _ in Timeout(timeout):
        selector = selectors.DefaultSelector()  # type: selectors.BaseSelector
        with selector:
            for addr in addrs_set:
                sock = socket.socket()
                sock.setblocking(False)
                try:
                    sock.connect(addr)
                except BlockingIOError:
                    pass
                selector.register(sock, selectors.EVENT_READ, data=addr)

            etime = time.time() + tcp_timeout
            ltime = etime - time.time()
            while ltime > 0:
                # convert to greater or equal integer
                for key, _ in selector.select(timeout=int(ltime + 0.99999)):
                    selector.unregister(key.fileobj)
                    try:
                        key.fileobj.getpeername()  # type: ignore
                        addrs_set.remove(key.data)
                    except OSError as exc:
                        if exc.errno == errno.ENOTCONN:
                            pass
                ltime = etime - time.time()

        if not addrs_set:
            break


