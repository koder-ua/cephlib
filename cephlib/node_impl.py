import os
import zlib
import time
import json
import socket
import logging
import tempfile
import subprocess
from typing import Union, cast, Optional, Tuple, Dict

import paramiko

from agent import agent

from .node import NodeInfo
from .ssh import connect as ssh_connect
from .node import IRPCNode, ISSHHost


logger = logging.getLogger("wally")


class SSHHost(ISSHHost):
    def __init__(self, conn: paramiko.SSHClient, info: NodeInfo) -> None:
        self.conn = conn
        self.info = info

    def __str__(self) -> str:
        return self.node_id

    @property
    def node_id(self) -> str:
        return self.info.node_id

    def put_to_file(self, path: Optional[str], content: bytes) -> str:
        if path is None:
            path = self.run("mktemp", nolog=True).strip()

        logger.debug("PUT %s bytes to %s", len(content), path)

        with self.conn.open_sftp() as sftp:
            with sftp.open(path, "wb") as fd:
                fd.write(content)

        return path

    def disconnect(self):
        self.conn.close()

    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        if not nolog:
            logger.debug("SSH:{0} Exec {1!r}".format(self, cmd))

        transport = self.conn.get_transport()
        session = transport.open_session()

        try:
            session.set_combine_stderr(True)
            stime = time.time()

            session.exec_command(cmd)
            session.settimeout(1)
            session.shutdown_write()
            output = ""

            while True:
                try:
                    ndata = session.recv(1024).decode("utf-8")
                    if not ndata:
                        break
                    output += ndata
                except socket.timeout:
                    pass

                if time.time() - stime > timeout:
                    raise OSError(output + "\nExecution timeout")

            code = session.recv_exit_status()
        finally:
            found = False

            if found:
                session.close()

        if code != 0:
            templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
            raise OSError(templ.format(self, cmd, code, output))

        return output


class LocalHost(ISSHHost):
    def __str__(self):
        return "<Local>"

    def get_ip(self) -> str:
        return 'localhost'

    def put_to_file(self, path: Optional[str], content: bytes) -> str:
        if path is None:
            fd, path = tempfile.mkstemp(text=False)
            os.close(fd)
        else:
            dir_name = os.path.dirname(path)
            os.makedirs(dir_name, exist_ok=True)

        with open(path, "wb") as fd2:
            fd2.write(content)

        return path

    def run(self, cmd: str, timeout: int = 60, nolog: bool = False) -> str:
        proc = subprocess.Popen(cmd, shell=True,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        stdout_data_b, _ = proc.communicate()
        stdout_data = stdout_data_b.decode("utf8")

        if proc.returncode != 0:
            templ = "SSH:{0} Cmd {1!r} failed with code {2}. Output: {3}"
            raise OSError(templ.format(self, cmd, proc.returncode, stdout_data))

        return stdout_data

    def disconnect(self):
        pass


def get_rpc_server_code() -> Tuple[bytes, Dict[str, bytes]]:
    # setup rpc data
    if agent.__file__.endswith(".pyc"):
        path = agent.__file__[:-1]
    else:
        path = agent.__file__

    master_code = open(path, "rb").read()

    plugins = {}  # type: Dict[str, bytes]
    cli_path = os.path.join(os.path.dirname(path), "cli_plugin.py")
    plugins["cli"] = open(cli_path, "rb").read()

    fs_path = os.path.join(os.path.dirname(path), "fs_plugin.py")
    plugins["fs"] = open(fs_path, "rb").read()

    return master_code, plugins


def connect(info: Union[str, NodeInfo], conn_timeout: int = 60) -> ISSHHost:
    if info == 'local':
        return LocalHost()
    else:
        info_c = cast(NodeInfo, info)
        return SSHHost(ssh_connect(info_c.ssh_creds, conn_timeout), info_c)


class RPCNode(IRPCNode):
    """Node object"""

    def __init__(self, conn: agent.SimpleRPCClient, info: NodeInfo) -> None:
        self.info = info
        self.conn = conn

    def __str__(self) -> str:
        return "Node({!r})".format(self.info)

    def __repr__(self) -> str:
        return str(self)

    @property
    def node_id(self) -> str:
        return self.info.node_id

    def get_file_content(self, path: str, expanduser: bool = False, compress: bool = True) -> bytes:
        logger.debug("GET %s from %s", path, self.info)
        if expanduser:
            path = self.conn.fs.expanduser(path)
        res = self.conn.fs.get_file(path, compress)
        logger.debug("Download %s bytes from remote file %s from %s", len(res), path, self.info)
        if compress:
            res = zlib.decompress(res)
        return res

    def run(self, cmd: str, timeout: int = 60, nolog: bool = False, check_timeout: float = 0.01) -> str:
        if not nolog:
            logger.debug("Node %s - run %s", self.node_id, cmd)

        cmd_b = cmd.encode("utf8")
        proc_id = self.conn.cli.spawn(cmd_b, timeout=timeout, merge_out=True)
        out = ""

        while True:
            code, outb, _ = self.conn.cli.get_updates(proc_id)
            out += outb.decode("utf8")
            if code is not None:
                break
            time.sleep(check_timeout)

        if code != 0:
            templ = "Node {} - cmd {!r} failed with code {}. Output: {!r}."
            raise OSError(templ.format(self.node_id, cmd, code, out))

        return out

    def copy_file(self, local_path: str, remote_path: str = None,
                  expanduser: bool = False,
                  compress: bool = False) -> str:

        if expanduser:
            remote_path = self.conn.fs.expanduser(remote_path)

        data = open(local_path, 'rb').read()  # type: bytes
        return self.put_to_file(remote_path, data, compress=compress)

    def put_to_file(self, path: Optional[str], content: bytes, expanduser: bool = False, compress: bool = False) -> str:
        if expanduser:
            path = self.conn.fs.expanduser(path)
        if compress:
            content = zlib.compress(content)
        return self.conn.fs.store_file(path, content, compress)

    def stat_file(self, path: str, expanduser: bool = False) -> Dict[str, int]:
        if expanduser:
            path = self.conn.fs.expanduser(path)
        return self.conn.fs.file_stat(path)

    def __exit__(self, x, y, z) -> bool:
        self.disconnect(stop=True)
        return False

    def upload_plugin(self, name: str, code: bytes, version: str = None) -> None:
        self.conn.server.load_module(name, version, code)

    def disconnect(self, stop: bool = False) -> None:
        if stop:
            logger.debug("Stopping RPC server on %s", self.info)
            self.conn.server.stop()

        logger.debug("Disconnecting from %s", self.info)
        self.conn.disconnect()
        self.conn = None


def get_node_python_27(node: ISSHHost) -> Optional[str]:
    python_cmd = None  # type: Optional[str]
    try:
        python_cmd = node.run('which python2.7').strip()
    except Exception:
        pass

    if python_cmd is None:
        try:
            if '2.7' in node.run('python --version'):
                python_cmd = node.run('which python').strip()
        except Exception:
            pass

    return python_cmd


def setup_rpc(node: ISSHHost,
              rpc_server_code: bytes,
              plugins: Dict[str, bytes] = None,
              port: int = 0,
              log_level: str = None,
              sudo: bool = False) -> IRPCNode:

    logger.debug("Setting up RPC connection to {}".format(node.info))
    python_cmd = get_node_python_27(node)
    if python_cmd:
        logger.debug("python2.7 on node {} path is {}".format(node.info, python_cmd))
    else:
        logger.error(("Can't find python2.7 on node {}. " +
                      "Install python2.7 and rerun test").format(node.info))
        raise ValueError("Python not found")

    code_file = node.put_to_file(None, rpc_server_code)
    ip = node.info.ssh_creds.addr.host

    log_file = None  # type: Optional[str]
    if log_level:
        log_file = node.run("mktemp", nolog=True).strip()
        cmd = "{} {} --log-level={} server --listen-addr={}:{} --daemon --show-settings"
        cmd = cmd.format(python_cmd, code_file, log_level, ip, port) + " --stdout-file={}".format(log_file)
        logger.info("Agent logs for node {} stored remotely in file {}, log level is {}".format(
            node.node_id, log_file, log_level))
    else:
        cmd = "{} {} --log-level=CRITICAL server --listen-addr={}:{} --daemon --show-settings"
        cmd = cmd.format(python_cmd, code_file, ip, port)

    if sudo:
        cmd = "sudo {}".format(cmd)

    params_js = node.run(cmd).strip()
    params = json.loads(params_js)

    node.info.params.update(params)

    port = int(params['addr'].split(":")[1])
    rpc_conn = agent.connect((ip, port))

    rpc_node = RPCNode(rpc_conn, node.info)
    rpc_node.rpc_log_file = log_file

    if plugins is not None:
        try:
            for name, code in plugins.items():
                rpc_node.upload_plugin(name, code)
        except Exception:
            rpc_node.disconnect(True)
            raise

    return rpc_node
