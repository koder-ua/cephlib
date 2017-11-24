import json
import time
import socket
import logging
import subprocess

from .common import run_ssh, run_ssh
from . import sensors_rpc_plugin

from agent.agent import connect

from agent import agent, cli_plugin, fs_plugin

logger = logging.getLogger("rpc")
get_code = lambda x: open(x.__file__.rsplit(".", 1)[0] + ".py").read()

# load plugins via relative path
agent_code = get_code(agent)
cli_plugin_code = get_code(cli_plugin)
fs_plugin_code = get_code(fs_plugin)
load_plugin_code = get_code(sensors_rpc_plugin)


def init_node(node_name, ssh_opts, with_sudo=False):
    """init RPC connection to node
    Upload rpc code, start daemon, open RPC connection, upload plugins
    """
    try:
        for python_cmd in ('python2.7', 'python'):
            try:
                out = run_ssh(node_name, ssh_opts, python_cmd + ' --version', merge_err=True).decode('utf8')
            except subprocess.SubprocessError:
                continue

            if '2.7' in out:
                break
        else:
            raise AssertionError("Failed to run python2.7 on node {0}".format(node_name))

        path = '/tmp/ceph_agent.py'
        run_ssh(node_name, ssh_opts, "'cat > {0}'".format(path), input_data=agent_code.encode('utf8'))

        log_file = '/tmp/ceph_agent.log'
        ip = socket.gethostbyname(node_name)
        cmd_templ = '{0} {1} server --listen-addr={2}:0 --daemon --show-settings --stdout-file={3}'
        cmd = ("sudo " if with_sudo else "") + cmd_templ.format(python_cmd, path, ip, log_file)
        out = run_ssh(node_name, ssh_opts, cmd).decode('utf8')
        data_j = json.loads(out)
        daemon_pid = data_j["daemon_pid"]

        rpc = connect(data_j["addr"].split(":"))
        rpc.server.load_module("fs", None, fs_plugin_code)
        rpc.server.load_module("cli", None, cli_plugin_code)
        rpc.server.load_module("sensors", None, load_plugin_code)
    except Exception:
        logger.exception("During init")
        raise

    return rpc, daemon_pid


def rpc_run(rpc, cmd, timeout=60, input_data=None, node_name=None, start_timeout=0.01, check_timeout=0.1, log=True):
    if log:
        logger.debug("%s: %s", node_name, cmd)

    pid = rpc.cli.spawn(cmd, timeout=timeout, input_data=input_data)
    out = b""
    err = b""

    time.sleep(start_timeout)

    while True:
        ecode, dout, derr = rpc.cli.get_updates(pid)
        out += dout
        err += err
        if ecode is not None:
            if ecode == 0:
                return out
            else:
                raise subprocess.CalledProcessError(ecode, cmd, output=out, stderr=err)
        time.sleep(check_timeout)
