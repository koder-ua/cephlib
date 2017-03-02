import json
import time
import socket
import logging


from .common import check_output_ssh, CmdResult, run_ssh
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


def init_node(node_name, ssh_opts):
    """init RPC connection to node
    Upload rpc code, start daemon, open RPC connection, upload plugins
    """
    try:
        res = run_ssh(node_name, ssh_opts, 'python2.7 --version')
        python_cmd = None
        if res.code != 0:
            res = run_ssh(node_name, ssh_opts, 'python --version')
            if res.code == 0 and '2.7' in res.out:
                python_cmd = 'python'
        else:
            python_cmd = 'python2.7'

        assert python_cmd, "Failed to run python2.7 on node {0}".format(node_name)

        path = '/tmp/ceph_agent.py'
        check_output_ssh(node_name, ssh_opts, "'cat > {0}'".format(path), input=agent_code)

        log_file = '/tmp/ceph_agent.log'
        ip = socket.gethostbyname(node_name)
        cmd = '{} {} server --listen-addr={}:0 --daemon --show-settings --stdout-file={}'
        out = check_output_ssh(node_name, ssh_opts, cmd.format(python_cmd, path, ip, log_file))
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


def rpc_run_ch(node, cmd, timeout=60, input_data=None):
    res = rpc_run(node, cmd, timeout, input_data)
    assert res.code == 0, "{!r} is failed with code {}. output is {!r}".format(cmd, res.code, res.out)
    return res.out


def rpc_run(node, cmd, timeout=60, input_data=None):
    logger.debug("%r: %s", node.name, cmd)
    pid = node.rpc.cli.spawn(cmd, timeout=timeout, input_data=input_data)
    out = ""
    err = ""

    time.sleep(0.1)

    while True:
        ecode, dout, derr = node.rpc.cli.get_updates(pid)
        out += dout
        err += err
        if ecode == 0:
            return CmdResult(0, out)
        elif ecode is not None:
            return CmdResult(ecode, err)
        time.sleep(0.5)
