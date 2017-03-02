import os
import time
import Queue
import socket
import atexit
import logging
import tempfile
import threading
import subprocess
import collections


logger = logging.getLogger("cmd")


RSMAP = [('K', 1024),
         ('M', 1024 ** 2),
         ('G', 1024 ** 3),
         ('T', 1024 ** 4)]


def b2ssize(value):
    value = int(value)
    if value < 1024:
        return str(value) + " "

    # make mypy happy
    scale = 1
    name = ""

    for name, scale in RSMAP:
        if value < 1024 * scale:
            if value % scale == 0:
                return "{0} {1}i".format(value // scale, name)
            else:
                return "{0:.1f} {1}i".format(float(value) / scale, name)

    return "{0}{1}i".format(value // scale, name)


CmdResult = collections.namedtuple("CmdResult", ["code", "out"])


def run(cmd, log=True, input=None):
    if log:
        logger.debug("CMD: %r", cmd)

    if input is None:
        stdin = None
    else:
        stdin = subprocess.PIPE

    p = subprocess.Popen(cmd, shell=True,
                         stdin=stdin,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out = p.communicate(input)
    code = p.wait()

    if 0 == code:
        return CmdResult(0, out[0])
    else:
        return CmdResult(code, out[0] + out[1])


def run_ssh(host, ssh_opts, cmd, no_retry=False, max_retry=3, input=None):
    if no_retry:
        max_retry = 0

    logger.debug("SSH:%s: %r", host, cmd)
    while True:
        cmd = "ssh {0} {1} {2}".format(ssh_opts, host, cmd)
        res = run(cmd, False, input=input)
        if res.code == 0 or max_retry <= 0:
            return res

        max_retry -= 1
        time.sleep(1)
        logger.warning("Retry SSH:%s: %r. Err is %r", host, cmd, res.out)


def check_output(cmd, log=True, input=None):
    assert isinstance(log, bool)
    code, out = run(cmd, log=log, input=input)
    assert code == 0, "{0!r} failed with code {1}. Out\n{2}".format(cmd, code, out)
    return out


def check_output_ssh(host, ssh_opts, cmd, no_retry=False, max_retry=3, input=None):
    code, out = run_ssh(host, ssh_opts, cmd, no_retry=no_retry, max_retry=max_retry, input=input)
    assert code == 0, "{0!r} failed with code {1}. Out\n{2}".format(cmd, code, out)
    return out


def setup_loggers(loggers, default_level=logging.INFO, log_fname=None):
    sh = logging.StreamHandler()
    sh.setLevel(default_level)
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    colored_formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
    sh.setFormatter(colored_formatter)
    handlers = [sh]

    if log_fname is not None:
        fh = logging.FileHandler(log_fname)
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)

    for logger in loggers:
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        logger.addHandler(sh)

        for handler in handlers:
            logger.addHandler(handler)

    root_logger = logging.getLogger()
    root_logger.handlers = []



def prun(runs, thcount):
    res_q = Queue.Queue()
    input_q = Queue.Queue()

    for num, run_info in enumerate(runs):
        input_q.put((num, run_info))

    def worker():
        while True:
            try:
                pos, (func, args, kwargs) = input_q.get(False)
            except Queue.Empty:
                return

            try:
                res_q.put((pos, True, func(*args, **kwargs)))
            except Exception as exc:
                res_q.put((pos, False, exc))

    ths = [threading.Thread(target=worker) for i in range(thcount)]

    for th in ths:
        th.daemon = True
        th.start()

    for th in ths:
        th.join()

    results = []
    while not res_q.empty():
        results.append(res_q.get())

    return [(ok, val) for _, ok, val in sorted(results)]


def pmap(func, data, thcount=32 ):
    return prun([(func, [val], {}) for val in data], thcount)


def get_sshable_hosts(nodes, ssh_opts, thcount=32):
    def check_host(node):
        try:
            socket.gethostbyname(node.name)
            if check_output_ssh(node.name, ssh_opts, 'pwd').code == 0:
                return node.name
        except socket.gaierror:
            pass

    results = pmap(check_host, nodes, thcount=thcount)
    return [res for ok, res in results if ok and res is not None]


FILES_TO_REMOVE = []


def tmpnam(remove_after=True):
    fd, name = tempfile.mkstemp()
    os.close(fd)
    if remove_after:
        FILES_TO_REMOVE.append(name)
    return name


def clean_tmp_files():
    for fname in FILES_TO_REMOVE:
        try:
            os.unlink(fname)
        except IOError:
            pass
    FILES_TO_REMOVE[:] = []


atexit.register(clean_tmp_files)
