import os
import re
import sys
import time
import json
import socket
import atexit
import logging
import tempfile
import threading
import subprocess
import collections

from .pyver import queue, raise_me, logging_config


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
        logger.debug("CMD %r", cmd)

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

    logger.debug("SSH %s %r", host, cmd)
    while True:
        cmd = "ssh {0} {1} {2}".format(ssh_opts, host, cmd)
        res = run(cmd, False, input=input)
        if res.code == 0 or max_retry <= 0:
            return res

        max_retry -= 1
        time.sleep(1)
        logger.warning("Retry SSH:%s: %r. Err is %r", host, cmd, res.out)


def check_output(cmd, log=True, input=None, out_limit=200):
    assert isinstance(log, bool)
    code, out = run(cmd, log=log, input=input)
    assert code == 0, "{0!r} failed with code {1}. Out\n{2}".format(cmd, code, out[-out_limit:])
    return out


def check_output_ssh(host, ssh_opts, cmd, no_retry=False, max_retry=3, input=None, out_limit=200):
    code, out = run_ssh(host, ssh_opts, cmd, no_retry=no_retry, max_retry=max_retry, input=input)
    assert code == 0, "{0!r} failed with code {1}. Out\n{2}".format(cmd, code, out[-out_limit:])
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


def setup_logging(log_config_fname=None, log_file=None, log_level=None, log_config_obj=None):
    if log_config_obj:
        assert not log_config_fname
        log_config = log_config_obj
    else:
        log_config = json.load(open(log_config_fname))

    if log_file is not None:
        log_config["handlers"]["log_file"]["filename"] = log_file

    if log_level is not None:
        log_config["handlers"]["console"]["level"] = log_level

    logging_config.dictConfig(log_config)


def prun(runs, thcount):
    res_q = queue.Queue()
    input_q = queue.Queue()

    for num, run_info in enumerate(runs):
        input_q.put((num, run_info))

    def worker():
        while True:
            try:
                pos, (func, args, kwargs) = input_q.get(False)
            except queue.Empty:
                return
            except Exception:
                logger.exception("In worker thread")

            try:
                res_q.put((pos, True, func(*args, **kwargs)))
            except Exception as exc:
                res_q.put((pos, False, (exc, sys.exc_info()[2])))

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


def pmap(func, data, thcount=32):
    return prun([(func, [val], {}) for val in data], thcount)


def prun_check(runs, thcount):
    res = []
    for ok, val in prun(runs, thcount):
        if not ok:
            exc, tb = val
            raise_me(exc, tb)
        res.append(val)
    return res


def pmap_check(func, data, thcount=32):
    res = []
    for ok, val in pmap(func, data, thcount=thcount):
        if not ok:
            exc, tb = val
            raise_me(exc, tb)
        res.append(val)
    return res


def get_sshable_hosts(addrs, ssh_opts, thcount=32):
    def check_host(addr):
        try:
            if not re.match(r"\d+\.\d+\.\d+\.\d+$", addr):
                socket.gethostbyname(addr)
            if run_ssh(addr, ssh_opts, 'pwd').code == 0:
                return addr
        except socket.gaierror:
            pass

    results = pmap(check_host, addrs, thcount=thcount)
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


class AttredDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def float2str(val, digits=3):
    if digits < 1:
        raise ValueError("digits must be >= 1")

    if val < 0:
        return '-' + float2str(-val, digits=digits)

    if val < 1E-10:
        return '0'

    if val < 0.1:
        return ("{0:.%se}" % (digits - 1,)).format(val)

    if val < 1:
        return ("{0:.%sf}" % (digits,)).format(val)

    if val < 10 ** digits and (isinstance(val, int) or val >= 10 ** (digits - 1)):
        return str(int(val))

    for idx in range(1, digits):
        if val < 10 ** idx:
            return ("{0:%s.%sf}" % (idx, digits - idx)).format(val)

    for idx in range(1, 4):
        if val < 10 ** (idx + digits):
            return str(int(val) // (10 ** idx) * (10 ** idx))

    return "{0:.2e}".format(val)


def floats2str(vals, digits=3, width=8):
    if digits < 1:
        raise ValueError("digits must be >= 1")

    svals = [float2str(val) for val in vals]
    max_after_dot = 0
    max_before_dot = 0

    for sval in svals:
        if 'e' not in sval and 'E' not in sval:
            if '.' in sval:
                bf, af = sval.split('.')
                max_after_dot = max(max_after_dot, len(af))
                max_before_dot = max(max_before_dot, len(bf))
            else:
                max_before_dot = max(max_before_dot, len(sval))

    if max_after_dot > 0:
        format_dt = "{:>%ss}.{:<%ss}" % (width - 1 - max_after_dot, max_after_dot)
        format = "{:>%ss}%s" % (width - 1 - max_after_dot, " " * (1 + max_after_dot))
    else:
        format_dt = None
        format = "{:>%ss}" % (width,)

    result = []
    for sval in svals:
        if 'e' in sval or 'E' in sval:
            result.append(sval)
        else:
            if '.' in sval:
                result.append(format_dt.format(*sval.split('.')))
            else:
                result.append(format.format(sval))
    return result

