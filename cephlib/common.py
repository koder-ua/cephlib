import os
import re
import time
import json
import math
import socket
import atexit
import logging
import tempfile
import ipaddress
import subprocess
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Iterator, cast, Union, Tuple, Callable, TypeVar, List, Any, Optional, IO, Dict

try:
    import psutil
except ImportError:
    psutil = None

import logging.config as logging_config
from .types import Number, TNumber

logger = logging.getLogger("cephlib")


# command execution ----------------------------------------------------------------------------------------------------

def run_locally(cmd: Union[str, List[str]], input_data: bytes = None, timeout: int = 20,
                log: bool = True, merge_err: bool = False) -> bytes:

    if log:
        logger.debug("CMD %r", cmd)

    if isinstance(cmd, str):
        shell = True
        cmd_str = cmd
    else:
        shell = False
        cmd_str = " ".join(cmd)

    proc = subprocess.Popen(cmd,
                            shell=shell,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        stdout_data, stderr_data = proc.communicate(input_data, timeout=timeout)
    except TimeoutError:
        if psutil is not None:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        else:
            proc.kill()
        proc.wait(0.1)
        raise RuntimeError("Local process timeout: " + cmd_str)

    if 0 != proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd_str, stdout_data + stderr_data)
    return stdout_data + (stderr_data if merge_err else b'')


def run_ssh(host: str, ssh_opts: str, cmd: str, no_retry: bool = False, max_retry: int = 3, timeout: int = 20,
            input_data: bytes = None, merge_err: bool = False) -> bytes:
    if no_retry:
        max_retry = 0

    ssh_cmd = "ssh {0} {1} {2}".format(ssh_opts, host, cmd)
    logger.debug("SSH %s %r", host, cmd)
    while True:
        try:
            return run_locally(ssh_cmd, input_data=input_data, timeout=timeout, log=False, merge_err=merge_err)
        except (subprocess.CalledProcessError, TimeoutError) as lexc:
            if max_retry == 0:
                raise
            exc = lexc

        err = exc.output
        if isinstance(err, bytes):
            err = err.decode('utf8')

        if err:
            logger.warning("SSH error for host %s. Cmd: %r. Err is %r. Will retry", host, cmd, err)
        else:
            logger.warning("SSH error for host %s. Cmd: %r. Most probably host " +
                           "is unreachable via ssh. Will retry", host, cmd)

        max_retry -= 1
        time.sleep(1)


def get_sshable_hosts(addrs: Iterable[str], ssh_opts: str, thcount: int = 32) -> List[str]:
    def check_host(addr):
        try:
            if not re.match(r"\d+\.\d+\.\d+\.\d+$", addr):
                socket.gethostbyname(addr)
            if run_ssh(addr, ssh_opts, 'pwd'):
                return addr
        except (subprocess.CalledProcessError, socket.gaierror):
            return None

    with ThreadPoolExecutor(thcount) as executor:
        return [addr for addr in executor.map(check_host, addrs) if addr is not None]


def which(program: str) -> Optional[str]:
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    for path in os.environ["PATH"].split(os.pathsep):
        path = path.strip('"')
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return exe_file

    return None

# ------- Loggers ------------------------------------------------------------------------------------------------------


def setup_loggers(loggers: List[logging.Logger], default_level: int = logging.INFO, log_fname: str = None) -> None:
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

    for clogger in loggers:
        clogger.setLevel(logging.DEBUG)
        clogger.handlers = []
        clogger.addHandler(sh)

        for handler in handlers:
            clogger.addHandler(handler)

    root_logger = logging.getLogger()
    root_logger.handlers = []


def setup_logging(log_config_fname: str = None, log_file: str = None, log_level: str = None,
                  log_config_obj: Dict[str, Any] = None) -> None:
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


FILES_TO_REMOVE = []  # type: List[str]


def tmpnam(remove_after: bool = True, **kwargs) -> str:
    fd, name = tempfile.mkstemp(**kwargs)
    os.close(fd)
    if remove_after:
        FILES_TO_REMOVE.append(name)
    return name


def clean_tmp_files() -> None:
    for fname in FILES_TO_REMOVE:
        try:
            os.unlink(fname)
        except IOError:
            pass
    FILES_TO_REMOVE[:] = []


atexit.register(clean_tmp_files)


class AttredDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


# --- Number formatting functions --------------------------------------------------------------------------------------

def greater_digit_pos(val: Number) -> int:
    return int(math.floor(math.log10(val))) + 1


def round_digits(val: TNumber, num_digits: int = 3) -> TNumber:
    npow = 10 ** (greater_digit_pos(val) - num_digits)
    return type(val)(int(val / npow) * npow)


def float2str(val: float, digits: int = 3) -> str:
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


def floats2str(vals: List[float], digits: int = 3, width: int = 8) -> List[str]:
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
        format_val = "{:>%ss}%s" % (width - 1 - max_after_dot, " " * (1 + max_after_dot))
    else:
        format_dt = None
        format_val = "{:>%ss}" % (width,)

    result = []
    for sval in svals:
        if 'e' in sval or 'E' in sval:
            result.append(sval)
        else:
            if '.' in sval:
                result.append(format_dt.format(*sval.split('.')))
            else:
                result.append(format_val.format(sval))
    return result

# ----------------------------------------------------------------------------------------------------------------------

class Timeout(Iterable[float]):
    def __init__(self, timeout: int, message: str = None, min_tick: int = 1, no_exc: bool = False) -> None:
        self.end_time = time.time() + timeout
        self.message = message
        self.min_tick = min_tick
        self.prev_tick_at = time.time()
        self.no_exc = no_exc

    def tick(self) -> bool:
        current_time = time.time()

        if current_time > self.end_time:
            if self.message:
                msg = "Timeout: {}".format(self.message)
            else:
                msg = "Timeout"

            if self.no_exc:
                return False

            raise TimeoutError(msg)

        sleep_time = self.min_tick - (current_time - self.prev_tick_at)
        if sleep_time > 0:
            time.sleep(sleep_time)
            self.prev_tick_at = time.time()
        else:
            self.prev_tick_at = current_time

        return True

    def __iter__(self) -> Iterator[float]:
        return cast(Iterator[float], self)

    def __next__(self) -> float:
        if not self.tick():
            raise StopIteration()
        return self.end_time - time.time()


def is_ip(data: str) -> bool:
    try:
        ipaddress.ip_address(data)
        return True
    except ValueError:
        return False


def parse_creds(creds: str) -> Tuple[str, str, str]:
    """Parse simple credentials format user[:passwd]@host"""
    user, passwd_host = creds.split(":", 1)

    if '@' not in passwd_host:
        passwd, host = passwd_host, None
    else:
        passwd, host = passwd_host.rsplit('@', 1)

    return user, passwd, host


def get_ip_for_target(target_ip: str) -> str:
    if not is_ip(target_ip):
        target_ip = socket.gethostbyname(target_ip)

    first_dig = map(int, target_ip.split("."))
    if first_dig == 127:
        return '127.0.0.1'

    data = run_locally('ip route get to'.split(" ") + [target_ip]).decode("utf8")
    data_line = data.split("\n")[0].strip()

    rr1 = r'{0} via [.0-9]+ dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr1 = rr1.replace(" ", r'\s+')
    rr1 = rr1.format(target_ip.replace('.', r'\.'))

    rr2 = r'{0} dev (?P<dev>.*?) src (?P<ip>[.0-9]+)$'
    rr2 = rr2.replace(" ", r'\s+')
    rr2 = rr2.format(target_ip.replace('.', r'\.'))

    res1 = re.match(rr1, data_line)
    res2 = re.match(rr2, data_line)

    if res1 is not None:
        return res1.group('ip')

    if res2 is not None:
        return res2.group('ip')

    raise OSError("Can't define interface for {0}".format(target_ip))


def open_for_append_or_create(fname: str) -> IO[str]:
    if not os.path.exists(fname):
        return cast(IO[str], open(fname, "w"))

    fd = open(fname, 'r+')
    fd.seek(0, os.SEEK_END)
    return cast(IO[str], fd)


def flatten(data: Iterable[Any]) -> List[Any]:
    res = []
    for i in data:
        if isinstance(i, (list, tuple, set)):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res


@contextlib.contextmanager
def empty_ctx(val: Any = None) -> Iterator[Any]:
    yield val


def to_ip(host_or_ip: str) -> str:
    # translate hostname to address
    try:
        ipaddress.ip_address(host_or_ip)
        return host_or_ip
    except ValueError:
        ip_addr = socket.gethostbyname(host_or_ip)
        return ip_addr


def sec_to_str(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return "{}:{:02d}:{:02d}".format(h, m, s)


# def get_time_after_interval(seconds: int) -> datetime.datetime:
#     return datetime.datetime.now() + datetime.timedelta(0, seconds)


FM_FUNC_INPUT = TypeVar("FM_FUNC_INPUT")
FM_FUNC_RES = TypeVar("FM_FUNC_RES")


def flatmap(func: Callable[[FM_FUNC_INPUT], Iterable[FM_FUNC_RES]],
            inp_iter: Iterable[FM_FUNC_INPUT]) -> Iterator[FM_FUNC_RES]:
    for val in inp_iter:
        for res in func(val):
            yield res


def shape2str(shape: Iterable[int]) -> str:
    return "*".join(map(str, shape))


def str2shape(shape: str) -> Tuple[int, ...]:
    return tuple(map(int, shape.split('*')))


Tp = TypeVar('Tp')


def find(lst: List[Tp], check: Callable[[Tp], bool], default: Tp = None) -> Tp:
    for obj in lst:
        if check(obj):
            return obj
    return default
