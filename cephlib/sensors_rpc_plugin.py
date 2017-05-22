import re
import os
import sys
import json
import time
import zlib
import array
import pprint
import struct
import os.path
import logging
import threading
import traceback
import subprocess
import collections
from distutils import spawn


try:
    import libvirt
except ImportError:
    libvirt = None


try:
    from agent_module import Pool, noraise, BIO, tostr, IS_PYTHON3, Promote  # type: ignore
except ImportError:
    noraise = lambda x: x


try:
    from ceph_daemon import admin_socket
except ImportError:
    admin_socket = None


mod_name = "sensors"
__version__ = (0, 1)


logger = logging.getLogger("agent.sensors")
SensorsMap = {}
SENSOR2DEV_TYPE = {}


class Sensor(object):
    def __init__(self, params, allowed=None, disallowed=None):
        self.params = params
        self.allowed = allowed
        self.disallowed = disallowed
        self.allowed_names = set()

    def add_data(self, device, name, value):
        pass

    def collect(self):
        pass

    def get_updates(self):
        pass

    @classmethod
    def unpack_results(cls, device, metric, data, typecode):
        pass

    def init(self):
        pass

    def stop(self):
        pass


class ArraysSensor(Sensor):
    typecode = 'L'

    def __init__(self, params, allowed=None, disallowed=None):
        Sensor.__init__(self, params, allowed, disallowed)
        self.data = collections.defaultdict(lambda: array.array(self.typecode))
        self.prev_vals = {}

    def add_data(self, device, name, value):
        self.data[(device, name)].append(value)

    def add_relative(self, device, name, value):
        key = (device, name)
        pval = self.prev_vals.get(key)
        if pval is not None:
            if (value - pval) < 0:
                logger.error("Failed data in ArraysSensor.add_relative::%s. pval(=%s)>value(=%s). %s::%s",
                             self.__class__.__name__, pval, value, device, name)
                self.data[key].append(0)
            else:
                self.data[key].append(value - pval)
        self.prev_vals[key] = value

    def get_updates(self):
        res = self.data
        self.data = collections.defaultdict(lambda: array.array(self.typecode))
        return {key: (arr.typecode, arr.tostring()) for key, arr in res.items()}

    @classmethod
    def unpack_results(cls, device, metric, packed, typecode):
        arr = array.array(typecode)
        if sys.version_info >= (3, 0, 0):
            arr.frombytes(packed)
        else:
            arr.fromstring(packed)
        return arr

    def is_dev_accepted(self, name):
        dev_ok = True

        if self.disallowed is not None:
            dev_ok = all(not name.startswith(prefix) for prefix in self.disallowed)

        if dev_ok and self.allowed is not None:
            dev_ok = any(name.startswith(prefix) for prefix in self.allowed)

        return dev_ok


time_array_typechar = ArraysSensor.typecode


def provides(name, dev_tp=None):
    def closure(cls):
        SensorsMap[name] = cls
        if dev_tp is not None:
            SENSOR2DEV_TYPE[name] = dev_tp
        return cls
    return closure


def get_pid_list(disallowed_prefixes, allowed_prefixes):
    """Return pid list from list of pids and names"""
    # exceptions
    disallowed = disallowed_prefixes if disallowed_prefixes is not None else []
    if allowed_prefixes is None:
        # if nothing setted - all ps will be returned except setted
        result = [pid for pid in os.listdir('/proc')
                  if pid.isdigit() and pid not in disallowed]
    else:
        result = []
        for pid in os.listdir('/proc'):
            if pid.isdigit() and pid not in disallowed:
                name = get_pid_name(pid)
                if pid in allowed_prefixes or any(name.startswith(val) for val in allowed_prefixes):
                    # this is allowed pid?
                    result.append(pid)
    return result


def get_pid_name(pid):
    """Return name by pid"""
    try:
        with open(os.path.join('/proc/', pid, 'cmdline'), 'r') as pidfile:
            try:
                cmd = pidfile.readline().split()[0]
                return os.path.basename(cmd).rstrip('\x00')
            except IndexError:
                # no cmd returned
                return "<NO NAME>"
    except IOError:
        # upstream wait any string, no matter if we couldn't read proc
        return "no_such_process"


@provides("block-io", 'block')
class BlockIOSensor(ArraysSensor):
    #  1 - major number
    #  2 - minor mumber
    #  3 - device name
    #  4 - reads completed successfully
    #  5 - reads merged
    #  6 - sectors read
    #  7 - time spent reading (ms)
    #  8 - writes completed
    #  9 - writes merged
    # 10 - sectors written
    # 11 - time spent writing (ms)
    # 12 - I/Os currently in progress
    # 13 - time spent doing I/Os (ms)
    # 14 - weighted time spent doing I/Os (ms)

    SECTOR_SIZE = 512

    io_values_pos = [
        (3, 'reads_completed', True, 1),
        (5, 'sectors_read', True, SECTOR_SIZE),
        (6, 'rtime', True, 1),
        (7, 'writes_completed', True, 1),
        (9, 'sectors_written', True, SECTOR_SIZE),
        (10, 'wtime', True, 1),
        (11, 'io_queue', False, 1),
        (12, 'io_time', True, 1),
        (13, 'weighted_io_time', True, 1)
    ]
    rbd_dev = re.compile(r"rbd\d+$")

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        if self.disallowed is None:
            self.disallowed = ('ram', 'loop')

        for line in open('/proc/diskstats'):
            vals = line.split()
            dev_name = vals[2]
            if self.is_dev_accepted(dev_name):
                if not dev_name[-1].isdigit() or self.rbd_dev.match(dev_name):
                    self.allowed_names.add(dev_name)

        self.collect(init_rel=True)

    def collect(self, init_rel=False):
        for line in open('/proc/diskstats'):
            vals = line.split()
            dev_name = vals[2]

            if dev_name not in self.allowed_names:
                continue

            for pos, name, aggregated, coef in self.io_values_pos:
                vl = int(vals[pos]) * coef

                if dev_name == 'sdc' and name == 'io_time':
                    if not os.path.exists("/tmp/sdc_iotime.log"):
                        mode = 'w'
                    else:
                        mode = 'r+'

                    with open("/tmp/sdc_iotime.log", mode) as fd:
                        fd.seek(0, os.SEEK_END)
                        fd.write("{0}\n".format(vl))

                if aggregated:
                    self.add_relative(dev_name, name, vl)
                elif not init_rel:
                    self.add_data(dev_name, name, int(vals[pos]))


@provides("vm-io", 'block')
class VMIOSensor(ArraysSensor):
    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)
        self.per_node_io = {}

        if libvirt:
            self.conn = libvirt.openReadOnly(None)
            self.collect(init_rel=True)
        else:
            self.conn = None

    def collect(self, init_rel=False):
        if self.conn is None:
            return

        cum_stats = [0, 0, 0, 0]

        for vm in self.conn.listAllDomains():
            if vm.isActive():
                did = vm.ID()
                vm_stat = vm.blockStats('')
                if did in self.per_node_io:
                    prev_dstat = self.per_node_io[did]
                    cum_stats[0] += vm_stat[0] - prev_dstat[0]
                    cum_stats[1] += vm_stat[1] - prev_dstat[1]
                    cum_stats[2] += vm_stat[2] - prev_dstat[2]
                    cum_stats[3] += vm_stat[3] - prev_dstat[3]
                else:
                    self.per_node_io[did] = vm_stat[:-1]

        self.add_data("vm_io", "reads_completed", cum_stats[0])
        self.add_data("vm_io", "bytes_read", cum_stats[1])
        self.add_data("vm_io", "writes_completed", cum_stats[2])
        self.add_data("vm_io", "bytes_written", cum_stats[3])


def get_interfaces():
    for name in os.listdir("/sys/class/net"):
        fpath = os.path.join("/sys/class/net", name)

        if not os.path.islink(fpath):
            continue

        while os.path.islink(fpath):
            fpath = os.path.abspath(
                os.path.join(os.path.dirname(fpath),
                             os.readlink(fpath)))

        yield '/devices/virtual/' not in fpath, name


@provides("net-io", 'eth')
class NetIOSensor(ArraysSensor):
    net_values_pos = [
        (0, 'recv_bytes', True),
        (1, 'recv_packets', True),
        (8, 'send_bytes', True),
        (9, 'send_packets', True),
    ]

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        assert self.allowed is None
        assert self.disallowed is None

        for _, _, aggregated in self.net_values_pos:
            assert aggregated, "Non-aggregated values is not supported in net sensor"

        self.allowed_names.update(dev_name for is_phy, dev_name in get_interfaces() if is_phy)
        self.collect(init_rel=True)

    def collect(self, init_rel=False):
        for line in open('/proc/net/dev').readlines()[2:]:
            dev_name, stats = line.split(":", 1)
            dev_name = dev_name.strip()
            if dev_name in self.allowed_names:
                vals = stats.split()
                for pos, name, _ in self.net_values_pos:
                    vl = int(vals[pos])
                    self.add_relative(dev_name, name, vl)


def pid_stat(pid):
    """Return total cpu usage time from process"""
    # read /proc/pid/stat
    with open(os.path.join('/proc/', pid, 'stat'), 'r') as pidfile:
        proctimes = pidfile.readline().split()
    # get utime from /proc/<pid>/stat, 14 item
    utime = proctimes[13]
    # get stime from proc/<pid>/stat, 15 item
    stime = proctimes[14]
    # count total process used time
    return float(int(utime) + int(stime))


@provides("perprocess-cpu")
class ProcCpuSensor(ArraysSensor):
    def collect(self):
        # TODO(koder): fixed list of PID's must be given
        for pid in get_pid_list(self.disallowed, self.allowed):
            try:
                self.add_data(get_pid_name(pid), pid, pid_stat(pid))
            except IOError:
                # probably proc has already terminated, skip it
                continue


def get_mem_stats(pid):
    """Return memory data of pid in format (private, shared)"""

    fname = '/proc/{0}/{1}'.format(pid, "smaps")
    lines = open(fname).readlines()

    shared = 0
    private = 0
    pss = 0

    # add 0.5KiB as this avg error due to truncation
    pss_adjust = 0.5

    for line in lines:
        if line.startswith("Shared"):
            shared += int(line.split()[1])

        if line.startswith("Private"):
            private += int(line.split()[1])

        if line.startswith("Pss"):
            pss += float(line.split()[1]) + pss_adjust

    # Note Shared + Private = Rss above
    # The Rss in smaps includes video card mem etc.

    if pss != 0:
        shared = int(pss - private)

    return (private, shared)


def get_ram_size():
    """Return RAM size in Kb"""
    with open("/proc/meminfo") as proc:
        mem_total = proc.readline().split()
    return int(mem_total[1])


@provides("perprocess-ram")
class ProcRamSensor(ArraysSensor):
    def collect(self):
        # TODO(koder): fixed list of PID's nust be given
        for pid in get_pid_list(self.disallowed, self.allowed):
            try:
                dev_name = get_pid_name(pid)

                private, shared = get_mem_stats(pid)
                total = private + shared
                sys_total = get_ram_size()
                usage = float(total) / sys_total

                sensor_name = "{0}({1})".format(dev_name, pid)

                self.add_data(sensor_name, "private_mem", private)
                self.add_data(sensor_name, "shared_mem", shared),
                self.add_data(sensor_name, "used_mem", total),
                self.add_data(sensor_name, "mem_usage_percent", int(usage * 100))
            except IOError:
                # permission denied or proc die
                continue


@provides("system-cpu", 'cpu')
class SystemCPUSensor(ArraysSensor):
    # 0 - cpu name
    # 1 - user: normal processes executing in user mode
    # 2 - nice: niced processes executing in user mode
    # 3 - system: processes executing in kernel mode
    # 4 - idle: twiddling thumbs
    # 5 - iowait: waiting for I/O to complete
    # 6 - irq: servicing interrupts
    # 7 - softirq: servicing softirqs

    cpu_values_pos = [
        (1, 'user', True),
        (2, 'nice', True),
        (3, 'sys', True),
        (4, 'idle', True),
        (5, 'iowait', True),
        (6, 'irq', True),
        (7, 'sirq', True),
        (8, 'steal', True),
        (9, 'guest', True),
    ]

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        assert self.allowed is None
        assert self.disallowed is None

        for _, _, aggregated in self.cpu_values_pos:
            assert aggregated, "Non-aggregated values is not supported in cpu sensor"

        self.collect(init_rel=True)

    def collect(self, init_rel=False):
        # calculate core count
        core_count = 0

        for line in open('/proc/stat'):
            vals = line.split()
            dev_name = vals[0]

            if dev_name == 'cpu':
                for pos, name, _ in self.cpu_values_pos:
                    self.add_relative(dev_name, name, int(vals[pos]))
            elif dev_name == 'procs_blocked' and not init_rel:
                self.add_data("cpu", "procs_blocked", int(vals[1]))
            elif dev_name.startswith('cpu') and not init_rel:
                core_count += 1

        if not init_rel:
            # procs in queue
            TASKSPOS = 3
            vals = open('/proc/loadavg').read().split()
            ready_procs = vals[TASKSPOS].partition('/')[0]

            # dec on current proc
            procs_queue = (float(ready_procs) - 1) / core_count
            self.add_data("cpu", "procs_queue_x10", int(procs_queue * 10))


@provides("system-ram")
class SystemRAMSensor(ArraysSensor):
    # return this values or setted in allowed
    ram_fields = ['MemTotal', 'MemFree', 'Buffers', 'Cached', 'SwapCached',
                  'Dirty', 'Writeback', 'SwapTotal', 'SwapFree']

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)

        if self.allowed is None:
            self.allowed = self.ram_fields

        self.allowed_fields = set()
        for line in open('/proc/meminfo'):
            field_name = line.split()[0].rstrip(":")
            if self.is_dev_accepted(field_name):
                self.allowed_fields.add(field_name)

    def collect(self):
        for line in open('/proc/meminfo'):
            vals = line.split()
            field = vals[0].rstrip(":")
            if field in self.allowed_fields:
                self.add_data("ram", field, int(vals[1]))


def get_val(dct, path):
    if '/' in path:
        root, next = path.split('/', 1)
        return get_val(dct[root], next)
    return dct[path]


CephOp = collections.namedtuple("CephOp", "age descr_idx duration initiated_at status stages extra1 extra2")
ALL_STAGES = [
    "queued_for_pg",
    "reached_pg",
    "started",
    "commit_queued_for_journal_write",
    "waiting_for_subop",
    "write_thread_in_journal_buffer",
    "op_commit",
    "op_applied",
    "sub_op_applied",
    "journaled_completion_queued",
    "commit_sent",
    "done",
    "subop_commit_rec",
    "subop_apply_rec"
]


CEPH_OP_DESCR_IDX = {
    "osd_repop_reply": 0,
    "osd_op": 1,
    "osd_repop": 2
}


STAGE_TIME_FORMAT = 'I'
STAGE_TIME_FORMAT_SZ = struct.calcsize(STAGE_TIME_FORMAT)
MAX_STAGE_TIME = 2 ** 32 - 1


waiting_subop = "waiting for subops from"
sub_op_commit = "sub_op_commit_rec from"
sub_op_applied = "sub_op_applied_rec from"


ceph_op_format = "!BfBfL" + STAGE_TIME_FORMAT * len(ALL_STAGES)
OP_SIZE = struct.calcsize(ceph_op_format)
VERSION = 1
BOR = 'EF34'
EOR = 'AB12'
assert struct.calcsize('!B') == 1


def pack_ceph_op(op):
    extra = struct.pack("!B" + STAGE_TIME_FORMAT * len(op.extra1), len(op.extra1), *op.extra1)
    extra += struct.pack("!B" + STAGE_TIME_FORMAT * len(op.extra2), len(op.extra2), *op.extra2)
    dt = struct.pack(ceph_op_format, VERSION, op.age, op.descr_idx, op.duration,
                     op.initiated_at / 1000000,
                     *[(0 if tm is None else tm) for tm in op.stages]) + extra
    return BOR + dt + EOR


def unpack_ceph_op(stream):
    assert BOR == stream.read(len(BOR))
    vls = struct.unpack(ceph_op_format, stream.read(OP_SIZE))
    version, age, descr_idx, duration, init_at = vls[:5]
    stages = list(vls[5:])
    assert version == VERSION

    sz1 = ord(stream.read(1))
    if sz1:
        extra_stages1 = list(struct.unpack("!" + STAGE_TIME_FORMAT * sz1, stream.read(STAGE_TIME_FORMAT_SZ * sz1)))
    else:
        extra_stages1 = []

    sz2 = ord(stream.read(1))
    if sz2:
        extra_stages2 = list(struct.unpack("!" + STAGE_TIME_FORMAT * sz2, stream.read(STAGE_TIME_FORMAT_SZ * sz2)))
    else:
        extra_stages2 = []

    assert EOR == stream.read(len(EOR))
    return CephOp(age, descr_idx, duration, init_at * 1000000, None, stages, extra_stages1, extra_stages2)


def to_ctime_ms(time_str):
    dt_s, micro_sec = time_str.split('.')
    dt = time.strptime(dt_s, '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(dt) * 1000000 + int(micro_sec))


LN_FORMAT = '!I'
LN_FORMAT_SZ = struct.calcsize(LN_FORMAT)


def merge_strings(strs):
    res = ""
    for string in strs:
        res += struct.pack(LN_FORMAT, len(string)) + string
    return res


def unmerge_strings(blob):
    offset = 0
    while offset < len(blob):
        ln, = struct.unpack(LN_FORMAT, blob[offset: offset + LN_FORMAT_SZ])
        yield blob[offset + LN_FORMAT_SZ: offset + LN_FORMAT_SZ + ln]
        offset += LN_FORMAT_SZ + ln


@provides("ceph")
class CephSensor(ArraysSensor):

    historic_duration = 2
    historic_size = 200

    def __init__(self, *args, **kwargs):
        ArraysSensor.__init__(self, *args, **kwargs)
        self.cluster = self.params.get('cluster', 'ceph')
        self.prev_vals = {}

        sources = self.params.get('sources', [])
        self.historic = {} if 'historic' in sources else None
        self.historic_js = {} if 'historic_js' in sources else None
        self.perf_dump = {} if 'perf_dump' in sources else None

        self.prev_historic = set()

        if self.params['osds'] == "all":
            self.osd_ids = []
            for name in os.listdir('/var/lib/ceph/osd'):
                rr = re.match(r"ceph-\d+", name)
                if rr:
                    self.osd_ids.append(name.split("-")[1])
        else:
            self.osd_ids = self.params['osds'][:]

        if 'historic' in self.params.get('sources', {}):
            for osd_id in self.osd_ids:
                self.prev_vals[osd_id] = self.set_osd_historic(self.historic_duration, self.historic_size, osd_id)

    def run_ceph_daemon_cmd(self, osd_id, args):
        asok = "/var/run/ceph/{0}-osd.{}.asok".format(self.cluster, osd_id)
        if admin_socket:
            res = admin_socket(asok, list(args.split()))
        else:
            res = subprocess.check_output("ceph daemon {} {}".format(asok, args), shell=True)

        return res

    def store_to_db(self, metric, osd_id, data):
        key = "{0}.{1}.{2}".format(metric, osd_id, int(time.time() * 100))
        self.disk_stor[key] = data

    def collect(self):
        for osd_id in self.osd_ids:
            if self.historic is not None:
                ops_json = self.run_ceph_daemon_cmd(osd_id, "dump_historic_ops").strip()
                ops_json_z = zlib.compress(ops_json)
                if self.historic_js is not None:
                    self.historic_js.setdefault(osd_id, []).append(ops_json_z)

                curr = set()

                new_ops = []
                for op in json.loads(ops_json)['Ops']:
                    curr.add(op['description'])
                    if op['description'] in self.prev_historic:
                        continue
                    op_obj = self.parse_op(op)
                    if op_obj:
                        new_ops.append(op_obj)

                data = "".join(pack_ceph_op(op_obj) for op_obj in new_ops)
                data_z = zlib.compress(data)
                self.historic.setdefault(osd_id, []).append(data_z)

                self.prev_historic = curr

            if 'in_flight' in self.params.get('sources', []):
                raise NotImplementedError()

            if self.perf_dump is not None:
                data = self.run_ceph_daemon_cmd(osd_id, 'perf dump')
                data_z = zlib.compress(data)
                self.perf_dump.setdefault(osd_id, []).append(data_z)


    def set_osd_historic(self, duration, keep, osd_id):
        data = json.loads(self.run_ceph_daemon_cmd(osd_id, "dump_historic_ops"))
        self.run_ceph_daemon_cmd(osd_id, "config set osd_op_history_duration {}".format(duration))
        self.run_ceph_daemon_cmd(osd_id, "config set osd_op_history_size {}".format(keep))
        return (data["duration to keep"], data["num to keep"])

    @classmethod
    def parse_op(cls, op):
        descr = op['description']
        if descr.startswith('osd_repop('):
            # slave op
            assert len(op['type_data']) == 2
            tp, steps = op['type_data']
        elif descr.startswith('osd_op('):
            # master op
            assert len(op['type_data']) == 3
            tp, _, steps = op['type_data']
        elif descr.startswith('osd_repop_reply('):
            # ?????
            assert len(op['type_data']) == 2
            tp, steps = op['type_data']
        else:
            logger.warning("Can't parse op %r\n%r", descr, op)
            return None
            # raise ValueError("Can't parse op {0!r}".format(descr))

        step_timings = [None] * len(ALL_STAGES)
        extra1 = []
        extra2 = []
        initiated_at = to_ctime_ms(op['initiated_at'])
        assert steps[0]['event'] == 'initiated'

        for step in steps[1:]:
            name = step['event']
            tm = to_ctime_ms(step['time']) - initiated_at

            if tm > MAX_STAGE_TIME:
                tm = MAX_STAGE_TIME

            if name.startswith(waiting_subop):
                name = 'waiting_for_subop'

            try:
                step_timings[ALL_STAGES.index(name)] = tm
            except ValueError:
                if name.startswith(sub_op_commit):
                    extra1.append(tm)
                elif name.startswith(sub_op_applied):
                    extra2.append(tm)
                else:
                    raise ValueError("Unknown stage type {0!r}".format(name))
        return CephOp(age=op['age'],
                      descr_idx=CEPH_OP_DESCR_IDX[descr.split("(", 1)[0]],
                      duration=op['duration'],
                      initiated_at=initiated_at,
                      status=tp,
                      stages=step_timings,
                      extra1=extra1,
                      extra2=extra2)

    def stop(self):
        for osd_id, (duration, keep) in self.prev_vals.items():
            self.prev_vals[osd_id] = self.set_osd_historic(duration, keep, osd_id)

    def get_updates(self):
        res = super(CephSensor, self).get_updates()
        if self.disk_stor:
            raise NotImplementedError("Updates from disk aren't implemented")
        else:
            if self.historic:
                for osd_id, packed_ops in self.historic.items():
                    res[("osd{0}".format(osd_id), "historic")] = (None, merge_strings(packed_ops))
                self.historic = {}

            if self.historic_js:
                for osd_id, ops in self.historic_js.items():
                    res[("osd{0}".format(osd_id), "historic_js")] = (None, merge_strings(ops))
                self.historic_js = {}

            if self.perf_dump:
                for osd_id, ops in self.perf_dump.items():
                    res[("osd{0}".format(osd_id), "perf_dump")] = (None, merge_strings(ops))
                self.perf_dump = {}

        return res

    @classmethod
    def unpack_historic(cls, packed):
        return cls.unpack_historic_fd(BIO(packed), len(packed))

    @classmethod
    def unpack_historic_fd(cls, fd, size):
        while fd.tell() < size:
            yield unpack_ceph_op(fd)

    @classmethod
    def unpack_results(cls, device, metric, packed_z, typecode):
        raise NotImplementedError()

    @staticmethod
    def split_results(metric, packed_z):
        packed = (zlib.decompress(chunk) for chunk in unmerge_strings(packed_z))
        if metric == 'historic':
            if IS_PYTHON3:
                return b"".join(packed)
            return "".join(packed)
        elif metric in ('historic_js', 'perf_dump'):
            if IS_PYTHON3:
                return ("[" + ",\n".join(chunk.decode('utf8').strip() for chunk in packed) + "]").encode('utf8')
            return "[" + ",\n".join(chunk.strip() for chunk in packed) + "]"
        else:
            assert False, "Unknown metric {0!r}".format(metric)


class SensorsData(object):
    def __init__(self):
        self.cond = threading.Condition()
        self.collected_at = array.array(time_array_typechar)
        self.stop = False
        self.sensors = {}
        self.data_fd = None  # temporary file to store results
        self.promoted_exc = None


def collect(sensors_config):
    curr = {}
    for name, config in sensors_config.items():
        params = {'config': config}

        if "allow" in config:
            params["allowed_prefixes"] = config["allow"]

        if "disallow" in config:
            params["disallowed_prefixes"] = config["disallow"]

        curr[name] = SensorsMap[name](**params)
    return curr


def sensors_bg_thread(sensors_config, sdata, collect_tout=1.0):
    try:
        sensors_config = sensors_config.copy()
        pool_sz = sensors_config.pop("pool_sz", 32)
        pool = Pool(pool_sz) if pool_sz != 0 else None

        # prepare sensor classes
        with sdata.cond:
            sdata.sensors = {}
            for name, config in sensors_config.items():
                params = {'params': config}

                if "allow" in config:
                    params["allowed_prefixes"] = config["allow"]

                if "disallow" in config:
                    params["disallowed_prefixes"] = config["disallow"]

                sdata.sensors[name] = SensorsMap[name](**params)
                sdata.sensors[name].init()

        next_collect_at = time.time() + collect_tout

        while not sdata.stop:
            dtime = next_collect_at - time.time()
            if dtime > 0:
                with sdata.cond:
                    sdata.cond.wait(dtime)

            next_collect_at += collect_tout

            if sdata.stop:
                break

            ctm = time.time()
            with sdata.cond:
                sdata.collected_at.append(int(ctm * 1000))
                if pool is not None:

                    def caller(x):
                        return x()

                    for msg, tb, exc_cls_name in pool.map(caller, [sensor.collect for sensor in sdata.sensors.values()]):
                        if tb:
                            sdata.promoted_exc = Promote(msg, tb, exc_cls_name)
                            break
                else:
                    for sensor in sdata.sensors.values():
                        sensor.collect()

                etm = time.time()
                sdata.collected_at.append(int(etm * 1000))

    except Exception as exc:
        logger.exception("In sensor BG thread")
        sdata.promoted_exc = Promote(str(exc), traceback.format_exc(), type(exc).__name__)
    finally:
        for sensor in sdata.sensors.values():
            sensor.stop()


sensors_thread = None
sdata = None  # type: SensorsData


sensor_units = {
    "collected_at": "ms",

    "system-cpu.idle": "",
    "system-cpu.nice": "",
    "system-cpu.user": "",
    "system-cpu.sys": "",
    "system-cpu.iowait": "",
    "system-cpu.irq": "",
    "system-cpu.sirq": "",
    "system-cpu.steal": "",
    "system-cpu.guest": "",

    "system-cpu.procs_blocked": "",
    "system-cpu.procs_queue_x10": "",

    "net-io.recv_bytes": "B",
    "net-io.recv_packets": "",
    "net-io.send_bytes": "B",
    "net-io.send_packets": "",

    "block-io.io_queue": "",
    "block-io.io_time": "ms",
    "block-io.reads_completed": "",
    "block-io.rtime": "ms",
    "block-io.sectors_read": "B",
    "block-io.sectors_written": "B",
    "block-io.writes_completed": "",
    "block-io.wtime": "ms",
    "block-io.weighted_io_time": "ms"
}


def unpack_rpc_updates(res_tuple):
    """
    :param res_tuple: 
    :return: Iterator[sensor_path:str, data: Any, is_parsed: bool] 
    """
    offset_map, compressed_blob, compressed_collected_at_b = res_tuple
    blob = zlib.decompress(compressed_blob)
    collected_at_b = zlib.decompress(compressed_collected_at_b)
    collected_at = array.array(time_array_typechar)

    if IS_PYTHON3:
        collected_at.frombytes(collected_at_b)
    else:
        collected_at.fromstring(collected_at_b)

    yield 'collected_at', collected_at, True, sensor_units['collected_at']

    # TODO: data is unpacked/repacked here with no reason
    for sensor_path, (offset, size, typecode) in offset_map.items():
        sensor_path = sensor_path.decode("utf8")
        sensor_name, device, metric = sensor_path.split('.', 2)
        units = sensor_units.get("{0}.{1}".format(sensor_name, metric), "")
        if sensor_name == 'ceph' and metric in {'historic', 'historic_js', 'perf_dump'}:
            yield sensor_path, CephSensor.split_results(metric, blob[offset:offset + size]), False, units
        else:
            sensor_data = SensorsMap[sensor_name].unpack_results(device,
                                                                 metric,
                                                                 blob[offset:offset + size],
                                                                 typecode.decode("ascii") if typecode else None)
            yield sensor_path, sensor_data, True, units


@noraise
def rpc_start(sensors_config):
    global sensors_thread
    global sdata

    if array.array('L').itemsize != 8:
        message = "Python array.array('L') items should be 8 bytes in size, not {0}." + \
                  " Can't provide sensors on this platform. Disable sensors in config and retry"
        raise ValueError(message.format(array.array('L').itemsize))

    if sensors_thread is not None:
        raise ValueError("Thread already running")

    sdata = SensorsData()
    sensors_thread = threading.Thread(target=sensors_bg_thread, args=(sensors_config, sdata))
    sensors_thread.daemon = True
    sensors_thread.start()

    logger.info("Sensors started with config %s", pprint.pformat(sensors_config))


@noraise
def rpc_get_updates():
    t = time.time()
    if sdata is None:
        raise ValueError("No sensor thread running")

    offset_map = collected_at = None
    blob = ""

    with sdata.cond:
        if sdata.promoted_exc:
            raise sdata.promoted_exc

        offset_map = {}
        for sensor_name, sensor in sdata.sensors.items():
            for (device, metric), (typecode, val) in sensor.get_updates().items():
                offset_map["{0}.{1}.{2}".format(sensor_name, device, metric)] = (len(blob), len(val), typecode)
                blob += val

        collected_at = sdata.collected_at
        sdata.collected_at = array.array(sdata.collected_at.typecode)

    res = offset_map, zlib.compress(blob), zlib.compress(collected_at.tostring())
    dt = int((time.time() - t) * 1000)
    tlen = len(res[1]) + len(res[2]) + sum(map(len, offset_map)) + 16 * len(offset_map)
    logger.debug("Send sensor updates. Total size is ~%sKiB. Prepare time is %sms", tlen // 1024, dt)
    return res


@noraise
def rpc_stop():
    global sensors_thread
    global sdata

    logger.info("Sensors stop requested")

    if sensors_thread is None:
        raise ValueError("No sensor thread running")

    sdata.stop = True
    with sdata.cond:
        sdata.cond.notify_all()

    sensors_thread.join()

    if sdata.promoted_exc:
        raise sdata.promoted_exc

    res = rpc_get_updates()

    sensors_thread = None
    sdata = None

    return res


@noraise
def rpc_find_pids_for_cmd(bname):
    bin_path = spawn.find_executable(bname)

    if not bin_path:
        raise NameError("Can't found binary path for {0!r}".format(bname))

    res = []
    for name in os.listdir('/proc'):
        if name.isdigit() and os.path.isdir(os.path.join('/proc', name)):
            exe = os.path.join('/proc', name, 'exe')
            if os.path.exists(exe) and os.path.islink(exe) and bin_path == os.readlink(exe):
                res.append(int(name))

    logger.debug("Find pids for binary %s = %s", bname, res)

    return res
