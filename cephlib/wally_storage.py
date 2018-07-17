from typing import List
from .node import NodeInfo


from .istorage import IStorage


class WallyDB:
    suite_cfg_r = r'results/{suite_id}\.info\.yml'

    job_root = r'results/{suite_id}\.{job_id}/'
    job_cfg_r = job_root + r'info\.yml'

    # time series, data from load tool, sensor is a tool name
    ts_r = job_root + r'{node_id}\.{sensor}\.{metric}\.{tag}'

    # statistica data for ts
    stat_r = job_root + r'{node_id}\.{sensor}\.{metric}\.stat\.yaml'

    # sensor data
    sensor_data_r = r'sensors/{node_id}_{sensor}\.{dev}\.{metric}\.{tag}'
    sensor_time_r = r'sensors/{node_id}_collected_at\.csv'

    report_root = 'report/'
    plot_r = r'{suite_id}\.{job_id}/{node_id}\.{sensor}\.{dev}\.{metric}\.{tag}'
    txt_report = report_root + '{suite_id}_report.txt'

    job_extra = 'meta/{suite_id}.{job_id}/{name}.yaml'

    job_cfg = job_cfg_r.replace("\\.", '.')
    suite_cfg = suite_cfg_r.replace("\\.", '.')
    ts = ts_r.replace("\\.", '.')
    stat = stat_r.replace("\\.", '.')
    sensor_data = sensor_data_r.replace("\\.", '.')
    sensor_time = sensor_time_r.replace("\\.", '.')
    plot = plot_r.replace("\\.", '.')

    all_nodes = 'nodes/all.yml'
    nodes_params = 'nodes/params.js'
    rpc_logs = "rpc_logs/{node_id}.txt"
    config = 'config.yml'
    run_interval = 'run_interval.yml'
    resource_usage_rel = 'resource_usage'
    res_code = 'result_code.yml'
    pgs_io = 'sensors/ceph.pgs_dump.{time}.bz2'
    pools_io = 'sensors/ceph.pools_df.{time}.bz2'
    ceph_metric = 'sensors/ceph.{node_id}.{metric}.{time}.{tag}'


def find_nodes_by_roles(storage: IStorage, roles: List[str]) -> List[str]:
    nodes = storage.load_list(NodeInfo, WallyDB.all_nodes)
    roles_s = set(roles)
    return [node.node_id for node in nodes if roles_s.intersection(node.roles)]

