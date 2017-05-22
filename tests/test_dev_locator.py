import contextlib
from typing import Iterator, Dict, Tuple, List, Set
from unittest.mock import patch

import cephlib.storage_selectors
from cephlib.storage import make_storage
from cephlib.types import DataSource
from cephlib.wally_storage import WallyDB
from cephlib.sensor_storage import SensorStorage, ISensorStorage
from cephlib.node import NodeInfo, ConnCreds

from test_storage import in_temp_dir


@contextlib.contextmanager
def temp_stor() -> Iterator[ISensorStorage]:
    with in_temp_dir() as root:
        with make_storage(root, existing=False) as storage:
            yield SensorStorage(storage, WallyDB)


def test_dev_locator():
    dev_locator_dct = [
        {"node21": [{"sd[kc]": 'nomagic'}]},
        {"role=testnode": [{'rbd0': 'client_disk'}]},
        {"node2.*": [{"sd[kc]": 'magic'}]},
        {"role=ceph-osd": [
            {'sdk': 'storage_disk'},
            {"sd[g-z]": ['storage_disk', 'ceph_storage']},
            {"sd[c-f]": ['storage_disk', 'ceph_journal']}
            ]
        },
    ]

    nodes_roles_and_devs = {
        'node1': ('1.1.1.1:22', {'testnode'}, {
            'rbd0': ('hdd', ['client_disk'])
        }),
        'node2': ('2.2.2.2:22', {'ceph-osd'}, {
            'sdg': ('hdd', ['storage_disk', 'ceph_storage']),
            'sdc': ('hdd', ['magic']),
            'sdk': ('hdd', ['magic'])
        }),
        'node3': ('3.3.3.3:22', {'compute'}, {
            'sdk': ('hdd', []),
            'sdc': ('hdd', [])
        }),
        'node21': ('21.21.21.21:22', {'mon'}, {
            'sdk': ('hdd', ['nomagic']),
            'sdc': ('hdd', ['nomagic'])
        }),
        'node22': ('22.22.22.22:22', {'ceph-osd'}, {
            'sdk': ('hdd', ['magic']),
            'sdc': ('hdd', ['magic'])
        })
    }  # type: Dict[str, Tuple[str, Set[str], Dict[str, Tuple[str, List[str]]]]]

    nodes = []

    with temp_stor() as storage:  # type: ISensorStorage
        for node_name, (stor_id, roles, devs) in nodes_roles_and_devs.items():
            for dev, (tp, _) in devs.items():
                ds = DataSource(node_id=stor_id, sensor='sensor', dev=dev, metric='metric', tag='csv')
                ds.verify()
                storage.append_sensor([1], ds, 'units')

            ip, port = stor_id.split(":")
            creds = ConnCreds(ip, 'test', port=port)
            nodes.append(NodeInfo(creds, roles, hostname=node_name))

        with patch.dict(cephlib.storage_selectors.DEFAULT_ROLES, {}):
            cephlib.storage_selectors.update_storage_selector(storage, dev_locator_dct, nodes)

        all_dev_roles = set()
        for node_name, (stor_id, roles, devs) in nodes_roles_and_devs.items():
            for dev, (_, roles) in devs.items():
                for role in roles:
                    all_dev_roles.add(role)
                    node_dev = [(ds.node_id, ds.dev) for ds in storage.iter_sensors(dev_role=role)]
                    assert (stor_id, dev) in node_dev, "{},{},{}".format(node_name, dev, role)

                for role in all_dev_roles - set(roles):
                    node_dev = [(ds.node_id, ds.dev) for ds in storage.iter_sensors(dev_role=role)]
                    assert (stor_id, dev) not in node_dev, "{},{},{}".format(node_name, dev, role)
