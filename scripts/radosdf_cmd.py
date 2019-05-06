import sys

from cephlib import RadosDF


df1 = RadosDF.from_json(open(sys.argv[1]).read())
df2 = RadosDF.from_json(open(sys.argv[2]).read())

pools1 = {pool.name: pool for pool in df1.pools}
pools2 = {pool.name: pool for pool in df2.pools}

ios1 = {name: (pool.write_ops * 3 + pool.read_ops) for name, pool in pools1.items()}
ios2 = {name: (pool.write_ops * 3 + pool.read_ops) for name, pool in pools2.items()}

delta = {name: ios1[name] - ios2[name] for name in ios1}

for name, dio in sorted(delta.items(), key=lambda x: -x[1]):
    print(f"{name:>30s}  {dio // 3600:>10d}")
