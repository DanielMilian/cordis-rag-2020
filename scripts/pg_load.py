import sys, numpy as np, pandas as pd, psycopg
from pgvector.psycopg import register_vector

SIZE, OV = int(sys.argv[1]), int(sys.argv[2])
tag = f"bge-small_{SIZE}_{OV}"
table = f"chunks_{SIZE}_{OV}"
E = np.load(f"data/processed/emb_{tag}.npy")
ids = pd.read_parquet(f"data/processed/emb_{tag}_ids.parquet")
assert len(E) == len(ids) and E.shape[1] == 384 and E.dtype == np.float32
env = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)
conn = psycopg.connect(host="localhost", port=5433, dbname=env["POSTGRES_DB"],
                       user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
register_vector(conn)
cur = conn.cursor()
cur.execute(f"DROP TABLE IF EXISTS {table}")
cur.execute(f"CREATE TABLE {table} (pos int PRIMARY KEY, chunk_id text, project_id text, emb vector(384))")
rows = [(i, str(c), str(p), E[i]) for i, (c, p) in enumerate(zip(ids.chunk_id, ids.project_id))]
cur.executemany(f"INSERT INTO {table} VALUES (%s,%s,%s,%s)", rows)
conn.commit()
cur.execute(f"SELECT count(*) FROM {table}")
print("rows in table:", cur.fetchone()[0], "expected:", len(E))
cur.execute(f"SELECT emb FROM {table} ORDER BY pos")
back = np.array([r[0].to_list() for r in cur.fetchall()], dtype=np.float32)
print("roundtrip shape:", back.shape, "max abs diff:", float(np.abs(back - E).max()))
