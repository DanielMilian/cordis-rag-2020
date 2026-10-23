import re
import pandas as pd

REC_SPLIT = re.compile(r'\r?\n(?="\d+";")')   # new record = newline + "<id>";"
SEP = '";"'


def read_cordis_csv(path, max_bad_frac=0.001) -> pd.DataFrame:
    text = open(path, encoding="utf-8-sig", newline="").read()
    header, _, body = text.partition("\n")
    names = header.strip().strip('"').split(SEP)

    rows, bad = [], []
    for rec in REC_SPLIT.split(body.strip("\r\n")):
        rec = rec.strip()
        if rec.startswith('"'):
            rec = rec[1:]
        if rec.endswith('"'):
            rec = rec[:-1]
        f = rec.split(SEP)
        if len(f) != len(names):
            bad.append(f[0][:20])
            continue
        rows.append([x.replace('""', '"') for x in f])

    total = len(rows) + len(bad)
    print(f"[csv_io] records={total} parsed={len(rows)} malformed={len(bad)}")
    assert len(bad) / total <= max_bad_frac, f"too many malformed rows: {bad[:10]}"

    df = pd.DataFrame(rows, columns=names).replace("", pd.NA)
    df["objective"] = df["objective"].str.strip('"')   # stray leading/trailing quote
    return df
