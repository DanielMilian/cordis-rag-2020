import csv, datetime as dt, subprocess
from pathlib import Path

F = Path("results/results.csv")
COLS = ["ts", "git", "dirty", "method", "config", "size", "overlap", "split", "n",
        "hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"]

def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

def check_split(split, method, config, size, overlap, allow_test=False):
    if split != "test":
        return
    if not allow_test:
        raise SystemExit("test split needs --allow-test (run once, at very end)")
    if F.exists():
        for r in csv.DictReader(open(F)):
            key = (r["split"], r["method"], r["config"], r["size"], r["overlap"])
            if key == ("test", method, config, str(size), str(overlap)):
                raise SystemExit("test already logged for this method/config")

def log_result(method, config, size, overlap, split, n, metrics, allow_test=False):
    check_split(split, method, config, size, overlap, allow_test)
    row = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
           "git": _git("rev-parse", "--short", "HEAD"),
           "dirty": bool(_git("status", "--porcelain", "--", ".", ":!results")),
           "method": method, "config": config, "size": size, "overlap": overlap,
           "split": split, "n": n, **metrics}
    F.parent.mkdir(exist_ok=True)
    new = not F.exists()
    if not new:
        assert next(csv.reader(open(F))) == COLS, "results.csv header mismatch"
    with open(F, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)

