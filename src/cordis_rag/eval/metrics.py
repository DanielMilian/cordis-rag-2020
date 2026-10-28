import numpy as np


def project_ranking(hits):
    seen, out = set(), []
    for _, pid, _ in hits:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def gold_ranks(rankings, golds):
    return np.array([r.index(g) + 1 if g in r else np.inf for r, g in zip(rankings, golds)])


def summarize(ranks, ks=(1, 3, 5, 10)):
    out = {f"hit@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr@10"] = float(np.where(ranks <= 10, 1 / ranks, 0).mean())
    return out


def bootstrap_ci(ranks, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(ranks[rng.integers(0, len(ranks), len(ranks))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
