import glob, json, re
import pandas as pd

df = pd.read_parquet("data/processed/projects.parquet").set_index("id")
SELF = re.compile(r"\b(the|this|that|our)\s+(project|study|proposal|abstract|action|programme|grant|consortium|researcher|authors?|team|system|platform|tool|sensor|device|bioprinter|network|conference)\b", re.I)


def words(t): return re.findall(r"\w+", t.lower())
def ngrams(t, n=4):
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


bad = 0
for f in sorted(glob.glob("eval_data/review_*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if not r["keep"]:
            continue
        p, q = df.loc[r["project_id"]], r["question"]
        fl = []
        shared = ngrams(q) & ngrams(p["objective"])
        if shared:
            fl.append("copied 4-gram: " + " ".join(sorted(shared)[0]))
        a = p["acronym"]
        if isinstance(a, str) and len(a) >= 3 and re.search(rf"\b{re.escape(a)}\b", q, re.I):
            fl.append(f"acronym {a}")
        if p["title"] and p["title"].lower() in q.lower():
            fl.append("title in question")
        if SELF.search(q):
            fl.append("self-reference")
        if re.match(r"\s*how many\b", q, re.I):
            fl.append("how-many")
        if fl:
            bad += 1
            print(r["qid"], "|", "; ".join(fl), "\n  Q:", q, "\n")
print("flagged:", bad)
