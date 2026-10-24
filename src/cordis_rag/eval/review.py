import json, re, sys, textwrap
import pandas as pd

P = sys.argv[1] if len(sys.argv) > 1 else "eval_data/candidates.jsonl"
rows = [json.loads(l) for l in open(P)]
df = pd.read_parquet("data/processed/projects.parquet").set_index("id")
abst, acr = df["objective"], df["acronym"]


def flags(r):
    q, f = r["question"], []
    a = acr.get(r["project_id"])
    if isinstance(a, str) and len(a) >= 3 and a.lower() in q.lower():
        f.append(f"acronym '{a}' in question")
    if re.match(r"\s*how many\b", q, re.I):
        f.append("starts with 'How many' (often generic)")
    if len(q.split()) > 30:
        f.append("long / maybe compound")
    return f


def save():
    with open(P, "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)


for r in rows:
    if r["keep"] is not None:
        continue
    print("\n" + "=" * 80)
    print(textwrap.fill(abst[r["project_id"]], 100))
    print(f"\n[{r['qid']} / {r['style']}]\nQ: {r['question']}\nA: {r['answer']}")
    fl = flags(r)
    if fl:
        print("!! FLAGS:", "; ".join(fl))
    c = input("keep? y / n / e(dit question) / q(uit): ").strip().lower()
    if c == "q":
        break
    if c == "e":
        r["question"] = input("new question: ").strip()
        na = input("new answer (Enter = keep current): ").strip()
        if na:
            r["answer"] = na
        c = "y"
    r["keep"] = (c == "y")
    save()

save()
print("kept:", sum(1 for r in rows if r["keep"]), "/", len(rows), "| undecided:", sum(1 for r in rows if r["keep"] is None))
