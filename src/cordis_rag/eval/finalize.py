import glob, json, random

rows = [json.loads(l) for f in sorted(glob.glob("eval_data/review_*.jsonl")) for l in open(f)]
kept = [r for r in rows if r["keep"]]
assert len({r["project_id"] for r in kept}) == len(kept), "duplicate project in eval set"
random.Random(42).shuffle(kept)
cut = int(0.4 * len(kept))
dev, test = kept[:cut], kept[cut:]
for name, part in (("dev", dev), ("test", test)):
    with open(f"eval_data/eval_{name}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in part)
print(f"reviewed={len(rows)} kept={len(kept)} dev={len(dev)} test={len(test)}")
