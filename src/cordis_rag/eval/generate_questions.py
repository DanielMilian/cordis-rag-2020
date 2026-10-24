import json, os, re
from collections import Counter
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ["LLM_MODEL"]
TARGET = int(os.environ.get("N_CANDIDATES", 120))
SEED = 42
OUT, ATT = Path("eval_data/candidates.jsonl"), Path("data/processed/gen_attempted.txt")

STYLES = {
    "specific": "Ask about one concrete technical detail (a method, material, target group, quantity or goal) stated in the abstract.",
    "paraphrase": "Ask a conceptual question the way a researcher who does not know this project exists would search for it, using general terms and synonyms instead of the abstract's own wording.",
}
PROMPT = """You write evaluation questions for a retrieval system over EU research project abstracts.

Abstract:
\"\"\"{text}\"\"\"

Write ONE question that this abstract answers and that is specific enough that few other projects would answer it equally well. Style: {style}

Rules:
- The question must make sense to someone who has never seen this abstract. Never say "the project", "this study", "the proposal" or similar.
- Include the distinctive subject matter (the specific problem, material, organism, method or domain).
- One sentence, at most 25 words, a single question (no second "and how..." part).
- Do not mention the project name, acronym or title.
- Do not copy any run of 4 or more consecutive words from the abstract.
- Also give a reference answer of 1-2 sentences using only facts in the abstract.

Bad question (generic, refers to the project): "What technology does the project develop?"
Good question (self-contained, specific; invented example): "Which sensing approach has been proposed for detecting microplastics in drinking water without sample preparation?"

Return only JSON: {{"question": "...", "answer": "..."}}"""

SELF_REF = re.compile(r"\b(the|this|that|our)\s+(project|study|proposal|abstract|action|programme|grant|consortium|research project)\b", re.I)


def words(t): return re.findall(r"\w+", t.lower())
def ngrams(t, n=4):
    w = words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check(q, row):
    if len(words(q)) < 8:
        return "too_short"
    if SELF_REF.search(q):
        return "self_ref"
    if q.count("?") > 1 or len(words(q)) > 30 or re.search(r"\band\s+(what|how|why)\b", q, re.I):
        return "compound"
    if ngrams(q) & ngrams(row.objective):
        return "4gram_copy"
    if row.acronym and len(str(row.acronym)) >= 3 and str(row.acronym).lower() in q.lower():
        return "acronym"
    if row.title and row.title.lower() in q.lower():
        return "title"
    return None


def ask(row, style, why):
    for _ in range(2):
        r = client.chat.completions.create(
            model=MODEL, temperature=0.7, max_tokens=200,
            messages=[{"role": "user", "content": PROMPT.format(text=row.objective, style=STYLES[style])}])
        m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
        try:
            d = json.loads(m.group(0))
            q, a = d["question"].strip(), d["answer"].strip()
        except Exception:
            why["bad_json"] += 1
            continue
        bad = check(q, row)
        if bad is None:
            return {"question": q, "answer": a}
        why[bad] += 1
    return None


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/projects.parquet")
    elig = set(json.load(open("data/processed/eligible_ids.json")))
    pool = df[df["id"].isin(elig)].sample(frac=1, random_state=SEED)
    OUT.parent.mkdir(exist_ok=True)
    done = {json.loads(l)["project_id"] for l in OUT.open()} if OUT.exists() else set()
    tried = set(ATT.read_text().split()) if ATT.exists() else set()
    n, why = len(done), Counter()
    for row in pool.itertuples():
        if n >= TARGET:
            break
        if row.id in done or row.id in tried:
            continue
        style = list(STYLES)[n % 2]
        d = ask(row, style, why)
        with ATT.open("a") as f:
            f.write(row.id + "\n")
        if d is None:
            continue
        rec = {"qid": f"q{n:03d}", "project_id": row.id, "style": style, **d,
               "generator": MODEL, "keep": None}
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        print(n, style, "|", d["question"], flush=True)
    print(f"\n{n} candidates. rejected attempts by reason: {dict(why)}")
