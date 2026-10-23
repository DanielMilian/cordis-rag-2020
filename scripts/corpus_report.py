import pandas as pd

df = pd.read_parquet("data/processed/projects.parquet")
w = df["objective"].str.split().str.len()
print("words per abstract:\n", w.describe(percentiles=[.1, .5, .9]))
print("\n% abstracts > 200 words:", round((w > 200).mean() * 100, 1))
print("% abstracts > 400 words:", round((w > 400).mean() * 100, 1))
print("\nstatus:\n", df["status"].value_counts())
print("\nfundingScheme (top 10):\n", df["fundingScheme"].value_counts().head(10))
print("\nkeywords missing:", df["keywords"].isna().mean().round(3))
print("\nnear-dup check (same first 60 chars):",
      df["objective"].str[:60].duplicated().sum())
