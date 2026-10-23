import pandas as pd
from langdetect import detect, DetectorFactory
from src.cordis_rag.csv_io import read_cordis_csv

DetectorFactory.seed = 0
s = pd.read_parquet("data/processed/projects.parquet")
pop = read_cordis_csv("data/raw/cordis/project.csv")

print("\n== fundingScheme: population vs sample ==")
print(pd.concat([pop["fundingScheme"].value_counts(normalize=True).rename("pop"),
                 s["fundingScheme"].value_counts(normalize=True).rename("sample")],
                axis=1).round(3).head(8))

print("\n== near-dup prefixes ==")
k = s["objective"].str[:60]
d = s[k.duplicated(keep=False)].assign(prefix=k).sort_values("prefix")
print(d[["id", "acronym", "fundingScheme", "prefix"]].to_string())

print("\n== language (300 sampled) ==")
print(s["objective"].sample(300, random_state=0).map(detect).value_counts())
