import sys 
import pandas as pd


if len(sys.argv) < 2:
    sys.exit("usage: python scripts/inspect_raw.py <path/to/project.csv>")


df = pd.read_csv(sys.argv[1], sep=None, engine="python", nrows=5, dtype=str)
print("COLUMNS:", df.columns.tolist())
pd.set_option("display.max_colwidth", 300)
print(df.iloc[0])
