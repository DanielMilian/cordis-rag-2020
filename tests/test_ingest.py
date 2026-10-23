from src.cordis_rag.chunking import chunk_text
from src.cordis_rag.csv_io import read_cordis_csv


def test_chunk_overlap_and_coverage():
    t = " ".join(str(i) for i in range(500))
    ch = chunk_text(t, 100, 20)
    assert all(len(c.split()) <= 100 for c in ch)
    assert ch[1].split()[:20] == ch[0].split()[-20:]
    assert set(" ".join(ch).split()) == set(t.split())


def test_unescaped_quote_row(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text('"id";"objective"\n"1";"normal; text"\n"2";""Starts with quote; has semis"\n',
                 encoding="utf-8")
    df = read_cordis_csv(p)
    assert len(df) == 2
    assert df.loc[1, "objective"].startswith("Starts with quote")
