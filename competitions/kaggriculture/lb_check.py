import zipfile, csv, io

z = zipfile.ZipFile("lb/kaggriculture.zip")
n = z.namelist()[0]
rows = list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
print("columns:", list(rows[0].keys()))
sc = "Score" if "Score" in rows[0] else "score"
tn = "TeamName" if "TeamName" in rows[0] else "teamName"
print("total teams:", len(rows))
print("top:", rows[0][sc], "|", rows[0][tn])
print("median:", rows[len(rows) // 2][sc])
print("bottom:", rows[-1][sc])
for i, r in enumerate(rows):
    if "onnolly" in r[tn].lower() or r[sc] == "243.7":
        print(f"OURS: rank {i+1}/{len(rows)}  score {r[sc]}  team {r[tn]}")
# show the score at a few percentiles for context
for pct in (0.1, 0.25, 0.5, 0.75):
    print(f"  p{int(pct*100)} score: {rows[int(len(rows)*pct)][sc]}")
