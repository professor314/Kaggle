import zipfile, csv, io, glob
z = zipfile.ZipFile(sorted(glob.glob("lb2/*.zip"))[-1])
n = z.namelist()[0]
rows = list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding="utf-8")))
sc = "Score" if "Score" in rows[0] else "score"
tn = "TeamName" if "TeamName" in rows[0] else "teamName"
print("total teams:", len(rows), "| top:", rows[0][sc], "| median:", rows[len(rows)//2][sc])
for i, r in enumerate(rows):
    if "onnolly" in r[tn].lower():
        print(f"OURS: rank {i+1}/{len(rows)}  score {r[sc]}")
for pct in (0.1, 0.25, 0.5):
    print(f"  p{int(pct*100)}: {rows[int(len(rows)*pct)][sc]}")
