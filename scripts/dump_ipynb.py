"""Print an .ipynb's cells as plain text (ASCII-safe for the Windows console)."""
import json
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
for i, c in enumerate(nb["cells"]):
    print(f"----- [{i}] {c['cell_type']}")
    print("".join(c["source"]).encode("ascii", "replace").decode()[:limit])
