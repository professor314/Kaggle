"""Verifiable Quantum Circuit Compilation - v1: package the official sample submission.

Zips the competition's sample_submission/ tree (invsqrt.kmb, dqi.kmb, dfthcsos/*) into
/kaggle/working/submission.zip with paths relative to sample_submission/. This is the
reference baseline; scoring is the geometric mean of C_ref / C_task over the three tasks.
"""
import glob, os, zipfile
from pathlib import Path

root = next(Path(p) for p in glob.glob("/kaggle/input/**/sample_submission", recursive=True) if os.path.isdir(p))
print("sample root:", root)
members = sorted(p for p in root.rglob("*") if p.is_file())
out = Path("/kaggle/working/submission.zip")
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in members:
        arc = p.relative_to(root).as_posix()
        zf.write(p, arc)
        print(f"  {arc}  {p.stat().st_size}")
with zipfile.ZipFile(out) as zf:
    assert zf.testzip() is None
print("wrote", out, out.stat().st_size, "bytes,", len(members), "members")
