"""Enveda CASMI26 - spectral nearest-neighbour baseline (CPU).

Task: for each test molecule (one or more MS/MS spectra), output up to 25 candidate SMILES,
';'-separated, best first.

Baseline: library search. Bin every MS/MS spectrum (0.01 Da bins up to 1000 m/z, sqrt
intensities, L2-normalised), take cosine similarity between each test spectrum and the
train spectra, and rank train SMILES by their best similarity to any spectrum of the test
molecule. A small bonus is added when the precursor m/z matches within 10 ppm and the adduct
is the same, since then the candidate has the right mass.

train.parquet is ~3 GB, so it is read in row-group batches with only the needed columns,
and capped at MAX_TRAIN spectra to stay within kernel RAM.
"""
import glob
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, vstack


def _find_input(slug):
    for p in (f"/kaggle/input/{slug}", f"/kaggle/input/competitions/{slug}"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/kaggle/input/**/{slug}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{slug} not found. Inputs: {glob.glob('/kaggle/input/*')}")


DATA = _find_input("enveda-CASMI26-molecule-id-mass-spectra")
OUT = "/kaggle/working"
BIN = 0.01
MAX_MZ = 1000.0
NBINS = int(MAX_MZ / BIN)
MAX_TRAIN = 600_000
TOPK = 25

pf = pq.ParquetFile(os.path.join(DATA, "train.parquet"))
print("train schema:", pf.schema_arrow)
names = pf.schema_arrow.names
smiles_col = next(c for c in names if "smiles" in c.lower())
cols = [c for c in ("ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", smiles_col) if c in names]
print("using", cols, "| rows", pf.metadata.num_rows, "| row groups", pf.num_row_groups)


def to_sparse(mzs_col, ints_col):
    rows, colidx, vals = [], [], []
    for i, (mz, it) in enumerate(zip(mzs_col, ints_col)):
        mz = np.asarray(mz, dtype=np.float64)
        it = np.sqrt(np.clip(np.asarray(it, dtype=np.float64), 0, None))
        keep = (mz > 0) & (mz < MAX_MZ)
        b = (mz[keep] / BIN).astype(np.int64)
        rows.append(np.full(len(b), i))
        colidx.append(b)
        vals.append(it[keep])
    n = len(mzs_col)
    if n == 0:
        return csr_matrix((0, NBINS))
    m = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(colidx))), shape=(n, NBINS))
    m.sum_duplicates()
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
    norms[norms == 0] = 1
    return csr_matrix(m.multiply(1 / norms[:, None]))


def parse_list(v):
    if isinstance(v, str):
        return [float(x) for x in v.strip("[]").replace(",", " ").split()]
    return v


mats, meta = [], []
seen = 0
for g in range(pf.num_row_groups):
    df = pf.read_row_group(g, columns=cols).to_pandas()
    df = df[df[smiles_col].notna()]
    if seen + len(df) > MAX_TRAIN:
        df = df.iloc[: MAX_TRAIN - seen]
    mz = df["ms2_mzs"].map(parse_list)
    it = df["ms2_normalized_intensities"].map(parse_list)
    mats.append(to_sparse(mz.tolist(), it.tolist()))
    meta.append(df[[c for c in ("precursor_mz", "adduct", smiles_col) if c in df.columns]].reset_index(drop=True))
    seen += len(df)
    if g % 10 == 0:
        print(f"row group {g}: {seen} spectra")
    if seen >= MAX_TRAIN:
        break

L = vstack(mats).tocsr()
lib = pd.concat(meta, ignore_index=True)
print("library", L.shape)

test = pd.read_parquet(os.path.join(DATA, "test.parquet"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
T = to_sparse(test["ms2_mzs"].map(parse_list).tolist(), test["ms2_normalized_intensities"].map(parse_list).tolist())

lib_mz = lib["precursor_mz"].to_numpy() if "precursor_mz" in lib else None
lib_add = lib["adduct"].astype(str).to_numpy() if "adduct" in lib else None
lib_smi = lib[smiles_col].astype(str).to_numpy()

best = {}  # molecule_id -> {smiles: score}
LT = L.T.tocsc()
for start in range(0, T.shape[0], 64):
    S = (T[start:start + 64] @ LT).toarray()
    for j in range(S.shape[0]):
        r = test.iloc[start + j]
        s = S[j]
        if lib_mz is not None and pd.notna(r["precursor_mz"]):
            ppm = np.abs(lib_mz - r["precursor_mz"]) / r["precursor_mz"] * 1e6
            s = s + 0.5 * ((ppm < 10) & (lib_add == str(r["adduct"])))
        top = np.argpartition(-s, 200)[:200]
        d = best.setdefault(r["molecule_id"], {})
        for k in top:
            smi = lib_smi[k]
            if s[k] > d.get(smi, -1):
                d[smi] = s[k]

fallback = sample["smiles"].iloc[0].split(";")[0]
out = []
for mid in sample["molecule_id"]:
    d = best.get(mid, {})
    ranked = [smi for smi, _ in sorted(d.items(), key=lambda kv: -kv[1])][:TOPK]
    if not ranked:
        ranked = [fallback]
    out.append(";".join(ranked))
sub = pd.DataFrame({"molecule_id": sample["molecule_id"], "smiles": out})
sub.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(sub.head(), sub.shape)
