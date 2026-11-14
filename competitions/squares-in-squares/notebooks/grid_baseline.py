"""Squares in Squares - grid baseline + local score.

Score (from the public Square Packing Lab notebook's total_score): sum over n=1..1000 of
side(n)/sqrt(n) - 1, lower is better. side(n) = side of the smallest axis-aligned square
containing all n unit squares (x, y = centres, angle in degrees).

Baseline: k x k grid with k = ceil(sqrt(n)), filled row by row. Valid by construction
(no overlaps). This reproduces the sample submission; it is the floor to improve on.
"""
import math
import os

import numpy as np
import pandas as pd

OUT = "submissions"
os.makedirs(OUT, exist_ok=True)


def grid(n):
    k = math.ceil(math.sqrt(n))
    return [(i % k + 0.5, i // k + 0.5, 0.0) for i in range(n)]


def side(sq):
    """Bounding square side of unit squares given (x, y, angle_deg)."""
    pts = []
    for x, y, a in sq:
        t = math.radians(a)
        c, s = math.cos(t), math.sin(t)
        for dx, dy in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)):
            pts.append((x + c * dx - s * dy, y + s * dx + c * dy))
    p = np.array(pts)
    return float(max(p[:, 0].max() - p[:, 0].min(), p[:, 1].max() - p[:, 1].min()))


rows, total = [], 0.0
for n in range(1, 1001):
    sq = grid(n)
    total += side(sq) / math.sqrt(n) - 1
    rows += [(n, j, x, y, a) for j, (x, y, a) in enumerate(sq)]

sub = pd.DataFrame(rows, columns=["n", "square_id", "x", "y", "angle"])
sub.insert(0, "id", range(len(sub)))
sub.to_csv(os.path.join(OUT, "grid_baseline.csv"), index=False)
print(f"rows {len(sub)}  local score {total:.6f}")

sample = pd.read_csv("data/sample_submission.csv")
print("matches sample layout:", list(sample.columns) == list(sub.columns) and len(sample) == len(sub))
