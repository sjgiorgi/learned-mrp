# -*- coding: utf-8 -*-
"""End-to-end test of the size x bias grid. Synthetic world tuned so methods
SEPARATE: many areas, modest population per area, strong demographic->outcome
signal with interactions, so that under heavy subsampling + bias the naive mean
degrades and correction methods help."""

import numpy as np
import pandas as pd
import torch

from size_bias_grid import run_grid, summarize

rng = np.random.default_rng(1)
device = torch.device("cpu")

N_AREAS = 80
AGE = list(range(6)); GEN = [0, 1]; MARR = [0, 1]

true_age = rng.normal(0, 1.2, len(AGE))
true_gen = rng.normal(0, 0.6, len(GEN))
true_marr = rng.normal(0, 0.6, len(MARR))
true_inter = rng.normal(0, 1.0, (len(AGE), len(MARR)))
area_eff = rng.normal(0, 0.6, N_AREAS)

def cmean(a, g, m):
    return true_age[a] + true_gen[g] + true_marr[m] + true_inter[a, m]

rows = []
for area in range(N_AREAS):
    comp = rng.dirichlet(np.ones(len(AGE) * len(GEN) * len(MARR)) * 0.5)  # uneven
    pop = max(150, int(rng.normal(400, 100)))   # modest pop per area
    ids = rng.choice(len(comp), size=pop, p=comp)
    for cid in ids:
        a = cid // (len(GEN) * len(MARR)); rem = cid % (len(GEN) * len(MARR))
        g = rem // len(MARR); m = rem % len(MARR)
        y = cmean(a, g, m) + area_eff[area] + rng.normal(0, 1.0)
        rows.append((area, a, g, m, y))

pop_df = pd.DataFrame(rows, columns=["area", "age_bin", "gen", "married", "y"])
print(f"population: {len(pop_df)} people, {N_AREAS} areas, "
      f"~{len(pop_df)//N_AREAS}/area")

DEMS = ["age_bin", "gen", "married"]
ps_frames = {}
for area in range(N_AREAS):
    sub = pop_df[pop_df.area == area]
    frac = (sub.groupby(DEMS).size() / len(sub)).reset_index()
    frac.columns = DEMS + ["census_frac"]
    ps_frames[area] = frac

# run a small grid (keep it fast for the smoke test)
df = run_grid(
    pop_df, DEMS, ps_frames, device,
    sizes=(0.03, 0.15),
    biases=(0.0, 3.0),
    seeds=(0, 1),
    deepmrp_configs=("classical_mrp", "full_deep_mrp"),
    bias_target={"age_bin": "LOW", "married": [1]},
    y_col="y",
)

summ = summarize(df)
pd.set_option("display.width", 120)
print("\n=== size x bias grid (mean Pearson r over seeds) ===")
piv = summ.pivot_table(index=["size", "bias"], columns="method", values="r_mean")
print(piv.round(3))
print("\n(rows: data fraction x bias strength; higher r = better recovery)")
