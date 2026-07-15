# -*- coding: utf-8 -*-
"""Smoke test for the real MRP baseline on synthetic data shaped like the
voting/Gallup setup: people in areas, demographic cells, a person-level outcome
with cell + area structure, and a known area-level ground truth (full-sample
mean). We then subsample non-randomly (manufacture selection bias) and check
that MRP recovers the full-sample area values better than the naive subsample
mean."""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from mrp_baseline import MRPMixedLM, MRPCellMeans, raking_estimate

rng = np.random.default_rng(0)

# --- synthetic world -------------------------------------------------------- #
N_AREAS = 120
DEMS = ["age_bin", "gen", "married"]
AGE_BINS = list(range(6))         # 6 age bins
GEN = [0, 1]
MARRIED = [0, 1]

# true cell effects (with interactions -> nonlinear in the marginal sense)
true_age = rng.normal(0, 1.0, size=len(AGE_BINS))
true_gen = rng.normal(0, 0.5, size=len(GEN))
true_marr = rng.normal(0, 0.5, size=len(MARRIED))
# an interaction term age x married (the thing a linear HLM misses)
true_inter = rng.normal(0, 0.8, size=(len(AGE_BINS), len(MARRIED)))

def cell_mean(a, g, m):
    return true_age[a] + true_gen[g] + true_marr[m] + true_inter[a, m]

# per-area random intercept
area_effect = rng.normal(0, 0.7, size=N_AREAS)

rows = []
area_truth = {}
for area in range(N_AREAS):
    # population composition for this area (census fractions over the 6x2x2=24 cells)
    comp = rng.dirichlet(np.ones(len(AGE_BINS) * len(GEN) * len(MARRIED)))
    pop = max(300, int(rng.normal(1500, 400)))
    # draw the "population" people
    cell_ids = rng.choice(len(comp), size=pop, p=comp)
    truth_vals = []
    cells = []
    for cid in cell_ids:
        a = cid // (len(GEN) * len(MARRIED))
        rem = cid % (len(GEN) * len(MARRIED))
        g = rem // len(MARRIED)
        m = rem % len(MARRIED)
        y = cell_mean(a, g, m) + area_effect[area] + rng.normal(0, 1.0)
        truth_vals.append(y)
        cells.append((area, a, g, m, y))
    area_truth[area] = np.mean(truth_vals)   # FULL-SAMPLE area mean = ground truth
    rows.extend(cells)

pop_df = pd.DataFrame(rows, columns=["area", "age_bin", "gen", "married", "y"])
print(f"population: {len(pop_df)} people across {N_AREAS} areas")

# census poststratification frames: true area cell fractions
ps_frames = {}
for area in range(N_AREAS):
    sub = pop_df[pop_df.area == area]
    frac = (sub.groupby(["age_bin", "gen", "married"]).size() / len(sub)).reset_index()
    frac.columns = ["age_bin", "gen", "married", "census_frac"]
    ps_frames[area] = frac

# --- manufacture NON-RANDOM selection bias (oversample young + married) ------ #
def selection_prob(row):
    p = 0.15
    if row.age_bin <= 1:      # young strongly oversampled
        p *= 4.0
    if row.age_bin >= 4:      # old strongly undersampled
        p *= 0.25
    if row.married == 1:      # married oversampled
        p *= 2.0
    return min(p, 1.0)

keep = pop_df.apply(lambda r: rng.random() < selection_prob(r), axis=1)
samp_df = pop_df[keep].copy()
print(f"biased sample: {len(samp_df)} people "
      f"({100*len(samp_df)/len(pop_df):.0f}% of population)")

# --- fit MRP on the biased sample, poststratify to census ------------------- #
# Real MRP carries area information (area-level predictors or area random
# effects); without it, poststratification only adjusts demographic mix and
# cannot recover between-area differences. We give the baselines an area offset
# equal to the biased-sample area mean minus the biased-sample grand mean -- the
# simplest honest area term -- so the comparison reflects MRP-as-practised.
samp_grand = samp_df["y"].mean()
area_offset = {a: (samp_df[samp_df.area == a]["y"].mean() - samp_grand)
               for a in sorted(area_truth.keys())}

mrp = MRPMixedLM(DEMS, spatial_group="area", area_random_effect=False)
mrp.fit(samp_df, y_col="y")
print("MixedLM fit failed?", mrp._fit_failed)

eb = MRPCellMeans(DEMS, spatial_group="area").fit(samp_df, y_col="y")

# --- evaluate recovery of full-sample area truth ---------------------------- #
areas = sorted(area_truth.keys())
truth = np.array([area_truth[a] for a in areas])

# naive: biased subsample area mean (no correction)
naive = np.array([samp_df[samp_df.area == a]["y"].mean() for a in areas])

# raking (no pooling)
rake = np.array([raking_estimate(samp_df[samp_df.area == a], DEMS,
                                 ps_frames[a], y_col="y") for a in areas])

# real MRP (REML pooling) + area offset
mrp_est = np.array([mrp.predict_area(ps_frames[a]) + area_offset[a] for a in areas])

# EB fallback + area offset
eb_est = np.array([eb.predict_area(ps_frames[a]) + area_offset[a] for a in areas])

def report(name, est):
    m = ~np.isnan(est)
    r = pearsonr(est[m], truth[m])[0]
    mse = mean_squared_error(truth[m], est[m])
    print(f"  {name:18s}  r = {r:.3f}   MSE = {mse:.3f}   (n={m.sum()})")

print("\nRecovery of full-sample area means from the BIASED sample:")
report("naive (no corr)", naive)
report("raking", rake)
report("MRP (REML)", mrp_est)
report("MRP (EB fallback)", eb_est)
