# -*- coding: utf-8 -*-
"""End-to-end test using synthetic DataFrames in the EXACT notebook schema:
- USER df indexed by 'cnty', continuous age/inc/gen/edu + score columns
- OUTCOME df indexed by 'cnty' with area-level target columns
- CENSUS df indexed by 'cnty' with MARGINAL columns matching CENSUS_TABLE_COLS

Exercises the adapter (binning + outer-product ps_frames + external target) and
runs a tiny grid for BOTH a same-construct (Gallup-like) and a cross-construct
(voting-like) target."""

import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize

rng = np.random.default_rng(3)
device = torch.device("cpu")

# ---- notebook-style config (Gallup/Twitter) ------------------------------- #
DEMOGRAPHICS = ["age", "gen"]           # keep small for a fast test
TWITTER_BINS = {
    "age": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],   # 11 bins
    "gen": [-np.inf, 0, np.inf],                                # 2 bins
}
CENSUS_TABLE_COLS = {
    "age": ['total_15to19','total_20to24','total_25to29','total_30to34',
            'total_35to39','total_40to44','total_45to49','total_50to54',
            'total_55to59','total_60to64','total_65plus'],
    "gen": ['male_perc', 'female_perc'],
}

N_AREAS = 60
fips = [f"{12000+i}" for i in range(N_AREAS)]   # fake 5-digit-ish FIPS
fips = [int(f) for f in fips]

# true age-bin and gen effects on a person score
true_age = rng.normal(0, 1.0, 11)
true_gen = rng.normal(0, 0.6, 2)
area_eff = dict(zip(fips, rng.normal(0, 0.6, N_AREAS)))

user_rows = []
census_rows = []
outcome_rows = []
for f in fips:
    pop = max(120, int(rng.normal(300, 80)))
    # area age distribution over 11 bins, gender split
    age_dist = rng.dirichlet(np.ones(11) * 0.6)
    gen_male = rng.uniform(0.4, 0.6)
    # census marginals (as percentages, like the real file)
    census_rows.append({
        "cnty": f, "population": pop,
        **{c: 100*age_dist[i] for i, c in enumerate(CENSUS_TABLE_COLS["age"])},
        "male_perc": 100*gen_male, "female_perc": 100*(1-gen_male),
    })
    # draw users from a BIASED-ish area sample (younger overrepresented)
    samp_age_dist = age_dist * np.linspace(2.0, 0.5, 11)
    samp_age_dist /= samp_age_dist.sum()
    ages_bin = rng.choice(11, size=pop, p=samp_age_dist)
    # map bin -> a continuous age inside the bin
    bin_lo = TWITTER_BINS["age"][:-1]; bin_hi = TWITTER_BINS["age"][1:]
    score_vals = []
    for ab in ages_bin:
        lo, hi = bin_lo[ab], min(bin_hi[ab], 90)
        age_cont = rng.uniform(lo, hi)
        gen_cont = rng.normal(0.5 if rng.random() < gen_male else -0.5, 0.3)
        gb = 0 if gen_cont < 0 else 1
        score = true_age[ab] + true_gen[gb] + area_eff[f] + rng.normal(0, 1.0)
        user_rows.append({"cnty": f, "user_id": rng.integers(1e9),
                          "age": age_cont, "gen": gen_cont,
                          "inc": rng.normal(45000, 12000), "edu": rng.integers(0,2),
                          "SWB_LADDER": score, "registered_repub": float(score > 0)})
        score_vals.append(score)
    # area-level TRUE ladder = census-weighted true cell means (the GT)
    true_area = sum(age_dist[i]*true_age[i] for i in range(11)) \
                + gen_male*true_gen[1] + (1-gen_male)*true_gen[0] + area_eff[f]
    # voting target: a DIFFERENT construct -- nonlinear fn of the same world
    vote_share = 1/(1+np.exp(-(true_area - 0.0)))   # logistic of area latent
    outcome_rows.append({"cnty": f, "ladder": true_area, "repub_perc": vote_share})

USER = pd.DataFrame(user_rows).set_index("cnty")
CENSUS = pd.DataFrame(census_rows).set_index("cnty")
OUTCOME = pd.DataFrame(outcome_rows).set_index("cnty")
print("USER", USER.shape, "| CENSUS", CENSUS.shape, "| OUTCOME", OUTCOME.shape)

# ============ SAME-CONSTRUCT (Gallup-like): SWB_LADDER -> ladder ============ #
print("\n========== SAME-CONSTRUCT: SWB_LADDER -> ladder ==========")
pop_df, ps_frames, area_target, dem_bin_cols = build_dataset(
    USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
    spatial_group="cnty", person_outcome="SWB_LADDER", area_target_col="ladder",
)
print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, "
      f"dem cols: {dem_bin_cols}")
print("ps_frame example cells:", len(ps_frames[list(ps_frames)[0]]))

df = run_grid(pop_df, dem_bin_cols, ps_frames, device,
              sizes=(0.05, 0.3), biases=(0.0, 3.0), seeds=(0,),
              deepmrp_configs=("classical_mrp", "full_deep_mrp"),
              area_target=area_target)
print(summarize(df).pivot_table(index=["size","bias"], columns="method",
                                values="r_mean").round(3))

# ============ CROSS-CONSTRUCT (voting-like): registered_repub -> repub_perc = #
print("\n========== CROSS-CONSTRUCT: registered_repub -> repub_perc ==========")
pop_df2, ps_frames2, area_target2, dem_bin_cols2 = build_dataset(
    USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
    spatial_group="cnty", person_outcome="registered_repub",
    area_target_col="repub_perc",
)
df2 = run_grid(pop_df2, dem_bin_cols2, ps_frames2, device,
               sizes=(0.05, 0.3), biases=(0.0, 3.0), seeds=(0,),
               deepmrp_configs=("classical_mrp", "full_deep_mrp"),
               area_target=area_target2)
print(summarize(df2).pivot_table(index=["size","bias"], columns="method",
                                 values="r_mean").round(3))
print("\n(cross-construct: naive = 'registration share as vote share';"
      " deep can learn the proxy->target conversion)")
