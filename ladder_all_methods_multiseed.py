# -*- coding: utf-8 -*-
"""Multi-seed, all-methods comparison on the ladder outcome, full sample, all
4 demographics (age, inc, gen, edu), using the corrected user_table_redist.csv.
Reuses run_grid directly so naive/raking/mrp/classical_mrp/full_deep_mrp/
ps_nn_legacy are all scored the same way we've validated throughout.

Note: with size=1.0 (no subsampling), naive/raking/mrp are deterministic --
the sample composition doesn't change across seeds, so those three will be
identical every "seed." Only classical_mrp/full_deep_mrp/ps_nn_legacy vary
by seed (train/val county split + weight init), which is exactly what we
want to characterize.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize

DATA_DIR = "data"
SEEDS = (0, 1, 2, 3, 4, 5)

DEMOGRAPHICS = ["age", "inc", "gen", "edu"]
TWITTER_BINS = {
    "age": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
    "inc": [-np.inf, 10000, 15000, 25000, 35000, 50000, 75000, 100000, 150000, 200000, np.inf],
    "gen": [-np.inf, 0, np.inf],
    "edu": [0, 1, 5],
}
CENSUS_TABLE_COLS = {
    "age": ['total_15to19', 'total_20to24', 'total_25to29', 'total_30to34',
            'total_35to39', 'total_40to44', 'total_45to49', 'total_50to54',
            'total_55to59', 'total_60to64', 'total_65plus'],
    "inc": ['incomelt10k', 'income10kto14999', 'income15kto24999', 'income25kto34999',
            'income35kto49999', 'income50kto74999', 'income75kto99999',
            'income100kto149999', 'income150kto199999', 'incomegt200k'],
    "gen": ['male_perc', 'female_perc'],
    "edu": ['perc_high_school_or_higher', 'perc_bach_or_higher'],
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print("loading CSVs...", flush=True)
    user = pd.read_csv(f"{DATA_DIR}/user_table_redist.csv")
    outcome = pd.read_csv(f"{DATA_DIR}/gallup_outcomes.csv")
    census = pd.read_csv(f"{DATA_DIR}/census_acs2015_5yr_demographics.csv")
    user["cnty"] = user["cnty"].astype(str).str.zfill(5)
    outcome["cnty"] = outcome["cnty"].astype(str).str.zfill(5)
    census["cnty"] = census["cnty"].astype(str).str.zfill(5)
    user = user.set_index("cnty"); outcome = outcome.set_index("cnty"); census = census.set_index("cnty")

    pop_df, ps_frames, area_target, dem_cols = build_dataset(
        user, outcome, census, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
        spatial_group="cnty", person_outcome="SWB_LADDER", area_target_col="ladder",
    )
    print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem_cols: {dem_cols}", flush=True)

    df = run_grid(
        pop_df, dem_cols, ps_frames, device,
        sizes=(1.0,), biases=(0.0,), seeds=SEEDS,
        deepmrp_configs=("classical_mrp", "full_deep_mrp"),
        area_target=area_target,
    )
    df.to_csv("/tmp/ladder_all_methods_multiseed_raw.csv", index=False)
    print("\n=== per-seed r by method ===")
    piv = df.pivot_table(index="seed", columns="method", values="r")
    print(piv.round(3))

    print("\n=== summary across seeds (mean/std) ===")
    summ = df.groupby("method")["r"].agg(["mean", "std", "min", "max"])
    print(summ.round(3))
    print("\npaper's Table 2 life satisfaction learned post-strat: r = .526")


if __name__ == "__main__":
    main()
