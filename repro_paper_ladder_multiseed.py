# -*- coding: utf-8 -*-
"""Multi-seed version of repro_paper_ladder.py. Each seed gets a different
80/20 county train/val split AND a different weight init (both driven by
`seed` in eval_ps_nn_legacy/train_ps_nn_legacy), mirroring the actual
notebook's train_test_split(..., random_state=None) -- a fresh random split
every run, not a fixed one. Reports the distribution of r across seeds so we
can judge whether the paper's r=.526 is a plausible single draw from this
port, or genuinely outside its range.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import eval_ps_nn_legacy, eval_naive
from scipy.stats import pearsonr

DATA_DIR = "data"
SEEDS = [0, 1, 2, 3, 4, 5]

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
    print(f"device: {device}")
    print("loading CSVs...")
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
    print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem_cols: {dem_cols}")

    areas = sorted(area_target.keys())
    t = np.array([area_target[a] for a in areas])
    naive_r = pearsonr(np.asarray(eval_naive(pop_df, areas, "y")), t)[0]
    print(f"naive (full sample, no split) r = {naive_r:.3f}")

    rs = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---", flush=True)
        est_map, held_out = eval_ps_nn_legacy(
            pop_df, area_target, areas, dem_cols, ps_frames, device,
            seed=seed, y_col="y", epochs=100,
        )
        est_arr = np.array([est_map[a] for a in held_out])
        truth_arr = np.array([area_target[a] for a in held_out])
        r = pearsonr(est_arr, truth_arr)[0]
        rs.append(r)
        print(f"seed {seed}: r = {r:.3f}  (n={len(held_out)})", flush=True)

    rs = np.array(rs)
    print(f"\n=== summary over {len(SEEDS)} seeds ===")
    print(f"per-seed r: {[round(x, 3) for x in rs]}")
    print(f"mean = {rs.mean():.3f}   std = {rs.std():.3f}   "
          f"min = {rs.min():.3f}   max = {rs.max():.3f}")
    print(f"paper's Table 2 life satisfaction learned post-strat: r = .526")


if __name__ == "__main__":
    main()
