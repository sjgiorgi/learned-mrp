# -*- coding: utf-8 -*-
"""Run the learned-MRP pipeline on the real Gallup/census/Twitter-user data in
data/. Mirrors test_adapter.py's schema (which was copied from the real
notebook constants) but points adapter.build_dataset at the actual CSVs.

Gotcha fixed here: gallup_outcomes.csv 'cnty' is a zero-padded 5-digit FIPS
string ("01001"), but census_acs2015_5yr_demographics.csv and user_table.csv
store 'cnty' as an unpadded int (1001). Without zero-padding all three to a
common 5-digit string, the area intersection in build_dataset silently drops
every county in states 01-09.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize

DATA_DIR = "data"

DEMOGRAPHICS = ["age", "gen"]
TWITTER_BINS = {
    "age": [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 150],
    "gen": [-np.inf, 0, np.inf],
}
CENSUS_TABLE_COLS = {
    "age": ['total_15to19', 'total_20to24', 'total_25to29', 'total_30to34',
            'total_35to39', 'total_40to44', 'total_45to49', 'total_50to54',
            'total_55to59', 'total_60to64', 'total_65plus'],
    "gen": ['male_perc', 'female_perc'],
}


def load_real_data():
    user = pd.read_csv(f"{DATA_DIR}/user_table_redist.csv")
    outcome = pd.read_csv(f"{DATA_DIR}/gallup_outcomes.csv")
    census = pd.read_csv(f"{DATA_DIR}/census_acs2015_5yr_demographics.csv")

    # normalise FIPS to zero-padded 5-digit strings across all three sources
    user["cnty"] = user["cnty"].astype(str).str.zfill(5)
    outcome["cnty"] = outcome["cnty"].astype(str).str.zfill(5)
    census["cnty"] = census["cnty"].astype(str).str.zfill(5)

    user = user.set_index("cnty")
    outcome = outcome.set_index("cnty")
    census = census.set_index("cnty")
    return user, outcome, census


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("loading CSVs...")
    USER, OUTCOME, CENSUS = load_real_data()
    print(f"USER {USER.shape} | OUTCOME {OUTCOME.shape} | CENSUS {CENSUS.shape}")
    print(f"counties in common: "
          f"{len(set(USER.index) & set(OUTCOME.index) & set(CENSUS.index))}")

    pop_df, ps_frames, area_target, dem_cols = build_dataset(
        USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
        spatial_group="cnty", person_outcome="SWB_LADDER",
        area_target_col="ladder",
    )
    print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem cols: {dem_cols}")

    # small/fast grid first -- proves the real-data path works end to end
    # before committing to the full sweep (2000+ areas x many configs is slow
    # on CPU).
    df = run_grid(
        pop_df, dem_cols, ps_frames, device,
        sizes=(0.05, 0.3), biases=(0.0, 3.0), seeds=(0,),
        deepmrp_configs=("classical_mrp", "full_deep_mrp"),
        area_target=area_target,
    )
    print(summarize(df).pivot_table(index=["size", "bias"], columns="method",
                                     values="r_mean").round(3))


if __name__ == "__main__":
    main()
