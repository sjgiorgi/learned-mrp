# -*- coding: utf-8 -*-
"""Run the learned-MRP pipeline on all 4 Gallup outcomes in data/gallup_outcomes.csv,
each paired with its matching person-level proxy score in data/user_table.csv:

    DEP_SCORE  -> depression
    ANX_SCORE  -> worry
    SWB_HAPPY  -> happy
    SWB_LADDER -> ladder

Loads the 3 CSVs once and reuses them across outcomes. Deliberately a SMALL grid
(one size, one bias, one seed) per outcome for debugging -- not the full sweep.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize

DATA_DIR = "data"

DEMOGRAPHICS = ["inc", "edu"]
TWITTER_BINS = {
    "inc": [-np.inf, 10000, 15000, 25000, 35000, 50000, 75000, 100000, 150000, 200000, np.inf],
    "edu": [0, 1, 5],
}
CENSUS_TABLE_COLS = {
    "inc": ['incomelt10k', 'income10kto14999', 'income15kto24999', 'income25kto34999',
            'income35kto49999', 'income50kto74999', 'income75kto99999',
            'income100kto149999', 'income150kto199999', 'incomegt200k'],
    "edu": ['perc_high_school_or_higher', 'perc_bach_or_higher'],
}

OUTCOME_PAIRS = [
    ("DEP_SCORE", "depression"),
    ("ANX_SCORE", "worry"),
    ("SWB_HAPPY", "happy"),
    ("SWB_LADDER", "ladder"),
]


def load_real_data():
    user = pd.read_csv(f"{DATA_DIR}/user_table_redist.csv")
    outcome = pd.read_csv(f"{DATA_DIR}/gallup_outcomes.csv")
    census = pd.read_csv(f"{DATA_DIR}/census_acs2015_5yr_demographics.csv")

    # normalise FIPS to zero-padded 5-digit strings across all three sources
    user["cnty"] = user["cnty"].astype(str).str.zfill(5)
    outcome["cnty"] = outcome["cnty"].astype(str).str.zfill(5)
    census["cnty"] = census["cnty"].astype(str).str.zfill(5)

    return user.set_index("cnty"), outcome.set_index("cnty"), census.set_index("cnty")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("loading CSVs...")
    USER, OUTCOME, CENSUS = load_real_data()
    print(f"USER {USER.shape} | OUTCOME {OUTCOME.shape} | CENSUS {CENSUS.shape}")

    for person_outcome, area_target_col in OUTCOME_PAIRS:
        print(f"\n{'='*70}\n{person_outcome} -> {area_target_col}\n{'='*70}")
        pop_df, ps_frames, area_target, dem_cols = build_dataset(
            USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
            spatial_group="cnty", person_outcome=person_outcome,
            area_target_col=area_target_col,
        )
        print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}")

        # small grid: one size, one bias, one seed -- for debugging only
        df = run_grid(
            pop_df, dem_cols, ps_frames, device,
            sizes=(0.1,), biases=(0.0,), seeds=(0,),
            deepmrp_configs=("classical_mrp", "full_deep_mrp"),
            area_target=area_target,
        )
        print(summarize(df).pivot_table(index=["size", "bias"], columns="method",
                                         values="r_mean").round(3))


if __name__ == "__main__":
    main()
