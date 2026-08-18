# -*- coding: utf-8 -*-
"""Run the learned-MRP pipeline on Florida voter data: person-level party
registration predicts area-level 2020 GOP vote share (cross-construct, same
pattern as run_real_data.py's Gallup task).

Unlike the Gallup case, all three frames adapter.build_dataset wants (person
records, census marginals, area ground truth) live pre-joined in one table,
election_data/voting_data_florida.csv -- the block-level census percentages
and 2020 election results were already merged onto every voter row. This
script just splits that flat table back into the three frames build_dataset
expects.

Area unit: 2020 Census block (GEOID20). Demographics: age, gender, marital
status -- the only three with matching census marginal columns. Religion is
in the raw data but has no census marginal (the Census Bureau does not
collect religion), so it can't be a post-stratification variable here.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid

DATA_DIR = "election_data"
SPATIAL_GROUP = "GEOID20"

DEMOGRAPHICS = ["age", "gen", "married"]

# Bin edges must align with CENSUS_TABLE_COLS below: bin k -> census column k.
TWITTER_BINS = {
    "age": [18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 150],
    "gen": [-np.inf, 0.5, np.inf],       # gender_int: 0=male, 1=female
    "married": [-np.inf, 0.5, np.inf],   # married_int: 0=not married, 1=married
}
CENSUS_TABLE_COLS = {
    "age": [
        "vote_age_18_and_19_years_perc", "vote_age_20_to_24_years_perc",
        "vote_age_25_to_29_years_perc", "vote_age_30_to_34_years_perc",
        "vote_age_35_to_39_years_perc", "vote_age_40_to_44_years_perc",
        "vote_age_45_to_49_years_perc", "vote_age_50_to_54_years_perc",
        "vote_age_55_to_59_years_perc", "vote_age_60_and_64_years_perc",
        "vote_age_65_and_69_years_perc", "vote_age_70_to_74_years_perc",
        "vote_age_75_to_79_years_perc", "vote_age_80_to_84_years_perc",
        "vote_age_85_years_and_over_perc",
    ],
    "gen": ["gender_male_perc", "gender_female_perc"],
    "married": ["not_married_perc", "married_perc"],
}


def load_real_data():
    df = pd.read_csv(f"{DATA_DIR}/voting_data_florida.csv")
    df = df.dropna(subset=[SPATIAL_GROUP])
    df[SPATIAL_GROUP] = df[SPATIAL_GROUP].astype(np.int64)

    user = df[[SPATIAL_GROUP, "age_int", "gender_int", "married_int",
               "registered_repub"]].set_index(SPATIAL_GROUP)
    user = user.rename(columns={"age_int": "age", "gender_int": "gen",
                                 "married_int": "married"})

    area_cols = ([SPATIAL_GROUP] + CENSUS_TABLE_COLS["age"]
                 + CENSUS_TABLE_COLS["gen"] + CENSUS_TABLE_COLS["married"]
                 + ["repub2020_perc_block"])
    area = (df[area_cols].drop_duplicates(subset=SPATIAL_GROUP)
            .set_index(SPATIAL_GROUP))

    census = area[CENSUS_TABLE_COLS["age"] + CENSUS_TABLE_COLS["gen"]
                  + CENSUS_TABLE_COLS["married"]]
    outcome = area[["repub2020_perc_block"]].rename(
        columns={"repub2020_perc_block": "repub_perc"})

    return user, outcome, census


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("loading CSVs...")
    USER, OUTCOME, CENSUS = load_real_data()
    print(f"USER {USER.shape} | OUTCOME {OUTCOME.shape} | CENSUS {CENSUS.shape}")
    print(f"blocks in common: "
          f"{len(set(USER.index) & set(OUTCOME.index) & set(CENSUS.index))}")

    pop_df, ps_frames, area_target, dem_cols = build_dataset(
        USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
        spatial_group=SPATIAL_GROUP, person_outcome="registered_repub",
        area_target_col="repub_perc",
    )
    print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem cols: {dem_cols}")

    # ONE pass, full data, no subsampling, no injected selection bias, no
    # search grid -- size=1.0/bias=0.0 is a single cell, not a sweep. Only
    # naive/raking/mrp + ps_nn_legacy run; DeepMRP configs and hybrid_mrp
    # stay off. cv_folds=10 -> ps_nn_legacy trains 10 independent models (one
    # per fold) and every kept area gets scored exactly once, pooled across
    # folds, instead of only the 20% held out by a single random split.
    # naive/raking/mrp are scored on that same full pooled set, so every
    # method in the comparison sees the same areas (see run_grid's
    # `needs_split`/cv_folds docstring). Costs ~10x the ps_nn_legacy training
    # compute of a single split -- up to 100 epochs x 10 folds.
    df = run_grid(
        pop_df, dem_cols, ps_frames, device,
        sizes=(1.0,), biases=(0.0,), seeds=(0,),
        include_classical=True, include_deep=False,
        include_legacy=True, include_hybrid=False,
        cv_folds=10,
        area_target=area_target,
    )
    print(df[["method", "r", "mse", "n"]].sort_values("method")
          .to_string(index=False))


if __name__ == "__main__":
    main()
