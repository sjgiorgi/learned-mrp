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

Multitask: one person-level proxy (registered_repub) scored against TWO
area targets, 2020 and 2016 GOP block vote share. Three ps_nn_legacy models
share the same areas and split -- 2020-only, 2016-only, and both (shared
weights) -- and each is scored on both elections, so the 2020-only model's
2016 score (and vice versa) is zero-shot transfer: do weights tuned on one
election correct the sample for the other?
"""

from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize
from run_full_analysis import multitask_tables

DATA_DIR = "election_data"
OUT_DIR = "results"
SPATIAL_GROUP = "GEOID20"

PERSON_OUTCOME = "registered_repub"
TARGETS = ["repub2020_perc_block", "repub2016_perc_block"]
TASKS = [(PERSON_OUTCOME, t) for t in TARGETS]

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
               PERSON_OUTCOME]].set_index(SPATIAL_GROUP)
    user = user.rename(columns={"age_int": "age", "gender_int": "gen",
                                 "married_int": "married"})

    area_cols = ([SPATIAL_GROUP] + CENSUS_TABLE_COLS["age"]
                 + CENSUS_TABLE_COLS["gen"] + CENSUS_TABLE_COLS["married"]
                 + TARGETS)
    area = (df[area_cols].drop_duplicates(subset=SPATIAL_GROUP)
            .set_index(SPATIAL_GROUP))

    census = area[CENSUS_TABLE_COLS["age"] + CENSUS_TABLE_COLS["gen"]
                  + CENSUS_TABLE_COLS["married"]]
    outcome = area[TARGETS]

    return user, outcome, census


def main():
    # CPU, not CUDA: ps_nn_legacy's per-area Python loop is kernel-launch
    # bound on tiny tensors, so a GPU is slower (see run_full_analysis --device)
    device = torch.device("cpu")
    print(f"device: {device}")
    print("loading CSVs...")
    USER, OUTCOME, CENSUS = load_real_data()
    print(f"USER {USER.shape} | OUTCOME {OUTCOME.shape} | CENSUS {CENSUS.shape}")
    print(f"blocks in common: "
          f"{len(set(USER.index) & set(OUTCOME.index) & set(CENSUS.index))}")

    t0 = time.time()
    pop_df, ps_frames, area_target, dem_cols = build_dataset(
        USER, OUTCOME, CENSUS, DEMOGRAPHICS, TWITTER_BINS, CENSUS_TABLE_COLS,
        spatial_group=SPATIAL_GROUP, tasks=TASKS,
    )
    print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem cols: {dem_cols}")

    # ONE pass, full data, no subsampling, no injected selection bias, no
    # search grid -- size=1.0/bias=0.0 is a single cell, not a sweep. Only
    # naive/raking/mrp + ps_nn_legacy run; DeepMRP configs and hybrid_mrp
    # stay off (single-task only). cv_folds=None -> ONE random 80/20 split,
    # and every method is scored on the same held-out 20%: at ~1 min/epoch
    # over ~270k blocks, 10-fold CV x 3 models (up to 30 trainings) doesn't
    # fit the time budget; 3 trainings does.
    df = run_grid(
        pop_df, dem_cols, ps_frames, device,
        sizes=(1.0,), biases=(0.0,), seeds=(0,),
        include_classical=True, include_deep=False,
        include_legacy=True, include_hybrid=False,
        cv_folds=None,
        area_target=area_target, tasks=TASKS,
        legacy_train_sets=[TARGETS] + [[t] for t in TARGETS],
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/florida_multitask_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"saved {out_path}  ({len(df)} rows, {(time.time() - t0) / 60:.1f} min)")
    print(df[["method", "task", "transfer", "r", "mse", "n"]]
          .sort_values(["task", "method"]).to_string(index=False))
    print()
    print(multitask_tables(summarize(df), TARGETS))


if __name__ == "__main__":
    main()
