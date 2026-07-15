# -*- coding: utf-8 -*-
"""
adapter.py
==========
Bridges the existing notebook DataFrames to the size_bias_grid / DeepMRP harness.

It consumes the THREE DataFrames exactly as the notebook loads them:
    USER_FACTORS_AND_FEATURES_DF  (indexed by spatial_group; raw continuous dems)
    COUNTY_OUTCOMES_DF            (indexed by spatial_group; area-level targets)
    CENSUS_DEMOGRAPHICS_DF        (indexed by spatial_group; MARGINAL census dists)

and produces:
    pop_df     : flat person-level frame [area, <dem>_bin..., y, <reliability?>]
    ps_frames  : {area -> DataFrame of (dem_bin cells, census_frac)}  joint frame
                 built by OUTER-PRODUCTING the census marginals (independence
                 assumption -- no joint census table exists in this data).
    area_target: {area -> ground-truth area value} sourced from COUNTY_OUTCOMES_DF
                 (this is what makes the VOTING task cross-construct: the target
                 is NOT the mean of the person-level proxy).

The demographic binning reuses the notebook's TWITTER_BINS so cells align with
the census columns named in CENSUS_TABLE_COLS.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import itertools
import numpy as np
import pandas as pd


def bin_users(user_df: pd.DataFrame, demographics: List[str],
              twitter_bins: Dict[str, list]) -> pd.DataFrame:
    """Add a <dem>_bin integer column for each demographic, using the notebook's
    bin boundaries. Bin index k corresponds to the k-th census column for that
    demographic, so cells align with the census marginals."""
    out = user_df.copy()
    for d in demographics:
        bins = twitter_bins[d]
        # right=False to match get_twitter_bin_counts' inclusive='left'
        out[f"{d}_bin"] = pd.cut(out[d], bins=bins, right=False, labels=False)
    return out


def census_marginals(census_row: pd.Series,
                     census_table_cols: Dict[str, list],
                     demographics: List[str]) -> Dict[str, np.ndarray]:
    """Return {dem -> fraction vector over that dem's bins} for one area.
    Census columns are percentages (or counts); we normalise each dem to sum 1."""
    out = {}
    for d in demographics:
        cols = census_table_cols[d]
        vals = census_row[cols].to_numpy(dtype=float)
        s = vals.sum()
        out[d] = vals / s if s > 0 else np.ones_like(vals) / len(vals)
    return out


def build_ps_frame(marg: Dict[str, np.ndarray],
                   demographics: List[str]) -> pd.DataFrame:
    """Outer-product the per-dem marginal fractions into a joint cell frame.
    Columns: <dem>_bin for each dem, plus census_frac. Assumes independence
    across demographics (the only option when no joint census table exists)."""
    bin_ranges = [range(len(marg[d])) for d in demographics]
    rows = []
    for combo in itertools.product(*bin_ranges):
        frac = 1.0
        for d, b in zip(demographics, combo):
            frac *= marg[d][b]
        row = {f"{d}_bin": b for d, b in zip(demographics, combo)}
        row["census_frac"] = frac
        rows.append(row)
    df = pd.DataFrame(rows)
    # numerical guard: renormalise to exactly 1
    df["census_frac"] = df["census_frac"] / df["census_frac"].sum()
    return df


def build_dataset(
    user_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    census_df: pd.DataFrame,
    demographics: List[str],
    twitter_bins: Dict[str, list],
    census_table_cols: Dict[str, list],
    spatial_group: str,
    person_outcome: str,          # e.g. 'SWB_LADDER' or 'registered_repub'
    area_target_col: str,         # e.g. 'ladder' or 'repub_perc'
    reliability_col: Optional[str] = None,
):
    """Returns (pop_df, ps_frames, area_target).

    person_outcome  : the person-level proxy column -> becomes pop_df['y'].
    area_target_col : the area-level ground-truth column in outcome_df. For
                      Gallup/Twitter this is the same construct as person_outcome
                      (e.g. SWB_LADDER -> ladder); for VOTING it differs
                      (registered_repub -> repub_perc), which is the
                      cross-construct case.
    """
    dem_bin_cols = [f"{d}_bin" for d in demographics]

    # --- person-level flat frame ---
    binned = bin_users(user_df.reset_index(), demographics, twitter_bins)
    keep_cols = [spatial_group] + dem_bin_cols + [person_outcome]
    if reliability_col is not None and reliability_col in binned.columns:
        keep_cols.append(reliability_col)
    pop = binned[keep_cols].dropna(subset=dem_bin_cols + [person_outcome]).copy()
    pop = pop.rename(columns={spatial_group: "area", person_outcome: "y"})
    pop[dem_bin_cols] = pop[dem_bin_cols].astype(int)

    # --- area target (ground truth), sourced from outcome_df ---
    # drop areas with a missing/NaN ground truth -- a NaN target poisons the
    # batched training loss (pearson_loss is computed jointly over the batch)
    # and crashes scoring, so it must be filtered here rather than downstream.
    target_col = outcome_df[area_target_col]
    area_target = target_col[target_col.notna()].to_dict()

    # --- ps_frames: only for areas present in BOTH user and census ---
    ps_frames = {}
    areas = sorted(set(pop["area"].unique())
                   & set(census_df.index)
                   & set(area_target.keys()))
    for a in areas:
        marg = census_marginals(census_df.loc[a], census_table_cols, demographics)
        ps_frames[a] = build_ps_frame(marg, demographics)

    # restrict pop and target to the common areas
    pop = pop[pop["area"].isin(areas)].copy()
    area_target = {a: area_target[a] for a in areas}

    return pop, ps_frames, area_target, dem_bin_cols
