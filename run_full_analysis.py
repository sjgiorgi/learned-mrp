# -*- coding: utf-8 -*-
"""The full analysis: size x bias sweep over the Gallup outcomes, scored with
real k-fold CV via run_grid's cv_folds option. Fully configurable via CLI
flags -- see `python3 run_full_analysis.py --help`.

Uses the CORRECTED user_table_redist.csv (calibrated age/income, min age 18,
matching the Gallup 18+ survey population) -- not user_table.csv, which has
uncalibrated demographics down to age 13 and was used for nothing past the
point this was discovered.

COST: this script prints an estimated total-trainings count before starting
based on your flags, so you can sanity-check before committing a remote
server to a long run. Roughly: n_cells = sizes x biases x outcomes x rounds;
n_trainings = n_cells x cv_folds x (n_deepmrp_configs + [1 if ps_nn_legacy]).
At ~3-5 min/training observed at these subsample sizes locally, a few
thousand trainings is a multi-day job single-threaded on CPU, less on GPU.

Checkpointing: results are written to disk immediately after each outcome
finishes (not just at the very end), so an interruption partway through
doesn't lose completed outcomes. Re-running only recomputes what's missing
is NOT implemented -- if interrupted, drop finished outcomes from --outcomes
before restarting, or accept recomputing everything.

Examples:
    # defaults: full sweep, all outcomes/demographics/methods, 10-fold CV
    python3 run_full_analysis.py

    # quick sanity check before committing to the full run
    python3 run_full_analysis.py --outcomes ladder --demographics age gen \\
        --cv-folds 3 --methods naive raking mrp full_deep_mrp ps_nn_legacy

    # repeated k-fold CV, MSE loss, only two DeepMRP configs
    python3 run_full_analysis.py --cv-folds 5 --rounds 3 --loss-type mse \\
        --methods naive raking mrp classical_mrp full_deep_mrp ps_nn_legacy
"""

from __future__ import annotations
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch

from adapter import build_dataset
from size_bias_grid import run_grid, summarize

DATA_DIR = "data"
OUT_DIR = "results"

DEFAULT_SIZES = (0.02, 0.05, 0.15, 0.5)
DEFAULT_BIASES = (0.0, 1.0, 3.0)

ALL_DEMOGRAPHICS = ["age", "inc", "gen", "edu"]
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

# CLI outcome name -> (person-level proxy column, area-level target column)
OUTCOME_LOOKUP = {
    "depression": ("DEP_SCORE", "depression"),
    "worry": ("ANX_SCORE", "worry"),
    "happy": ("SWB_HAPPY", "happy"),
    "ladder": ("SWB_LADDER", "ladder"),
}

DEEPMRP_METHOD_NAMES = ("classical_mrp", "learned_mr", "learned_p", "full_deep_mrp")
ALL_METHODS = ("naive", "raking", "mrp") + DEEPMRP_METHOD_NAMES + ("ps_nn_legacy", "hybrid_mrp")


def parse_args():
    p = argparse.ArgumentParser(
        description="Full size x bias x outcome analysis with configurable k-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sizes", nargs="+", type=float, default=list(DEFAULT_SIZES),
                    help="sample-size fractions to sweep. Use '--sizes 1.0' (with --biases 0.0) "
                         "for NO additional subsampling -- the full biased sample as-is, which is "
                         "what paper reproduction needs (the size/bias grid answers a different "
                         "question: how methods degrade under EXTRA sparsity/bias on top of what "
                         "the data already has)")
    p.add_argument("--biases", nargs="+", type=float, default=list(DEFAULT_BIASES),
                    help="selection-bias strengths to sweep. 0.0 = representative subsample, "
                         "no extra induced bias beyond what's already in the data")
    p.add_argument("--outcomes", nargs="+", choices=list(OUTCOME_LOOKUP), default=list(OUTCOME_LOOKUP),
                    help="which Gallup outcomes to run")
    p.add_argument("--demographics", nargs="+", choices=ALL_DEMOGRAPHICS, default=ALL_DEMOGRAPHICS,
                    help="which demographics to post-stratify on")
    p.add_argument("--methods", nargs="+", choices=list(ALL_METHODS), default=list(ALL_METHODS),
                    help="which methods to score. naive/raking/mrp are bundled as a group "
                         "(run_grid can't toggle them individually) -- including any one of "
                         "them runs all three.")
    p.add_argument("--cv-folds", type=int, default=10,
                    help="number of CV folds. 0 disables k-fold CV and falls back to a single "
                         "random 80/20 Monte Carlo holdout split per round (run_grid's original "
                         "behavior, cv_folds=None)")
    p.add_argument("--rounds", type=int, default=1,
                    help="number of repeated CV rounds (i.e. how many seeds). >1 with "
                         "--cv-folds>0 gives proper repeated k-fold CV")
    p.add_argument("--loss-type", choices=["pearson", "mse"], default="pearson",
                    help="training loss for DeepMRP and ps_nn_legacy. 'pearson' is what every "
                         "actual notebook training call used; 'mse' is what the paper's Eq. 9 "
                         "describes -- untested which one produced the reported numbers")
    return p.parse_args()


def load_real_data():
    user = pd.read_csv(f"{DATA_DIR}/user_table_redist.csv")
    outcome = pd.read_csv(f"{DATA_DIR}/gallup_outcomes.csv")
    census = pd.read_csv(f"{DATA_DIR}/census_acs2015_5yr_demographics.csv")

    user["cnty"] = user["cnty"].astype(str).str.zfill(5)
    outcome["cnty"] = outcome["cnty"].astype(str).str.zfill(5)
    census["cnty"] = census["cnty"].astype(str).str.zfill(5)

    return user.set_index("cnty"), outcome.set_index("cnty"), census.set_index("cnty")


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    outcome_pairs = [OUTCOME_LOOKUP[o] for o in args.outcomes]
    demographics = [d for d in ALL_DEMOGRAPHICS if d in args.demographics]  # stable order
    twitter_bins = {d: TWITTER_BINS[d] for d in demographics}
    census_table_cols = {d: CENSUS_TABLE_COLS[d] for d in demographics}

    include_classical = any(m in args.methods for m in ("naive", "raking", "mrp"))
    deepmrp_configs = tuple(m for m in DEEPMRP_METHOD_NAMES if m in args.methods)
    include_deep = len(deepmrp_configs) > 0
    include_legacy = "ps_nn_legacy" in args.methods
    include_hybrid = "hybrid_mrp" in args.methods

    cv_folds = args.cv_folds if args.cv_folds > 0 else None
    seeds = tuple(range(args.rounds))
    sizes = tuple(args.sizes)
    biases = tuple(args.biases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"sizes: {sizes}  biases: {biases}", flush=True)
    print(f"outcomes: {args.outcomes}", flush=True)
    print(f"demographics: {demographics}", flush=True)
    print(f"methods: classical={include_classical} (naive/raking/mrp) "
          f"deepmrp_configs={deepmrp_configs} ps_nn_legacy={include_legacy} "
          f"hybrid_mrp={include_hybrid}", flush=True)
    print(f"cv_folds={cv_folds}  rounds={args.rounds} (seeds={seeds})  "
          f"loss_type={args.loss_type}", flush=True)

    n_cells = len(sizes) * len(biases) * len(outcome_pairs) * len(seeds)
    n_neural_methods = len(deepmrp_configs) + (1 if include_legacy else 0) + (1 if include_hybrid else 0)
    effective_folds = cv_folds if cv_folds else 1
    n_trainings = n_cells * effective_folds * n_neural_methods
    print(f"grid: {len(sizes)} sizes x {len(biases)} biases x {len(outcome_pairs)} "
          f"outcomes x {len(seeds)} round(s) = {n_cells} cells "
          f"-> ~{n_trainings} total neural trainings", flush=True)

    print("loading CSVs...", flush=True)
    USER, OUTCOME, CENSUS = load_real_data()
    print(f"USER {USER.shape} | OUTCOME {OUTCOME.shape} | CENSUS {CENSUS.shape}", flush=True)

    all_dfs = []
    for person_outcome, area_target_col in outcome_pairs:
        t0 = time.time()
        print(f"\n{'='*70}\n{person_outcome} -> {area_target_col}\n{'='*70}", flush=True)

        pop_df, ps_frames, area_target, dem_cols = build_dataset(
            USER, OUTCOME, CENSUS, demographics, twitter_bins, census_table_cols,
            spatial_group="cnty", person_outcome=person_outcome,
            area_target_col=area_target_col,
        )
        print(f"pop_df: {pop_df.shape}, areas: {len(ps_frames)}, dem_cols: {dem_cols}",
              flush=True)

        df = run_grid(
            pop_df, dem_cols, ps_frames, device,
            sizes=sizes, biases=biases, seeds=seeds,
            deepmrp_configs=deepmrp_configs,
            include_classical=include_classical, include_deep=include_deep,
            include_legacy=include_legacy, include_hybrid=include_hybrid,
            cv_folds=cv_folds, loss_type=args.loss_type,
            area_target=area_target,
        )
        df["outcome"] = area_target_col

        # checkpoint immediately -- don't wait for all outcomes to finish
        out_path = f"{OUT_DIR}/{area_target_col}_raw.csv"
        df.to_csv(out_path, index=False)
        elapsed = (time.time() - t0) / 60
        print(f"saved {out_path}  ({len(df)} rows, {elapsed:.1f} min)", flush=True)
        print(summarize(df).pivot_table(index=["size", "bias"], columns="method",
                                         values="r_mean").round(3), flush=True)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(f"{OUT_DIR}/all_outcomes_raw.csv", index=False)
    print(f"\nsaved {OUT_DIR}/all_outcomes_raw.csv  ({len(combined)} rows total)")


if __name__ == "__main__":
    main()
