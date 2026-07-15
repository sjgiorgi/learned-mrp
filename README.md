# Learned-MRP: full pipeline

A full learned MRP where BOTH the multilevel-regression (M/R) step and the
post-stratification (P) step are learnable, with every component independently
switchable to its classical counterpart for the 2x2 ablation. Includes a real
MRP baseline and the size x bias evaluation grid.

## Files
- `deep_mrp.py`      Full learned MRP. `ablation_config()` gives the 2x2:
                     classical_mrp / learned_mr / learned_p / full_deep_mrp
                     (+ ps_nn_legacy = your original weighted-average model).
- `mrp_baseline.py`  REAL MRP baseline (REML random intercepts + census
                     poststratification) with an empirical-Bayes fallback for
                     singular covariances, plus a raking baseline.
- `featurize.py`     Turns per-area structures into pre-tensorised AreaBatch
                     objects (removes the per-county Python loop).
- `train_deepmrp.py` Modern training loop (AdamW, cosine schedule, seeding,
                     early stopping).
- `size_bias_grid.py` The load-bearing experiment: sweep (sample size x bias
                     severity), run every method in every cell against known
                     ground truth, return a tidy results table. `area_target`
                     arg supports the cross-construct case (score against an
                     external target, e.g. true vote share).
- `adapter.py`       Bridges the notebook DataFrames (USER/OUTCOME/CENSUS) to
                     the harness: bins continuous demographics with TWITTER_BINS,
                     builds joint ps_frames by OUTER-PRODUCTING census marginals
                     (independence assumption -- no joint census frame exists),
                     and sources the area target from the outcome file.
- `test_adapter.py`  End-to-end test on synthetic data in the EXACT notebook
                     schema, for both same-construct and cross-construct targets.
- `test_grid.py`, `smoke_mrp.py`  Additional smoke tests.

## Running on your real data
```python
from adapter import build_dataset
from size_bias_grid import run_grid, summarize

pop_df, ps_frames, area_target, dem_cols = build_dataset(
    USER_FACTORS_AND_FEATURES_DF, COUNTY_OUTCOMES_DF, CENSUS_DEMOGRAPHICS_DF,
    demographics=DEMOGRAPHICS, twitter_bins=TWITTER_BINS,
    census_table_cols=CENSUS_TABLE_COLS, spatial_group=spatial_group,
    person_outcome='SWB_LADDER',     # or 'registered_repub' for voting
    area_target_col='ladder',        # or 'repub_perc' for voting
)
df = run_grid(pop_df, dem_cols, ps_frames, device,
              area_target=area_target)   # external target = cross-construct
print(summarize(df))
```

## Known findings (from synthetic tests -- expect on real data too)
1. Classical methods (naive/raking/MRP) tie outside the sparse-and-biased
   corner. Methods only separate when data per area is thin AND selection is
   severe. Push the grid into that corner or everything ties.
2. DeepMRP overfits the FEW area-level targets and can lose to a mean. The
   flexibility is a liability at the area-data scale. Mitigate with heavy
   regularization, fewer parameters, or moving capacity to the user level
   (train the per-user pieces on the millions of users, not the ~thousands of
   areas).
3. MixedLM goes singular on sparse crossed cells -> EB fallback handles it.
4. The cross-construct (voting) path scores against true vote share while the
   naive baseline = "registration share as vote share"; only the learned method
   can calibrate the proxy->target conversion.

## Caveat baked into the data
`ps_frames` assume demographic INDEPENDENCE (marginals outer-producted) because
no joint census table exists in this data. Note this in the paper; if a joint
frame becomes available, pass it directly instead of using build_ps_frame.
