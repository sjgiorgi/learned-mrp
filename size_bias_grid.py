# -*- coding: utf-8 -*-
"""
size_bias_grid.py
=================
The load-bearing experiment: sweep (sample size  x  bias severity) and run every
method in every cell against the known full-sample area ground truth.

Key design choices that keep the comparison honest:
  * ONE subsample per (size, bias, seed) cell, fed IDENTICALLY to every method.
    No method gets a different sample than another in the same cell.
  * Ground truth is the FULL-SAMPLE area mean (the standard MRP self-validation
    target). Methods are scored on recovering it from the biased subsample.
  * Random subsampling = size axis (representative, tests variance).
    Non-random subsampling = bias axis (skewed, tests selection correction).
  * Methods compared: naive mean, raking, real MRP (REML+EB fallback), and the
    DeepMRP ablation configs (classical / learned-MR / learned-P / full).

Inputs (person-level long dataframe `pop_df`):
    columns: area, <demographics...>, y
Plus per-area census poststratification frames (cell -> census_frac).

Outputs: a tidy results dataframe with columns
    [size, bias, seed, method, r, mse]
ready for a heatmap or a LaTeX table.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from mrp_baseline import MRPMixedLM, MRPCellMeans, raking_estimate
from deep_mrp import DeepMRP, ablation_config
from featurize import build_area_batches, infer_dims
from train_deepmrp import train_deepmrp, set_seed
from ps_nn_legacy import PS_NN, infer_num_factors, train_ps_nn_legacy
from hybrid_mrp import HybridMRP, train_hybrid_mrp


# --------------------------------------------------------------------------- #
#  Subsampling: the two axes
# --------------------------------------------------------------------------- #
def subsample(pop_df: pd.DataFrame, demographics: List[str],
              size_frac: float, bias_strength: float,
              rng: np.random.Generator,
              bias_target: Optional[Dict] = None) -> pd.DataFrame:
    """Draw a subsample.

    size_frac in (0,1]   : expected fraction of people kept (the SIZE axis).
    bias_strength >= 0   : 0 -> representative; larger -> stronger non-random
                           selection (the BIAS axis). Selection multiplies the
                           keep-prob for people matching `bias_target` cells.

    bias_target: dict like {'age_bin': [0,1], 'married':[1]} naming the
                 over-sampled categories. If None, a default young+married skew.
    """
    if bias_target is None:
        bias_target = {demographics[0]: "LOW"}  # oversample low values of dem 0

    p = np.full(len(pop_df), size_frac, dtype=float)

    if bias_strength > 0:
        # multiplicative skew: boost matching rows, suppress the rest, then
        # renormalise so the expected kept fraction stays ~ size_frac.
        boost = np.ones(len(pop_df))
        for dem, cats in bias_target.items():
            col = pop_df[dem].to_numpy()
            if cats == "LOW":
                thresh = np.quantile(col, 0.34)
                match = col <= thresh
            elif cats == "HIGH":
                thresh = np.quantile(col, 0.66)
                match = col >= thresh
            else:
                match = np.isin(col, cats)
            boost = boost * np.where(match, 1.0 + bias_strength, 1.0)
        # suppress non-matching to keep overall size ~ constant
        boost = boost / boost.mean()
        p = np.clip(size_frac * boost, 0, 1)

    keep = rng.random(len(pop_df)) < p
    return pop_df[keep].copy()


# --------------------------------------------------------------------------- #
#  Ground truth
# --------------------------------------------------------------------------- #
def area_ground_truth(pop_df: pd.DataFrame, y_col: str = "y") -> pd.Series:
    return pop_df.groupby("area")[y_col].mean()


# --------------------------------------------------------------------------- #
#  Method evaluators -- each returns area_id -> estimate
# --------------------------------------------------------------------------- #
def eval_naive(samp_df, areas, y_col="y"):
    m = samp_df.groupby("area")[y_col].mean()
    return np.array([m.get(a, np.nan) for a in areas])


def eval_raking(samp_df, areas, demographics, ps_frames, y_col="y"):
    out = []
    for a in areas:
        sub = samp_df[samp_df.area == a]
        if len(sub) == 0:
            out.append(np.nan); continue
        out.append(raking_estimate(sub, demographics, ps_frames[a], y_col=y_col))
    return np.array(out)


def eval_mrp(samp_df, areas, demographics, ps_frames, y_col="y"):
    samp_grand = samp_df[y_col].mean()
    area_off = {a: (samp_df[samp_df.area == a][y_col].mean() - samp_grand)
                for a in areas}
    mrp = MRPMixedLM(demographics, spatial_group="area",
                     area_random_effect=False).fit(samp_df, y_col=y_col)
    out = []
    for a in areas:
        off = area_off.get(a, 0.0)
        off = 0.0 if np.isnan(off) else off
        out.append(mrp.predict_area(ps_frames[a]) + off)
    return np.array(out)


# --------------------------------------------------------------------------- #
#  DeepMRP evaluator: needs featurised batches + a quick train on the subsample
# --------------------------------------------------------------------------- #
def _featurize_from_df(samp_df, areas, demographics, ps_frames, device,
                       y_col="y"):
    """Turn the long subsample into the per-area structures build_area_batches
    expects (for DeepMRP), plus the per-demographic (unconcatenated) legacy
    structures ps_nn_legacy.PS_NN expects -- both are built from the exact
    same per-dem census_frac / sample-count / one-hot ingredients computed in
    this one pass, so DeepMRP and the legacy PS_NN see identical inputs."""
    cats = {d: sorted(samp_df[d].unique()) for d in demographics}
    # build per-area structures
    user_scores, census_counts, samp_counts, user_embs, targets = [], [], [], [], []
    legacy_areas = []
    kept_areas = []
    for a in areas:
        sub = samp_df[samp_df.area == a]
        if len(sub) < 2:
            continue
        kept_areas.append(a)
        scores = sub[y_col].to_numpy()
        user_scores.append(scores)
        # per-dem one-hots for users, census frac, sample frac
        c_counts, s_counts, embs = [], [], []
        psf = ps_frames[a]
        for d in demographics:
            k = len(cats[d])
            idx = {c: i for i, c in enumerate(cats[d])}
            # user one-hot
            oh = np.zeros((len(sub), k))
            for i, v in enumerate(sub[d].to_numpy()):
                oh[i, idx[v]] = 1.0
            embs.append(oh)
            # census counts from ps frame (sum census_frac within each category)
            cc = np.zeros(k)
            for _, row in psf.iterrows():
                cc[idx[row[d]]] += row["census_frac"]
            c_counts.append(cc)
            # sample counts
            sc = oh.sum(0)
            s_counts.append(sc)
        census_counts.append(c_counts)
        samp_counts.append(s_counts)
        user_embs.append(embs)

        legacy_areas.append({
            "user_scores": torch.tensor(scores, dtype=torch.float32, device=device),
            "census_bin_percents": [torch.tensor(cc, dtype=torch.float32, device=device)
                                     for cc in c_counts],
            "twitter_bin_counts": [torch.tensor(sc, dtype=torch.float32, device=device)
                                    for sc in s_counts],
            "user_dem_embeddings": [torch.tensor(oh, dtype=torch.float32, device=device)
                                     for oh in embs],
        })

    batches = build_area_batches(user_scores, census_counts, samp_counts,
                                 user_embs, device, cell_space="marginal")
    return batches, legacy_areas, kept_areas


def _train_val_split(kept_areas, seed, val_frac=0.2):
    """Deterministic 80/20 split over an area list, given a seed. Factored out
    so run_grid can compute ONE split per (size, bias, seed) cell and hand it
    to every method -- guaranteeing naive/raking/mrp/DeepMRP/ps_nn_legacy are
    all scored on the EXACT same held-out areas, rather than naive/raking/mrp
    seeing all areas while the neural methods only see their 20% val split
    (an apples-to-oranges comparison, since differences in r could then come
    from which counties are in each method's test set, not just method
    quality)."""
    n = len(kept_areas)
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(2, int(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    held_out_areas = [kept_areas[i] for i in val_idx]
    return tr_idx, val_idx, held_out_areas


def _kfold_splits(kept_areas, seed, k):
    """k disjoint folds over kept_areas, given a seed for the fold assignment.
    Returns a list of k (tr_idx, val_idx) index-array pairs whose val_idx sets
    partition range(len(kept_areas)) exactly -- every area is held out in
    exactly one fold, unlike repeated Monte Carlo holdout (_train_val_split
    called independently per seed), which can leave some areas untested and
    others tested multiple times purely by chance. `seed` reproducibly fixes
    the fold assignment; running the same k with different seeds gives
    repeated k-fold CV."""
    n = len(kept_areas)
    perm = np.random.RandomState(seed).permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    splits = []
    start = 0
    for size in fold_sizes:
        val_idx = perm[start:start + size]
        tr_idx = np.concatenate([perm[:start], perm[start + size:]])
        splits.append((tr_idx, val_idx))
        start += size
    return splits


def eval_deepmrp(samp_df, full_truth, areas, demographics, ps_frames, device,
                 config_name="full_deep_mrp", seed=0, y_col="y", epochs=60,
                 precomputed=None, split=None, loss_type="pearson"):
    """precomputed: optional (batches, legacy_areas, kept) from a prior
    _featurize_from_df call, to avoid re-featurizing when scoring multiple
    methods on the same sample. split: optional (tr_idx, val_idx) to force a
    specific train/val split (e.g. one shared across methods in run_grid)
    instead of drawing a fresh one here. Both default to None so direct calls
    (e.g. from the standalone repro scripts) work unchanged."""
    set_seed(seed)
    if precomputed is not None:
        batches, _, kept = precomputed
    else:
        batches, _, kept = _featurize_from_df(samp_df, areas, demographics, ps_frames,
                                              device, y_col=y_col)
    if len(kept) < 6:
        return {a: np.nan for a in areas}, kept
    y = np.array([full_truth[a] for a in kept])
    marg_dim, _ = infer_dims(batches)

    if split is not None:
        tr_idx, val_idx = split
    else:
        n = len(kept)
        perm = np.random.RandomState(seed).permutation(n)
        n_val = max(2, int(0.2 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_b = [batches[i] for i in tr_idx]; tr_y = y[tr_idx]
    va_b = [batches[i] for i in val_idx]; va_y = y[val_idx]

    cfg = ablation_config(config_name)
    model = DeepMRP(marg_dim=marg_dim, cell_space="marginal",
                    hidden=32, depth=2, **cfg)
    model, _ = train_deepmrp(model, tr_b, tr_y, va_b, va_y, device,
                             epochs=epochs, patience=12, seed=seed,
                             loss_type=loss_type)
    model.eval()
    # score ONLY on the held-out val split -- tr_b's areas were fit directly
    # against their true full_truth value (the thing being scored against),
    # so including them here would be training-label leakage into the metric.
    # Matches the original PS_NN notebook, which reported test() on its
    # held-out 20% split, never on the training counties.
    with torch.no_grad():
        preds = model(va_b).cpu().numpy()
    held_out = [kept[i] for i in val_idx]
    est = {a: float(preds[i]) for i, a in enumerate(held_out)}
    return est, held_out


def eval_ps_nn_legacy(samp_df, full_truth, areas, demographics, ps_frames, device,
                      seed=0, y_col="y", epochs=100,
                      precomputed=None, split=None, loss_type="pearson"):
    """The literal original PS_NN model (ps_nn_legacy.py), not DeepMRP's
    approximation of it. Same train/val discipline as eval_deepmrp: gradients
    only touch the 80% train split, scoring only happens on the held-out 20%.
    precomputed/split: see eval_deepmrp -- lets run_grid share one
    featurization + one train/val split across every method in a cell."""
    set_seed(seed)
    if precomputed is not None:
        _, legacy_areas, kept = precomputed
    else:
        _, legacy_areas, kept = _featurize_from_df(samp_df, areas, demographics,
                                                   ps_frames, device, y_col=y_col)
    if len(kept) < 6:
        return {a: np.nan for a in areas}, kept
    y = np.array([full_truth[a] for a in kept])
    num_factors = infer_num_factors(legacy_areas)

    if split is not None:
        tr_idx, val_idx = split
    else:
        n = len(kept)
        perm = np.random.RandomState(seed).permutation(n)
        n_val = max(2, int(0.2 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_a = [legacy_areas[i] for i in tr_idx]; tr_y = y[tr_idx]
    va_a = [legacy_areas[i] for i in val_idx]; va_y = y[val_idx]

    model = PS_NN(num_factors=num_factors, num_dem=len(demographics))
    model, _ = train_ps_nn_legacy(model, tr_a, tr_y, va_a, va_y, device,
                                  epochs=epochs, patience=3, seed=seed,
                                  loss_type=loss_type)
    model.eval()
    with torch.no_grad():
        preds = model(va_a).cpu().numpy()
    held_out = [kept[i] for i in val_idx]
    est = {a: float(preds[i]) for i, a in enumerate(held_out)}
    return est, held_out


def eval_hybrid_mrp(samp_df, full_truth, areas, demographics, ps_frames, device,
                    seed=0, y_col="y", epochs=100,
                    precomputed=None, split=None, loss_type="pearson"):
    """DeepMRP's MultilevelRegression (MR) + ps_nn_legacy's informed-smoothing
    weighting (P), jointly trained -- see hybrid_mrp.py. Same held-out
    discipline and precomputed/split sharing as eval_deepmrp/eval_ps_nn_legacy."""
    set_seed(seed)
    if precomputed is not None:
        batches, legacy_areas, kept = precomputed
    else:
        batches, legacy_areas, kept = _featurize_from_df(samp_df, areas, demographics,
                                                          ps_frames, device, y_col=y_col)
    if len(kept) < 6:
        return {a: np.nan for a in areas}, kept
    y = np.array([full_truth[a] for a in kept])
    marg_dim, _ = infer_dims(batches)

    if split is not None:
        tr_idx, val_idx = split
    else:
        n = len(kept)
        perm = np.random.RandomState(seed).permutation(n)
        n_val = max(2, int(0.2 * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_b = [batches[i] for i in tr_idx]; tr_la = [legacy_areas[i] for i in tr_idx]; tr_y = y[tr_idx]
    va_b = [batches[i] for i in val_idx]; va_la = [legacy_areas[i] for i in val_idx]; va_y = y[val_idx]

    model = HybridMRP(marg_dim=marg_dim, num_dem=len(demographics))
    model, _ = train_hybrid_mrp(model, tr_b, tr_la, tr_y, va_b, va_la, va_y, device,
                                epochs=epochs, patience=12, seed=seed,
                                loss_type=loss_type)
    model.eval()
    with torch.no_grad():
        preds = model(va_b, va_la).cpu().numpy()
    held_out = [kept[i] for i in val_idx]
    est = {a: float(preds[i]) for i, a in enumerate(held_out)}
    return est, held_out


# --------------------------------------------------------------------------- #
#  The grid
# --------------------------------------------------------------------------- #
def run_grid(pop_df: pd.DataFrame, demographics: List[str],
             ps_frames: Dict, device: torch.device,
             sizes=(0.02, 0.05, 0.15, 0.5),
             biases=(0.0, 1.0, 3.0),
             seeds=(0, 1, 2),
             deepmrp_configs=("classical_mrp", "learned_mr",
                              "learned_p", "full_deep_mrp"),
             y_col="y", bias_target=None,
             include_classical=True, include_deep=True, include_legacy=True,
             include_hybrid=True,
             area_target: Optional[Dict] = None,
             loss_type: str = "pearson",
             cv_folds: Optional[int] = None) -> pd.DataFrame:
    """area_target: optional {area -> ground-truth value}. If given, methods are
    scored against THIS (e.g. true vote share from the outcome file -- the
    cross-construct case). If None, ground truth is the full-sample mean of y
    (the same-construct self-validation case).
    loss_type: "pearson" (what every actual notebook training call used) or
    "mse" (what the paper's Eq. 9 describes -- untested which one produced
    the reported numbers). Applies to both DeepMRP and ps_nn_legacy training.
    cv_folds: None (default) -> current behavior: ONE random 80/20 train/val
        split per (size, bias, seed) cell (Monte Carlo holdout), same as
        every run so far. Set to an int k >= 2 to switch to real k-fold CV
        instead: kept areas are partitioned into k disjoint folds, k
        independent models are trained (one per fold), and their held-out
        predictions are pooled so every area is scored exactly once -- no
        area left untested, none tested twice, unlike repeated random
        holdout. `seed` still controls the (reproducible) fold assignment,
        so multiple `seeds` with cv_folds set gives repeated k-fold CV. Costs
        roughly k times the compute of a single-split run, since k models get
        trained per cell instead of 1."""
    if area_target is not None:
        truth_arr = dict(area_target)
        areas = sorted(truth_arr.keys())
    else:
        truth = area_ground_truth(pop_df, y_col)
        areas = sorted(truth.index.tolist())
        truth_arr = truth.to_dict()

    records = []

    def score(name, est_arr, eval_areas, size, bias, seed):
        est_arr = np.asarray(est_arr, dtype=float)
        t = np.array([truth_arr[a] for a in eval_areas], dtype=float)
        mask = ~np.isnan(est_arr) & ~np.isnan(t)
        if mask.sum() < 5:
            return
        r = pearsonr(est_arr[mask], t[mask])[0]
        mse = mean_squared_error(t[mask], est_arr[mask])
        records.append(dict(size=size, bias=bias, seed=seed,
                            method=name, r=r, mse=mse, n=int(mask.sum())))

    needs_split = include_deep or include_legacy or include_hybrid

    for size in sizes:
        for bias in biases:
            for seed in seeds:
                rng = np.random.default_rng(1000 * seed + int(100 * bias))
                samp = subsample(pop_df, demographics, size, bias, rng,
                                 bias_target=bias_target)
                if len(samp) < 10:
                    continue

                # ONE featurization per cell, shared by every method that
                # needs one. Split mode depends on cv_folds:
                #   cv_folds=None -> ONE random 80/20 split (as before);
                #     naive/raking/mrp and the neural methods are all scored
                #     on that same held-out 20%.
                #   cv_folds=k -> k disjoint folds; each neural method is
                #     trained k times (once per fold) and its held-out
                #     predictions are pooled across folds so every area gets
                #     scored exactly once. naive/raking/mrp are scored on the
                #     full kept-area set (the union of all folds' held-out
                #     areas), matching the neural methods' full coverage.
                # Either way, every method in the cell sees the SAME
                # eval_areas -- not just "each independently happens to
                # compute the same split," and not naive/raking/mrp seeing
                # more/fewer areas than the neural methods (which would make
                # differences in r partly reflect which counties are in each
                # method's test set, not just method quality).
                precomputed = None
                eval_areas = areas
                deep_est = {cfg: {} for cfg in deepmrp_configs}
                legacy_est = {}
                hybrid_est = {}

                if needs_split:
                    precomputed = _featurize_from_df(samp, areas, demographics,
                                                      ps_frames, device, y_col=y_col)
                    kept = precomputed[2]
                    min_needed = max(6, 2 * cv_folds) if cv_folds else 6
                    if len(kept) >= min_needed:
                        if cv_folds is None:
                            tr_idx, val_idx, held_out_areas = _train_val_split(kept, seed)
                            fold_splits = [(tr_idx, val_idx)]
                            eval_areas = held_out_areas
                        else:
                            fold_splits = _kfold_splits(kept, seed, cv_folds)
                            eval_areas = kept

                        for fold_i, (tr_idx, val_idx) in enumerate(fold_splits):
                            # distinct weight-init seed per fold; the fold
                            # ASSIGNMENT itself is fixed by `seed` above, via
                            # _train_val_split/_kfold_splits.
                            fold_seed = seed * 1000 + fold_i

                            if include_deep:
                                for cfg in deepmrp_configs:
                                    est_map, _ = eval_deepmrp(
                                        samp, truth_arr, areas, demographics, ps_frames,
                                        device, config_name=cfg, seed=fold_seed, y_col=y_col,
                                        precomputed=precomputed, split=(tr_idx, val_idx),
                                        loss_type=loss_type)
                                    deep_est[cfg].update(est_map)

                            if include_legacy:
                                est_map, _ = eval_ps_nn_legacy(
                                    samp, truth_arr, areas, demographics, ps_frames,
                                    device, seed=fold_seed, y_col=y_col,
                                    precomputed=precomputed, split=(tr_idx, val_idx),
                                    loss_type=loss_type)
                                legacy_est.update(est_map)

                            if include_hybrid:
                                est_map, _ = eval_hybrid_mrp(
                                    samp, truth_arr, areas, demographics, ps_frames,
                                    device, seed=fold_seed, y_col=y_col,
                                    precomputed=precomputed, split=(tr_idx, val_idx),
                                    loss_type=loss_type)
                                hybrid_est.update(est_map)
                    else:
                        precomputed = None  # too few areas to train; fall back

                if include_classical:
                    score("naive", eval_naive(samp, eval_areas, y_col),
                          eval_areas, size, bias, seed)
                    score("raking",
                          eval_raking(samp, eval_areas, demographics, ps_frames, y_col),
                          eval_areas, size, bias, seed)
                    score("mrp",
                          eval_mrp(samp, eval_areas, demographics, ps_frames, y_col),
                          eval_areas, size, bias, seed)

                if include_deep and precomputed is not None:
                    for cfg in deepmrp_configs:
                        est_arr = [deep_est[cfg].get(a, np.nan) for a in eval_areas]
                        score(f"deep:{cfg}", est_arr, eval_areas, size, bias, seed)

                if include_legacy and precomputed is not None:
                    est_arr = [legacy_est.get(a, np.nan) for a in eval_areas]
                    score("ps_nn_legacy", est_arr, eval_areas, size, bias, seed)

                if include_hybrid and precomputed is not None:
                    est_arr = [hybrid_est.get(a, np.nan) for a in eval_areas]
                    score("hybrid_mrp", est_arr, eval_areas, size, bias, seed)

    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds -> mean r per (size, bias, method)."""
    return (df.groupby(["size", "bias", "method"])
              .agg(r_mean=("r", "mean"), r_std=("r", "std"),
                   mse_mean=("mse", "mean"))
              .reset_index())
