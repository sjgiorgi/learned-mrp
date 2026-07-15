# -*- coding: utf-8 -*-
"""
featurize.py
============
Bridges the existing notebook data structures to the new DeepMRP module.

It consumes the *same* objects your current `load_data` already returns
(user_ids, user_features, census_bin_counts, twitter_bin_counts,
user_dem_embeddings, county_outcomes) and turns each area into a pre-tensorised
`AreaBatch`. Crucially, all tensor creation happens ONCE here, not inside the
training loop -- removing the per-area `torch.tensor(...)` reallocation that the
old PS_NN.forward did on every epoch.
"""

from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import torch

from deep_mrp import AreaBatch


def _multi_hot_census(census_bin_counts: Sequence[np.ndarray]) -> np.ndarray:
    """[ (k1,), (k2,), ... ] counts -> concatenated *fractions* (sum-1 per dem)."""
    parts = []
    for c in census_bin_counts:
        c = np.asarray(c, dtype=np.float64)
        s = c.sum()
        parts.append(c / s if s > 0 else c)
    return np.concatenate(parts, axis=0)


def _multi_hot_sample(twitter_bin_counts: Sequence[np.ndarray]) -> np.ndarray:
    parts = []
    for c in twitter_bin_counts:
        c = np.asarray(c, dtype=np.float64)
        s = c.sum()
        parts.append(c / s if s > 0 else c)
    return np.concatenate(parts, axis=0)


def _multi_hot_users(user_dem_embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """[ (u,k1), (u,k2), ... ] -> (u, K) concatenated one/multi-hot."""
    return np.concatenate(user_dem_embeddings, axis=1).astype(np.float64)


def _crossed_users(user_dem_embeddings: Sequence[np.ndarray]):
    """Outer-product the per-dem one-hots into a single crossed one-hot (u, C)
    and the integer cell id per user. Done in numpy once, per area."""
    enc = user_dem_embeddings[0].astype(np.float64)             # (u, k0)
    for e in user_dem_embeddings[1:]:
        u = enc.shape[0]
        enc = np.einsum("ui,uj->uij", enc, e.astype(np.float64)).reshape(u, -1)
    cell_index = enc.argmax(axis=1).astype(np.int64)            # (u,)
    return enc, cell_index


def _crossed_census(census_bin_counts: Sequence[np.ndarray]) -> np.ndarray:
    """Outer-product census marginals into crossed-cell fractions (C,).
    NOTE: this assumes independence across demographics for the crossed census
    frame -- the standard assumption when a full joint frame is unavailable.
    If you HAVE a joint census table, pass it in directly instead."""
    frac = []
    for c in census_bin_counts:
        c = np.asarray(c, dtype=np.float64)
        s = c.sum()
        frac.append(c / s if s > 0 else c)
    cross = frac[0]
    for f in frac[1:]:
        cross = np.outer(cross, f).ravel()
    return cross


def build_area_batches(
    user_scores_per_area: Sequence[np.ndarray],
    census_bin_counts: Sequence[Sequence[np.ndarray]],
    twitter_bin_counts: Sequence[Sequence[np.ndarray]],
    user_dem_embeddings: Sequence[Sequence[np.ndarray]],
    device: torch.device,
    cell_space: str = "marginal",
    reliability_per_area: Optional[Sequence[np.ndarray]] = None,
) -> List[AreaBatch]:
    """Returns a list of AreaBatch, one per area, all tensors on `device`.

    Inputs mirror the existing notebook structures:
      user_scores_per_area : list over areas of (u,) arrays   (e.g. user_features['SWB_LADDER'])
      census_bin_counts    : list over areas of [ (k_d,) per dem ]
      twitter_bin_counts   : list over areas of [ (k_d,) per dem ]
      user_dem_embeddings  : list over areas of [ (u,k_d) per dem ]
      reliability_per_area : optional list over areas of (u,) outcome-independent
                             reliability (e.g. log #tweets). If None -> ones.
    """
    batches: List[AreaBatch] = []
    n_areas = len(user_scores_per_area)

    for j in range(n_areas):
        scores = np.asarray(user_scores_per_area[j], dtype=np.float64)
        u = scores.shape[0]

        census_marg = _multi_hot_census(census_bin_counts[j])
        sample_marg = _multi_hot_sample(twitter_bin_counts[j])
        user_marg = _multi_hot_users(user_dem_embeddings[j])

        if reliability_per_area is not None:
            rel = np.asarray(reliability_per_area[j], dtype=np.float64)
        else:
            rel = np.ones(u, dtype=np.float64)

        kw = dict(
            user_scores=torch.tensor(scores, dtype=torch.float32, device=device),
            user_cell_marg=torch.tensor(user_marg, dtype=torch.float32, device=device),
            census_marg=torch.tensor(census_marg, dtype=torch.float32, device=device),
            sample_marg=torch.tensor(sample_marg, dtype=torch.float32, device=device),
            user_reliability=torch.tensor(rel, dtype=torch.float32, device=device),
        )

        if cell_space == "crossed":
            user_cross, cell_index = _crossed_users(user_dem_embeddings[j])
            census_cross = _crossed_census(census_bin_counts[j])
            kw.update(
                user_cell_cross=torch.tensor(user_cross, dtype=torch.float32, device=device),
                census_cross=torch.tensor(census_cross, dtype=torch.float32, device=device),
                cell_index=torch.tensor(cell_index, dtype=torch.long, device=device),
            )

        batches.append(AreaBatch(**kw))

    return batches


def infer_dims(batches: List[AreaBatch]):
    """Returns (marg_dim, cross_dim or None) from a built batch list."""
    marg_dim = batches[0].user_cell_marg.shape[1]
    cross_dim = (batches[0].user_cell_cross.shape[1]
                 if batches[0].user_cell_cross is not None else None)
    return marg_dim, cross_dim
