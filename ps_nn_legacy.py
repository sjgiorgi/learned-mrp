# -*- coding: utf-8 -*-
"""
ps_nn_legacy.py
================
Faithful port of the ORIGINAL model from gallup_with_sampling_neurips.py (the
`PS_NN` class + `train_model`), kept as its own literal baseline rather than
folded into DeepMRP's `ps_nn_legacy` ablation config in deep_mrp.py -- that
config (mr_mode='identity', p_mode='learned') is only an *approximate*
reduction of this model: it uses bilinear attention and drops the original's
informed smoothing. This module is the actual original architecture, so it
can be scored as a real baseline rather than an analogy.

Kept faithful to what the notebook actually ran:
  * use_smoothing='multiple'  (one learned smoothing pair per demographic --
    used for the SWB_LADDER/SWB_HAPPY models; DEP/ANX used 'single', a minor
    variant not reproduced here)
  * combine='multi-hot', distance='l1'  (the only combination the notebook's
    four training calls actually used)

Architecture, unchanged:
  * 'informed smoothing': per-demographic Twitter bin percentages are pulled
    toward the census percentages by a LEARNED amount before comparison --
    softens the census-vs-sample mismatch signal for sparse cells.
  * user_features = |twitter_pct - census_pct| * user_one_hot  (masked L1
    distance between census and smoothed-sample composition, restricted to
    the user's own cell across all demographics)
  * a single Linear(num_factors, 1) maps that to a per-user logit
  * softmax over users in the county -> weights -> weighted mean of
    user_scores = the area estimate
"""
from __future__ import annotations
from copy import deepcopy
from typing import List, Dict, Optional, Sequence, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_deepmrp import set_seed, pearson_loss, _metrics


class PS_NN(nn.Module):
    def __init__(self, num_factors: int, num_dem: int):
        super().__init__()
        self.num_factors = num_factors
        self.num_dem = num_dem
        # 'multiple' informed smoothing: one learned smoothing pair per demographic
        self.smoothing = nn.Parameter(torch.full((num_dem,), 10.0))
        self.smoothing2 = nn.Parameter(torch.full((num_dem,), 10.0))
        self.fc1 = nn.Linear(num_factors, 1)

    def compute_weights(self, area: Dict) -> torch.Tensor:
        """The P step alone, for ONE county: informed smoothing -> masked L1
        distance -> fc1 -> softmax -> weights (u,). Factored out of forward()
        so other models (e.g. hybrid_mrp.py) can reuse this exact weighting
        mechanism with a different per-user prediction than raw user_scores.

        area: per-county dict with keys:
             census_bin_percents  [ (k_d,) per demographic, sums to 1 ]
             twitter_bin_counts   [ (k_d,) per demographic, RAW sample counts ]
             user_dem_embeddings  [ (u, k_d) one-hot per demographic ]
        """
        census_bin_percents = area["census_bin_percents"]
        twitter_bin_counts = area["twitter_bin_counts"]
        user_dems = area["user_dem_embeddings"]

        twitter_bin_percents = []
        for i in range(self.num_dem):
            sm = self.smoothing[i]
            sm2 = self.smoothing2[i]
            k_d = twitter_bin_counts[i].shape[0]
            denom = twitter_bin_counts[i].sum() + sm * k_d
            pct = (twitter_bin_counts[i] + sm * census_bin_percents[i]) / denom
            pct = torch.sigmoid(sm2 * pct)
            twitter_bin_percents.append(pct)

        census_encoding = torch.cat(census_bin_percents, dim=0)      # (num_factors,)
        twitter_encoding = torch.cat(twitter_bin_percents, dim=0)    # (num_factors,)
        user_encodings = torch.cat(user_dems, dim=1)                 # (u, num_factors)

        user_features = torch.abs(twitter_encoding - census_encoding) * user_encodings
        user_features = self.fc1(user_features).squeeze(-1)         # (u,)
        user_weights = F.softmax(user_features, dim=0) * user_features.shape[0]
        return user_weights

    def forward(self, legacy_areas: List[Dict]) -> torch.Tensor:
        """legacy_areas: list of per-county dicts with keys:
             user_scores          (u,) single-task, or (u, T) multitask
             census_bin_percents  [ (k_d,) per demographic, sums to 1 ]
             twitter_bin_counts   [ (k_d,) per demographic, RAW sample counts ]
             user_dem_embeddings  [ (u, k_d) one-hot per demographic ]

        Returns (n_areas,) single-task or (n_areas, T) multitask. Multitask
        shares ONE set of weights per county across all T tasks: the weights
        depend only on demographics/census, never on the outcome, so each
        task's column of user_scores is averaged with the same weights.
        """
        county_weighted_avgs = []
        for area in legacy_areas:
            user_weights = self.compute_weights(area)
            scores = area["user_scores"]
            w = user_weights[:, None] if scores.dim() == 2 else user_weights
            county_weighted_avg = (w * scores).sum(0) / user_weights.sum()
            county_weighted_avgs.append(county_weighted_avg)

        return torch.stack(county_weighted_avgs)


def infer_num_factors(legacy_areas: List[Dict]) -> int:
    return sum(t.shape[0] for t in legacy_areas[0]["census_bin_percents"])


def _loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Single-task: the plain loss. Multitask ((n, T) pred/target): the mean
    of the per-task losses, each over only the areas where that task's
    target is non-NaN (targets can have different area coverage)."""
    if loss_type == "pearson":
        fn = pearson_loss
    elif loss_type == "mse":
        fn = F.mse_loss
    else:
        raise ValueError(loss_type)
    if pred.dim() == 1:
        return fn(pred, target)
    losses = []
    for t in range(pred.shape[1]):
        m = ~torch.isnan(target[:, t])
        losses.append(fn(pred[m, t], target[m, t]))
    return torch.stack(losses).mean()


def _task_metrics(pred: np.ndarray, target: np.ndarray):
    """(r, mse) single-task; the mean over tasks of the NaN-masked per-task
    (r, mse) multitask -- one number each, so early stopping / LR decay pick
    the checkpoint that is best across all tasks together."""
    if pred.ndim == 1:
        return _metrics(pred, target)
    rs, mses = [], []
    for t in range(pred.shape[1]):
        m = ~np.isnan(target[:, t])
        r, mse = _metrics(pred[m, t], target[m, t])
        rs.append(r); mses.append(mse)
    return float(np.mean(rs)), float(np.mean(mses))


def train_ps_nn_legacy(
    model: PS_NN,
    train_areas: List[Dict], y_train: np.ndarray,
    val_areas: List[Dict], y_val: np.ndarray,
    device: torch.device,
    epochs: int = 100, lr: float = 0.5, patience: int = 3, decays: int = 1,
    loss_type: str = "pearson", seed: int = 0,
    train_tasks: Optional[Sequence[int]] = None,
) -> Tuple[PS_NN, dict]:
    """Matches notebook_NeurIPS.ipynb's train_model exactly: gradients only
    ever touch train_areas, val_areas only pick the best checkpoint -- AND
    the same 2-stage schedule: on `patience` epochs without val improvement,
    decay LR by 10x, reload the best checkpoint, and keep training, up to
    `decays` such cycles before finally stopping. lr=0.5 is the notebook's
    actual default (gallup_with_sampling_neurips.py had quietly dropped it
    to 0.1 and dropped this decay step -- a later, diverged copy).

    loss_type: "pearson" (what every actual training call in the notebook
    used) or "mse" (what the paper's Eq. 9 describes as the loss -- the code
    and the paper text disagree on this, and it's untested which one
    actually produced Table 2's numbers). Mirrors train_deepmrp's loss_type
    so DeepMRP and ps_nn_legacy can be compared under the same loss.

    Multitask: pass (n, T) y_train/y_val (NaN = missing target) alongside
    (u, T) user_scores. The loss is the mean of per-task losses and val_r /
    val_mse are means over tasks; with 1-D targets this is exactly the
    single-task path. train_tasks: optional column indices -- the loss AND
    checkpoint selection see only those tasks, so the others' targets are
    never touched in training (the model still outputs every column, which
    is what makes held-out-outcome transfer scoring possible)."""
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val = np.asarray(y_val)
    cols = list(train_tasks) if train_tasks is not None else None
    if cols is not None:
        yt, y_val = yt[:, cols], y_val[:, cols]

    best_val = -np.inf
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    total_decays = 0
    history = {"val_r": [], "val_mse": []}

    t_start = time.time()
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_areas)
        if cols is not None:
            pred = pred[:, cols]
        loss = _loss(pred, yt, loss_type)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = model(val_areas).cpu().numpy()
        if cols is not None:
            vp = vp[:, cols]
        val_r, val_mse = _task_metrics(vp, y_val)
        history["val_r"].append(val_r)
        history["val_mse"].append(val_mse)

        elapsed = time.time() - t_start
        print(f"  [ps_nn_legacy] epoch {ep + 1}/{epochs}  "
              f"loss={loss.item():.4f}  val_r={val_r:.4f}  val_mse={val_mse:.4f}  "
              f"best_val_r={best_val:.4f}  "
              f"elapsed={elapsed:.1f}s  ({elapsed / (ep + 1):.2f}s/epoch)",
              flush=True)

        if val_r > best_val:
            best_val = val_r
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                total_decays += 1
                if total_decays > decays:
                    print(f"  [ps_nn_legacy] early stop at epoch {ep + 1} "
                          f"(no improvement, out of decays)", flush=True)
                    break
                model.load_state_dict(best_state)
                for g in optimizer.param_groups:
                    g["lr"] /= 10
                no_improve = 0
                print(f"  [ps_nn_legacy] epoch {ep + 1}: no improvement, "
                      f"decaying lr to {optimizer.param_groups[0]['lr']:.2e} "
                      f"(decay {total_decays}/{decays})", flush=True)

    model.load_state_dict(best_state)
    print(f"  [ps_nn_legacy] done: best_val_r={best_val:.4f} "
          f"over {len(history['val_r'])} epochs", flush=True)
    return model, history
