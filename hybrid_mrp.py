# -*- coding: utf-8 -*-
"""
hybrid_mrp.py
=============
Combines DeepMRP's MR step (MultilevelRegression: deep MLP + partial pooling)
with ps_nn_legacy's P step (informed smoothing + a single Linear + softmax),
instead of DeepMRP's own bilinear-attention P step.

Why: full_deep_mrp and ps_nn_legacy differ in BOTH their MR step and their P
step at once, so a win or loss for either one doesn't tell you which piece is
responsible. ps_nn_legacy's P step is lower-capacity (~26 params vs. ~1,600+
for DeepMRP's attention), includes learned informed smoothing (DeepMRP's
attention has none), and has empirically outperformed DeepMRP's learned-P
across most outcomes tested this session. Holding P fixed at that
already-validated mechanism and swapping in a genuinely learned, pooled MR
step in place of ps_nn_legacy's raw user_scores isolates one clean question:
does a learned per-user prediction beat the raw proxy score, given the best
known weighting mechanism? That's the question full_deep_mrp vs. ps_nn_legacy
couldn't cleanly answer on its own.

Both components are reused as-is (MultilevelRegression from deep_mrp.py,
PS_NN.compute_weights from ps_nn_legacy.py) -- this module only wires them
together and trains them jointly, end-to-end, on one loss.
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_mrp import AreaBatch, MultilevelRegression
from ps_nn_legacy import PS_NN
from train_deepmrp import set_seed, pearson_loss, _metrics


class HybridMRP(nn.Module):
    """MR = MultilevelRegression (deep, partially pooled); P = ps_nn_legacy's
    informed-smoothing + linear + softmax weighting. Jointly trained."""

    def __init__(self, marg_dim: int, num_dem: int, hidden: int = 32, depth: int = 2,
                 dropout: float = 0.1, pool: bool = True):
        super().__init__()
        self.mr = MultilevelRegression(marg_dim, hidden, depth, mode="deep",
                                       dropout=dropout, pool=pool)
        self.ps = PS_NN(num_factors=marg_dim, num_dem=num_dem)

    def forward_area(self, batch: AreaBatch, legacy_area: Dict) -> torch.Tensor:
        user_enc = batch.user_cell_marg                            # (u, K)
        # same evidence calc DeepMRP.forward_area uses for its pooling gate
        evidence = (user_enc * (user_enc.sum(0))).sum(-1)           # (u,)
        yhat = self.mr(user_enc, evidence)                          # (u,) learned per-user prediction

        w = self.ps.compute_weights(legacy_area)                   # (u,) ps_nn_legacy's weights

        area_est = (w * yhat).sum() / (w.sum() + 1e-9)
        return area_est

    def forward(self, batches: List[AreaBatch], legacy_areas: List[Dict]) -> torch.Tensor:
        ests = [self.forward_area(b, la) for b, la in zip(batches, legacy_areas)]
        return torch.stack(ests)


def train_hybrid_mrp(
    model: HybridMRP,
    train_batches: List[AreaBatch], train_legacy: List[Dict], y_train: np.ndarray,
    val_batches: List[AreaBatch], val_legacy: List[Dict], y_val: np.ndarray,
    device: torch.device,
    epochs: int = 100, lr: float = 1e-2, ps_lr: float = 0.5,
    weight_decay: float = 1e-4,
    patience: int = 15, loss_type: str = "pearson", seed: int = 0,
) -> Tuple[HybridMRP, dict]:
    """Same discipline as train_deepmrp: gradients only touch the train split,
    val is used solely to pick the best checkpoint. AdamW + cosine schedule,
    matching DeepMRP's (more modern) training loop rather than ps_nn_legacy's
    notebook-era one, since this model's MR half is DeepMRP's.

    Two learning rates, not one: the MR half (model.mr) trains at DeepMRP's
    usual scale (lr, default 1e-2), but the P half (model.ps) is literally
    ps_nn_legacy's smoothing+fc1 parameters, which we already established
    empirically need a ~50x larger LR (0.5, matching train_ps_nn_legacy's own
    default) to actually converge within a comparable epoch budget -- a
    single shared optimizer at DeepMRP's LR would leave the smoothing
    constants stuck near their initialization, undertrained relative to how
    ps_nn_legacy is trained standalone."""
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW([
        {"params": model.mr.parameters(), "lr": lr},
        {"params": model.ps.parameters(), "lr": ps_lr},
    ], weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    best_val = -np.inf
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history = {"train_r": [], "val_r": [], "val_mse": []}

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(train_batches, train_legacy)
        if loss_type == "pearson":
            loss = pearson_loss(pred, yt)
        elif loss_type == "mse":
            loss = F.mse_loss(pred, yt)
        else:
            raise ValueError(loss_type)
        loss.backward()
        opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = model(val_batches, val_legacy).cpu().numpy()
        val_r, val_mse = _metrics(vp, np.asarray(y_val))
        with torch.no_grad():
            tp = model(train_batches, train_legacy).cpu().numpy()
        tr_r, _ = _metrics(tp, np.asarray(y_train))

        history["train_r"].append(tr_r)
        history["val_r"].append(val_r)
        history["val_mse"].append(val_mse)

        if val_r > best_val:
            best_val = val_r
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                break

    model.load_state_dict(best_state)
    return model, history
