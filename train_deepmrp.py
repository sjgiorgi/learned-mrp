# -*- coding: utf-8 -*-
"""
train_deepmrp.py
================
Modernised training loop for DeepMRP. Replaces the old Adam + manual /10 decay
with AdamW + cosine schedule, global seeding, and clean early stopping. Operates
on List[AreaBatch] (pre-tensorised) and a parallel list/array of area targets.
"""

from __future__ import annotations
from copy import deepcopy
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from deep_mrp import AreaBatch, DeepMRP


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    vp = pred - pred.mean()
    vt = target - target.mean()
    denom = (vp.pow(2).sum().sqrt() * vt.pow(2).sum().sqrt()) + 1e-12
    return -(vp * vt).sum() / denom


def _metrics(pred: np.ndarray, target: np.ndarray):
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    r = pearsonr(pred, target)[0]
    mse = mean_squared_error(target, pred)
    return r, mse


def train_deepmrp(
    model: DeepMRP,
    train_batches: List[AreaBatch], y_train: np.ndarray,
    val_batches: List[AreaBatch], y_val: np.ndarray,
    device: torch.device,
    epochs: int = 100, lr: float = 1e-2, weight_decay: float = 1e-4,
    patience: int = 15, loss_type: str = "pearson", seed: int = 0,
    verbose: bool = False,
) -> Tuple[DeepMRP, dict]:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32, device=device)
    best_val = -np.inf
    best_state = deepcopy(model.state_dict())
    no_improve = 0
    history = {"train_r": [], "val_r": [], "val_mse": []}

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(train_batches)
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
            vp = model(val_batches).cpu().numpy()
        val_r, val_mse = _metrics(vp, np.asarray(y_val))
        with torch.no_grad():
            tp = model(train_batches).cpu().numpy()
        tr_r, _ = _metrics(tp, np.asarray(y_train))

        history["train_r"].append(tr_r)
        history["val_r"].append(val_r)
        history["val_mse"].append(val_mse)
        if verbose:
            print(f"  ep {ep:3d}  train_r={tr_r:.3f}  val_r={val_r:.3f}  val_mse={val_mse:.3f}")

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
