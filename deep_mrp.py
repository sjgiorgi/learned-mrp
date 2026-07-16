# -*- coding: utf-8 -*-
"""
deep_mrp.py
===========
A *full* learned MRP: both the multilevel-regression step (M/R) and the
post-stratification step (P) are learnable, and EACH can be independently
switched to its classical counterpart so the same class runs every cell of the
2x2 ablation:

        P = census-weighted        P = learned
    --------------------------------------------------------
 MR = linear (HLM-like)   | classical MRP        | learned-P only
 MR = deep  (nonlinear)   | learned-MR only      | full deep-MRP
    --------------------------------------------------------

Design notes (why it is built this way):

* The MR step predicts a *cell-level outcome* with PARTIAL POOLING. The pooling
  -- not the nonlinearity -- is what makes it "multilevel". Sparse cells are
  shrunk toward their parent (marginal) means via a learned, per-cell gate. If
  you drop the pooling you have ridge-regression-with-extra-steps, which is the
  thing reviewers (rightly) reject. So pooling is a first-class, ablatable part.

* The P step produces weights. Classical P uses *known census fractions*
  (nothing learned). Learned P relaxes the within-cell-representativeness
  assumption that classical P makes: it lets the weighting depend on the
  census-vs-sample MISMATCH, i.e. it corrects residual selection the census
  margins cannot see. It is implemented as attention whose QUERY is the
  population (census) target and whose KEYS are the per-user cell encodings.

* The final area estimate post-stratifies the MR cell predictions by the
  (census or learned) cell weights:  y_area = sum_c  w_c * yhat_c .
  When MR is the identity-on-observed-scores and P is learned attention, this
  reduces exactly to the original PS_NN weighted-average -- so the old model is
  a special case, which keeps the ablation honest.

The module is written to consume the PRE-TENSORISED, per-area batch produced by
`featurize.py` (no per-area python loop with in-loop tensor allocation).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Per-area input container
# --------------------------------------------------------------------------- #
@dataclass
class AreaBatch:
    """One area's pre-tensorised inputs. All tensors already live on DEVICE.

    Shapes (let u = #users in the area, K = total marginal-bin width,
            C = #crossed cells if used):
      user_scores        : (u,)        person-level proxy value (the thing aggregated)
      user_cell_marg     : (u, K)      per-user multi-hot over marginal bins
      user_cell_cross    : (u, C)      per-user one-hot over crossed cells (optional)
      census_marg        : (K,)        census fractions per marginal bin (population target)
      census_cross       : (C,)        census fractions per crossed cell (optional)
      sample_marg        : (K,)        sample fractions per marginal bin
      cell_index         : (u,)  long  crossed-cell id per user (optional; for pooling)
      user_reliability   : (u,)        outcome-INDEPENDENT reliability signal
                                       (e.g. log #tweets) or all-ones if unused
    """
    user_scores: torch.Tensor
    user_cell_marg: torch.Tensor
    census_marg: torch.Tensor
    sample_marg: torch.Tensor
    user_cell_cross: Optional[torch.Tensor] = None
    census_cross: Optional[torch.Tensor] = None
    cell_index: Optional[torch.Tensor] = None
    user_reliability: Optional[torch.Tensor] = None


# --------------------------------------------------------------------------- #
#  The MR (multilevel regression) step
# --------------------------------------------------------------------------- #
class MultilevelRegression(nn.Module):
    """Predicts a per-user (equivalently per-cell) outcome with partial pooling.

    mode = 'linear' : a single linear map over the cell encoding. This is the
                      HLM-analog baseline (still pooled, but no interactions).
    mode = 'deep'   : an MLP over the cell encoding -> captures interactions a
                      linear multilevel model cannot without manual terms.

    Partial pooling: a learned per-user gate g in (0,1) blends the cell-specific
    prediction with a pooled (global / marginal) prediction. g is driven by the
    cell's *evidence* (how many users share the cell) so data-poor cells shrink
    toward the pooled value -- the neural analog of random-effect shrinkage.
    """

    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2,
                 mode: str = "deep", dropout: float = 0.1, pool: bool = True):
        super().__init__()
        self.mode = mode
        self.pool = pool

        if mode == "linear":
            self.cell_net = nn.Linear(in_dim, 1)
        elif mode == "deep":
            layers: List[nn.Module] = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.GELU(),
                           nn.LayerNorm(hidden), nn.Dropout(dropout)]
                d = hidden
            layers += [nn.Linear(d, 1)]
            self.cell_net = nn.Sequential(*layers)
        else:
            raise ValueError(f"invalid MR mode: {mode}")

        # global (fully-pooled) intercept -- the thing sparse cells shrink toward
        self.global_mean = nn.Parameter(torch.zeros(1))

        if pool:
            # gate driven by log-evidence: more users in a cell -> trust the
            # cell-specific prediction more (gate -> 1).
            self.pool_gate = nn.Sequential(nn.Linear(1, 1))
            # init so that with ~e users the gate sits near 0.5
            with torch.no_grad():
                self.pool_gate[0].weight.fill_(1.0)
                self.pool_gate[0].bias.fill_(-1.0)

    def forward(self, cell_enc: torch.Tensor,
                cell_evidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """cell_enc: (u, in_dim) ; cell_evidence: (u,) count of users in the
        user's cell (for pooling). Returns yhat: (u,) per-user cell prediction."""
        cell_pred = self.cell_net(cell_enc).squeeze(-1)            # (u,)
        if not self.pool or cell_evidence is None:
            return cell_pred + self.global_mean
        log_ev = torch.log1p(cell_evidence).unsqueeze(-1)          # (u,1)
        g = torch.sigmoid(self.pool_gate(log_ev)).squeeze(-1)      # (u,) in (0,1)
        # shrink sparse-cell predictions toward the global mean
        return g * cell_pred + (1.0 - g) * self.global_mean


# --------------------------------------------------------------------------- #
#  The P (post-stratification / reweighting) step
# --------------------------------------------------------------------------- #
class PostStratify(nn.Module):
    """Produces per-user weights used to aggregate cell predictions to the area.

    mode = 'census' : NON-learned. Each user is weighted so that, after
                      aggregation, cells match their known census fractions
                      (classical post-stratification). Nothing to learn here --
                      this is the principled baseline.
    mode = 'learned': attention. QUERY = census-vs-sample mismatch (population
                      target relative to what we observed); KEYS = per-user cell
                      encoding. Lets the weighting correct residual WITHIN-cell
                      selection that census margins cannot express.
    """

    def __init__(self, in_dim: int, d: int = 64, mode: str = "learned",
                 use_reliability: bool = True, dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.use_reliability = use_reliability
        if mode == "learned":
            # query comes from the mismatch vector (same width as cell enc)
            self.q_proj = nn.Linear(in_dim, d)
            self.k_proj = nn.Linear(in_dim, d)
            self.scale = d ** 0.5
            # d=64 gives this step ~3.3k params (more than the MR step's MLP
            # at the same default hidden size) with NOTHING to regularize it
            # before this -- a likely source of the overfitting this ablation
            # cell shows on small area-level training sets. proj_dropout on
            # q/k mirrors standard attention-layer dropout; attn_dropout on
            # the post-softmax weights is standard attention-weight dropout.
            # Both are inverted dropout (PyTorch default), so the downstream
            # w.sum()-normalisation in DeepMRP.forward_area stays correct
            # under whatever weights a given dropout draw actually produces.
            self.proj_dropout = nn.Dropout(dropout)
            self.attn_dropout = nn.Dropout(dropout)
            if use_reliability:
                # reliability enters the logit additively (log-space multiplier)
                self.rel_proj = nn.Linear(1, 1)
        elif mode != "census":
            raise ValueError(f"invalid P mode: {mode}")

    def forward(self, batch: AreaBatch, user_cell_enc: torch.Tensor,
                census_enc: torch.Tensor, sample_enc: torch.Tensor):
        """Returns (weights (u,), aux dict).  Weights are normalised to sum to u
        so that a uniform weighting recovers the plain mean."""
        u = user_cell_enc.shape[0]

        if self.mode == "census":
            # classical P: weight_i = census_frac(cell_i) / sample_frac(cell_i)
            # implemented via the per-user cell encoding dotted with the ratio.
            ratio = census_enc / (sample_enc + 1e-9)               # (in_dim,)
            w = (user_cell_enc * ratio).sum(-1)                    # (u,)
            w = torch.clamp(w, min=1e-6)
            w = w / w.sum() * u
            return w, {}

        # learned P: attention with population-mismatch query
        mismatch = (census_enc - sample_enc)                       # (in_dim,)
        q = self.proj_dropout(self.q_proj(mismatch))                # (d,)
        k = self.proj_dropout(self.k_proj(user_cell_enc))           # (u, d)
        logits = (k @ q) / self.scale                              # (u,)

        if self.use_reliability and batch.user_reliability is not None:
            rel = batch.user_reliability.unsqueeze(-1)             # (u,1)
            logits = logits + self.rel_proj(rel).squeeze(-1)       # (u,)

        alpha = torch.softmax(logits, dim=0)                       # (u,) sums to 1
        alpha = self.attn_dropout(alpha)                           # attention-weight dropout
        w = alpha * u                                              # sums to u (in expectation)
        # effective sample size implied by the weights (Kish) -- exposed for the
        # uncertainty / "30 good vs 30 bad users" analyses.
        n_eff = (w.sum() ** 2) / (w.pow(2).sum() + 1e-9)
        return w, {"alpha": alpha, "n_eff": n_eff, "logits": logits}


# --------------------------------------------------------------------------- #
#  Full learned-MRP
# --------------------------------------------------------------------------- #
class DeepMRP(nn.Module):
    """Full learned MRP. Set mr_mode/p_mode to walk the 2x2 ablation.

    cell_space = 'marginal' : MR & P operate on the multi-hot marginal-bin
                              encoding (dense, ~K dims). Safe default.
    cell_space = 'crossed'  : operate on the crossed-cell one-hot (sparse, C
                              dims). This is where interactions + pooling matter
                              and where "PS can't do this" is demonstrated.
    """

    def __init__(self, marg_dim: int, cross_dim: Optional[int] = None,
                 cell_space: str = "marginal",
                 mr_mode: str = "deep", p_mode: str = "learned",
                 hidden: int = 64, depth: int = 2, d_attn: int = 32,
                 dropout: float = 0.1, pool: bool = True,
                 use_reliability: bool = True):
        super().__init__()
        self.cell_space = cell_space
        in_dim = marg_dim if cell_space == "marginal" else cross_dim
        assert in_dim is not None, "cross_dim required for cell_space='crossed'"
        self.in_dim = in_dim

        self.mr = MultilevelRegression(in_dim, hidden, depth, mr_mode,
                                       dropout, pool)
        self.ps = PostStratify(in_dim, d_attn, p_mode, use_reliability, dropout)
        self.mr_mode, self.p_mode = mr_mode, p_mode

    # -- encoding selector ---------------------------------------------------- #
    def _pick(self, batch: AreaBatch):
        if self.cell_space == "marginal":
            return (batch.user_cell_marg, batch.census_marg, batch.sample_marg)
        else:
            assert batch.user_cell_cross is not None
            # crossed sample fractions derived from the users present
            samp = batch.user_cell_cross.mean(0)
            return (batch.user_cell_cross, batch.census_cross, samp)

    # -- single area ---------------------------------------------------------- #
    def forward_area(self, batch: AreaBatch, return_aux: bool = False):
        user_enc, census_enc, sample_enc = self._pick(batch)      # (u,D),(D,),(D,)

        # cell evidence for pooling: how many users share each user's cell
        if batch.cell_index is not None and self.cell_space == "crossed":
            counts = torch.bincount(batch.cell_index,
                                    minlength=self.in_dim).float()
            evidence = counts[batch.cell_index]                   # (u,)
        else:
            # marginal: approximate evidence by the user's own multi-hot mass
            evidence = (user_enc * (user_enc.sum(0))).sum(-1)

        # --- MR: predict per-user cell outcome ---
        if self.mr_mode == "identity":
            yhat = batch.user_scores                              # PS_NN special case
        else:
            yhat = self.mr(user_enc, evidence)                    # (u,)

        # --- P: weights ---
        w, aux = self.ps(batch, user_enc, census_enc, sample_enc) # (u,)

        # --- post-stratify: weighted aggregate of cell predictions ---
        area_est = (w * yhat).sum() / (w.sum() + 1e-9)
        if return_aux:
            aux = dict(aux)
            aux["yhat"] = yhat.detach()
            aux["w"] = w.detach()
            return area_est, aux
        return area_est

    # -- batch of areas ------------------------------------------------------- #
    def forward(self, batches: List[AreaBatch], return_aux: bool = False):
        ests = []
        auxes = []
        for b in batches:
            if return_aux:
                e, a = self.forward_area(b, return_aux=True)
                auxes.append(a)
            else:
                e = self.forward_area(b)
            ests.append(e)
        ests = torch.stack(ests)
        return (ests, auxes) if return_aux else ests


# --------------------------------------------------------------------------- #
#  Convenience: the four ablation configs
# --------------------------------------------------------------------------- #
def ablation_config(name: str):
    """Returns kwargs (mr_mode, p_mode) for the 2x2 plus the PS_NN special case."""
    table = {
        "classical_mrp":  dict(mr_mode="linear",  p_mode="census"),
        "learned_mr":     dict(mr_mode="deep",    p_mode="census"),
        "learned_p":      dict(mr_mode="linear",  p_mode="learned"),
        "full_deep_mrp":  dict(mr_mode="deep",    p_mode="learned"),
        "ps_nn_legacy":   dict(mr_mode="identity", p_mode="learned"),  # old model
    }
    if name not in table:
        raise KeyError(f"unknown ablation '{name}'; options: {list(table)}")
    return table[name]
