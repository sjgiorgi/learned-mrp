# -*- coding: utf-8 -*-
"""
mrp_baseline.py
===============
A *real* MRP baseline -- the Gelman/Park-Bafumi two-stage procedure -- not a
linearised stand-in inside the neural module. Use this in the ablation table so
the comparison is honest:

  Stage 1 (multilevel regression, the "MR"):
      fit  y_person ~ (1 | cell_demographic_factors)
      i.e. random intercepts per demographic level, estimated by REML. The
      variance components give *principled partial pooling*: sparse cells shrink
      toward the grand mean by an amount the data chooses, not a tuned constant.
      This is the property a hand-rolled ridge version does NOT have, and the
      reason this -- not my in-module 'classical_mrp' config -- is the baseline a
      reviewer will accept.

  Stage 2 (poststratification, the "P"):
      predict the outcome for every census cell, then take the census-fraction-
      weighted average over cells to get the area estimate.

Two fitters are provided:
  * MRPMixedLM  -- statsmodels MixedLM, REML random intercepts (lme4-equivalent,
                   fast, the recommended default here).
  * MRPCellMeans -- a transparent empirical-Bayes shrinkage fallback (no extra
                   deps) for sanity-checking / when MixedLM won't converge.

Both expose .fit(person_df) and .predict_area(area_frame) so they drop into the
same evaluation loop as DeepMRP.

Expected person-level dataframe columns:
    spatial_group   : area id
    <demographics>  : categorical demographic columns (e.g. age_bin, gen, married)
    y               : the person-level outcome/proxy being modelled
Expected poststratification frame per area:
    a table of (cell demographic values) -> census_fraction in that area.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

import statsmodels.formula.api as smf


# --------------------------------------------------------------------------- #
#  Real MRP via REML random intercepts (statsmodels MixedLM)
# --------------------------------------------------------------------------- #
class MRPMixedLM:
    """Two-stage MRP. Random intercepts for ONE grouping factor are native to
    MixedLM; additional demographic factors enter as variance-component groups
    via vc_formula. For the common small-cell case we pool the *cell* (the
    crossed demographic id) as the random effect, which is the textbook MRP
    setup (random effect per poststratification cell)."""

    def __init__(self, demographics: List[str], spatial_group: str,
                 area_random_effect: bool = True):
        self.demographics = demographics
        self.spatial_group = spatial_group
        self.area_random_effect = area_random_effect
        self.result = None
        self._grand_mean = None

    def _cell_id(self, df: pd.DataFrame) -> pd.Series:
        return df[self.demographics].astype(str).agg("|".join, axis=1)

    def fit(self, person_df: pd.DataFrame, y_col: str = "y"):
        df = person_df.copy()
        df["_cell"] = self._cell_id(df)
        self._grand_mean = df[y_col].mean()

        # Random intercept per poststratification cell (partial pooling across
        # cells). Optionally also a random intercept per area. We fit the cell
        # effect as the primary grouping factor; the area effect (if requested)
        # is added as a variance component.
        vc = {}
        if self.area_random_effect:
            vc[self.spatial_group] = f"0 + C({self.spatial_group})"

        try:
            md = smf.mixedlm(
                f"{y_col} ~ 1",
                data=df,
                groups=df["_cell"],
                vc_formula=vc if vc else None,
            )
            self.result = md.fit(reml=True, method="lbfgs", maxiter=200)
            # random_effects extraction can itself fail on a singular cov even
            # when .fit() "succeeds" -- so it must be inside the try.
            re = self.result.random_effects
            self._cell_effect = {k: float(v.iloc[0]) for k, v in re.items()}
            self._intercept = float(self.result.fe_params["Intercept"])
        except Exception as e:  # convergence / singular -- fall back gracefully
            self.result = None
            self._fallback = MRPCellMeans(self.demographics, self.spatial_group)
            self._fallback.fit(person_df, y_col)
            self._fit_failed = str(e)
            return self

        self._fit_failed = None
        return self

    def predict_cell(self, cell_values: Dict[str, str]) -> float:
        if self.result is None:
            return self._fallback.predict_cell(cell_values)
        cell = "|".join(str(cell_values[d]) for d in self.demographics)
        eff = self._cell_effect.get(cell, 0.0)  # unseen cell -> shrink to mean
        return self._intercept + eff

    def predict_area(self, ps_frame: pd.DataFrame,
                     frac_col: str = "census_frac") -> float:
        """ps_frame: rows = cells for ONE area, with demographic columns and a
        census_frac column summing to 1. Returns the poststratified estimate.

        NOTE: cell keys are built with per-column astype(str) BEFORE any
        row-wise access. ps_frame.iterrows() would upcast each row to a
        single dtype (float, because census_frac is float), turning int bin
        ids like 0 into "0.0" -- which then never matches the "0" keys built
        from the person-level sample, silently falling back to the "unseen
        cell" default for every cell."""
        cells = ps_frame[self.demographics].astype(str).agg("|".join, axis=1)
        fracs = ps_frame[frac_col].to_numpy()
        if self.result is None:
            preds = cells.map(lambda c: self._fallback._predict_from_cell_key(c))
        else:
            preds = cells.map(lambda c: self._intercept + self._cell_effect.get(c, 0.0))
        return float(np.dot(fracs, preds.to_numpy()))


# --------------------------------------------------------------------------- #
#  Transparent empirical-Bayes fallback (no heavy deps)
# --------------------------------------------------------------------------- #
class MRPCellMeans:
    """Empirical-Bayes shrinkage of cell means toward the grand mean -- an
    explicit, inspectable partial-pooling baseline. Useful as a sanity check and
    when MixedLM does not converge. shrinkage(cell) = n_cell / (n_cell + kappa),
    with kappa estimated from between- vs within-cell variance."""

    def __init__(self, demographics: List[str], spatial_group: str):
        self.demographics = demographics
        self.spatial_group = spatial_group

    def _cell_id(self, df):
        return df[self.demographics].astype(str).agg("|".join, axis=1)

    def fit(self, person_df: pd.DataFrame, y_col: str = "y"):
        df = person_df.copy()
        df["_cell"] = self._cell_id(df)
        self.grand_mean = df[y_col].mean()

        g = df.groupby("_cell")[y_col]
        cell_mean = g.mean()
        cell_n = g.size()
        cell_var = g.var(ddof=1).fillna(df[y_col].var(ddof=1))

        # method-of-moments kappa: within-var / between-var
        within = cell_var.mean()
        between = cell_mean.var(ddof=1)
        self.kappa = float(within / between) if between > 0 else 10.0

        shrink = cell_n / (cell_n + self.kappa)
        pooled = shrink * cell_mean + (1 - shrink) * self.grand_mean
        self.cell_pred = pooled.to_dict()
        return self

    def predict_cell(self, cell_values: Dict[str, str]) -> float:
        cell = "|".join(str(cell_values[d]) for d in self.demographics)
        return self._predict_from_cell_key(cell)

    def _predict_from_cell_key(self, cell: str) -> float:
        return self.cell_pred.get(cell, self.grand_mean)

    def predict_area(self, ps_frame: pd.DataFrame,
                     frac_col: str = "census_frac") -> float:
        # see MRPMixedLM.predict_area: build cell keys column-wise, never via
        # ps_frame.iterrows() (which upcasts int bin ids to float and breaks
        # the string match against the person-level sample's cell keys).
        cells = ps_frame[self.demographics].astype(str).agg("|".join, axis=1)
        fracs = ps_frame[frac_col].to_numpy()
        preds = cells.map(self._predict_from_cell_key).to_numpy()
        return float(np.dot(fracs, preds))


# --------------------------------------------------------------------------- #
#  Raking baseline (the simplest classical P; no regression at all)
# --------------------------------------------------------------------------- #
def raking_estimate(person_df: pd.DataFrame, demographics: List[str],
                    ps_frame: pd.DataFrame, y_col: str = "y",
                    frac_col: str = "census_frac") -> float:
    """Classical poststratification by raw cell means (no pooling). This is the
    'no-M' baseline: just reweight observed cell means by census fractions. If
    learned-MRP can't beat THIS, the regression isn't earning its place."""
    df = person_df.copy()
    df["_cell"] = df[demographics].astype(str).agg("|".join, axis=1)
    cell_mean = df.groupby("_cell")[y_col].mean()
    grand = df[y_col].mean()
    # build cell keys column-wise (see MRPMixedLM.predict_area for why
    # ps_frame.iterrows() silently breaks this string match)
    cells = ps_frame[demographics].astype(str).agg("|".join, axis=1)
    fracs = ps_frame[frac_col].to_numpy()
    preds = cells.map(lambda c: cell_mean.get(c, grand)).to_numpy()
    return float(np.dot(fracs, preds))
