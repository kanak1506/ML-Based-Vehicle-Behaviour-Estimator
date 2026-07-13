"""
feature_contribution_plots.py
==============================
Generates feature-vs-target visualisations for the 7 engineered features
(4 Roll Gradient predictors + 3 Understeer Gradient predictors) shown on the
"Engineered Features" slide.

Two complementary views are produced per target, because they answer
different questions:

1. Raw scatter (feature vs target, coloured by Type)
   -> "What does the physical relationship look like in the data?"
   Marginal / uncontrolled — a strong trend here can be confounded by other
   correlated features (e.g. Roll_Index correlates with Track_Width_Squared).

2. Partial Dependence + ICE (from the trained model pipeline)
   -> "What has the model actually learned this feature contributes,
      holding the other predictors fixed?"
   This is the better basis for an "importance / contribution" claim,
   since it reflects the fitted model rather than a raw 2-D projection.

Saves to Reports/Plots/:
    feature_vs_target_rg.png   feature_vs_target_ug.png
    pdp_ice_rg.png             pdp_ice_ug.png

Usage:
    python feature_contribution_plots.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import partial_dependence

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Palette (matches generate_plots.py) ────────────────────────────────────
BG    = "#FFFFFF"
CARD  = "#F4F7FB"
BORD  = "#D0D9E8"
BLUE  = "#1565C0"
TEXT  = "#1A2B3C"
GREY  = "#5A7194"
AMB   = "#E65100"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORD,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        TEXT,
    "grid.color":        BORD,
    "grid.alpha":        0.7,
    "grid.linestyle":    "--",
    "legend.facecolor":  BG,
    "legend.edgecolor":  BORD,
    "legend.labelcolor": TEXT,
    "legend.fontsize":   9,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.facecolor": BG,
})

OUT = os.path.join("Reports", "Plots")
os.makedirs(OUT, exist_ok=True)


def save(name: str) -> None:
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  OK  {name}")


# ── Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv("Data/processed_dataset.csv")
with open("Models/outlier_vehicles.json") as f:
    outlier_row_indices = json.load(f)["outlier_row_indices"]

# Row-level exclusion — must match notebooks 03/04 exactly, since these plots
# are computed against the actual training set.
dfc = df.drop(index=outlier_row_indices, errors="ignore").reset_index(drop=True)
dfc = dfc[dfc["Understeer Gradient"] <= 60].reset_index(drop=True)

rg_m = joblib.load("Models/rg_model.joblib")
ug_m = joblib.load("Models/ug_model.joblib")

type_cols = {"3W": BLUE, "4W": AMB}

# Slide labels differ slightly from column names — mapped here for titles only.
RG_PLOT_FEATURES = [
    ("Roll_Index",                "Roll Index\nRI = Mass x Zcg / Track Width"),
    ("ARB_Stiffness_Index",       "ARB Stiffness Index\nASI = log1p(D_ARB^4 / TW)"),
    ("Track_Width_Squared",       "Track Width Squared\nTW^2"),
    ("Tire_Width_Pressure_Ratio", "Tire Compliance Index\nTCI = Tire Width / Front Pressure"),
]
UG_PLOT_FEATURES = [
    ("Roll_Stiffness_Ratio", "Roll Stiffness Ratio\nRSR = ASI / TW"),
    ("Front_WD",             "Front Weight Distribution\nFWD = Front Load / Mass"),
    ("Zcg_Wheelbase_Ratio",  "Zcg / Wheelbase Ratio"),
]


# ── 1. Raw scatter: feature vs target ──────────────────────────────────────

def scatter_grid(features, target_col, target_label, X, fname, ncols):
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    fig.suptitle(f"Engineered Features vs {target_label} (raw / marginal)",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.02)

    for i, (col, label) in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        x = X[col].values
        y = X[target_col].values

        for t, c in type_cols.items():
            mask = X["Type"] == t
            if mask.sum():
                ax.scatter(x[mask], y[mask], color=c, s=55, alpha=0.8,
                           edgecolors=BG, linewidths=0.5, label=t, zorder=3)

        # OLS trend line + Spearman rank correlation (robust to nonlinearity/outliers)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=TEXT, ls="--", lw=1.3, zorder=2)
        rho, p = stats.spearmanr(x, y)

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(f"{target_label} (deg/g)")
        ax.text(0.04, 0.93, f"Spearman rho = {rho:.2f}  (p={p:.3f})",
                transform=ax.transAxes, fontsize=9, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=BORD, alpha=0.9))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    save(fname)


scatter_grid(RG_PLOT_FEATURES, "Roll Gradient", "Roll Gradient", dfc,
             "feature_vs_target_rg.png", ncols=2)
scatter_grid(UG_PLOT_FEATURES, "Understeer Gradient", "Understeer Gradient", dfc,
             "feature_vs_target_ug.png", ncols=3)


# ── 2. Partial Dependence + ICE from the trained model pipeline ───────────

def pdp_grid(model, features, target_label, X, fname, ncols):
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    fig.suptitle(f"Model-Learned Contribution — {target_label}\n"
                 "(Partial Dependence, other features held at observed distribution; "
                 "thin lines = per-vehicle ICE curves)",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.04)

    for i, (col, label) in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        pd_res = partial_dependence(model, X, [col], kind="both", grid_resolution=30)
        grid = pd_res["grid_values"][0]
        ice = pd_res["individual"][0]      # shape (n_samples, grid_res)
        avg = pd_res["average"][0]         # shape (grid_res,)

        for row in ice:
            ax.plot(grid, row, color=BLUE, alpha=0.15, lw=1.0, zorder=2)
        ax.plot(grid, avg, color=TEXT, lw=2.4, zorder=3, label="Average (PDP)")

        # Rug plot of actual observed feature values (n=35, so worth showing —
        # PDP can extrapolate past the region where data actually exists).
        ax.plot(X[col].values, np.full_like(X[col].values, ax.get_ylim()[0], dtype=float),
                "|", color=GREY, markersize=8, alpha=0.6, zorder=1, clip_on=False)

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(f"Predicted {target_label} (deg/g)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    save(fname)


X_rg = dfc[list(rg_m.named_steps["prep"].feature_names_in_)].copy()
X_ug = dfc[list(ug_m.named_steps["prep"].feature_names_in_)].copy()

pdp_grid(rg_m, RG_PLOT_FEATURES, "Roll Gradient", X_rg, "pdp_ice_rg.png", ncols=2)
pdp_grid(ug_m, UG_PLOT_FEATURES, "Understeer Gradient", X_ug, "pdp_ice_ug.png", ncols=3)

print("\nDone. Plots saved to Reports/Plots/")
