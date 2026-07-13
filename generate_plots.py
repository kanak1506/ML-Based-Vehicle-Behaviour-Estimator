"""
generate_plots.py
=================
Run once after training (or after retraining) to regenerate all dashboard
figures. Saves to Reports/Plots/ using a consistent dark automotive palette.

Usage:
    python generate_plots.py
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
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.feature_engineering import RG_FEATURES, UG_FEATURES

# ── Palette (light, automotive blue) ────────────────────────────────────────────
BG    = "#FFFFFF"
CARD  = "#F4F7FB"
BORD  = "#D0D9E8"
BLUE  = "#1565C0"   # primary blue
BLU2  = "#1976D2"
TEXT  = "#1A2B3C"
GREY  = "#5A7194"
GRN   = "#2E7D32"
AMB   = "#E65100"
RED   = "#C62828"
PURP  = "#6A1B9A"

VEH_PALETTE = [
    "#1565C0", "#E65100", "#2E7D32", "#6A1B9A",
    "#B71C1C", "#00695C", "#F9A825", "#01579B", "#880E4F",
]

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
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.titlepad":     10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.facecolor": BG,
    "savefig.bbox":      "tight",
    "figure.dpi":        100,
})

OUT = os.path.join("Reports", "Plots")
os.makedirs(OUT, exist_ok=True)


def save(name: str) -> None:
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"  OK  {name}")


# ── Load ─────────────────────────────────────────────────────────────────────────

df = pd.read_csv("Data/processed_dataset.csv")
with open("Models/outlier_vehicles.json") as f:
    outlier_row_indices = json.load(f)["outlier_row_indices"]

# Row-level exclusion — must match notebooks 03/04 exactly, since these plots
# recompute residuals with cross_val_predict on the actual training set.
dfc = df.drop(index=outlier_row_indices, errors="ignore").reset_index(drop=True)
dfc = dfc[dfc["Understeer Gradient"] <= 60].reset_index(drop=True)

rg_m = joblib.load("Models/rg_model.joblib")
ug_m = joblib.load("Models/ug_model.joblib")

X_rg = dfc[RG_FEATURES].copy()
y_rg = dfc["Roll Gradient"].values
X_ug = dfc[UG_FEATURES].copy()
y_ug = dfc["Understeer Gradient"].values
groups = dfc["Vehicle"].values
vehicles = np.unique(groups)
vcol = {v: VEH_PALETTE[i % len(VEH_PALETTE)] for i, v in enumerate(vehicles)}

logo = LeaveOneGroupOut()

print("Computing LOOCV predictions ...")
yp_rg = cross_val_predict(clone(rg_m), X_rg, y_rg, cv=logo, groups=groups)
yp_ug = cross_val_predict(clone(ug_m), X_ug, y_ug, cv=logo, groups=groups)
res_rg = y_rg - yp_rg
res_ug = y_ug - yp_ug

rmse_rg = np.sqrt(mean_squared_error(y_rg, yp_rg))
r2_rg   = r2_score(y_rg, yp_rg)
rmse_ug = np.sqrt(mean_squared_error(y_ug, yp_ug))
r2_ug   = r2_score(y_ug, yp_ug)

print(f"  RG  RMSE={rmse_rg:.3f}  R²={r2_rg:.3f}")
print(f"  UG  RMSE={rmse_ug:.3f}  R²={r2_ug:.3f}")
print("\nGenerating plots ...")


# ── 1. Target Distributions ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Target Variable Distributions", fontsize=14, fontweight="bold", color=TEXT, y=1.02)

type_cols = {"3W": BLUE, "4W": AMB}
targets = [
    ("Roll Gradient",      "Roll Gradient (deg/g)",      axes[0]),
    ("Understeer Gradient","Understeer Gradient (deg/g)", axes[1]),
]
for col, xlabel, ax in targets:
    for t, c in type_cols.items():
        sub = dfc[dfc["Type"] == t][col].dropna()
        if len(sub):
            ax.hist(sub, bins=8, alpha=0.55, color=c, label=t, edgecolor=BORD, linewidth=0.6)
    mu, md = dfc[col].mean(), dfc[col].median()
    ax.axvline(mu, color=TEXT, ls="--", lw=1.3, label=f"Mean {mu:.1f}")
    ax.axvline(md, color=GREY,  ls=":",  lw=1.3, label=f"Median {md:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(True)

plt.tight_layout()
save("target_distributions.png")


# ── 2. Actual vs Predicted (LOOCV) ────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("LOOCV: Actual vs Predicted", fontsize=14, fontweight="bold", color=TEXT, y=1.02)

for ax, y_true, y_pred, rmse, r2, title in [
    (ax1, y_rg, yp_rg, rmse_rg, r2_rg, "Roll Gradient"),
    (ax2, y_ug, yp_ug, rmse_ug, r2_ug, "Understeer Gradient"),
]:
    for veh in vehicles:
        mask = groups == veh
        ax.scatter(y_true[mask], y_pred[mask], color=vcol[veh],
                   s=60, edgecolors=BG, linewidths=0.5, label=veh, zorder=3)

    lo = min(y_true.min(), y_pred.min()) - 1
    hi = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], color=TEXT, ls="--", lw=1.2, label="Perfect", zorder=2)

    ax.set_xlabel(f"Actual (deg/g)")
    ax.set_ylabel(f"Predicted (deg/g)")
    ax.set_title(title)
    ax.text(0.04, 0.93, f"RMSE = {rmse:.2f}\nR² = {r2:.3f}",
            transform=ax.transAxes, fontsize=10, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=BORD, alpha=0.9))
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True)

plt.tight_layout()
save("actual_vs_predicted.png")


# ── 3. Residual Analysis (2 × 2 grid) ────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold", color=TEXT, y=1.01)

pairs = [
    (axes[0, 0], axes[0, 1], yp_rg, res_rg, "Roll Gradient",      BLUE),
    (axes[1, 0], axes[1, 1], yp_ug, res_ug, "Understeer Gradient", AMB),
]

for ax_scatter, ax_hist, y_pred, res, label, color in pairs:
    # Residuals vs Predicted
    ax_scatter.scatter(y_pred, res, color=color, alpha=0.7, s=55,
                       edgecolors=BG, linewidths=0.5, zorder=3)
    ax_scatter.axhline(0, color=TEXT, ls="--", lw=1.2)
    ax_scatter.set_xlabel("Predicted (deg/g)")
    ax_scatter.set_ylabel("Residual (deg/g)")
    ax_scatter.set_title(f"{label} — Residuals vs Predicted")
    ax_scatter.grid(True)

    # Residual histogram
    ax_hist.hist(res, bins=10, color=color, alpha=0.7, edgecolor=BORD, linewidth=0.6)
    mu_r, sd_r = res.mean(), res.std()
    ax_hist.axvline(0,    color=TEXT, ls="--", lw=1.2, label="Zero")
    ax_hist.axvline(mu_r, color=GREY,  ls=":",  lw=1.2, label=f"Mean {mu_r:.2f}")
    ax_hist.set_xlabel("Residual (deg/g)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{label} — Residual Distribution")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True)

plt.tight_layout()
save("residuals.png")


# ── 4. QQ Plots ───────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Residual Normality (Q-Q Plots)", fontsize=14, fontweight="bold", color=TEXT, y=1.02)

for ax, res, title, color in [
    (ax1, res_rg, "Roll Gradient",      BLUE),
    (ax2, res_ug, "Understeer Gradient", AMB),
]:
    (osm, osr), (slope, intercept, _) = stats.probplot(res)
    ax.scatter(osm, osr, color=color, s=50, edgecolors=BG, linewidths=0.4, zorder=3)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color=TEXT, ls="--", lw=1.2, label="Normal line")
    stat_w, p_w = stats.shapiro(res)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"{title}")
    ax.text(0.04, 0.93, f"Shapiro p = {p_w:.3f}",
            transform=ax.transAxes, fontsize=10, color=TEXT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=BORD, alpha=0.9))
    ax.legend(fontsize=9)
    ax.grid(True)

plt.tight_layout()
save("qq_plots.png")


# ── 5. Feature Importance — RG (Ridge Coefficients) ──────────────────────────

prep = rg_m.named_steps["prep"]
raw_names = prep.get_feature_names_out()
clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
coefs = rg_m.named_steps["model"].regressor_.coef_

coef_df = (
    pd.DataFrame({"Feature": clean_names, "Coefficient": coefs})
    .reindex(pd.Series(coefs).abs().sort_values(ascending=True).index)
)

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = [GRN if c > 0 else RED for c in coef_df["Coefficient"]]
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors,
               edgecolor=BORD, linewidth=0.5, height=0.6)
ax.axvline(0, color=TEXT, lw=0.8, ls="--")
ax.set_xlabel("Standardised Coefficient")
ax.set_title("Roll Gradient — Ridge Regression Coefficients\n(Yeo-Johnson transformed target; green = increases RG, red = decreases RG)")
ax.grid(True, axis="x")
plt.tight_layout()
save("feature_importance_rg.png")


# ── 6. Feature Importance — UG (Permutation Importance) ──────────────────────

perm = permutation_importance(
    ug_m, X_ug, y_ug,
    n_repeats=15, random_state=42,
    scoring="neg_root_mean_squared_error",
)
perm_df = (
    pd.DataFrame({
        "Feature": UG_FEATURES,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    })
    .sort_values("Importance", ascending=True)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(9, 4.5))
colors_p = [BLUE if v > 0 else RED for v in perm_df["Importance"]]
ax.barh(perm_df["Feature"], perm_df["Importance"],
        xerr=perm_df["Std"], color=colors_p,
        edgecolor=BORD, linewidth=0.5, height=0.6,
        error_kw={"ecolor": GREY, "linewidth": 1.2, "capsize": 3})
ax.axvline(0, color=TEXT, lw=0.8, ls="--")
ax.set_xlabel("Mean RMSE Increase when Permuted")
ax.set_title("Understeer Gradient — MLP Permutation Importance\n(higher = feature is more important; negative = shuffling helps, indicating noise)")
ax.grid(True, axis="x")
plt.tight_layout()
save("feature_importance_ug.png")


# ── 7. Model Comparison Bar Chart ─────────────────────────────────────────────

rg_comp = pd.read_csv("Models/model_comparison_rg.csv")
ug_comp = pd.read_csv("Models/model_comparison_ug.csv")

def short_name(s: str) -> str:
    return s.replace(" (Tuned)", "").replace("Regressor", "").replace("Neural Network", "NN").strip()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Comparison — LOOCV Metrics", fontsize=14, fontweight="bold", color=TEXT, y=1.02)

for ax, comp, title, target_col in [
    (axes[0], rg_comp, "Roll Gradient", "LOOCV RMSE (deg/g)"),
    (axes[1], ug_comp, "Understeer Gradient", "LOOCV RMSE (deg/g)"),
]:
    models  = [short_name(m) for m in comp["Model"]]
    rmses   = comp[target_col].values
    r2s     = comp["LOOCV R^2"].values
    x       = np.arange(len(models))
    w       = 0.38

    bar_rmse = ax.bar(x - w / 2, rmses, w, label="RMSE (deg/g)",
                      color=BLUE, edgecolor=BORD, linewidth=0.5)
    ax2 = ax.twinx()
    bar_r2 = ax2.bar(x + w / 2, np.clip(r2s, 0, None), w, label="R²",
                     color=AMB, edgecolor=BORD, linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color=TEXT)
    ax.set_ylabel("RMSE (deg/g)", color=BLUE)
    ax2.set_ylabel("R²", color=BLU2)
    ax2.tick_params(colors=BLU2)
    ax.set_title(title)
    ax.yaxis.label.set_color(BLUE)

    lines = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=AMB)]
    ax.legend(lines, ["RMSE (deg/g)", "R²"], fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 1.1)

plt.tight_layout()
save("model_comparison.png")


# ── 8. Per-Vehicle RMSE ───────────────────────────────────────────────────────

pv_rg, pv_ug = {}, {}
for veh in vehicles:
    mask = groups == veh
    pv_rg[veh] = np.sqrt(mean_squared_error(y_rg[mask], yp_rg[mask]))
    pv_ug[veh] = np.sqrt(mean_squared_error(y_ug[mask], yp_ug[mask]))

veh_labels = [v.replace("_", " ") for v in vehicles]
x = np.arange(len(vehicles))
w = 0.35

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x - w / 2, [pv_rg[v] for v in vehicles], w,
       label="RG RMSE", color=BLUE, edgecolor=BORD, linewidth=0.5)
ax.bar(x + w / 2, [pv_ug[v] for v in vehicles], w,
       label="UG RMSE", color=AMB, edgecolor=BORD, linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(veh_labels, rotation=25, ha="right", fontsize=10)
ax.set_ylabel("LOOCV RMSE (deg/g)")
ax.set_title("Per-Vehicle LOOCV RMSE\n(each vehicle held out once; model trained on remaining 8)")
ax.legend(fontsize=10)
ax.grid(True, axis="y")
plt.tight_layout()
save("per_vehicle_rmse.png")


# ── 9. Correlation Heatmap (model features only) ─────────────────────────────

import seaborn as sns

feat_cols = list(dict.fromkeys(RG_FEATURES + UG_FEATURES))  # unique, order-preserved
num_feats = [f for f in feat_cols if dfc[f].dtype != object]
corr = dfc[num_feats + ["Roll Gradient", "Understeer Gradient"]].corr(method="spearman")

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", linewidths=0.5,
    cmap="RdYlBu_r", center=0, vmin=-1, vmax=1,
    ax=ax, cbar_kws={"shrink": 0.75},
    annot_kws={"size": 9},
)
ax.set_title("Spearman Correlation — Model Features & Targets", fontsize=13, pad=12)
ax.tick_params(axis="x", rotation=40, labelsize=10)
ax.tick_params(axis="y", rotation=0,  labelsize=10)
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
plt.tight_layout()
save("correlation_heatmap_features.png")


# ── 10. Evaluation Metric Distributions Across LOGO Folds ────────────────────
# LOGO CV already holds out one vehicle per fold (2-8 runs each). Rather than
# report a single pooled RMSE/MAE/R², this shows how much each metric varies
# fold-to-fold — directly relevant given only 9 independent vehicle geometries.

def fold_metrics(y_true, y_pred, groups, vehicles):
    rows = []
    for v in vehicles:
        mask = groups == v
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "Vehicle": v,
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "n": mask.sum(),
        })
    return pd.DataFrame(rows)

def bootstrap_r2(y_true, y_pred, groups, vehicles, n_boot=1000, seed=42):
    # R² isn't additive across tiny per-vehicle folds (2-run folds give wild,
    # near-meaningless values e.g. -40). Vehicle-level bootstrap of the pooled
    # score is the statistically valid way to show R²'s fold-to-fold uncertainty.
    rng = np.random.default_rng(seed)
    r2s = []
    for _ in range(n_boot):
        sample = rng.choice(vehicles, size=len(vehicles), replace=True)
        idx = np.concatenate([np.where(groups == v)[0] for v in sample])
        yt, yp = y_true[idx], y_pred[idx]
        if np.var(yt) > 0:
            r2s.append(r2_score(yt, yp))
    return np.array(r2s)

fm_rg = fold_metrics(y_rg, yp_rg, groups, vehicles)
fm_ug = fold_metrics(y_ug, yp_ug, groups, vehicles)
boot_r2_rg = bootstrap_r2(y_rg, yp_rg, groups, vehicles)
boot_r2_ug = bootstrap_r2(y_ug, yp_ug, groups, vehicles)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle(
    "Evaluation Metric Distributions Across LOGO Folds\n"
    "RMSE/MAE: each dot = one held-out vehicle, box = IQR across the 10 folds.  "
    "R²: 1000x vehicle-level bootstrap of the pooled score (per-fold R² is undefined for 2-run folds).",
    fontsize=12, fontweight="bold", color=TEXT, y=1.06,
)

rng = np.random.default_rng(42)
for ax, col, ylabel in [(axes[0], "RMSE", "RMSE (deg/g)"), (axes[1], "MAE", "MAE (deg/g)")]:
    data = [fm_rg[col].values, fm_ug[col].values]
    bp = ax.boxplot(data, positions=[0, 1], widths=0.45, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], [BLUE, AMB]):
        patch.set_facecolor(c)
        patch.set_alpha(0.20)
        patch.set_edgecolor(c)
    for median in bp["medians"]:
        median.set_color(TEXT)
        median.set_linewidth(1.5)

    for x_pos, fm in [(0, fm_rg), (1, fm_ug)]:
        jitter = rng.uniform(-0.08, 0.08, size=len(fm))
        for (_, row), j in zip(fm.iterrows(), jitter):
            ax.scatter(x_pos + j, row[col], color=vcol[row["Vehicle"]], s=55,
                       edgecolors=BG, linewidths=0.6, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Roll Gradient", "Understeer Gradient"])
    ax.set_ylabel(ylabel)
    ax.set_title(col)
    ax.grid(True, axis="y")

ax = axes[2]
vp = ax.violinplot([boot_r2_rg, boot_r2_ug], positions=[0, 1], showextrema=False, widths=0.6)
for body, c in zip(vp["bodies"], [BLUE, AMB]):
    body.set_facecolor(c)
    body.set_alpha(0.35)
    body.set_edgecolor(c)
for x_pos, boot, pooled in [(0, boot_r2_rg, r2_rg), (1, boot_r2_ug, r2_ug)]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ax.plot([x_pos, x_pos], [lo, hi], color=TEXT, lw=1.5, zorder=3)
    ax.scatter([x_pos], [pooled], color=TEXT, s=70, zorder=4, marker="D", label="Pooled LOOCV R²" if x_pos == 0 else None)
ax.axhline(0, color=GREY, ls=":", lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Roll Gradient", "Understeer Gradient"])
ax.set_ylabel("R²")
ax.set_title("R² (bootstrap, 95% CI whisker)")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, axis="y")

handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=vcol[v],
                       markersize=8, label=v.replace("_", " ")) for v in vehicles]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
save("metric_distributions.png")


print("\nDone. All plots saved to Reports/Plots/")
