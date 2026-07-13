"""
Vehicle Dynamics ML — full retrain pipeline (raw -> processed -> trained models)
=================================================================================

Single source of truth for turning an edited raw dataset into new model
artifacts, callable from the Streamlit dashboard's "Data & Retrain" tab.
Mirrors the manual workflow in Notebooks 02/03/04 (feature engineering +
outlier detection, then per-target nested LOOCV training) but the notebooks
remain the exploratory reference and are not driven by this module.

run_full_retrain() writes nothing to the live Data/Models paths -- everything
is staged under a caller-supplied directory so the dashboard can preview
metrics before promoting (promote_staging()) or discarding (discard_staging()).

A one-off Xcg reference-frame correction for a single vehicle, which
Notebook 02 used to apply by row position, has been baked permanently into
"Data/Vehicle Dynamics Dataset.xlsx" (position-based patches are unsafe once
rows can be inserted/deleted from the dashboard), so run_feature_pipeline()
below only needs the position-independent corrections.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import sklearn
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from src.feature_engineering import RG_FEATURES, UG_FEATURES, add_engineered_features
from src.training import get_model_suite, run_nested_cv, save_model_with_metadata

__all__ = [
    "load_raw_dataset",
    "save_raw_dataset",
    "validate_raw_dataset",
    "run_feature_pipeline",
    "train_target",
    "run_full_retrain",
    "promote_staging",
    "discard_staging",
]

RAW_SHEET_NAME = "Datasheet"

# Same 9 features Notebook 02 Section 5 feeds to the robust Mahalanobis
# outlier detector. Order doesn't matter to MinCovDet but is kept stable
# for readability.
OUTLIER_FEATURES = [
    "Mass", "Wheelbase", "Track_Width", "Zcg",
    "Front_WD", "Xcg", "ARB_Diameter",
    "Front_Pressure", "Rear_Pressure",
]

# Raw columns dropped after feature engineering + outlier detection
# (Notebook 02 Section 7) -- captured by engineered features or zero-variance.
REDUNDANT_COLS = [
    "Front_Pressure", "Rear_Pressure",
    "Front_Load", "Rear_Load", "Rear_WD", "Xcg",
    "ARB_Diameter",
    "Damper_Configuration",
]


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Read the raw vehicle dataset from the workbook's Datasheet sheet."""
    return pd.read_excel(path, sheet_name=RAW_SHEET_NAME)


def save_raw_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Write df back to the Datasheet sheet only, preserving the workbook's
    other sheets (Extracted Data, Sheet1, Feature Dictionary).
    """
    try:
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=RAW_SHEET_NAME, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot save '{path}' -- it looks like the file is open in "
            "Excel or another program. Close it there and try again."
        ) from exc


def validate_raw_dataset(df: pd.DataFrame) -> list[str]:
    """Row-level validation for the raw dataset editor. Empty list = valid."""
    errors: list[str] = []
    required_positive = [
        "Front_Load", "Rear_Load", "Front_Pressure", "Rear_Pressure",
        "Track_Width", "Tire_Width", "Wheelbase",
    ]

    for i, row in df.iterrows():
        label = f"Row {i + 1} ({row.get('Vehicle', '?')})"

        if not str(row.get("Vehicle", "")).strip():
            errors.append(f"{label}: Vehicle name is required.")
        if not str(row.get("Type", "")).strip():
            errors.append(f"{label}: Type is required.")

        for col in required_positive:
            val = row.get(col)
            if pd.isna(val) or float(val) <= 0:
                errors.append(f"{label}: {col} must be greater than zero.")

        for col in ("Zcg", "Xcg"):
            if pd.isna(row.get(col)):
                errors.append(f"{label}: {col} is required.")

        arb = row.get("ARB_Diameter")
        if not pd.isna(arb) and float(arb) < 0:
            errors.append(f"{label}: ARB_Diameter cannot be negative.")

        for tgt in ("Roll Gradient", "Understeer Gradient"):
            if pd.isna(row.get(tgt)):
                errors.append(f"{label}: {tgt} is required for training.")

    if df["Vehicle"].nunique(dropna=True) < 2:
        errors.append(
            "At least 2 distinct vehicles are required for vehicle-level cross-validation."
        )
    return errors


def run_feature_pipeline(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ports Notebook 02 sections 2-3-4-5-7: imputation, engineered features
    (single source of truth: src/feature_engineering.py), row-level robust
    Mahalanobis outlier detection, and the redundant-column drop. Section 8
    (correlation/VIF/MI analysis-only plots) is not ported -- unrelated to
    training.

    Returns (processed_df, outlier_info) matching the shape of
    Data/processed_dataset.csv and Models/outlier_vehicles.json.
    """
    df = raw_df.copy().reset_index(drop=True)

    # Impute missing configuration/ARB values (Notebook 02 Section 3).
    # add_engineered_features() internally treats NaN ARB_Diameter as 0 for
    # its own formulas but does not write that back to the column, so the
    # explicit fillna here is required before outlier detection below.
    df["Suspension_Configuration"] = df["Suspension_Configuration"].fillna("Unknown")
    df["Damper_Configuration"] = df["Damper_Configuration"].fillna("Unknown")
    df["ARB_Diameter"] = df["ARB_Diameter"].fillna(0.0)

    # Engineered features. This also enforces Mass = Front_Load + Rear_Load,
    # satisfying the Notebook 02 Section 2 correction.
    df = add_engineered_features(df)
    df["Rear_WD"] = df["Rear_Load"] / df["Mass"]

    # Robust Mahalanobis outlier detection (Notebook 02 Section 5).
    X_out = df[OUTLIER_FEATURES].to_numpy(dtype=float)
    mcd = MinCovDet(random_state=42, support_fraction=0.8)
    mcd.fit(X_out)
    md_squared = mcd.mahalanobis(X_out)
    threshold = chi2.ppf(0.99, len(OUTLIER_FEATURES))
    is_outlier = (md_squared > threshold).astype(int)

    veh_summary = (
        pd.DataFrame({"Vehicle": df["Vehicle"], "Is_Outlier": is_outlier})
        .groupby("Vehicle")["Is_Outlier"].agg(["sum", "count"])
    )
    veh_summary.columns = ["flagged_runs", "total_runs"]
    fully_excluded = veh_summary[
        veh_summary["flagged_runs"] == veh_summary["total_runs"]
    ].index.tolist()
    partially = veh_summary[
        (veh_summary["flagged_runs"] > 0) & (veh_summary["flagged_runs"] < veh_summary["total_runs"])
    ]

    outlier_info = {
        "outlier_row_indices": df.index[is_outlier == 1].tolist(),
        "outlier_vehicles": fully_excluded,
        "partially_flagged_vehicles": {
            veh: {"flagged_runs": int(r["flagged_runs"]), "total_runs": int(r["total_runs"])}
            for veh, r in partially.iterrows()
        },
    }

    processed = df.drop(columns=REDUNDANT_COLS, errors="ignore")
    return processed, outlier_info


def train_target(
    processed_df: pd.DataFrame,
    outlier_row_indices: list[int],
    outlier_vehicles: list[str],
    feature_list: list[str],
    target_col: str,
    extra_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> tuple[Any, dict, pd.DataFrame]:
    """
    Shared logic behind Notebooks 03/04: drop outlier rows, build X/y/groups,
    run nested vehicle-level LOOCV, and pick the champion model. Returns
    (fitted_model, metadata_dict, comparison_df) matching the schema of
    Models/{rg,ug}_metadata.json and Models/model_comparison_{rg,ug}.csv.
    """
    df_filtered = processed_df.drop(index=outlier_row_indices, errors="ignore").reset_index(drop=True)
    if extra_filter is not None:
        df_filtered = extra_filter(df_filtered)

    keep_cols = ["Vehicle", target_col] + feature_list
    df_final = df_filtered[[c for c in keep_cols if c in df_filtered.columns]].copy()

    X = df_final[feature_list].copy()
    y = df_final[target_col]
    groups = df_final["Vehicle"].values
    n_vehicles = len(set(groups))

    results_df, best_estimators = run_nested_cv(X, y, groups, get_model_suite())
    best_row = results_df.iloc[0]
    best_model = best_estimators[best_row["Model"]]

    metadata = {
        "target": target_col,
        "trained_date": datetime.now().isoformat(timespec="seconds"),
        "n_training_records": int(len(X)),
        "n_training_vehicles": int(n_vehicles),
        "training_vehicles": sorted(set(groups)),
        "outlier_vehicles_excluded": outlier_vehicles,
        "features_used": list(X.columns),
        "best_model_class": best_row["Model"],
        "loocv_rmse": float(best_row["LOOCV RMSE (deg/g)"]),
        "loocv_mae": float(best_row["LOOCV MAE (deg/g)"]),
        "loocv_r2": float(best_row["LOOCV R^2"]),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
    }
    return best_model, metadata, results_df


def run_full_retrain(raw_df: pd.DataFrame, staging_dir: str) -> dict:
    """
    Full raw -> processed -> trained-models pipeline, staged under
    staging_dir. Never touches live Data/Models paths.
    """
    staging = Path(staging_dir)
    (staging / "Data").mkdir(parents=True, exist_ok=True)
    (staging / "Models").mkdir(parents=True, exist_ok=True)

    processed_df, outlier_info = run_feature_pipeline(raw_df)
    processed_df.to_csv(staging / "Data" / "processed_dataset.csv", index=False)
    with open(staging / "Models" / "outlier_vehicles.json", "w") as f:
        json.dump(outlier_info, f, indent=2)

    outlier_row_indices = outlier_info["outlier_row_indices"]
    outlier_vehicles = outlier_info["outlier_vehicles"]

    rg_model, rg_metadata, rg_comparison = train_target(
        processed_df, outlier_row_indices, outlier_vehicles,
        RG_FEATURES, "Roll Gradient",
    )
    ug_model, ug_metadata, ug_comparison = train_target(
        processed_df, outlier_row_indices, outlier_vehicles,
        UG_FEATURES, "Understeer Gradient",
        # Notebook 04 caps a small number of extreme Understeer Gradient
        # values beyond the standard row-level outlier filter.
        extra_filter=lambda d: d[d["Understeer Gradient"] <= 60].copy(),
    )

    save_model_with_metadata(rg_model, str(staging / "Models" / "rg_model.joblib"), rg_metadata)
    save_model_with_metadata(ug_model, str(staging / "Models" / "ug_model.joblib"), ug_metadata)
    rg_comparison.to_csv(staging / "Models" / "model_comparison_rg.csv", index=False)
    ug_comparison.to_csv(staging / "Models" / "model_comparison_ug.csv", index=False)

    return {
        "rg": {"model": rg_model, "metadata": rg_metadata, "comparison": rg_comparison},
        "ug": {"model": ug_model, "metadata": ug_metadata, "comparison": ug_comparison},
        "processed_df": processed_df,
        "outlier_info": outlier_info,
    }


def promote_staging(
    staging_dir: str,
    raw_df: pd.DataFrame,
    raw_excel_path: str,
    data_root: str = ".",
    promote_rg: bool = True,
    promote_ug: bool = True,
) -> None:
    """
    Promote staged retrain outputs to the live Data/Models paths.

    The raw Excel is written via save_raw_dataset() (sheet-preserving), not
    copied from staging, since staging never carries the workbook's other
    sheets. RG and UG model artifacts can be promoted independently so one
    target's regression doesn't force reverting the other (no-regression
    protocol: ask when targets split).
    """
    staging = Path(staging_dir)
    root = Path(data_root)

    save_raw_dataset(raw_df, raw_excel_path)
    shutil.copy2(staging / "Data" / "processed_dataset.csv", root / "Data" / "processed_dataset.csv")
    shutil.copy2(staging / "Models" / "outlier_vehicles.json", root / "Models" / "outlier_vehicles.json")

    if promote_rg:
        for name in ("rg_model.joblib", "rg_metadata.json", "model_comparison_rg.csv"):
            shutil.copy2(staging / "Models" / name, root / "Models" / name)
    if promote_ug:
        for name in ("ug_model.joblib", "ug_metadata.json", "model_comparison_ug.csv"):
            shutil.copy2(staging / "Models" / name, root / "Models" / name)


def discard_staging(staging_dir: str) -> None:
    """Delete staged retrain outputs without touching any live files."""
    shutil.rmtree(staging_dir, ignore_errors=True)
