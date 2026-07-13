"""
Vehicle Dynamics Feature Engineering — Single Source of Truth
=============================================================

All feature computation lives here. Both the training notebooks and the
Streamlit dashboard import from this module, guaranteeing identical preprocessing.

`add_engineered_features(df)` is the single implementation. For dashboard
inference, a one-row DataFrame is built from raw inputs and passed through the
same function used at training time:

    row = pd.DataFrame([{"Front_Load": ..., "Rear_Load": ..., ...}])
    features = add_engineered_features(row)

`build_inference_row(raw_inputs)` does this mapping automatically, so the
dashboard needs no direct knowledge of the column names.

Categorical encoding
--------------------
Categorical columns (Type, Tire_Make, Suspension_Configuration) are returned
as raw strings. The sklearn Pipeline inside each saved .joblib fits its own
OneHotEncoder on those strings during training — no separate encoder file is
needed, no hardcoded binary mapping exists. New or unseen categories at
inference time produce an all-zeros OHE row (handle_unknown='ignore'), which
the model treats as the reference (first dropped) category. No crash.

Adding a new numerical feature
-------------------------------
1. Compute it in add_engineered_features() below.
2. Add its column name to RG_FEATURES or UG_FEATURES if it should be a model input.
3. Retrain the relevant notebook (kernel → restart and run all).
4. app.py and get_model_input() adapt automatically — no other changes needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "RG_FEATURES",
    "UG_FEATURES",
    "add_engineered_features",
    "build_inference_row",
    "get_model_input",
    "validate_inputs",
]

# ── Canonical feature lists ───────────────────────────────────────────────────
# These lists define exactly which features go into each model.
# Notebooks 03 and 04 import these constants so the feature lists are
# defined in one place only. To add or remove a feature from a model:
#   1. Edit the list below.
#   2. Re-run the relevant model notebook (kernel → restart and run all).
#   3. app.py picks it up automatically via get_model_input().

RG_FEATURES: list[str] = [
    "Roll_Index",
    "Track_Width_Squared",
    "ARB_Stiffness_Index",
    "Tire_Width_Pressure_Ratio",
    "ARB_Present",
    "Type",
]

UG_FEATURES: list[str] = [
    # Ablation study (n=32, 9-vehicle LOOCV) confirmed these 4 numerical features
    # are optimal. Removed features and why:
    #   Test_condition:        constant per vehicle — confounded with vehicle identity
    #   Tire_Stress_Difference: r=0.73 with Pressure_Ratio — collinear noise on tiny folds
    #   Pressure_Ratio:        collinear with TSD; removing both together outperforms
    #                          either alone (LOOCV R² 0.096→0.160, +0.064)
    "Front_WD",
    "Roll_Stiffness_Ratio",
    "ARB_Present",
    "Zcg_Wheelbase_Ratio",
    "Tire_Make",
    "Type",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features from raw vehicle measurements.

    Works for both batch training DataFrames and single-row inference DataFrames.
    This is the single implementation for all feature engineering.

    Required input columns:
        Front_Load, Rear_Load, Zcg, Track_Width, Tire_Width,
        Front_Pressure, Rear_Pressure, ARB_Diameter, Wheelbase.

    Categorical columns (Type, Tire_Make, Suspension_Configuration) are
    passed through as raw strings so the Pipeline's fitted OneHotEncoder
    handles encoding automatically. handle_unknown='ignore' in the OHE
    produces all-zeros for unseen categories at inference — no crash.

    Physics display indices (Physics_RG_Index, UG_Physics_Index) are computed
    here but are not model inputs. They are used by the dashboard's
    Physics Background panel for interpretability only.
    """
    out = df.copy()

    mass = out["Front_Load"] + out["Rear_Load"]
    tw   = out["Track_Width"].clip(lower=1e-9)
    fp   = out["Front_Pressure"].clip(lower=1e-9)
    rp   = out["Rear_Pressure"].clip(lower=1e-9)
    arb  = out["ARB_Diameter"].fillna(0.0).clip(lower=0.0)
    tw2  = tw ** 2
    arb_si       = np.log1p(arb ** 4 / tw)
    avg_pressure = (fp + rp) / 2.0

    out["Mass"]                      = mass
    out["Front_WD"]                  = out["Front_Load"] / mass
    out["Roll_Index"]                = (mass * out["Zcg"]) / tw
    out["Track_Width_Squared"]       = tw2
    out["Pressure_Ratio"]            = fp / rp
    out["Tire_Stress_Difference"]    = (out["Front_Load"] / fp) - (out["Rear_Load"] / rp)
    out["Tire_Width_Pressure_Ratio"] = out["Tire_Width"] / fp
    out["ARB_Present"]               = (arb > 0).astype(int)
    out["ARB_Stiffness_Index"]       = arb_si
    out["Roll_Stiffness_Ratio"]      = arb_si / tw
    out["Physics_RG_Index"]          = (mass * out["Zcg"]) / (avg_pressure * out["Tire_Width"] * tw2)
    rear_wd = out["Rear_Load"] / mass
    out["UG_Physics_Index"]          = ((out["Front_WD"] / fp) - (rear_wd / rp)) / out["Tire_Width"].clip(lower=1e-9)
    out["Zcg_Wheelbase_Ratio"]       = out["Zcg"] / out["Wheelbase"].clip(lower=1e-9)

    return out


# Mapping from raw_inputs dict keys (app.py convention) to DataFrame column names
# expected by add_engineered_features(). Kept here so build_inference_row and
# any future inference scripts share a single mapping.
_RAW_INPUTS_TO_COLUMNS: dict[str, str] = {
    "front_load":        "Front_Load",
    "rear_load":         "Rear_Load",
    "zcg":               "Zcg",
    "wheelbase":         "Wheelbase",
    "track_width":       "Track_Width",
    "tire_width":        "Tire_Width",
    "rim_diameter":      "Rim_Diameter",
    "front_pressure":    "Front_Pressure",
    "rear_pressure":     "Rear_Pressure",
    "arb_diameter":      "ARB_Diameter",
    "vehicle_type":      "Type",
    "tire_make":         "Tire_Make",
    "suspension_config": "Suspension_Configuration",
    "test_condition":    "Test condition",
    "ycg":               "Ycg",
}


def build_inference_row(raw_inputs: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame from dashboard raw inputs and apply feature engineering.

    Translates the raw_inputs dict (lowercase keys from app.py) into a DataFrame
    with the column names expected by add_engineered_features(), then returns the
    fully engineered feature row ready for model inference.
    """
    row = {col: raw_inputs[key] for key, col in _RAW_INPUTS_TO_COLUMNS.items()
           if key in raw_inputs}
    return add_engineered_features(pd.DataFrame([row]))


def get_model_input(model, all_features: pd.DataFrame) -> pd.DataFrame:
    """
    Select exactly the features a model needs, in the correct order.

    Reads the required column names directly from the fitted Pipeline's
    ColumnTransformer so the list is always in sync with what the model
    was trained on. Extra columns in all_features are silently ignored.
    fill_value=0.0 handles any column absent from add_engineered_features()
    (equivalent to the OHE reference category for categorical columns).
    """
    required = list(model.named_steps["prep"].feature_names_in_)
    return all_features.reindex(columns=required, fill_value=0.0)


def validate_inputs(
    front_load: float,
    rear_load: float,
    front_pressure: float,
    rear_pressure: float,
    track_width: float,
    tire_width: float,
    wheelbase: float,
    **_kwargs,
) -> list[str]:
    """Validate raw inputs. Returns list of error strings (empty = valid)."""
    errors: list[str] = []
    if front_load + rear_load <= 0:
        errors.append("Front + Rear load must be greater than zero.")
    if front_load < 0:
        errors.append("Front load cannot be negative.")
    if rear_load < 0:
        errors.append("Rear load cannot be negative.")
    if front_pressure <= 0:
        errors.append("Front tire pressure must be greater than zero.")
    if rear_pressure <= 0:
        errors.append("Rear tire pressure must be greater than zero.")
    if track_width <= 0:
        errors.append("Track width must be greater than zero.")
    if tire_width <= 0:
        errors.append("Tire width must be greater than zero.")
    if wheelbase <= 0:
        errors.append("Wheelbase must be greater than zero.")
    return errors
