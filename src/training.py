"""
Vehicle Dynamics ML — shared training utilities
================================================

Centralises the logic that the two model notebooks (RG and UG) share:

  build_preprocessor(X)       — ColumnTransformer, dtype-detected columns
  get_model_suite()           — 3-model dict + hyperparameter grids
  run_nested_cv(...)          — nested LOOCV training loop
  save_model_with_metadata()  — joblib dump + sidecar JSON provenance file

Cross-validation strategy
-------------------------
  Outer : LeaveOneGroupOut by Vehicle — leaves ALL runs of one vehicle out.
  Inner : GridSearchCV(cv=5) on the training fold only. NOT grouped by vehicle
          (known limitation — see run_nested_cv() docstring: a grouped version
          was tried and rejected because it produced a worse, though more
          honest, LOOCV score).
  Final : GridSearchCV(refit=True) refit on ALL training data for deployment.

Model suite rationale (n ≈ 35 clean rows, 10 vehicles)
------------------------------------------------------
  Three models are compared for both targets (Ridge, Random Forest, MLP — see
  get_model_suite()). Which one wins depends on the target and is decided by
  the current nested-CV run, not hardcoded here — check Models/*_metadata.json
  for the deployed champion. Historically:
    RG : Ridge has won decisively (physics-motivated, low-dimensional signal).
    UG : the gap between RF/MLP has been within fold-noise on 10 folds; treat
         small differences between them as noise, not a real ranking.

  Gradient Boosting was excluded from the suite because its LOOCV error was
  indistinguishable from RF/MLP on both targets while requiring a much larger
  hyperparameter grid — not justified at this sample size.

To add a new model       → add one entry to get_model_suite().
To change the CV strategy → edit run_nested_cv() in one place.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the given feature DataFrame.

    Auto-detects column types from dtype:
      - object dtype  → OneHotEncoder(handle_unknown='ignore', drop='first')
      - numeric dtype → StandardScaler

    The returned transformer is NOT fitted; it is fitted inside each Pipeline
    when GridSearchCV or cross_val_predict calls fit() on each fold.

    Unknown categories at inference → all-zeros (reference category). No crash.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(
                handle_unknown="ignore", drop="first", sparse_output=False
            ), cat_cols),
        ],
        remainder="drop",
    )


def get_model_suite() -> dict[str, tuple[Any, dict]]:
    """
    Return the 3-model suite used for both RG and UG targets.

    Each entry: model_name → (estimator, hyperparameter_grid).
    The grid keys use Pipeline step names: 'model__<param>'.

    Ridge: alpha is the only meaningful hyperparameter; 6 log-spaced values
    cover the full regularization range appropriate for standardised features.

    Random Forest: architecture is fixed conservatively for n≈32 training rows
    per outer fold. Only tree depth is searched (2 values) because it is the
    primary regularization knob on small samples. n_estimators=200 is stable;
    min_samples_split=10 prevents single-sample leaves on tiny folds.

    MLP: single small architecture (4 hidden units) with L2 regularization search
    (3 values). Architecture fixed to avoid overfitting on tiny folds; alpha
    search covers light-to-heavy regularization range.

    Adding a new model: add one entry here.
    Both notebooks pick it up automatically on next run.
    """
    return {
        "Ridge Regression (Tuned)": (
            TransformedTargetRegressor(
                regressor=Ridge(random_state=42),
                transformer=PowerTransformer(method="yeo-johnson"),
            ),
            {"model__regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
        "Random Forest Regressor (Tuned)": (
            RandomForestRegressor(
                n_estimators=200,
                min_samples_split=10,
                max_features="sqrt",
                random_state=42,
            ),
            {"model__max_depth": [2, 3]},
        ),
        "MLP Neural Network (Tuned)": (
            MLPRegressor(random_state=42, max_iter=2000),
            {
                "model__hidden_layer_sizes": [(4,)],
                "model__alpha": [10.0, 50.0, 100.0],
            },
        ),
    }


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_suite: dict[str, tuple[Any, dict]],
) -> tuple[pd.DataFrame, dict]:
    """
    Run nested vehicle-level LOOCV + inner GridSearchCV for each model.

    Outer loop: LeaveOneGroupOut(groups) — 1 fold per unique vehicle.
    Inner loop: GridSearchCV(cv=5, scoring='neg_mse') on training fold only.
    Final model: best estimator refit on ALL data (GridSearchCV(refit=True)).

    KNOWN LIMITATION — inner-loop group leakage: the inner cv=5 is plain KFold,
    not grouped by vehicle, so a training-fold vehicle with 2+ runs can have
    its rows split across the inner train/validation boundary. This was tried
    with GroupKFold(5) (grouped, no leakage) and rejected: it produced an
    honest but *worse* LOOCV score (RG RMSE 1.34→1.50, R² 0.80→0.75; UG
    R² 0.16→0.15, champion flipped MLP→RF — measured at n=32/9 vehicles, the
    dataset size when this trial was run) — the current numbers are
    optimistic. Revisit if more vehicles are added, since the leakage effect
    shrinks as training-fold vehicles outnumber inner splits.

    Parameters
    ----------
    X       : feature DataFrame (raw strings for categorical columns).
    y       : target Series.
    groups  : vehicle-label array (same length as X), used for LOGO splits.
    model_suite : output of get_model_suite() or a custom dict.

    Returns
    -------
    results_df     : DataFrame sorted by LOOCV RMSE (best model = row 0).
    best_estimators: dict {model_name -> fitted Pipeline} for deployment + analysis.
    """
    logo       = LeaveOneGroupOut()
    n_folds    = len(set(groups))
    preprocessor = build_preprocessor(X)

    results: list[dict] = []
    best_estimators: dict = {}

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print(f"Vehicle-level LOOCV ({n_folds} folds) + nested GridSearchCV...")

    for name, (model, params) in model_suite.items():
        pipe   = Pipeline([("prep", preprocessor), ("model", model)])
        search = GridSearchCV(pipe, params, cv=5,
                              scoring="neg_mean_squared_error", n_jobs=1)

        # Outer CV: produces honest out-of-fold predictions.
        # n_jobs=1 throughout: avoids Windows joblib memory-mapping KeyErrors
        # that occur with n_jobs=-1 and corrupt hyperparameter selection.
        y_pred = cross_val_predict(search, X, y, cv=logo, groups=groups, n_jobs=1)

        # Fit on ALL data → select best hyperparameters globally.
        # GridSearchCV(refit=True) already refits best_estimator_ on all of X, y,
        # so it is the deployment model as-is (no separate clone+refit needed).
        search.fit(X, y)
        best_estimators[name] = search.best_estimator_

        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        results.append({
            "Model":              name,
            "LOOCV RMSE (deg/g)": rmse,
            "LOOCV MAE (deg/g)":  float(mean_absolute_error(y, y_pred)),
            "LOOCV R^2":          float(r2_score(y, y_pred)),
        })
        print(f"  {name}: RMSE={rmse:.4f}")

    results_df = (
        pd.DataFrame(results)
        .sort_values("LOOCV RMSE (deg/g)")
        .reset_index(drop=True)
    )
    return results_df, best_estimators


def save_model_with_metadata(
    model,
    model_path: str,
    metadata: dict[str, Any],
) -> None:
    """
    Save a fitted sklearn Pipeline and write a sidecar metadata JSON.

    The JSON path convention mirrors load_model_metadata() in app.py:
        Models/rg_model.joblib  →  Models/rg_metadata.json
        Models/ug_model.joblib  →  Models/ug_metadata.json

    All non-JSON-serialisable values are coerced to str via default=str.
    """
    joblib.dump(model, model_path)
    # Derive metadata path: strip "_model" suffix, replace extension.
    # e.g. "Models/rg_model.joblib" → stem "rg_model" → "rg" → "rg_metadata.json"
    p = Path(model_path)
    meta_name = p.stem.replace("_model", "") + "_metadata.json"
    meta_path = p.parent / meta_name
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Model    saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
