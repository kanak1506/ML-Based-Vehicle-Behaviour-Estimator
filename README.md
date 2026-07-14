# ML-Based Vehicle Behaviour Estimator

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Predicts Roll Gradient (RG) and Understeer Gradient (UG) — two standard vehicle-handling metrics — from physical vehicle specifications**

<!-- TODO: screenshot of the Streamlit dashboard's prediction tab -->
<!-- TODO: sample evaluation figure from Reports/ (e.g. predicted vs. actual RG/UG scatter) -->

---

## Problem Statement

Estimating how a vehicle will handle — its roll gradient and understeer/oversteer tendency — normally requires physical prototypes and instrumented test tracks. This project builds a scikit-learn pipeline (Ridge, Random Forest, MLP) with nested, vehicle-grouped cross-validation that predicts both metrics directly from a vehicle's design specs (loads, CG height, track width, tyre data, suspension config), before a physical test is ever run. A Streamlit dashboard wraps the pipeline for interactive prediction, retraining, and diagnostics.

> **Scope of this repo.** This contains the ML pipeline and dashboard code
> only. The underlying test dataset, trained model artifacts, and exploratory
> notebooks were produced on proprietary vehicle test data and are not
> published here. Everything below documents the code and the data schema it
> expects, so the pipeline can be run end-to-end against any compatible
> dataset of your own.

---

## Project structure

```
.
├── app.py                          # Streamlit dashboard — prediction, diagnostics, retrain UI
├── generate_plots.py                # Regenerates evaluation/diagnostic figures after training
├── feature_contribution_plots.py    # Feature-vs-target and partial-dependence figures
├── requirements.txt                 # Pinned dependencies (Python 3.10.2)
├── src/
│   ├── feature_engineering.py       # Single source of truth for all engineered features
│   ├── training.py                  # Shared nested-CV loop, model suite, save utilities
│   └── data_pipeline.py             # Full retrain pipeline: raw → processed → trained models
└── README.md

Not included here (proprietary — see note above), but expected at runtime:
  Data/       raw + processed dataset
  Models/     trained .joblib pipelines, metadata, comparison CSVs
  Reports/    evaluation plots (written by generate_plots.py)
```

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py     # → http://localhost:8501
```

The dashboard's prediction tab needs `Models/rg_model.joblib` and
`Models/ug_model.joblib` to exist. Use the **Retraining** section below to
produce them from your own dataset.

---

## Expected input schema

Supply a dataset (Excel or CSV) with one row per test run and these columns.
`src/feature_engineering.py` derives everything else from them.

### Raw inputs

| Column | Unit | Notes |
|---|---|---|
| `Front_Load`, `Rear_Load` | kg | Static axle loads |
| `Zcg` | mm | CG height above ground |
| `Wheelbase` | mm | Front-to-rear axle distance |
| `Track_Width` | mm | Lateral tyre-centre distance |
| `Tire_Width` | mm | Tyre section width |
| `Front_Pressure`, `Rear_Pressure` | psi | Cold inflation pressures |
| `ARB_Diameter` | mm | Anti-roll bar diameter (0 = not fitted) |
| `Test condition` | m | Steady-state circle radius (10 or 30) |
| `Type` | — | Vehicle type: `3W` or `4W` |
| `Tire_Make` | — | e.g. `MRF`, `TVS`, `CEAT` (new makes are handled gracefully) |
| `Suspension_Configuration` | — | Spring/damper setup string |

### Engineered features

| Feature | Formula | Model |
|---|---|---|
| `Roll_Index` | (Mass × Zcg) / Track_Width | RG |
| `Track_Width_Squared` | Track_Width² | RG |
| `Tire_Width_Pressure_Ratio` | Tire_Width / Front_Pressure | RG |
| `ARB_Stiffness_Index` | log₁ₚ(D⁴ / Track_Width) | RG |
| `ARB_Present` | 1 if ARB_Diameter > 0 else 0 | RG, UG |
| `Front_WD` | Front_Load / Mass | UG |
| `Roll_Stiffness_Ratio` | ARB_Stiffness_Index / Track_Width | UG |
| `Zcg_Wheelbase_Ratio` | Zcg / Wheelbase | UG |
| `Type`, `Tire_Make` | raw string → one-hot inside the pipeline | RG, UG |

`OneHotEncoder(drop='first', handle_unknown='ignore')` is fit inside the
sklearn `Pipeline` on each training fold. Unknown categories at inference
resolve to all-zeros (the reference category); the dashboard surfaces a
warning when this happens rather than silently predicting from it.

---

## Model validation

Both targets use **nested cross-validation**:

- **Outer loop:** `LeaveOneGroupOut` grouped by vehicle — every run of a
  given vehicle is held out together, so no chassis appears in both train
  and test within a fold. This is what makes the reported metrics an honest
  estimate rather than optimistic row-level leakage.
- **Inner loop:** `GridSearchCV(cv=5)` on the training fold only, for
  hyperparameter selection.
- **Final model:** `GridSearchCV(refit=True)` refits the selected estimator
  on the full clean training set.

## Reference results

Metrics from the last run against the original (private) dataset — included
to show what the pipeline achieves, not as a claim about your own data.

**Roll Gradient** — best: Ridge Regression

| Model | LOOCV RMSE (deg/g) | LOOCV MAE (deg/g) | LOOCV R² |
|---|---|---|---|
| **Ridge Regression (Tuned)** | **1.458** | **1.120** | **0.762** |
| MLP Neural Network (Tuned) | 1.717 | 1.250 | 0.669 |
| Random Forest Regressor (Tuned) | 1.911 | 1.672 | 0.591 |

RG follows a near-linear physics relationship, so regularised linear models
outperform tree-based ones.

**Understeer Gradient** — best: MLP Neural Network

| Model | LOOCV RMSE (deg/g) | LOOCV MAE (deg/g) | LOOCV R² |
|---|---|---|---|
| **MLP Neural Network (Tuned)** | **3.994** | **2.911** | **0.207** |
| Random Forest Regressor (Tuned) | 4.105 | 2.932 | 0.163 |
| Ridge Regression (Tuned) | 4.574 | 3.320 | −0.039 |

UG R² ≈ 0.21 — treat this model as a directional indicator, not a precise
estimate. The dominant driver (tyre lateral cornering stiffness, C_α) isn't
in the schema above; it's the highest-leverage feature to add if you're
extending this on your own data.

---

## Retraining

`src/data_pipeline.py` is the single source of truth for turning a raw
dataset into new model artifacts — it's what the dashboard's "Data & Retrain"
tab calls, and it's independent of any exploratory notebook:

```
load_raw_dataset(path)        # read the raw Excel/CSV
validate_raw_dataset(df)      # schema/sanity checks, returns a list of issues
run_feature_pipeline(raw_df)  # feature engineering + outlier detection
run_full_retrain(raw_df, staging_dir)   # nested CV + train, writes to a staging dir
promote_staging(...)          # copy staged artifacts into Data/ and Models/
discard_staging(staging_dir)  # drop a staging run without promoting
```

`run_full_retrain()` never touches the live `Data/`/`Models/` paths directly —
everything lands in a staging directory first, so you can compare before/after
metrics and decide whether to promote. This is also exposed as a UI flow in
`app.py`'s "Data & Retrain" tab.

---

## Extending

**New engineered feature:**
1. Add the formula in `engineer_features()` in `src/feature_engineering.py`,
   mirrored in `add_engineered_features()` in the same file (used for batch
   training).
2. Add the column name to `RG_FEATURES` or `UG_FEATURES` if it should be a
   model input.
3. Retrain. `app.py` adapts automatically — no dashboard code changes needed.

**New model:** add one dict entry to `get_model_suite()` in `src/training.py`;
the retrain pipeline picks it up on the next run.

---

## Dashboard behavior

`app.py` accepts raw vehicle inputs and returns predicted RG/UG in real time.
It reads feature lists and category options from the saved model metadata
at startup, rather than hardcoding them, so it adapts to whatever the current
models were trained on. Notable behaviors:

- Slider ranges are derived from the training data's observed range.
- Inputs outside that range trigger an extrapolation warning.
- Unrecognized categorical values (e.g. a new tyre brand) trigger an explicit
  warning rather than silently falling back to the reference category.
- If a saved model expects a feature that the current `feature_engineering.py`
  doesn't produce, the dashboard raises a hard error instead of predicting
  from a wrong/missing value.

---

## License

[MIT](LICENSE) © 2026 Kanak Potdar
