"""
Vehicle Dynamics Prediction Dashboard
======================================
Dark automotive dashboard theme.

Run `python generate_plots.py` once (or after retraining) to refresh figures.
Run `streamlit run app.py` to start the dashboard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src import data_pipeline
from src.feature_engineering import build_inference_row, get_model_input, validate_inputs

RAW_EXCEL_PATH = os.path.join("Data", "Vehicle Dynamics Dataset.xlsx")
STAGING_DIR = "_staging"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Dynamics — ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ─── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 5px 7px;
    gap: 4px;
    border: 1px solid #C4D4E8;
    box-shadow: 0 1px 4px rgba(21,101,192,0.08);
    margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #3A5170;
    border-radius: 7px;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 24px;
    border: none;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background-color: #1565C0 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.30) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: #EBF1FA;
    color: #1565C0;
}

/* ─── Metrics ───────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #C4D4E8;
    border-left: 4px solid #1565C0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.09);
}

/* ─── Dividers ──────────────────────────────────────────────────────────────── */
hr { border-color: #C4D4E8 !important; margin: 20px 0 !important; }

/* ─── Images ────────────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border: 1px solid #C4D4E8;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.08);
}

/* ─── Dataframes ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #C4D4E8;
    border-radius: 10px;
    background: #FFFFFF;
    box-shadow: 0 1px 4px rgba(21,101,192,0.06);
}

/* ─── Info / Warning boxes ──────────────────────────────────────────────────── */
[data-testid="stInfo"]    { border-left: 4px solid #1565C0; background: #EBF1FA; }
[data-testid="stWarning"] { border-left: 4px solid #E65100; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_training_meta() -> dict:
    """
    Reads from processed_dataset.csv (never file-locked) for counts and
    categorical options. Columns dropped during preprocessing (Front_Load,
    Rear_Load, Front_Pressure, Rear_Pressure, ARB_Diameter, Xcg) are loaded
    from the Excel file only as a best-effort; falls back to known training
    ranges when Excel is locked (e.g. open in Microsoft Excel).
    """
    try:
        csv_path = os.path.join("Data", "processed_dataset.csv")
        df = pd.read_csv(csv_path)

        # Row-level outlier exclusion: outlier_row_indices flags individual
        # anomalous runs; outlier_vehicles lists only vehicles with NO surviving
        # runs (fully excluded); partially_flagged_vehicles keeps some runs.
        outlier_path = os.path.join("Models", "outlier_vehicles.json")
        if os.path.exists(outlier_path):
            with open(outlier_path) as f:
                _outlier_info = json.load(f)
            outlier_row_indices = _outlier_info.get("outlier_row_indices", [])
            outliers = _outlier_info.get("outlier_vehicles", [])
            partially_flagged = _outlier_info.get("partially_flagged_vehicles", {})
        else:
            outlier_row_indices, outliers, partially_flagged = [], [], {}
        clean = df.drop(index=outlier_row_indices, errors="ignore")

        def _s(series) -> dict:
            return {"min": float(series.min()), "max": float(series.max()), "median": float(series.median())}

        # Columns present in the processed CSV
        csv_cols = ["Mass", "Wheelbase", "Track_Width", "Tire_Width", "Rim_Diameter", "Ycg", "Zcg"]
        result = {c: _s(clean[c].dropna()) for c in csv_cols if c in clean.columns}

        # Columns dropped from processed CSV — load from Excel if available,
        # otherwise use known training-data ranges as fallback
        _excel_defaults = {
            "Front_Load":     {"min": 145.0,  "max": 283.0,  "median": 224.0},
            "Rear_Load":      {"min": 246.0,  "max": 1425.0, "median": 602.0},
            "Front_Pressure": {"min": 26.0,   "max": 70.0,   "median": 34.0},
            "Rear_Pressure":  {"min": 26.0,   "max": 73.0,   "median": 46.0},
            "ARB_Diameter":   {"min": 15.0,   "max": 20.0,   "median": 20.0},
            "Xcg":            {"min": 800.0,  "max": 1400.0, "median": 1050.0},
        }
        try:
            raw = pd.read_excel(
                os.path.join("Data", "Vehicle Dynamics Dataset.xlsx"),
                sheet_name="Datasheet",
            )
            raw_clean = raw.drop(index=outlier_row_indices, errors="ignore")
            arb_pos = raw_clean["ARB_Diameter"].dropna()
            arb_pos = arb_pos[arb_pos > 0]
            for col in ["Front_Load", "Rear_Load", "Front_Pressure", "Rear_Pressure", "Xcg"]:
                if col in raw_clean.columns:
                    result[col] = _s(raw_clean[col].dropna())
            result["ARB_Diameter"] = _s(arb_pos) if len(arb_pos) else _excel_defaults["ARB_Diameter"]
            result["categorical"] = {
                "Type":  sorted(raw["Type"].dropna().unique().tolist()),
                "Tire_Make": sorted(raw["Tire_Make"].dropna().unique().tolist()),
                "Suspension_Configuration": sorted(
                    raw["Suspension_Configuration"].fillna("Unknown").unique().tolist()
                ),
                "Damper_Configuration": sorted(
                    raw["Damper_Configuration"].fillna("Unknown").unique().tolist()
                ) if "Damper_Configuration" in raw.columns else ["Unknown"],
            }
        except Exception:
            # Excel locked or unavailable — use CSV categoricals + known defaults
            for col, vals in _excel_defaults.items():
                result.setdefault(col, vals)
            result["categorical"] = {
                "Type":  sorted(clean["Type"].dropna().unique().tolist()),
                "Tire_Make": sorted(clean["Tire_Make"].dropna().unique().tolist()),
                "Suspension_Configuration": sorted(
                    clean["Suspension_Configuration"].fillna("Unknown").unique().tolist()
                ),
                "Damper_Configuration": ["Unknown"],
            }

        result["n_clean"]                    = len(clean)
        result["n_total"]                    = len(df)
        result["n_clean_vehicles"]           = clean["Vehicle"].nunique()
        result["outlier_vehicles"]           = outliers
        result["partially_flagged_vehicles"] = partially_flagged
        return result
    except Exception as _e:
        st.error(f"load_training_meta failed: {_e}")
        return {}


@st.cache_resource(show_spinner=False)
def load_models():
    rg = joblib.load(os.path.join("Models", "rg_model.joblib"))
    ug = joblib.load(os.path.join("Models", "ug_model.joblib"))
    return rg, ug


@st.cache_data(show_spinner=False)
def load_metadata(name: str) -> dict:
    path = os.path.join("Models", f"{name}_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Resources ─────────────────────────────────────────────────────────────────

meta = load_training_meta()

try:
    rg_model, ug_model = load_models()
except Exception as exc:
    st.error(f"Cannot load models — {exc}")
    st.stop()

rg_meta       = load_metadata("rg")
ug_meta       = load_metadata("ug")
rg_loocv_rmse = rg_meta.get("loocv_rmse")
rg_loocv_r2   = rg_meta.get("loocv_r2")
ug_loocv_rmse = ug_meta.get("loocv_rmse")
ug_loocv_r2   = ug_meta.get("loocv_r2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def show_plot(filename: str, caption: str = "", width: int | None = None) -> None:
    """Display a pre-generated plot from Reports/Plots/."""
    path = os.path.join("Reports", "Plots", filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=width is None)
    else:
        st.info(f"Run `python generate_plots.py` to generate **{filename}**.")


def stat_card(label: str, value: str, sub: str = "", accent: str = "#1565C0") -> str:
    return f"""
    <div style="background:#FFFFFF; border:1px solid #C4D4E8;
                border-top:4px solid {accent}; border-radius:10px;
                padding:18px 14px; text-align:center; height:100%;
                box-shadow:0 2px 10px rgba(21,101,192,0.10);">
        <div style="color:#4A6080; font-size:10.5px; text-transform:uppercase;
                    letter-spacing:1.5px; margin-bottom:8px; font-weight:600;">{label}</div>
        <div style="color:#0D1B2A; font-size:32px; font-weight:800;
                    line-height:1.1; letter-spacing:-0.5px;">{value}</div>
        <div style="color:#4A6080; font-size:12px; margin-top:6px;">{sub}</div>
    </div>"""


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:4px;">
    <h1 style="margin:0; color:#0D1B2A; font-size:36px; font-weight:800;
               letter-spacing:-0.5px; line-height:1.1;">Vehicle Dynamics Predictor</h1>
</div>
""", unsafe_allow_html=True)

n_total     = meta.get("n_total", "?")
n_clean     = meta.get("n_clean", "?")
n_clean_veh = meta.get("n_clean_vehicles", "?")
n_rows_out  = (n_total - n_clean) if isinstance(n_total, int) and isinstance(n_clean, int) else "?"
rg_r2_str   = f"{rg_loocv_r2:.3f}" if rg_loocv_r2 is not None else "—"
ug_r2_str   = f"{ug_loocv_r2:.3f}" if ug_loocv_r2 is not None else "—"

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, "Total Records",    str(n_total),      "raw observations",         "#1565C0"),
    (c2, "Clean Records",    str(n_clean),      f"{n_rows_out} outlier runs removed", "#1E88E5"),
    (c3, "Unique Vehicles",  str(n_clean_veh),  "training groups",          "#42A5F5"),
    (c4, "Model Features",   "6",               "per target",               "#0288D1"),
    (c5, "RG Model R²",      rg_r2_str,         "Ridge · LOOCV",            "#4CAF50"),
    (c6, "UG Model R²",      ug_r2_str,         "MLP · LOOCV",              "#FF9800"),
]
for col, lbl, val, sub, acc in cards:
    with col:
        st.markdown(stat_card(lbl, val, sub, acc), unsafe_allow_html=True)

st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯  Prediction",
    "📊  Model Performance",
    "🔬  Feature Analysis",
    "📁  Dataset & Methodology",
    "🛠️  Data & Retrain",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    pred_container = st.container()

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.subheader("Vehicle Parameters")

    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown("**Vehicle & Test**")
        vehicle_name = st.text_input(
            "Vehicle ID / Name", value="",
            placeholder="e.g. Unit-014, Prototype-B …",
            help="Identifier for logging and reference. Not used in model predictions.",
        )

        type_opts = meta.get("categorical", {}).get("Type", ["3W", "4W"])
        vehicle_type = st.selectbox(
            "Vehicle Type", options=type_opts,
            index=type_opts.index("4W") if "4W" in type_opts else 0,
            help="3W = three-wheeler (auto-rickshaw). 4W = four-wheeler.",
        )
        test_condition = st.selectbox(
            "Test Track Radius (m)", options=[10, 30],
            help="Steady-state circle radius used during the test.",
        )

        st.markdown("**Mass & Centre of Gravity**")
        s = meta.get("Front_Load", {"min": 100, "max": 600, "median": 250})
        front_load = st.number_input("Front Axle Load (kg)", min_value=50, max_value=700,
                                     value=int(s["median"]), step=5,
                                     help="Static load on the front axle.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} kg")

        s = meta.get("Rear_Load", {"min": 200, "max": 1500, "median": 600})
        rear_load = st.number_input("Rear Axle Load (kg)", min_value=100, max_value=1600,
                                    value=int(s["median"]), step=5,
                                    help="Static load on the rear axle.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} kg")

        mass = front_load + rear_load
        st.caption(f"Total mass: **{mass} kg**  ·  Front bias: **{front_load / mass * 100:.1f}%**")

        s = meta.get("Zcg", {"min": 520, "max": 804, "median": 660})
        zcg = st.number_input("CG Height Zcg (mm)", min_value=300, max_value=1000,
                              value=int(s["median"]), step=5,
                              help="Height of the centre of gravity above ground.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} mm")

        s = meta.get("Ycg", {"min": -47, "max": 35, "median": 0})
        ycg = st.number_input("Lateral CG Offset Ycg (mm)", min_value=-60, max_value=60,
                              value=int(s["median"]), step=1,
                              help="Lateral CG offset from vehicle centre-line (positive = right).")

        s = meta.get("Xcg", {"min": 800, "max": 1400, "median": 1050})
        xcg = st.number_input("Longitudinal CG Position Xcg (mm)", min_value=500, max_value=2000,
                              value=int(s["median"]), step=5,
                              help="Distance of CG from front axle centre-line. Not currently used in models.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} mm")

    with col_b:
        st.markdown("**Chassis Geometry**")
        s = meta.get("Wheelbase", {"min": 2050, "max": 2165, "median": 2100})
        wheelbase = st.number_input("Wheelbase (mm)", min_value=1800, max_value=2500,
                                    value=int(s["median"]), step=10,
                                    help="Distance between front and rear axle centre-lines.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} mm")

        s = meta.get("Track_Width", {"min": 830, "max": 1500, "median": 1140})
        track_width = st.number_input("Track Width (mm)", min_value=700, max_value=1600,
                                      value=int(s["median"]), step=10,
                                      help="Lateral distance between tyre contact patches on the same axle.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} mm")

        st.markdown("**Tyres**")
        tire_make_opts = meta.get("categorical", {}).get("Tire_Make", ["MRF", "TVS", "CEAT"])
        tire_make = st.selectbox(
            "Tyre Manufacturer", options=tire_make_opts,
            help="CEAT is the OHE reference category (first-dropped). Unknown makes default to it.",
        )

        s = meta.get("Tire_Width", {"min": 95, "max": 120, "median": 114})
        tire_width = st.number_input("Tyre Width (mm)", min_value=80, max_value=160,
                                     value=int(s["median"]), step=5,
                                     help="Nominal section width (e.g. 120 in a 120/80-12 tyre).")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} mm")

        s = meta.get("Rim_Diameter", {"min": 8, "max": 12, "median": 12})
        rim_diameter = st.number_input("Rim Diameter (in)", min_value=6, max_value=16,
                                       value=int(s["median"]), step=1,
                                       help="Wheel rim diameter in inches.")
        st.caption(f"Training range: {s['min']:.0f}–{s['max']:.0f} in")

    with col_c:
        st.markdown("**Tyre Pressure**")
        s = meta.get("Front_Pressure", {"min": 26, "max": 70, "median": 34})
        front_pressure = st.number_input("Front Pressure (psi)", min_value=15, max_value=85,
                                          value=int(s["median"]), step=1)
        st.caption(f"Training: {s['min']:.0f}–{s['max']:.0f} psi")

        s = meta.get("Rear_Pressure", {"min": 26, "max": 73, "median": 46})
        rear_pressure = st.number_input("Rear Pressure (psi)", min_value=15, max_value=85,
                                         value=int(s["median"]), step=1)
        st.caption(f"Training: {s['min']:.0f}–{s['max']:.0f} psi")

        st.markdown("**Anti-Roll Bar (ARB)**")
        arb_diameter = st.number_input(
            "ARB Diameter (mm) — enter 0 if not fitted", min_value=0.0, max_value=25.0,
            value=0.0, step=0.5,
            help="Solid bar diameter. Torsional stiffness scales as D^4. Training range: 15–20 mm when fitted.",
        )
        if arb_diameter == 0:
            st.caption("No ARB fitted (ARB_Present = 0)")
        elif arb_diameter < 15:
            st.warning(f"ARB {arb_diameter} mm is below training range (15–20 mm). Extrapolation.")

        st.markdown("**Suspension**")
        susp_opts = meta.get("categorical", {}).get("Suspension_Configuration",
                                                    ["Existing_Shocker_Spring", "Modified_Shocker_Spring", "Unknown"])
        suspension_config = st.selectbox(
            "Suspension Configuration", options=susp_opts,
            index=susp_opts.index("Unknown") if "Unknown" in susp_opts else 0,
            help="Unknown = standard configuration (most common in training data).",
        )

        st.markdown("**Damper Configuration**")
        damper_opts = meta.get("categorical", {}).get("Damper_Configuration", ["Unknown"])
        damper_config = st.selectbox(
            "Damper Configuration", options=damper_opts,
            index=damper_opts.index("Unknown") if "Unknown" in damper_opts else 0,
            help="Not yet used in models — all current training data is 'Unknown'. Will be activated when damper measurements are collected.",
            disabled=True,
        )
        st.caption("No damper data collected yet — reserved for future use.")


# ── Validate and predict (runs on every rerun) ────────────────────────────────

raw_inputs = dict(
    front_load=float(front_load), rear_load=float(rear_load),
    zcg=float(zcg), wheelbase=float(wheelbase), track_width=float(track_width),
    tire_width=float(tire_width), rim_diameter=float(rim_diameter),
    front_pressure=float(front_pressure), rear_pressure=float(rear_pressure),
    arb_diameter=float(arb_diameter),
    vehicle_type=vehicle_type, tire_make=tire_make,
    suspension_config=suspension_config,
    test_condition=int(test_condition), ycg=float(ycg),
)

errors = validate_inputs(**raw_inputs)
if errors:
    for msg in errors:
        st.error(msg)
    st.stop()

try:
    all_features = build_inference_row(raw_inputs)
    X_rg = get_model_input(rg_model, all_features)
    X_ug = get_model_input(ug_model, all_features)
    predicted_rg = float(rg_model.predict(X_rg)[0])
    predicted_ug = float(ug_model.predict(X_ug)[0])
except Exception as exc:
    st.error(f"Prediction failed — {exc}")
    st.stop()


# ── Fill prediction container (top of Tab 1) ──────────────────────────────────

with pred_container:
    st.subheader("Predicted Results")

    # Pre-compute
    _rg_idx  = all_features["Physics_RG_Index"].iloc[0]
    _ug_idx  = all_features["UG_Physics_Index"].iloc[0]
    rg_label = "High roll tendency" if predicted_rg >= 10 else "Moderate roll tendency" if predicted_rg >= 5 else "Low roll tendency"
    ug_label = "Understeer" if predicted_ug > 0.2 else "Oversteer" if predicted_ug < -0.2 else "Near-neutral"
    rg_rmse_str = f"&plusmn; {rg_loocv_rmse:.2f}" if rg_loocv_rmse else "&mdash;"
    ug_rmse_str = f"&plusmn; {ug_loocv_rmse:.2f}" if ug_loocv_rmse else "&mdash;"

    # Extrapolation warnings
    zcg_max  = meta.get("Zcg", {}).get("max", 804)
    mass_max = meta.get("Mass", {}).get("max", 1664)
    tw_min   = meta.get("Track_Width", {}).get("min", 830)
    tw_max   = meta.get("Track_Width", {}).get("max", 1500)
    if zcg > zcg_max:
        st.warning(f"CG height ({zcg} mm) exceeds training maximum ({zcg_max:.0f} mm). Prediction extrapolates.")
    if mass > mass_max:
        st.warning(f"Total mass ({mass} kg) exceeds training maximum ({mass_max:.0f} kg). Prediction extrapolates.")
    if not (tw_min <= track_width <= tw_max):
        st.warning(f"Track width ({track_width} mm) is outside training range ({tw_min:.0f}–{tw_max:.0f} mm).")

    pc1, pc2 = st.columns(2, gap="large")

    with pc1:
        st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #C4D4E8; border-top:5px solid #1565C0;
            border-radius:12px; padding:24px; box-shadow:0 3px 12px rgba(21,101,192,0.11);">
  <div style="color:#1565C0; font-size:11px; font-weight:700; text-transform:uppercase;
              letter-spacing:1.8px; margin-bottom:14px;">Roll Gradient (RG)</div>
  <div style="color:#0D1B2A; font-size:44px; font-weight:800; line-height:1; margin-bottom:4px;">
    {predicted_rg:.3f} <span style="font-size:18px; font-weight:400; color:#4A6080;">deg/g</span>
  </div>
  <div style="color:#4A6080; font-size:12px; margin-bottom:16px;">
    {rg_rmse_str} deg/g model uncertainty (LOOCV RMSE)
  </div>
  <span style="background:#EBF1FA; color:#1565C0; border-radius:20px; padding:4px 14px;
               font-size:12px; font-weight:600;">{rg_label}</span>
</div>
""", unsafe_allow_html=True)

    with pc2:
        ug_warn_html = ""
        if ug_loocv_r2 is not None and ug_loocv_r2 < 0.3:
            ug_warn_html = f"""
  <div style="background:#FFF5F0; border:1px solid #F0C0A0; border-left:3px solid #E65100;
              border-radius:6px; padding:10px 12px; margin-top:14px;
              font-size:12px; color:#7A2E00; line-height:1.5;">
    <strong>Limited reliability</strong> &mdash; LOOCV R&sup2; = {ug_loocv_r2:.3f}.
    Tire cornering stiffness (C&alpha;) is not measured; treat as directional only.
  </div>"""

        st.markdown(f"""
<div style="background:#FFFFFF; border:1px solid #C4D4E8; border-top:5px solid #E65100;
            border-radius:12px; padding:24px; box-shadow:0 3px 12px rgba(230,81,0,0.09);">
  <div style="color:#E65100; font-size:11px; font-weight:700; text-transform:uppercase;
              letter-spacing:1.8px; margin-bottom:14px;">Understeer Gradient (UG)</div>
  <div style="color:#0D1B2A; font-size:44px; font-weight:800; line-height:1; margin-bottom:4px;">
    {predicted_ug:.3f} <span style="font-size:18px; font-weight:400; color:#4A6080;">deg/g</span>
  </div>
  <div style="color:#4A6080; font-size:12px; margin-bottom:16px;">
    {ug_rmse_str} deg/g model uncertainty (LOOCV RMSE)
    &nbsp;&middot;&nbsp; +ve = understeer &middot; &minus;ve = oversteer
  </div>
  <span style="background:#FFF0EB; color:#E65100; border-radius:20px; padding:4px 14px;
               font-size:12px; font-weight:600;">{ug_label}</span>
  {ug_warn_html}
</div>
""", unsafe_allow_html=True)

    st.divider()


# Feature table data (computed here so Tab 3 can use it)

_row = all_features.iloc[0]

def _fv(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)

rg_table = pd.DataFrame([
    {"Feature": "Roll_Index",               "Formula": "M x Zcg / Track_Width",        "Value": _fv(_row["Roll_Index"]),               "Effect on RG": "Higher roll moment / narrower track -> more body roll (up RG)"},
    {"Feature": "Track_Width_Squared",       "Formula": "Track_Width^2",                 "Value": _fv(_row["Track_Width_Squared"]),       "Effect on RG": "Wider track resists roll via spring stiffness -> less roll (down RG)"},
    {"Feature": "ARB_Stiffness_Index",       "Formula": "log1p(D^4 / Track_Width)",      "Value": _fv(_row["ARB_Stiffness_Index"]),       "Effect on RG": "Stiffer ARB adds torsional roll resistance -> less roll (down RG)"},
    {"Feature": "Tire_Width_Pressure_Ratio", "Formula": "Tire_Width / Front_Pressure",   "Value": _fv(_row["Tire_Width_Pressure_Ratio"]), "Effect on RG": "Wider / lower-pressure front tyre -> reduced front roll stiffness (down RG in dataset)"},
    {"Feature": "ARB_Present",               "Formula": "1 if ARB_Diameter > 0 else 0", "Value": _fv(_row["ARB_Present"]),               "Effect on RG": "Any ARB adds supplementary roll resistance"},
    {"Feature": "Type",                      "Formula": "Categorical: 3W / 4W",          "Value": _fv(_row["Type"]),                      "Effect on RG": "4W vehicles have wider track and lower relative CG -> lower RG on average"},
])

ug_table = pd.DataFrame([
    {"Feature": "Front_WD",            "Formula": "Front_Load / Total_Mass",            "Value": _fv(_row["Front_WD"]),            "Effect on UG": "Higher front weight fraction shifts lateral load transfer balance"},
    {"Feature": "Roll_Stiffness_Ratio","Formula": "log1p(D^4 / TW) / TW",   "Value": _fv(_row["Roll_Stiffness_Ratio"]),"Effect on UG": "Higher ARB stiffness per unit track width shifts understeer-oversteer balance"},
    {"Feature": "ARB_Present",         "Formula": "1 if ARB_Diameter > 0 else 0",      "Value": _fv(_row["ARB_Present"]),         "Effect on UG": "ARB redistributes lateral load transfer between axles"},
    {"Feature": "Zcg_Wheelbase_Ratio", "Formula": "Zcg / Wheelbase",                   "Value": _fv(_row["Zcg_Wheelbase_Ratio"]), "Effect on UG": "Higher CG / wheelbase -> more pitch load shift -> oversteer tendency (down UG)"},
    {"Feature": "Tire_Make",           "Formula": "Categorical: CEAT / MRF / TVS",     "Value": _fv(_row["Tire_Make"]),           "Effect on UG": "Tyre compound determines lateral cornering stiffness (Ca) — primary UG driver"},
    {"Feature": "Type",                "Formula": "Categorical: 3W / 4W",               "Value": _fv(_row["Type"]),                "Effect on UG": "3W and 4W have fundamentally different lateral dynamics"},
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════════

with tab2:

    # ── LOOCV Summary ─────────────────────────────────────────────────────────

    st.subheader("LOOCV Performance Summary")
    _n_folds = len(rg_meta.get("training_vehicles", [])) or "?"
    st.caption(
        f"Vehicle-level Leave-One-Group-Out CV ({_n_folds} folds). "
        "Each fold holds out all runs of one vehicle — no vehicle leaks into training."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        def _fmt(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        rg_rmse_v = _fmt(rg_meta.get("loocv_rmse"))
        rg_mae_v  = _fmt(rg_meta.get("loocv_mae"))
        rg_r2_v   = _fmt(rg_meta.get("loocv_r2"))
        st.markdown(f"""
        <div style="background:#F2F8F2; border:1px solid #B8D8B8; border-left:5px solid #2E7D32;
                    border-radius:10px; padding:20px; margin-bottom:8px;
                    box-shadow:0 2px 8px rgba(46,125,50,0.10);">
            <div style="color:#1B5E20; font-size:12px; font-weight:700;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:14px;">
                Roll Gradient — Ridge Regression
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; text-align:center;">
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #C8E0C8;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV RMSE</div>
                    <div style="color:#0D1B2A; font-size:24px; font-weight:800;">{rg_rmse_v}</div>
                    <div style="color:#4A6080; font-size:10px;">deg/g</div>
                </div>
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #C8E0C8;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV MAE</div>
                    <div style="color:#0D1B2A; font-size:24px; font-weight:800;">{rg_mae_v}</div>
                    <div style="color:#4A6080; font-size:10px;">deg/g</div>
                </div>
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #C8E0C8;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV R²</div>
                    <div style="color:#2E7D32; font-size:24px; font-weight:800;">{rg_r2_v}</div>
                    <div style="color:#4A6080; font-size:10px;">variance explained</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        ug_rmse_v = _fmt(ug_meta.get("loocv_rmse"))
        ug_mae_v  = _fmt(ug_meta.get("loocv_mae"))
        ug_r2_v   = _fmt(ug_meta.get("loocv_r2"))
        st.markdown(f"""
        <div style="background:#FFF5F0; border:1px solid #F0C0A0; border-left:5px solid #E65100;
                    border-radius:10px; padding:20px; margin-bottom:8px;
                    box-shadow:0 2px 8px rgba(230,81,0,0.10);">
            <div style="color:#BF360C; font-size:12px; font-weight:700;
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:14px;">
                Understeer Gradient — MLP Neural Network
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; text-align:center;">
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #F0C8B0;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV RMSE</div>
                    <div style="color:#0D1B2A; font-size:24px; font-weight:800;">{ug_rmse_v}</div>
                    <div style="color:#4A6080; font-size:10px;">deg/g</div>
                </div>
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #F0C8B0;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV MAE</div>
                    <div style="color:#0D1B2A; font-size:24px; font-weight:800;">{ug_mae_v}</div>
                    <div style="color:#4A6080; font-size:10px;">deg/g</div>
                </div>
                <div style="background:#FFFFFF; border-radius:8px; padding:12px 8px;
                            border:1px solid #F0C8B0;">
                    <div style="color:#4A6080; font-size:10px; text-transform:uppercase;
                                letter-spacing:1px; margin-bottom:4px;">LOOCV R²</div>
                    <div style="color:#E65100; font-size:24px; font-weight:800;">{ug_r2_v}</div>
                    <div style="color:#4A6080; font-size:10px;">variance explained</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Model Comparison ──────────────────────────────────────────────────────

    st.subheader("Model Comparison")
    show_plot("model_comparison.png",
              "Grouped bar: RMSE and R² across Ridge, MLP, and Random Forest for both targets.")

    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        rg_csv = os.path.join("Models", "model_comparison_rg.csv")
        if os.path.exists(rg_csv):
            st.markdown("**Roll Gradient**")
            st.dataframe(pd.read_csv(rg_csv).round(4), use_container_width=True, hide_index=True)
    with comp_c2:
        ug_csv = os.path.join("Models", "model_comparison_ug.csv")
        if os.path.exists(ug_csv):
            st.markdown("**Understeer Gradient**")
            st.dataframe(pd.read_csv(ug_csv).round(4), use_container_width=True, hide_index=True)
    st.caption(
        "Nested CV: GridSearchCV(cv=5) inside vehicle-level LOOCV. "
        "Outer folds produce honest out-of-sample predictions — no hyperparameter leakage."
    )

    st.divider()

    # ── Actual vs Predicted ───────────────────────────────────────────────────

    st.subheader("Actual vs Predicted (LOOCV)")
    show_plot("actual_vs_predicted.png",
              "Each point is one test run. Colour = vehicle. Dashed line = perfect prediction.")

    st.divider()

    # ── Residuals + QQ ────────────────────────────────────────────────────────

    st.subheader("Residual Analysis")
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        show_plot("residuals.png",
                  "Residuals vs predicted (left) and residual distribution (right) for both targets.")
    with res_c2:
        show_plot("qq_plots.png",
                  "Q-Q plots of LOOCV residuals. Shapiro-Wilk p-value annotated.")

    st.divider()

    # ── Per-Vehicle Performance ────────────────────────────────────────────────

    st.subheader("Per-Vehicle LOOCV RMSE")
    show_plot("per_vehicle_rmse.png",
              "Each vehicle is held out once. RG (blue) vs UG (amber) RMSE per vehicle.")
    st.caption(
        "High per-vehicle RMSE for a couple of vehicles (UG) reflects missing tire cornering "
        "stiffness data rather than a model failure."
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Feature Analysis
# ════════════════════════════════════════════════════════════════════════════════

with tab3:

    # ── Engineered Feature Values ─────────────────────────────────────────────

    st.subheader("Engineered Feature Values")
    st.caption("Derived from raw inputs via physics-motivated transformations. Values update live when you change inputs in the Prediction tab.")

    ft1, ft2 = st.columns(2)
    with ft1:
        st.markdown("**Roll Gradient — Model Features**")
        st.dataframe(rg_table, use_container_width=True, hide_index=True)
    with ft2:
        st.markdown("**Understeer Gradient — Model Features**")
        st.dataframe(ug_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── Feature Importance ────────────────────────────────────────────────────

    st.subheader("Feature Importance")
    fi_c1, fi_c2 = st.columns(2)
    with fi_c1:
        show_plot("feature_importance_rg.png",
                  "Ridge standardised coefficients (Yeo-Johnson transformed target).")
    with fi_c2:
        show_plot("feature_importance_ug.png",
                  "MLP permutation importance: mean RMSE increase when each feature is shuffled.")

    st.divider()

    # ── Correlation with Targets ──────────────────────────────────────────────

    st.subheader("Feature Correlations")
    corr_c1, corr_c2 = st.columns(2)
    with corr_c1:
        show_plot("correlation_with_targets.png",
                  "Pearson and Spearman correlation of all features with RG and UG.")
    with corr_c2:
        show_plot("mutual_information_scores.png",
                  "Mutual information captures non-linear associations ignored by Pearson r.")

    st.divider()

    # ── Engineered Features Scatter ───────────────────────────────────────────

    st.subheader("Engineered Feature Relationships")
    show_plot("engineered_features_scatter.png",
              "Scatter plots of key engineered features vs both targets, coloured by vehicle type.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Dataset & Methodology
# ════════════════════════════════════════════════════════════════════════════════

with tab4:

    # ── Dataset Overview ──────────────────────────────────────────────────────

    st.subheader("Dataset Overview")
    outlier_list      = meta.get("outlier_vehicles", [])
    partially_flagged = meta.get("partially_flagged_vehicles", {})
    training_vehs      = rg_meta.get("training_vehicles", [])
    n_fully_excluded    = len(outlier_list)

    ds1, ds2, ds3, ds4 = st.columns(4)
    for col, lbl, val, sub, acc in [
        (ds1, "Raw Observations", str(n_total), "47 rows × 21 columns", "#1565C0"),
        (ds2, "Clean for Training", str(n_clean), f"{n_rows_out} outlier runs removed", "#1E88E5"),
        (ds3, "Training Vehicles", str(len(training_vehs)), "Leave-One-Group-Out CV", "#42A5F5"),
        (ds4, "Fully Excluded Vehicles", str(n_fully_excluded), "Mahalanobis distance α=0.01", "#FF9800"),
    ]:
        with col:
            st.markdown(stat_card(lbl, val, sub, acc), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    if outlier_list:
        st.caption(f"Fully excluded (every run anomalous): {', '.join(outlier_list)}")
    if partially_flagged:
        parts = [
            f"{veh} ({info['flagged_runs']}/{info['total_runs']} runs excluded)"
            for veh, info in partially_flagged.items()
        ]
        st.caption(f"Partially flagged (some runs kept): {', '.join(parts)}")
    if training_vehs:
        st.caption(f"Training vehicles: {', '.join(training_vehs)}")

    st.divider()

    # ── Target Distributions ──────────────────────────────────────────────────

    st.subheader("Target Variable Distributions")
    show_plot("target_distributions.png",
              "RG and UG distributions in the clean dataset, coloured by vehicle type (3W vs 4W).")

    st.divider()

    # ── Correlation Heatmap ───────────────────────────────────────────────────

    st.subheader("Feature Correlation Heatmap")
    show_plot("correlation_heatmap_features.png",
              f"Spearman correlation matrix for model features and both targets (clean dataset, n={n_clean}).")

    st.divider()

    # ── Learning Curves ───────────────────────────────────────────────────────

    st.subheader("Learning Curves")
    lc1, lc2 = st.columns(2)
    with lc1:
        show_plot("learning_curve.png", "Roll Gradient — validation MAE vs training size.")
    with lc2:
        show_plot("ug_learning_curve.png", "Understeer Gradient — validation MAE vs training size.")
    st.caption(
        f"Validation error still decreasing at n={n_clean} for both targets, confirming that "
        "dataset size is the primary bottleneck for RG. For UG, the ceiling is primarily "
        "due to missing tire lateral cornering stiffness (Cα)."
    )

    st.divider()

    # ── Physics Background ────────────────────────────────────────────────────

    st.subheader("Physics Background")
    phys1, phys2 = st.columns(2)

    _rg_idx  = all_features["Physics_RG_Index"].iloc[0]
    _ug_idx  = all_features["UG_Physics_Index"].iloc[0]

    with phys1:
        st.markdown("**Roll Gradient**")
        st.code(
            "RG  proportional to  M * Zcg / K_phi\n"
            "K_phi  proportional to  P_avg * Tire_Width * Track_Width^2\n\n"
            "Physics_RG_Index = M*Zcg / (P_avg * TireW * TW^2)\n"
            f"                 = {_rg_idx:.4e}  (current input)",
            language="text",
        )
    with phys2:
        st.markdown("**Understeer Gradient**")
        st.code(
            "UG  proportional to  W_f/C_af  -  W_r/C_ar\n"
            "C_a  proportional to  Tire_Width * Pressure\n\n"
            "UG_Physics_Index = (FWD/FP - RWD/RP) / TireW\n"
            f"                 = {_ug_idx:.4f}  (current input)",
            language="text",
        )

    st.divider()

    # ── Methodology ───────────────────────────────────────────────────────────

    st.subheader("Modelling Methodology")
    ug_r2_disp = f"{ug_loocv_r2:.2f}" if ug_loocv_r2 is not None else "—"
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
**Cross-Validation**
- Vehicle-level Leave-One-Group-Out (LOGO) as outer loop
- {len(training_vehs)} folds — one per unique vehicle
- GridSearchCV(cv=5) as inner loop for hyperparameter tuning
- Outer loop is leakage-free (preprocessing fitted per training fold only);
  inner-loop tuning has a known un-grouped fold overlap — see `src/training.py`

**Models Evaluated**
- Ridge Regression + Yeo-Johnson target transform (best for RG)
- Random Forest (max_depth 2–3, n_estimators=200)
- MLP Neural Network (best for UG)
        """)
    with m2:
        st.markdown(f"""
**Feature Engineering**
- Physics-motivated ratios: Roll_Index, ARB_Stiffness_Index, Tire_Width_Pressure_Ratio
- All features derived in `src/feature_engineering.py` (single source of truth)
- Outlier detection: Robust Mahalanobis distance (MinCovDet, α=0.01), row-level
- Xcg reference-frame correction applied to one vehicle's raw records

**Known Limitations**
- n={n_clean} after outlier removal (small dataset)
- UG R²={ug_r2_disp} — tire lateral cornering stiffness (Cα) not measured
- Per-vehicle R² highly variable (LOGO variance with {len(training_vehs)} folds)
- Results should not be extrapolated to vehicles outside the training fleet
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data & Retrain
# ════════════════════════════════════════════════════════════════════════════════

with tab5:

    st.subheader("Edit Raw Dataset")
    st.caption(
        "Edit values or add/delete rows below, then run a retrain preview. "
        "Nothing on disk changes until you click Apply — the retrain runs "
        "against a staged copy first so you can compare LOOCV metrics "
        "before committing."
    )

    if "raw_df" not in st.session_state:
        st.session_state.raw_df = data_pipeline.load_raw_dataset(RAW_EXCEL_PATH)

    raw_column_config = {
        "Front_Load":         st.column_config.NumberColumn("Front_Load", min_value=0.0, help="kg"),
        "Rear_Load":          st.column_config.NumberColumn("Rear_Load", min_value=0.0, help="kg"),
        "Front_Pressure":     st.column_config.NumberColumn("Front_Pressure", min_value=0.0, help="psi"),
        "Rear_Pressure":      st.column_config.NumberColumn("Rear_Pressure", min_value=0.0, help="psi"),
        "Track_Width":        st.column_config.NumberColumn("Track_Width", min_value=0.0, help="mm"),
        "Tire_Width":         st.column_config.NumberColumn("Tire_Width", min_value=0.0, help="mm"),
        "Wheelbase":          st.column_config.NumberColumn("Wheelbase", min_value=0.0, help="mm"),
        "Rim_Diameter":       st.column_config.NumberColumn("Rim_Diameter", min_value=0.0, help="in"),
        "ARB_Diameter":       st.column_config.NumberColumn("ARB_Diameter", min_value=0.0, help="mm, 0 = not fitted"),
        "Zcg":                st.column_config.NumberColumn("Zcg", help="mm"),
        "Ycg":                st.column_config.NumberColumn("Ycg", help="mm"),
        "Xcg":                st.column_config.NumberColumn("Xcg", help="mm, front-axle reference"),
        "Mass":               st.column_config.NumberColumn("Mass", disabled=True, help="Auto-computed = Front_Load + Rear_Load"),
        "Roll Gradient":      st.column_config.NumberColumn("Roll Gradient", help="deg/g (training target)"),
        "Understeer Gradient": st.column_config.NumberColumn("Understeer Gradient", help="deg/g (training target)"),
    }

    edited_raw_df = st.data_editor(
        st.session_state.raw_df,
        num_rows="dynamic",
        use_container_width=True,
        key="raw_data_editor",
        column_config=raw_column_config,
    )

    btn_c1, btn_c2 = st.columns([1, 1])
    with btn_c1:
        run_clicked = st.button("🔄 Run Retrain Preview", type="primary")
    with btn_c2:
        if st.button("Reload from disk (discard edits)"):
            st.session_state.raw_df = data_pipeline.load_raw_dataset(RAW_EXCEL_PATH)
            st.session_state.pop("retrain_result", None)
            st.rerun()

    if run_clicked:
        validation_errors = data_pipeline.validate_raw_dataset(edited_raw_df)
        if validation_errors:
            for msg in validation_errors:
                st.error(msg)
        else:
            with st.spinner(
                "Running full retrain — feature engineering, outlier "
                "detection, nested LOOCV for both targets. This can take a "
                "couple of minutes on this dataset size."
            ):
                try:
                    retrain_result = data_pipeline.run_full_retrain(edited_raw_df, STAGING_DIR)
                    st.session_state.retrain_result = retrain_result
                    st.session_state.retrain_raw_df = edited_raw_df.copy()
                except Exception as exc:
                    st.error(f"Retrain failed — {exc}")

    if "retrain_result" in st.session_state:
        st.divider()
        st.subheader("Retrain Preview")

        result       = st.session_state.retrain_result
        new_raw_df   = st.session_state.retrain_raw_df
        live_raw_df  = data_pipeline.load_raw_dataset(RAW_EXCEL_PATH)
        row_delta    = len(new_raw_df) - len(live_raw_df)
        st.caption(f"Raw rows: {len(live_raw_df)} → {len(new_raw_df)} ({row_delta:+d})")

        apply_choice = {}
        cmp_c1, cmp_c2 = st.columns(2)
        for col, key, label, live_meta in [
            (cmp_c1, "rg", "Roll Gradient", rg_meta),
            (cmp_c2, "ug", "Understeer Gradient", ug_meta),
        ]:
            with col:
                new_meta  = result[key]["metadata"]
                old_rmse  = live_meta.get("loocv_rmse")
                new_rmse  = new_meta["loocv_rmse"]
                old_r2    = live_meta.get("loocv_r2")
                new_r2    = new_meta["loocv_r2"]
                improved  = old_rmse is not None and new_rmse < old_rmse
                icon      = "✅" if improved else "⚠️"

                st.markdown(f"**{label}** {icon}")
                old_rmse_str = f"{old_rmse:.3f}" if old_rmse is not None else "—"
                old_r2_str   = f"{old_r2:.3f}" if old_r2 is not None else "—"
                st.write(f"LOOCV RMSE: {old_rmse_str} → **{new_rmse:.3f}**")
                st.write(f"LOOCV R²: {old_r2_str} → **{new_r2:.3f}**")
                st.caption(f"Champion model: {new_meta['best_model_class']}")
                apply_choice[key] = st.checkbox(
                    f"Apply new {label} model", value=improved, key=f"apply_{key}"
                )

        st.caption(
            "The raw data edit and processed dataset are promoted either way "
            "on Apply — only the two checkboxes above control whether each "
            "target's model is replaced, so one target regressing doesn't "
            "force reverting the other."
        )

        apply_col, discard_col = st.columns(2)
        with apply_col:
            if st.button("✅ Apply", type="primary"):
                try:
                    data_pipeline.promote_staging(
                        STAGING_DIR, new_raw_df, RAW_EXCEL_PATH,
                        promote_rg=apply_choice["rg"], promote_ug=apply_choice["ug"],
                    )
                    with st.spinner("Regenerating plots..."):
                        subprocess.run(
                            [sys.executable, "generate_plots.py"],
                            cwd=os.path.abspath(os.path.dirname(__file__)),
                            check=False,
                        )
                    data_pipeline.discard_staging(STAGING_DIR)
                    st.session_state.pop("retrain_result", None)
                    st.session_state.pop("retrain_raw_df", None)
                    st.session_state.raw_df = data_pipeline.load_raw_dataset(RAW_EXCEL_PATH)
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Applied — reloading dashboard.")
                    st.rerun()
                except PermissionError as exc:
                    st.error(str(exc))
        with discard_col:
            if st.button("🗑️ Discard"):
                data_pipeline.discard_staging(STAGING_DIR)
                st.session_state.pop("retrain_result", None)
                st.session_state.pop("retrain_raw_df", None)
                st.info("Discarded — no live files were changed.")
                st.rerun()
