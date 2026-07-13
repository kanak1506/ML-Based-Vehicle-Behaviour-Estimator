# Vehicle Dynamics ML — Technical Report

### Predicting Roll Gradient and Understeer Gradient of Three- and Four-Wheeled Vehicles from Static Design Parameters using Machine Learning

---

> **Scope note.** This report explains the vehicle-dynamics physics and the
> ML methodology behind the project in depth, illustrated with the project's
> own real evaluation metrics (RMSE, MAE, R², correlation values, feature
> importances) so the findings are concrete rather than hypothetical. The
> underlying dataset and trained model files are proprietary and are not
> published alongside this report; wherever the original analysis referenced
> an individual vehicle by name, it has been replaced here with a generic
> label (Vehicle 1, Vehicle 2, …) so the methodology and the numbers can
> still be shown honestly without identifying anything about the source
> fleet.

---

## Project Overview

This project builds a machine-learning system that predicts two handling
characteristics of a vehicle — **Roll Gradient (RG)** and **Understeer
Gradient (UG)** — directly from static, easily-measured design parameters
(mass distribution, centre-of-gravity height, track width, wheelbase, tyre
size and pressure, anti-roll bar diameter, tyre brand, and vehicle type).

Both quantities are normally obtained only by building a physical prototype
and driving it through standardised circular-path tests. This project asks:
*can we predict them early, from a spreadsheet of design numbers, before a
prototype exists?* The answer built here is a qualified **yes for Roll
Gradient** (R² ≈ 0.76) and a **directional yes, not yet a precise one, for
Understeer Gradient** (R² ≈ 0.21), because the single physical quantity that
dominates understeer — tyre cornering stiffness — is not present anywhere in
the dataset.

The codebase contains:

- A **data pipeline** that cleans a 47-row/21-column spreadsheet of real
  vehicle test data down to 35 trustworthy rows and engineers 12
  physics-motivated features.
- **Model-training code** that fits and rigorously validates three candidate
  regression algorithms per target using vehicle-aware cross-validation.
- A **shared library** (`src/feature_engineering.py`, `src/training.py`) that
  is the single source of truth for every feature formula and every modelling
  decision, imported identically by training code and by the live app.
- A **Streamlit dashboard** (`app.py`) that lets an engineer type in a new
  vehicle's specifications and get an instant RG/UG prediction with
  uncertainty bounds and extrapolation warnings.
- **Plot-generation scripts** (`generate_plots.py`,
  `feature_contribution_plots.py`) that produce every diagnostic chart shown
  in the dashboard and reproduced in this report.

This report documents the entire pipeline end-to-end — the vehicle-dynamics
physics behind every formula, the mathematics behind every model, and the
concrete numerical results obtained — so that it is understandable to a
mechanical-engineering reader with only introductory exposure to machine
learning.

---

## Executive Summary

| Item | Roll Gradient (RG) | Understeer Gradient (UG) |
|---|---|---|
| Physical meaning | Body roll angle per g of lateral acceleration | Extra steering angle needed per g of lateral acceleration |
| Units | deg / g | deg / g |
| Best model | Ridge Regression (L2-regularised linear model, Yeo–Johnson target transform) | MLP Neural Network (1 hidden layer, 4 neurons) |
| Validation | Vehicle-level Leave-One-Group-Out (10 folds) + nested `GridSearchCV` | Same |
| LOOCV RMSE | **1.46 deg/g** | **3.99 deg/g** |
| LOOCV MAE | 1.12 deg/g | 2.91 deg/g |
| LOOCV R² | **0.762** | **0.207** |
| Model inputs | 6 features: `Roll_Index`, `Track_Width_Squared`, `ARB_Stiffness_Index`, `Tire_Width_Pressure_Ratio`, `ARB_Present`, `Type` | 6 features: `Front_WD`, `Roll_Stiffness_Ratio`, `ARB_Present`, `Zcg_Wheelbase_Ratio`, `Tire_Make`, `Type` |
| Dominant limiting factor | Dataset size (learning curve still descending at n = 35) | Missing tyre cornering-stiffness (Cα) measurement |
| Practical use | Reliable enough for early-stage ranking of design options | Directional only — tells you *which way* handling shifts, not the exact number |

The project's central engineering finding is **not** "which algorithm wins" —
it is that **Roll Gradient is a near-linear function of the measured
variables, while Understeer Gradient fundamentally is not**, because the
dataset is missing the one variable (tyre lateral cornering stiffness) that
the textbook physics equation says should dominate it. No amount of better
modelling can fix a missing input variable; this is documented honestly
throughout the code, the dashboard, and this report.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Vehicle Dynamics Background](#2-vehicle-dynamics-background)
3. [Roll Gradient](#3-roll-gradient)
4. [Understeer Gradient](#4-understeer-gradient)
5. [Dataset](#5-dataset)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Machine Learning Concepts](#7-machine-learning-concepts)
8. [Models Used](#8-models-used)
9. [Training Pipeline](#9-training-pipeline)
10. [Model Comparison](#10-model-comparison)
11. [Streamlit Dashboard](#11-streamlit-dashboard)
12. [Code Architecture](#12-code-architecture)
13. [Mathematical Appendix](#13-mathematical-appendix)
14. [Physics Appendix](#14-physics-appendix)
15. [ML Appendix](#15-ml-appendix)
16. [Results](#16-results)
17. [Future Work](#17-future-work)
18. [References](#18-references)

---

## 1. Problem Statement

### 1.1 Why this project exists

Every new vehicle — a three-wheeled cargo vehicle or a four-wheeled
commercial van — has to "handle" acceptably: it must go around a corner
predictably, without leaning over alarmingly or requiring the driver to fight
the steering wheel. Two numbers summarise this behaviour for engineers:

- **Roll Gradient (RG)** — how much the vehicle body tips sideways in a
  corner.
- **Understeer Gradient (UG)** — whether the vehicle tends to run wide
  (understeer) or spin-in (oversteer) as cornering speed increases.

Both numbers are legally and practically important: too much roll makes a
top-heavy cargo vehicle prone to rollover; too much oversteer makes a vehicle
dangerous for an average driver to control near the limit.

### 1.2 The current industry workflow

In the traditional workflow:

1. A vehicle is designed on paper/CAD with a target mass, CG height, track
   width, wheelbase, spring rates, anti-roll bar, and tyre specification.
2. A **physical prototype** is built.
3. The prototype is instrumented (steering-wheel angle sensor, lateral
   accelerometer, roll-angle sensor) and driven on a test track through a
   **constant-radius circular test** (this project's dataset records circle
   radii of 10 m and 30 m, consistent with the SAE J266 steady-state
   circular test procedure).
4. RG and UG are computed from the recorded steering angle, roll angle, and
   lateral acceleration as the vehicle speeds up around the fixed circle.
5. If the numbers are unacceptable, the *physical* vehicle is re-engineered
   (different springs, anti-roll bar, tyres) and **re-built and re-tested**.

### 1.3 Why physical testing is expensive

Step 2–4 above is the expensive part:

- Building even one prototype costs significant time and money (tooling,
  parts, assembly).
- Track testing requires a professional test track, instrumented sensors,
  and trained drivers.
- Each design iteration (new spring, new anti-roll bar, new tyre) in the
  traditional workflow means **another physical build-and-test cycle**.
- Early in a vehicle programme, engineers often want to compare 5–10 design
  variants (e.g. "what if we go 20 mm wider on track width, or add a 15 mm
  anti-roll bar?") — physically building all of them is not realistic.

### 1.4 The problem this project solves

This project asks whether RG and UG can be **predicted from the design
sheet alone**, before a prototype is built, using a statistical model trained
on **historical test data from vehicles that were already tested**. If such a
model is accurate enough, an engineer can:

- Screen many hypothetical design variants in seconds inside a dashboard.
- Get an early "is this design roughly OK" signal well before committing to
  a physical build.
- Understand *which* design parameters push RG/UG up or down, and by how
  much (interpretable feature effects), which is exactly what the
  dashboard's "Feature Analysis" tab and the Ridge coefficient plot provide.

The training data for this exercise is 47 real test runs across 15 distinct
vehicle chassis/configurations — a genuinely small sample, which shapes
almost every methodological decision in the codebase (described in
Sections 5–10).

---

## 2. Vehicle Dynamics Background

This section explains, in plain language, the vocabulary and physics that
everything else in the report builds on. A vehicle moving on a road can be
described by motion along and rotation about three axes (this is the
standard **SAE vehicle axis system**):

| Axis | Translation | Rotation |
|---|---|---|
| Longitudinal (x, nose-to-tail) | Acceleration / braking | **Roll** — tipping sideways |
| Lateral (y, side-to-side) | Cornering (lateral acceleration) | **Pitch** — nose diving / squatting |
| Vertical (z, up-down) | Bouncing (ride) | **Yaw** — rotating about a vertical axis (turning) |

### 2.1 Roll, pitch, and yaw

- **Roll** is the body's tendency to tip toward the outside of a turn — like
  a ship rolling in waves. It is resisted by the suspension springs and any
  anti-roll bar.
- **Pitch** is the nose-down "dive" under braking or nose-up "squat" under
  acceleration. It is not modelled directly in this project's targets, but the
  ratio of CG height to wheelbase (used in the `Zcg_Wheelbase_Ratio` feature)
  is a proxy for how strongly load shifts front-to-rear, which indirectly
  affects understeer.
- **Yaw** is the rotation that actually turns the vehicle around a corner.
  The rate of yaw rotation, together with the vehicle's forward speed,
  determines the radius of the turn.

### 2.2 Lateral acceleration ($a_y$)

When a vehicle goes around a curve of radius $R$ at speed $V$, it must
accelerate continuously toward the centre of the curve. This centre-seeking
acceleration is the **lateral acceleration**:

$$
a_y = \frac{V^2}{R}
$$

It is conventionally expressed in units of "g" (multiples of gravitational
acceleration, $g = 9.81\ \text{m/s}^2$), because both roll angle and the
extra steering angle needed scale almost linearly with $a_y$ expressed this
way — which is exactly why RG and UG are defined as "something per g."

### 2.3 Centre of gravity (CG), track width, wheelbase

- **Centre of Gravity (CG)** is the single point where the vehicle's entire
  weight can be considered to act. Its position is described by three
  coordinates in this dataset:
  - $X_{cg}$ — longitudinal position (distance from the front axle).
  - $Y_{cg}$ — lateral offset from the vehicle centreline (should be
    near zero for a symmetric vehicle; small manufacturing/loading offsets
    are common in practice).
  - $Z_{cg}$ — height above the ground. This is the single most important
    geometric number for roll behaviour: the higher the CG, the larger the
    overturning moment created by lateral acceleration.
- **Track width** ($T_W$) is the side-to-side distance between the centres
  of the left and right tyre contact patches on one axle. A wider track
  creates a longer lever arm resisting roll and load transfer — this is why
  track width is one of the strongest predictors of RG.
- **Wheelbase** ($L$) is the front-to-rear distance between axle centre
  lines. It sets the basic cornering geometry (Section 4.1) and, combined
  with CG height, describes how weight shifts between front and rear
  axles under acceleration/braking (`Zcg_Wheelbase_Ratio` feature).

### 2.4 Load transfer

When a vehicle corners, the outer tyres are pushed down harder and the inner
tyres are unloaded — this is **lateral load transfer**, and it happens
because the CG is above the ground and the lateral (inertial) force acts at
CG height while the tyres react at ground level, creating a moment that must
be balanced by extra vertical load on the outer wheels. The same idea applies
longitudinally (**pitch/longitudinal load transfer**) under acceleration and
braking. Load transfer matters for both RG (it is resisted by roll stiffness)
and UG (it changes each axle's *effective* cornering stiffness, since tyres
do not gain grip in direct proportion to added load — a real, physical,
non-linear tyre effect called *load sensitivity*, which is one reason a pure
weight-ratio feature like `Front_WD` only partially explains understeer).

### 2.5 Tyre slip angle and cornering stiffness

A tyre does not point exactly where it rolls when generating lateral (grip)
force. The small angular difference between where the tyre is aimed and
where it actually travels is the **slip angle** ($\alpha$). For modest slip
angles, the lateral force a tyre produces is approximately proportional to
slip angle:

$$
F_y \approx C_\alpha \cdot \alpha
$$

where $C_\alpha$ is the **tyre cornering stiffness** (force per degree of
slip angle) — a property of the tyre's construction, compound, size, and
inflation pressure, and also of the vertical load on it. **This quantity is
never directly measured in this project's dataset** — it can only be
inferred indirectly through *proxies* like tyre width and pressure. This gap
is the single biggest limitation of the Understeer Gradient model (explained
fully in Section 4 and revisited throughout the report).

### 2.6 Cornering, steering, and the "extra angle"

In a hypothetical vehicle with infinitely stiff, non-slipping tyres, the
front wheels would simply need to point at an angle equal to $L/R$ (in
radians) to trace a circle of radius $R$ — this is the **Ackermann steering
angle**, the purely geometric baseline. Real tyres need slip angles to
generate the cornering force, so the driver must add (or subtract) some
extra steering angle beyond the Ackermann baseline. **Understeer Gradient is
exactly the rate at which that extra angle grows with lateral
acceleration** (full derivation in Section 4).

### 2.7 Roll stiffness and suspension effects

**Roll stiffness** ($K_\phi$) is the total torque required to roll the
vehicle body by one degree, contributed by:

- The **suspension springs**, acting through the track width (a wider
  track means the same spring force creates more resisting torque — this
  is the physical reason `Track_Width_Squared` appears in the Roll Gradient
  formula).
- The **anti-roll bar (ARB)** — a torsion bar connecting the left and right
  suspension that only resists *body roll*, not simultaneous (both-wheels)
  bump motion. A stiffer/thicker ARB adds roll resistance without stiffening
  the ride over bumps, which is why ARBs are a favourite tuning tool. ARB
  torsional stiffness scales with the **fourth power of its diameter**
  (`ARB_Stiffness_Index` uses this).
- The **tyre's own vertical compliance** — a softer/wider tyre at lower
  pressure deflects more under the same load, subtly reducing the
  *effective* roll stiffness felt by the body (`Tire_Width_Pressure_Ratio`
  captures this).

### 2.8 Understeer, oversteer, and neutral steer

These three terms describe how a vehicle's cornering behaviour changes as
speed (and therefore lateral acceleration) increases, holding the steering
wheel angle *conceptually* fixed while asking "does the car turn more
tightly, the same, or less tightly than the geometric Ackermann angle would
suggest?":

| Behaviour | UG value | What happens as speed rises in a corner |
|---|---|---|
| **Understeer** | UG > 0 | The vehicle needs *more* steering angle than Ackermann; it tends to run wide (nose pushes toward the outside of the corner). Front tyres reach their slip-angle limit before the rears. |
| **Neutral steer** | UG = 0 | The extra steering angle needed stays constant; front and rear axles lose grip together. |
| **Oversteer** | UG < 0 | The vehicle needs *less* steering angle than Ackermann, or even opposite-lock; the rear end tends to swing wide first (yaw rotates faster than the driver commands). |

Mild, predictable understeer is the traditional safe default for
mass-market and commercial vehicles (it is self-correcting: an inattentive
driver who does nothing simply runs wide slowly, rather than spinning),
which is why UG is such a closely tracked handling metric in industry.

---

## 3. Roll Gradient

### 3.1 Definition

**Roll Gradient (RG)** is the rate at which the vehicle body rolls (tips
sideways) per unit of lateral acceleration, during steady-state cornering:

$$
RG = \frac{\Delta \phi}{\Delta a_y} \quad \left[\frac{\deg}{g}\right]
$$

where:

- $\phi$ = body roll angle, in degrees, measured relative to the road.
- $a_y$ = lateral acceleration, expressed in g's.

### 3.2 Physical meaning

RG answers the question: *"for every extra 0.1 g of hard cornering, how many
extra degrees does the body lean over?"* A small RG means a flat, stable
vehicle that inspires driver confidence and resists rollover; a large RG
means a "leany" top-heavy vehicle that visually and physically signals
approaching limits earlier (which can be a feature for driver warning, or a
liability for rollover risk, depending on how large it gets).

### 3.3 Units and typical values

RG is reported in **degrees of body roll per g of lateral acceleration**
(deg/g). In this project's clean dataset (n = 35), RG ranges from about 3.3
to 14.1 deg/g, with a mean of 8.4 deg/g (see Section 6 for the full
distribution).

### 3.4 Governing physics

The physical roll-moment balance is: the overturning moment created by
lateral acceleration acting at CG height must be balanced by the restoring
moment from total roll stiffness:

$$
M \cdot a_y \cdot Z_{cg} = K_\phi \cdot \phi
\qquad\Longrightarrow\qquad
RG = \frac{\phi}{a_y} \;\propto\; \frac{M \cdot Z_{cg}}{K_\phi}
$$

where:

- $M$ = total vehicle mass (kg).
- $Z_{cg}$ = CG height above ground (mm) — the lever arm of the overturning
  moment.
- $K_\phi$ = total roll stiffness of the vehicle (springs + anti-roll bar +
  tyre compliance effects, N·mm/deg).

Roll stiffness itself grows with the square of track width (springs act
through a wider lever arm) and with tyre/pressure effects, giving the
dashboard's *display-only* interpretability index (`Physics_RG_Index` in
`src/feature_engineering.py`, shown on the "Physics Background" panel of the
dashboard, **not** used as a model input):

$$
\text{Physics\_RG\_Index} = \frac{M \cdot Z_{cg}}{\bar{P} \cdot W_{tire} \cdot T_W^{2}}
$$

with $\bar{P}$ the average of front and rear tyre pressure and $W_{tire}$
the tyre section width.

### 3.5 Factors affecting Roll Gradient

| Factor | Effect on RG | Why |
|---|---|---|
| Higher CG ($Z_{cg}$) | ↑ RG | Longer overturning lever arm |
| Higher mass ($M$) | ↑ RG | Larger overturning moment for the same $a_y$ |
| Wider track ($T_W$) | ↓ RG | Roll-resisting moment scales with $T_W^2$ |
| Stiffer/thicker ARB | ↓ RG | Adds roll stiffness (scales as $D^4$) |
| Wider/lower-pressure tyre | ↑ RG (slightly) | More sidewall/carcass compliance softens the effective roll stiffness |

### 3.6 Typical patterns across vehicle types in this dataset

Lighter vehicles with a narrower track width sit at the low end of the RG
range, roughly 3–7 deg/g. Heavier vehicles with a higher CG relative to
their track width and ARB provision cluster toward the high end, roughly
8–14 deg/g — consistent with the governing physics in Section 3.4, where RG
scales with $M \cdot Z_{cg}$ and scales inversely with roll stiffness (which
itself grows with track width squared).

### 3.7 Engineering significance

RG is a first-order safety metric: excessive roll angle at high lateral
acceleration is both a **rollover propensity indicator** and a **driver
perception cue**. Vehicle programmes typically set an upper target RG (e.g.
"no more than X deg/g") and iterate track width, spring rate, and ARB
diameter until the target is met — exactly the kind of design search this
project's dashboard is meant to accelerate.

### 3.8 How Roll Gradient is computed in this project

RG is not derived from first-principles physics in the ML pipeline — it is
a **measured target value** already present in the raw dataset (`Roll
Gradient` column, from actual constant-radius vehicle testing). The
project's job is to *predict* this already-measured number from the design
parameters using the model described in Sections 8–10, using the six
engineered features listed in `RG_FEATURES`
(`src/feature_engineering.py`): `Roll_Index`, `Track_Width_Squared`,
`ARB_Stiffness_Index`, `Tire_Width_Pressure_Ratio`, `ARB_Present`, `Type`.

---

## 4. Understeer Gradient

### 4.1 Definition

**Understeer Gradient (UG)** comes from the classic bicycle-model steering
equation for steady-state circular motion:

$$
\delta = 57.3\,\frac{L}{R} \;+\; UG \cdot a_y
$$

where:

- $\delta$ = road-wheel steering angle (degrees) the driver must apply.
- $L$ = wheelbase (m).
- $R$ = turn radius (m).
- $a_y$ = lateral acceleration (g).
- $57.3\,L/R$ = the pure geometric (Ackermann) angle needed at very low
  speed, where tyre slip is negligible.
- $UG$ = **Understeer Gradient**, the extra steering angle required *per g*
  of lateral acceleration beyond the geometric baseline.

Rearranged, this is exactly how UG is measured on a physical test:

$$
UG = \frac{\delta - 57.3\,L/R}{a_y} \quad \left[\frac{\deg}{g}\right]
$$

### 4.2 Meaning of the sign

- **UG > 0 (Understeer):** the driver must add steering angle beyond
  Ackermann as speed rises. In this project's dashboard, `predicted_ug > 0.2`
  is labelled "Understeer".
- **UG = 0 (Neutral steer):** no extra steering angle is needed regardless of
  speed.
- **UG < 0 (Oversteer):** the driver must *reduce* steering angle (or add
  opposite lock) as speed rises; `predicted_ug < -0.2` is labelled
  "Oversteer" in `app.py`.

### 4.3 The classical formula (Olley's equation)

The standard vehicle-dynamics textbook expression (Milliken & Milliken;
Gillespie — see Section 18) decomposes UG into front and rear axle terms:

$$
UG = \frac{W_f}{C_{\alpha f}} - \frac{W_r}{C_{\alpha r}}
$$

where:

- $W_f, W_r$ = static (or load-transfer-corrected) weight on the front and
  rear axles.
- $C_{\alpha f}, C_{\alpha r}$ = front and rear **tyre cornering stiffness**
  (Section 2.5).

Physically: if the front axle carries proportionally more weight *for its
tyres' cornering stiffness* than the rear axle does, the front tyres run out
of grip (reach their slip-angle limit) first as cornering force increases —
this is understeer. If the rear axle is proportionally the weaker one, the
rear breaks away first — oversteer.

### 4.4 Why this project cannot fully reproduce that formula

The formula in Section 4.3 needs $C_{\alpha f}$ and $C_{\alpha r}$ — a
**dynamic, load- and slip-angle-dependent tyre property that is never
measured or supplied anywhere in this project's raw dataset.** The dataset
gives only *static* tyre facts: width, rim diameter, brand, and inflation
pressure. `src/feature_engineering.py` builds the best available *proxy*
for the ratio in Section 4.3, again purely for the dashboard's
interpretability panel (not a model input):

$$
\text{UG\_Physics\_Index} = \frac{1}{W_{tire}}\left(\frac{FWD}{P_f} - \frac{RWD}{P_r}\right)
$$

where $FWD, RWD$ are the front/rear weight *fractions* (`Front_WD`,
1 − `Front_WD`) and $P_f, P_r$ are front/rear tyre pressures — using tyre
width and pressure as a rough stand-in for cornering stiffness, since wider,
higher-pressure tyres generally do have higher (but not perfectly
predictable) cornering stiffness, and cornering stiffness sits in the
*denominator* of Olley's equation (Section 4.3), so tyre width belongs in
the denominator here too (a documented correction: an earlier version of
this formula had tyre width in the numerator instead, reversing the
relationship). This is explicitly acknowledged in the code and dashboard as
an approximation, not a substitute for a real $C_\alpha$ measurement.

### 4.5 How Understeer Gradient is computed (measured) in this project

Exactly like RG, UG is a **measured target** in the raw dataset
(`Understeer Gradient` column), obtained from the same constant-radius test
runs (steering angle, speed, and lateral acceleration recorded as the
vehicle is driven progressively faster around a fixed 10 m or 30 m circle).
The ML pipeline predicts this already-measured value from six engineered
features in `UG_FEATURES`: `Front_WD`, `Roll_Stiffness_Ratio`, `ARB_Present`,
`Zcg_Wheelbase_Ratio`, `Tire_Make`, `Type`.

### 4.6 Why the UG model's ceiling is low

Because the dominant physical term ($C_{\alpha f}, C_{\alpha r}$) is absent,
the model is forced to rely on secondary correlates (weight distribution,
roll stiffness split, CG/wheelbase ratio, tyre brand as a coarse stand-in
for compound/construction). This is why the best achievable LOOCV R² in this
project is only **≈ 0.21** — this is analysed in depth in Sections 6, 10,
and 16, and is the project's single most important engineering finding
about the *data*, not the *models*.

---

## 5. Dataset

### 5.1 Source

The raw data is a spreadsheet of vehicle test runs collected between
2023–2024 across a mixed fleet of three-wheeled and four-wheeled vehicles.
It contains **47 rows and 21 columns**, where each row is one test run of
one vehicle in one specific configuration (tyre brand, pressure, anti-roll
bar fitment, etc.) — the same physical vehicle chassis appears multiple
times (2 to 8 runs) with small configuration changes, across **10 unique
vehicle identities** after outlier rows are removed (15 unique vehicle
identities in the raw 47-row sheet).

### 5.2 Raw columns

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `Vehicle` | string | — | Chassis/model identifier (also the grouping key for cross-validation) |
| `Type` | categorical | — | `3W` (three-wheeler) or `4W` (four-wheeler) |
| `Damper_Configuration` | categorical | — | Damper setup; 100% missing in raw data, imputed to `Unknown` |
| `Suspension_Configuration` | categorical | — | Spring/shock setup string; mostly missing (31/47), imputed to `Unknown` |
| `Test condition` | int | m | Steady-state circular test radius: 10 or 30 |
| `Tire_Make` | categorical | — | `MRF`, `TVS`, `CEAT` |
| `Tire_Width` | float | mm | Tyre section width |
| `Rim_Diameter` | int | in | Wheel rim diameter |
| `ARB_Diameter` | float | mm | Anti-roll bar solid bar diameter (0/NaN = not fitted) |
| `Mass` | float | kg | Total vehicle mass |
| `Wheelbase` | int | mm | Front-rear axle distance |
| `Track_Width` | int | mm | Lateral tyre-centre distance |
| `Front_Load`, `Rear_Load` | float | kg | Static axle loads |
| `Xcg` | float | mm | Longitudinal CG position (from front axle) |
| `Ycg` | float | mm | Lateral CG offset |
| `Zcg` | float | mm | CG height above ground |
| `Front_Pressure`, `Rear_Pressure` | int | psi | Cold tyre inflation pressures |
| `Roll Gradient` | float | deg/g | **Target 1** |
| `Understeer Gradient` | float | deg/g | **Target 2** |

### 5.3 Why each raw feature matters physically

Every raw column above maps directly onto a term in the Section 2–4 physics:
mass/CG height/track width drive roll stiffness and moment (RG); axle
loads, tyre width/pressure, and ARB fitment drive the front/rear cornering
balance (UG); vehicle `Type` captures broad, unmodelled differences between
three- and four-wheeled chassis architectures (different suspension
kinematics, different tyre contact patch counts per axle, etc.).

### 5.4 Preprocessing

1. **Physical consistency enforcement:** `Mass` is recomputed as
   `Front_Load + Rear_Load`, guaranteeing the load-distribution features are
   internally consistent.
2. **Coordinate-frame correction:** two rows belonging to one vehicle had
   $X_{cg}$ recorded from the *rear* axle instead of the *front* axle
   (a documented data-entry inconsistency). The pipeline detects this via a
   threshold ($X_{cg} > 1300$ mm, physically implausible for that vehicle's
   wheelbase if measured from the front) and corrects it via
   $X_{cg,\text{front}} = L - X_{cg,\text{rear}}$, with an automatic
   assertion that the corrected value is physically valid
   ($0 < X_{cg} < L$).
3. **Missing-value imputation**, based on physical meaning rather than
   statistical guesswork:
   - `Suspension_Configuration`, `Damper_Configuration` → `"Unknown"` (NaN
     here means "standard/base configuration was used", not "value not
     recorded").
   - `ARB_Diameter` → `0` (NaN here means "no anti-roll bar fitted").
4. **Feature engineering** — see Section 5.5.
5. **Multivariate outlier detection and removal** — see Section 6.4.
6. **Column pruning** — raw columns that are now redundant with an
   engineered feature (`Front_Pressure`, `Rear_Pressure`, `Front_Load`,
   `Rear_Load`, `Xcg`, `ARB_Diameter`, the zero-variance
   `Damper_Configuration`) are dropped from the saved CSV, leaving 26
   columns for 47 rows (35 rows survive after outlier filtering at
   model-training time).

### 5.5 Engineered features

All feature formulas live in one function,
[`add_engineered_features()`](src/feature_engineering.py), used
identically by the training code (batch, at training time) and by the
dashboard (single-row, at inference time) — this is a deliberate design
choice explained further in Section 12.

| Feature | Formula | Physical rationale | Used by |
|---|---|---|---|
| `Mass` | $Front\_Load + Rear\_Load$ | Total vehicle weight | intermediate |
| `Front_WD` | $Front\_Load / Mass$ | Front weight fraction — cornering-balance driver | UG |
| `Roll_Index` | $Mass \cdot Z_{cg} / T_W$ | Roll moment magnitude relative to track (Section 3.4) | RG |
| `Track_Width_Squared` | $T_W^{2}$ | Roll-resisting moment scales with track² (Section 2.7) | RG |
| `Pressure_Ratio` | $P_f / P_r$ | Front/rear grip balance proxy | analysis only |
| `Tire_Stress_Difference` | $Front\_Load/P_f - Rear\_Load/P_r$ | Axle "tyre deflection" imbalance proxy | analysis only |
| `Tire_Width_Pressure_Ratio` | $W_{tire} / P_f$ | Front tyre compliance index | RG |
| `ARB_Present` | $\mathbb{1}[ARB_{diam} > 0]$ | Binary flag: any anti-roll bar fitted | RG, UG |
| `ARB_Stiffness_Index` | $\log_1 p\!\left(D_{ARB}^4 / T_W\right)$ | ARB torsional stiffness ($\propto D^4$), compressed for skew | RG |
| `Roll_Stiffness_Ratio` | $ARB\_Stiffness\_Index / T_W$ | ARB stiffness relative to spring-based stiffness | UG |
| `Zcg_Wheelbase_Ratio` | $Z_{cg} / L$ | Pitch/longitudinal load-transfer-rate proxy | UG |
| `Physics_RG_Index` | see Section 3.4 | Dashboard interpretability only | display only |
| `UG_Physics_Index` | see Section 4.4 | Dashboard interpretability only | display only |

`Roll_Stiffness_Ratio`'s denominator was corrected from $T_W^2$ to $T_W$ —
the roll-resisting moment from an anti-roll bar scales with $D_{ARB}^4
\times T_W$ while spring roll stiffness scales with $T_W^2$ (Section 2.7), so
their ratio should scale as $D_{ARB}^4 / T_W$, not $D_{ARB}^4 / T_W^3$. This
is a UG model input, so it was retrained and compared after the fix: LOOCV
RMSE was unchanged to 4 decimal places, so the more physically correct
scaling was kept at zero empirical cost.

Two engineered features (`Pressure_Ratio`, `Tire_Stress_Difference`) are
still *computed* for analysis and plotting but were **removed from the final
UG model inputs** after a documented ablation study found them collinear
with each other (Pearson $r = 0.73$) and their removal *raised* UG LOOCV R²
from 0.096 to 0.160 — a concrete demonstration that adding more engineered
features is not automatically better on a small dataset (elaborated in
Section 6.6 and Section 17). (That 0.160 figure was the UG LOOCV R² at the
time of this specific ablation, n=32/9 vehicles; the current baseline after
a later outlier-flagging fix is 0.207 at n=35/10 vehicles — see Section 16.)

### 5.6 Target variables

| Target | Unit | Clean-data range (n = 35) | Mean | Median |
|---|---|---|---|---|
| Roll Gradient | deg/g | 3.3 – 14.1 | 8.4 | 8.0 |
| Understeer Gradient | deg/g | −14.9 – +5.7 | −3.7 | −4.6 |

Notably, the *mean* Understeer Gradient across this fleet is **negative**
(oversteer-leaning by this metric), which is plausible for rear-heavy cargo
vehicles carrying loads mostly over/behind the rear axle — this is discussed
further in Section 6.

---

## 6. Exploratory Data Analysis

### 6.1 Data quality baseline

- **47 raw rows, 0 duplicate rows.**
- Missing values were confined to three columns and were all imputed based
  on physical meaning, as described in Section 5.4 — there was **no missing
  numerical measurement** (mass, geometry, pressures, targets) anywhere in
  the raw sheet.
- The initial exploration explicitly flags, before any modelling: *"n = 47
  records — extremely small; use LOOCV or stratified K-Fold (k ≤ 5)."* This
  single sentence, noted at the very start of the project, foreshadows
  nearly every methodological choice made afterwards (Sections 7–9).

### 6.2 Outlier screening at the feature level (IQR fences)

Univariate outlier counts (Tukey fences, $[Q_1 - 1.5\,IQR,\ Q_3 + 1.5\,IQR]$)
on the raw 47-row data:

| Feature | Outliers | % of data |
|---|---:|---:|
| Tire_Width | 12 | 25.5% |
| Wheelbase | 4 | 8.5% |
| Front_Load | 3 | 6.4% |
| Front_Pressure | 3 | 6.4% |
| Roll Gradient | 2 | 4.3% |
| Understeer Gradient | 2 | 4.3% |
| Ycg | 1 | 2.1% |
| Zcg | 1 | 2.1% |

Tire_Width's high outlier count simply reflects that the fleet mixes very
different tyre sizes across 3W and 4W vehicles (a real, expected range, not
a data error). Wheelbase, Front_Load, and Front_Pressure show moderate
variability for the same reason.

### 6.3 A single extreme data point, and why it matters

One row (from a single vehicle, at the 30 m test condition) has
`Understeer Gradient = 62.70 deg/g`, against a dataset median of −4.6. That
is more than five standard deviations from the mean — either a genuinely
unusual test result or a data issue. The analysis does **not** silently drop
it by eyeballing; instead it runs a **sensitivity check**: it recomputes
every feature's correlation rank with UG both with and without this single
row present. The result: the correlation *rank order* of several features
(`Front_Load`, `ARB_Diameter`, `Zcg`, `Track_Width`, `Ycg`) shifts by 4–11
places depending on whether this one row is included. This is a direct,
evidence-based illustration of a core small-sample-statistics lesson: **with
only 47 rows, a single extreme point can dominate a correlation analysis
and mislead feature selection.** This finding is exactly why the project
moved from ad-hoc, single-variable outlier capping toward the principled
**multivariate** method described next.

### 6.4 Multivariate outlier removal (the method actually used for modelling)

Rather than deleting rows based on one variable at a time, the pipeline uses
**robust Mahalanobis distance** to find entire vehicle configurations that
are jointly unusual across nine variables simultaneously
(`Mass, Wheelbase, Track_Width, Zcg, Front_WD, Xcg, ARB_Diameter,
Front_Pressure, Rear_Pressure`):

$$
D_M(x) = \sqrt{(x - \hat{\mu})^{T}\, \hat{\Sigma}^{-1} \,(x - \hat{\mu})}
$$

Using the *ordinary* sample mean $\hat\mu$ and covariance $\hat\Sigma$ would
be self-defeating here: true outliers pull the very statistics used to
detect them toward themselves (the **masking effect**), hiding their own
extremity. Instead the project uses **`MinCovDet`** (Minimum Covariance
Determinant), a robust estimator that finds the most tightly-clustered 80%
of the data (`support_fraction=0.8`) and computes $\hat\mu, \hat\Sigma$ from
*that* clean-looking core, making the resulting distances resistant to the
very outliers being searched for.

Under an assumption of (roughly) multivariate-normal data, $D_M^2$ follows a
chi-squared distribution with $d = 9$ degrees of freedom, so a statistically
principled cutoff is used:

$$
\text{threshold} = \chi^2_{0.99,\ d=9} = 21.666
$$

Any test run whose $D_M^2$ exceeds this threshold is flagged. The flagging is
**row-level**, not group-level: a vehicle chassis is excluded entirely only
if *every* one of its runs is flagged; a vehicle with a mix of flagged and
clean runs keeps its clean runs.

**Result: 12 rows excluded** — 5 vehicles are excluded entirely because
every run failed the threshold, plus 1 of a sixth vehicle's 4 runs — leaving
**35 clean rows across 10 vehicles** for all modelling (that sixth vehicle's
other 3 runs are physically normal, $D_M^2$ well below threshold, and are
retained as a 10th training vehicle). The excluded rows from the first of
those five vehicles include the UG = 62.70 extreme point from Section 6.3 —
the principled multivariate method independently corroborates the earlier
single-variable flag, which is reassuring evidence that the outlier removal
is not arbitrary.

*(Prior to a row-level fix, this step used group-level flagging — any
flagged run excluded the entire chassis — which discarded 15 rows across
the same 6 vehicles above, including the sixth vehicle's 3 normal runs
purely because its 4th run was bad. Fixed for the reason above.)*

### 6.5 Correlation structure

Two complementary correlation measures are used throughout:

- **Pearson correlation** ($r$) — strength of a *linear* relationship,
  $r \in [-1, 1]$.
- **Spearman correlation** ($\rho$) — strength of a *monotonic* (not
  necessarily linear) relationship, based on ranks rather than raw values,
  and therefore more robust to outliers and to the non-linear engineered
  features (like `Track_Width_Squared` or the `log1p`-transformed
  `ARB_Stiffness_Index`).

The feature/target Spearman heatmap (computed on the final RG_FEATURES +
UG_FEATURES set, n = 35) shows several important patterns:

- `Roll Gradient` correlates most strongly with `ARB_Present` (+0.50),
  `Track_Width_Squared` (−0.63), `ARB_Stiffness_Index` (+0.44), and
  `Roll_Index` (+0.41) — consistent with the RG physics in Section 3.
- `ARB_Stiffness_Index` and `ARB_Present` are themselves correlated at
  $\rho = 0.95$, and `Roll_Stiffness_Ratio` correlates at $\rho = 1.00$ with
  `ARB_Stiffness_Index` — a **near-perfect collinearity** in this
  particular 35-row sample, which is exactly why the RG and UG feature
  lists are deliberately built to use only *one* member of each collinear
  pair (RG keeps `ARB_Stiffness_Index`; UG keeps the related
  `Roll_Stiffness_Ratio` instead, never both).
- `Zcg_Wheelbase_Ratio` correlates at $\rho = 0.92$ with `Roll_Index` and
  at $\rho = -0.80$ with `Front_WD` — again a strong but *expected*
  geometric relationship (both involve $Z_{cg}$), managed by keeping each
  target's feature list free of redundant members of the same family.
- `Understeer Gradient`'s strongest correlations are much weaker than RG's:
  `Front_WD` (+0.49), `Zcg_Wheelbase_Ratio` (−0.55), `ARB_Present` (+0.42) —
  none exceed $|\rho| \approx 0.55$, an early visual signal (well before any
  model is trained) that UG will be harder to predict than RG.

### 6.6 Mutual Information (capturing non-linear signal)

Pearson/Spearman correlation only detects monotonic relationships.
**Mutual Information (MI)** additionally detects arbitrary non-linear
dependence between a feature and a target, at the cost of needing more data
to estimate reliably (a real concern at n = 35, so MI scores here are
read as a **secondary, confirmatory signal**, not a primary feature-selection
criterion).

The MI ranking (computed on the broader raw+engineered feature pool) puts
`Mass` at the top for *both* targets, followed by
`Tire_Width_Pressure_Ratio` and `Tire_Width` for RG, and `Zcg` and
`Tire_Stress_Difference` for UG — broadly consistent with, and
cross-validating, the Pearson/Spearman findings above.

### 6.7 Multicollinearity diagnostics (VIF)

A dedicated diagnostic step computes **Variance Inflation Factor (VIF)** —
how much a feature's variance is inflated because it can be linearly
predicted from the *other* features:

$$
VIF_i = \frac{1}{1 - R_i^2}
$$

where $R_i^2$ is the R² from regressing feature $i$ on all other features in
the analysis list. Computed on the deliberately broad analysis list (which
mixes engineered features *with* their own raw constituents, e.g.
`Roll_Index` alongside `Mass`, `Zcg`, and `Track_Width`), VIF values are
enormous (`Roll_Stiffness_Ratio`: 13,121; `ARB_Stiffness_Index`: 12,230;
`Track_Width`: 878) — but this is **expected and intentional**: the point of
this analysis is to show that raw inputs and their own engineered
derivatives are (by construction) redundant with each other. The *actual*
model feature lists (`RG_FEATURES`, `UG_FEATURES` in
`src/feature_engineering.py`) resolve this by explicit, hand-curated feature
selection — never using a raw input alongside a feature built directly
from it.

### 6.8 In-sample correlation improvements from feature engineering

On the clean 35-row data, a physically motivated engineered feature's
improvement on its raw constituent's correlation with the target was
quantified directly:

| Engineered feature | vs. raw feature | $\lvert r\rvert$ engineered | $\lvert r\rvert$ raw | Change |
|---|---|---:|---:|---:|
| `Roll_Index` | `Mass` | 0.407 | 0.084 | **+382%** |
| `Tire_Width_Pressure_Ratio` | `Tire_Width` | 0.505 | 0.020 | **+2410%** |
| `Pressure_Ratio` | `Front_Pressure` | 0.492 | 0.034 | **+1347%** |
| `Tire_Stress_Difference` | `Front_Load` | 0.319 | 0.164 | **+95%** |
| `ARB_Stiffness_Index` | `ARB_Diameter` | 0.491 | 0.487 | +0.7% |
| `Track_Width_Squared` | `Track_Width` | 0.662 | 0.664 | −0.3% |

An important caveat is reproduced here because it matters: **these numbers
are measured on the same data used to build the features (in-sample)**, so
they confirm the features encode *known* vehicle-dynamics physics correctly,
but they do **not** by themselves prove the features generalise to unseen
vehicles — that claim is only established later, by the out-of-sample,
vehicle-level cross-validation described in Sections 7–10.

### 6.9 Target distributions

Histograms of RG and UG split by vehicle `Type` show:

- **Roll Gradient** is right-skewed with two loose clusters — a low cluster
  around 3–7 deg/g (mostly 3W vehicles and lighter 4W configurations) and a
  higher cluster around 9–14 deg/g (heavier 4W vehicles). This bimodal,
  non-Gaussian shape is the direct motivation for the **Yeo–Johnson power
  transform** applied to the target inside the Ridge pipeline (Section 8.1)
  — without it, a plain Ridge regression's constant error-variance
  assumption is measurably violated, and accuracy suffers (documented
  concretely in Section 8.1).
- **Understeer Gradient** is more continuously spread from about −15 to +6
  deg/g, centred below zero (mean −3.6, median −4.6) — this fleet leans
  oversteer on average by this measure, plausibly reflecting rear-loaded
  cargo-vehicle weight distributions.

### 6.10 Feature-vs-target relationships (marginal, uncontrolled)

Plotting each model-input feature against its target, coloured by vehicle
`Type`, with a linear trend line and Spearman $\rho$, gives two important,
honest observations:

- Several apparent "trends" are really **step patterns between the two
  vehicle-type clusters** rather than smooth cause-and-effect relationships
  within a single vehicle type (e.g. `ARB_Present` is 0/1 by construction,
  and nearly all ARB-fitted vehicles in this sample happen to be the
  heavier 4W vehicles) — a classic **confounding** pattern in a
  small, non-randomised dataset. This exact confound resurfaces as a
  counter-intuitive Ridge coefficient sign in Section 8.1 and is explained
  there in full.
- Raw scatter plots are explicitly "marginal / uncontrolled" views,
  precisely because a strong 2-D trend can be produced by a *different*,
  correlated feature acting behind the scenes. The complementary **Partial
  Dependence + ICE plots** (discussed in Section 10.5) show what the
  *fitted model* actually learned once other features are statistically
  held fixed — a materially more trustworthy view of "does this feature
  really drive the target."

---

## 7. Machine Learning Concepts

This section explains, from first principles, every ML concept the project
relies on — written for a reader whose ML exposure may be limited to an
introductory course.

### 7.1 Training, validation, and test sets — and why this project blurs the last two

In a typical ML project with abundant data, one splits data three ways:

- **Training set** — rows the model directly learns its parameters from.
- **Validation set** — rows used to choose *hyperparameters* (settings the
  learning algorithm doesn't learn on its own, like Ridge's regularisation
  strength) without touching the final test set.
- **Test set** — rows touched exactly once, at the very end, to report an
  honest, unbiased estimate of real-world performance.

With only **35 usable rows across 10 vehicles**, carving out a separate,
untouched test set would leave far too little data to train on *and* too
little held out to trust the test estimate (a single held-out vehicle would
have almost no statistical power). The project instead uses **nested
cross-validation** (Section 7.4) to obtain a validation-style hyperparameter
search and an honest test-style performance estimate *simultaneously*, by
systematically rotating which data plays which role.

### 7.2 Bias and variance

Every model's prediction error can be conceptually decomposed into:

- **Bias** — error from a model that is too simple to capture the true
  pattern (e.g. fitting a straight line to a genuinely curved relationship).
  High-bias models *underfit*.
- **Variance** — error from a model that is so flexible it also fits the
  random noise in this *particular* training sample, so its predictions
  would look very different if trained on a slightly different sample of
  the same population. High-variance models *overfit*.

At n = 35, variance is the dominant risk: almost any sufficiently flexible
model can achieve a near-perfect fit to 35 training points while learning
nothing that generalises. This is why the project's model suite
(Section 8) deliberately favours simple, heavily-regularised, low-capacity
models over more flexible ones.

### 7.3 Overfitting and underfitting

- **Overfitting**: training error is low, but validation/test error is much
  higher — the model has memorised noise specific to the training rows.
- **Underfitting**: both training and validation error are high — the model
  is too simple (or too heavily regularised) even for the training data.

The project's **learning curves** (Section 7.8) plot exactly this
trade-off as a function of training-set size for each model.

### 7.4 Cross-validation, and why this project needs *two* layers of it

**K-fold cross-validation** splits data into $k$ folds; the model trains on
$k-1$ folds and is evaluated on the held-out fold, rotating which fold is
held out $k$ times, then averaging the $k$ scores — this uses every row for
both training and (eventually) testing, which is valuable when data is
scarce.

This project needs **two nested layers** of cross-validation for two
distinct reasons:

1. **Outer loop — honest generalisation, grouped by vehicle.** A vehicle
   chassis contributes 2–8 rows (different tyres/pressures/ARB
   configurations tested on the *same physical vehicle*). If a random
   row-level split let two runs of the *same* vehicle land in both train and
   test, the model would partly be tested on a chassis it has already
   partially "seen" — an optimistic leak. The project instead uses
   **`LeaveOneGroupOut`** grouped by the `Vehicle` column: every fold holds
   out **all** runs of one entire vehicle, so the model is always evaluated
   on a genuinely unseen chassis. With 10 unique vehicles, this gives exactly
   10 outer folds. This directly quantifies the size of the resulting
   optimism gap: naive row-level LOOCV would report R² ≈ 0.91, while honest
   vehicle-level LOOCV reports R² ≈ 0.67–0.80 for RG — nearly a 0.2–0.25 R²
   difference purely from leakage, a striking, concrete illustration of why
   grouping matters.
2. **Inner loop — hyperparameter tuning without leakage from the outer test
   vehicle** (see caveat below on a smaller, separate leakage source).
   Inside each of those 10 outer training folds, a further 5-fold
   `GridSearchCV` searches over each model's hyperparameter grid *using
   only that fold's training data*. Only after the inner search picks the
   best hyperparameters is the model evaluated once on the outer fold's
   held-out vehicle. This "nested" structure ensures the hyperparameter
   search itself never gets to peek at the data it will later be scored
   against — a common and often-overlooked source of over-optimistic
   reported metrics in small-data ML projects.
   **Caveat:** the inner 5-fold split is plain `KFold`, not grouped by
   vehicle, so a training-fold vehicle with 2+ runs can still have its own
   rows split across the inner train/validation boundary — a smaller-scale
   version of the same leakage the outer loop prevents. A grouped inner
   loop was tried and produced an honest but worse score, so it was
   reverted rather than kept (documented in the project's own training
   module). Reported inner-loop hyperparameter selection should be read
   with this caveat in mind, even though the headline LOOCV metrics
   themselves (from the outer loop) are unaffected.

This exact two-layer design is implemented once, in
[`run_nested_cv()`](src/training.py), and reused identically for both the
RG and UG targets — see Section 9 for the full pipeline walk-through.

### 7.5 Regularisation

Regularisation deliberately penalises model complexity to fight variance
(Section 7.2). This project uses two distinct forms:

- **L2 (Ridge) regularisation** — adds a penalty proportional to the sum of
  squared coefficients, shrinking them toward zero (Section 8.1).
- **Architectural regularisation** — deliberately using a *small,
  fixed* neural-network hidden layer (4 neurons) and a *shallow, fixed*
  random-forest depth (2–3 levels), rather than searching a large
  hyperparameter space that a 35-row dataset cannot statistically support
  (explained further in Section 8.2–8.3).

### 7.6 Feature scaling: standardisation

Before any model is fit, every numeric feature is **standardised**
(`StandardScaler`, fitted only on the training fold — see Section 9):

$$
z = \frac{x - \mu}{\sigma}
$$

This rescales every feature to zero mean and unit variance. It matters for
two of the three models used here:

- **Ridge** — the L2 penalty treats every coefficient's magnitude equally,
  so features must be on comparable scales or the penalty would unfairly
  punish naturally large-magnitude features (e.g. `Track_Width_Squared`,
  which is on the order of $10^6$) more than naturally small ones.
- **MLP** — gradient-based neural network training converges faster and
  more reliably when inputs are on comparable scales.

(Random Forest is scale-invariant — it only ever compares a feature to
itself at different split thresholds — but scaling it does no harm and
keeps one shared preprocessing pipeline for all three models, which is a
simplicity choice, not a modelling requirement.)

*Standardisation vs. normalisation:* "normalisation" typically rescales a
feature to a fixed $[0, 1]$ range, whereas "standardisation" (used here)
centres on the mean and scales by the standard deviation, which is generally
the better choice when a feature's distribution is not bounded and may
contain outliers — appropriate given Section 6's findings about this
dataset's distributions.

### 7.7 Categorical encoding: One-Hot Encoding

Categorical features (`Type`, `Tire_Make`) are converted to numbers via
**One-Hot Encoding**: each category becomes its own 0/1 column. The pipeline
uses `drop='first'` (one category, e.g. `CEAT` for `Tire_Make`, becomes the
implicit "reference" — all-zeros — category, avoiding redundant columns)
and `handle_unknown='ignore'` (a category never seen during training, e.g.
a brand-new tyre brand, is encoded as all-zeros at inference rather than
crashing — it is silently treated as the reference category, with the
dashboard explicitly warning the user when this happens; see Section 11).

### 7.8 Learning curves

A **learning curve** plots training error and validation error as a
function of *how much training data* is used, holding the model and
hyperparameters fixed. Two diagnostic patterns matter:

- If validation error is still clearly *decreasing* as training size grows
  (and has not caught up to training error), the model is **data-limited** —
  more data would likely help, and the current gap between train/validation
  error reflects genuine finite-sample variance, not a flawed model.
- If validation error has flattened while training error stays much lower,
  the gap reflects **irreducible overfitting** at the current model
  complexity — more data would help less than simplifying the model.

This project's learning curves show validation MAE still trending downward
for all three model families at the maximum available training size
(~25–26 samples in the 5-fold curve), for **both** targets — the documented
conclusion is that **dataset size, not model choice, is the primary
bottleneck for Roll Gradient**, while for **Understeer Gradient the ceiling
is additionally capped by the missing tyre cornering-stiffness variable**
(Section 4.6) regardless of how much more data of the *same* kind were
collected.

### 7.9 Residuals

A **residual** is the signed prediction error for a single row:

$$
e_i = y_i - \hat{y}_i
$$

Residual analysis is one of the most information-dense diagnostics
available for a small dataset:

- **Residuals vs. predicted** should show a random, structureless scatter
  around zero. A curved or funnel-shaped pattern signals unexplained
  non-linearity or non-constant error variance (heteroscedasticity).
- **Residual histograms / Q-Q plots** check whether residuals are
  approximately normally distributed, an assumption behind several
  classical statistical tools (though not strictly required for the tree
  and neural-network models used here).

A **Shapiro-Wilk normality test** on the LOOCV residuals shows RG residuals
pass comfortably (p = 0.391 — no evidence against normality), while UG
residuals **fail** (p = 0.010 — meaningful evidence of non-normality),
driven by a handful of vehicles with unusually large errors (see the
per-vehicle breakdown in Section 10.4) — concrete, numeric confirmation
that the UG model's difficulty is concentrated in specific vehicles rather
than spread evenly across the fleet.

### 7.10 Evaluation metrics: MAE, RMSE, R² — and why all three are reported

- **Mean Absolute Error (MAE)**:
  $$MAE = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert$$
  The average size of a prediction error, in the target's original units
  (deg/g) — easy to communicate to a non-statistician ("on average, the
  model is off by about 1.1 deg/g for Roll Gradient").
- **Root Mean Squared Error (RMSE)**:
  $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
  Also in the target's original units, but squaring before averaging
  penalises large individual errors more heavily than MAE does — useful
  because a single badly-mispredicted vehicle (Section 7.9) is arguably a
  more serious practical failure than several small ones, and RMSE will
  always be $\geq$ MAE, with the *gap* between them indicating how
  unevenly errors are distributed across rows.
- **R² (coefficient of determination)**:
  $$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$
  The fraction of the target's total variance that the model explains,
  relative to a naive baseline that always predicts the mean $\bar y$.
  $R^2 = 1$ is a perfect model; $R^2 = 0$ means "no better than always
  guessing the mean"; **$R^2$ can be negative**, meaning the model is
  actively *worse* than that naive baseline (this genuinely happens for
  Ridge on UG in this project's own model-comparison table — Section 10 —
  and for several individual vehicle folds in the UG per-vehicle table,
  Section 10.4).

Reporting all three together avoids the trap of any single metric telling
an incomplete story — e.g. a model could have deceptively good MAE if it is
right on most vehicles but catastrophically wrong on one or two (RMSE and
R² would immediately reveal this; MAE alone might not).

### 7.11 Why R² is unstable with only 10 cross-validation folds — bootstrap confidence

Because there are only 10 vehicles (10 outer folds), and some vehicles
contribute as few as 2 rows to a fold, a *per-fold* R² is frequently
undefined or meaningless (R² needs variance in the held-out target to be
computable at all — with 2 points it is a near-arbitrary number, and the
per-vehicle table literally reports figures like R² = −39.8 for a 2-run
fold to illustrate this). To responsibly communicate how much the *overall
pooled* R² could plausibly vary if a slightly different set of 10 vehicles
had been sampled, the plotting code runs a **vehicle-level bootstrap**
(resampling entire vehicles with replacement 1,000 times and recomputing
the pooled R² each time). This confirms visually that the RG R² estimate is
tightly clustered and clearly positive, while the UG R² estimate has a very
wide bootstrap distribution straddling zero — meaning the "0.21" headline
number for UG should be read as **directionally positive but not tightly
pinned down**, given the sample size.

---

## 8. Models Used

All three models share one preprocessing step (`build_preprocessor()` in
`src/training.py` — `StandardScaler` for numeric columns, `OneHotEncoder`
for categorical columns, auto-detected by pandas dtype) wrapped in a single
`sklearn.Pipeline`, and are compared under identical nested cross-validation
(Section 7.4, Section 9). Gradient Boosting was evaluated during
development and **removed** from the final suite — its LOOCV error was
statistically indistinguishable from Random Forest's on both targets, while
needing a much larger, statistically unjustified hyperparameter grid (16
combinations) at this sample size — a documented, deliberate simplicity
decision rather than an oversight. A follow-up review later proposed more
advanced boosting variants for UG specifically — physics-informed residual
boosting, extremely regularized stumps, and CatBoost — and declined all
three for the same underlying reason (see Section 17, item 8): the
bottleneck is the missing $C_\alpha$ signal and the small sample size,
neither of which any learner choice can fix.

### 8.1 Ridge Regression (best model for Roll Gradient)

**How it works.** Ordinary Linear Regression finds coefficients
$\beta$ that minimise the sum of squared errors:

$$
\hat\beta_{OLS} = \arg\min_\beta \sum_i (y_i - X_i\beta)^2
$$

With many correlated features and few rows (Section 6.7), OLS coefficients
become unstable — small data changes can swing them wildly. **Ridge
Regression** adds an L2 penalty on coefficient size:

$$
\hat\beta_{Ridge} = \arg\min_\beta \left[\sum_i (y_i - X_i\beta)^2 \;+\; \alpha \sum_j \beta_j^2\right]
$$

The **loss function** is therefore ordinary squared error plus a
complexity penalty; $\alpha$ (the model's one hyperparameter here) controls
how hard that penalty bites — $\alpha = 0$ recovers plain OLS, larger
$\alpha$ shrinks every coefficient further toward zero, trading a little
bias for a (hopefully much larger) reduction in variance (Section 7.2) —
exactly the trade a 35-row dataset needs.

**Target transform.** Section 6.9 showed Roll Gradient's distribution is
right-skewed and loosely bimodal, which violates the constant-error-variance
assumption behind ordinary least-squares fitting. The pipeline wraps Ridge
in a `TransformedTargetRegressor` with a **Yeo–Johnson power transform**,
which fits a monotonic, per-problem power transform to make the *target*
distribution closer to Gaussian before fitting, then automatically inverts
the transform on predictions. This is not a cosmetic step: **a documented
ablation removing this transform increased RG RMSE from 1.34 to 2.23 (a
66% increase) and dropped R² from 0.80 to 0.44** (recorded at n=32/9
vehicles, the dataset size at the time of this ablation; Random Forest
briefly became "best" only because the *unwrapped* Ridge got measurably
worse, not because RF improved) — a concrete before/after number
demonstrating that this preprocessing choice is load-bearing, not
decorative. The current baseline (n=35/10 vehicles, transform retained) is
RMSE=1.46, R²=0.762 (Section 16) — the qualitative conclusion (removing the
transform is a large regression) is unaffected by the later dataset change.

**Hyperparameters used.** `alpha ∈ {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}`
(log-spaced, searched by the inner `GridSearchCV`) — six values are judged
sufficient to span from "almost no regularisation" to "very heavy
regularisation" for standardised features.

**Advantages.** Highly interpretable (each standardised coefficient
directly says "how many standard deviations does the transformed target
move per standard deviation of this feature"); robust on small, noisy
datasets; fast; explicit uncertainty behaviour.

**Disadvantages.** Assumes an (approximately, after transform) linear
relationship; cannot represent genuine interaction/non-linear effects
between features unless they are engineered in manually; sensitive to
collinearity (mitigated here by careful, hand-curated feature selection —
Section 6.7).

**Why selected.** RG's underlying physics (Section 3.4) is genuinely close
to a product-of-ratios relationship that becomes linear after a log/power
transform — Ridge's assumptions match the physics unusually well for this
particular target, which is exactly why Ridge wins decisively over both
non-linear alternatives (Section 10).

**Interpreting the fitted coefficients (a genuine, non-obvious finding).**
The standardised Ridge coefficients for RG are: `ARB_Present` +1.03,
`ARB_Stiffness_Index` −0.83, `Track_Width_Squared` −0.65, `Roll_Index`
+0.52, `Tire_Width_Pressure_Ratio` −0.23, `Type_4W` −0.17. Most of these
match physical expectation directly (taller/heavier-relative-to-track
vehicles roll more; wider track and stiffer ARB roll less). The one
coefficient that looks physically backwards at first glance is
`ARB_Present`: a *positive* effect implies "having any anti-roll bar is
associated with *more* roll", which contradicts the purpose of an ARB.
Section 6.5 already showed why: `ARB_Present` and `ARB_Stiffness_Index` are
correlated at $\rho = 0.95$ in this 35-row sample — nearly every vehicle
that has an ARB at all happens to be one of the heavier, higher-CG 4W
vehicles that would have higher RG *regardless*. Ridge, faced with two
highly correlated inputs, is free to assign a large positive weight to one
and a compensating large negative weight to the other while their **sum**
still points the physically correct direction (the combined ARB effect,
`ARB_Present`+`ARB_Stiffness_Index` together, nets to a roll-*reducing*
contribution for any actually-fitted ARB, since the stiffness index term
dominates in magnitude for realistic diameters). This is flagged here
explicitly as a **known limitation of interpreting individual coefficients
under multicollinearity** — the model's *predictions* remain sound, but a
reader should not conclude "fitting an ARB increases roll" from this single
number in isolation.

### 8.2 Random Forest Regressor (secondary / nonlinear check for both targets)

**How it works.** A Random Forest is an ensemble of many **decision trees**,
each trained on a bootstrap-resampled subset of the training rows and a
random subset of features at each split (`max_features='sqrt'`). Each tree
recursively splits the data to minimise the variance of the target within
each resulting group; the forest's prediction is the **average** of all
individual trees' predictions. Averaging many de-correlated trees (bagging)
reduces variance without much loss of the individual trees' low bias.

**Loss / splitting criterion.** Each split is chosen to minimise the
weighted variance (mean squared error) of the target within the two
resulting child nodes — conceptually the regression analogue of Gini
impurity used for classification trees.

**Hyperparameters used.** `n_estimators=200` (fixed — enough trees for a
stable average, more would not change results meaningfully),
`min_samples_split=10` (fixed — prevents a tree from creating a leaf from a
single sample, an easy way to overfit on ~32 training rows per outer fold),
`max_features='sqrt'` (fixed), and **only `max_depth ∈ {2, 3}` is
searched** by the inner grid — tree depth is the single most
influential regularisation knob at this sample size, and the grid is
deliberately kept tiny (2 values) rather than exhaustively searched,
because a larger grid on ~32 rows would not be statistically distinguishable
from noise.

**Advantages.** Captures non-linear relationships and feature interactions
automatically, requires no target transform, robust to outliers and feature
scale, provides a complementary (Gini-based) feature-importance view.

**Disadvantages.** Much less interpretable than Ridge; prone to overfitting
if depth/leaf-size are not tightly controlled on small data (which is why
this project deliberately fixes most of its architecture rather than
tuning it freely); step-function-like predictions can extrapolate poorly
outside the training range (relevant for the dashboard's out-of-range
warnings, Section 11).

**Why selected.** Included specifically as a conservative, tightly-leashed
non-linear cross-check on Ridge — "does allowing non-linearity and
interactions actually improve on a straight physics-motivated linear
model?" For RG, the answer is no (RF RMSE 1.91 vs. Ridge's 1.46 — Section
10); for UG, RF is close behind the winning MLP (RMSE 4.10 vs. 3.99 — a gap
well within fold-to-fold noise, Section 7.11), meaning **RF and MLP should
be read as roughly tied for UG**, not "MLP is definitively better."

### 8.3 MLP Neural Network (best model for Understeer Gradient)

**How it works.** A Multi-Layer Perceptron is a small feed-forward neural
network: inputs are combined through a layer of weighted sums and a
non-linear activation function, producing "hidden" values, which are then
combined again (another weighted sum) to produce the final numeric
prediction. Weights are learned by **backpropagation** — repeatedly
computing how much each weight contributed to the current prediction error
and nudging it in the error-reducing direction (gradient descent).

For a single hidden layer with $h$ neurons, weights $W^{(1)} \in
\mathbb{R}^{p\times h}$, $W^{(2)} \in \mathbb{R}^{h\times 1}$, and
activation function $g$ (scikit-learn's default is ReLU):

$$
\hat y = W^{(2)\top}\, g\!\left(W^{(1)\top} x + b^{(1)}\right) + b^{(2)}
$$

**Loss function.** Mean squared error between predictions and targets,
plus an L2 weight penalty (`alpha`) analogous to Ridge's — the network is
trained to minimise
$\frac{1}{n}\sum_i (y_i-\hat y_i)^2 + alpha\sum \lvert W\rvert^2$.

**Hyperparameters used.** Architecture is **fixed** at a single hidden
layer of **4 neurons** — deliberately tiny, because with only ~32 training
rows per outer fold, a larger network has vastly more parameters than data
points and would simply memorise the training fold. Only the L2 penalty
`alpha ∈ {10.0, 50.0, 100.0}` is searched (note these values are much larger
than Ridge's typical range — an MLP with only 4 hidden units and this few
samples needs strong regularisation to avoid overfitting), and `max_iter =
2000` ensures the optimiser has enough iterations to converge.

**Advantages.** Can represent smooth non-linear relationships and feature
interactions that Ridge cannot; with a small, tightly-regularised
architecture, still highly constrained and resistant to overfitting.

**Disadvantages.** Least interpretable of the three models (no simple
coefficient reading — importance is assessed instead via **permutation
importance**, Section 10.5); more sensitive to random initialisation and
optimiser settings than Ridge or RF; historically the most fragile model to
tune, which is exactly why its architecture is fixed rather than searched.

**Why selected.** For Understeer Gradient, where the true relationship is
known to be more complex than a clean linear physics formula can capture
with the *available* variables (Section 4.4), allowing a small amount of
learned non-linearity provides a slight edge over both Ridge and RF — but
only a slight one (Section 10), and the project's own documentation
explicitly cautions this margin is close to fold-to-fold noise, not a
decisive algorithmic victory.

### 8.4 Why Ridge is *not* used (unwrapped) for Understeer Gradient

Plain Ridge Regression is still evaluated for UG as a linear baseline, and
it performs **worse than simply guessing the mean** (LOOCV R² = −0.06,
Section 10) — direct evidence that UG's relationship with the available
features is not well captured by a linear model, motivating the (modest)
preference for a non-linear model (MLP/RF) for this target specifically.

---

## 9. Training Pipeline

This section walks through the full, concrete sequence of steps that turns
the raw spreadsheet into a deployed prediction, in the order the code
actually executes it.

### 9.1 Step 1 — Load raw data

`pd.read_excel(...)` loads the raw dataset → 47 rows × 21 columns.

### 9.2 Step 2 — Clean and correct

- Enforce `Mass = Front_Load + Rear_Load`.
- Correct a coordinate-frame bug in one vehicle's $X_{cg}$ value
  (Section 5.4).
- Impute missing `Suspension_Configuration`, `Damper_Configuration`
  (→ `"Unknown"`) and `ARB_Diameter` (→ `0`).

### 9.3 Step 3 — Feature engineering

Call the shared `add_engineered_features(df)` function
(`src/feature_engineering.py`) to compute all 12 engineered columns listed
in Section 5.5, plus the two display-only physics indices. A built-in
sanity assertion checks `Front_WD + Rear_WD == 1.0` to numerical precision.

### 9.4 Step 4 — Outlier detection and removal

Robust Mahalanobis distance flags 12 rows for exclusion — row-level, not
group-level (Section 6.4): 5 vehicles are excluded entirely (every run
flagged) and 1 more vehicle loses only its 1 flagged run, keeping its 3
clean ones. The flagged row indices are saved to a shared outlier-tracking
file (not baked into the CSV as a column), so downstream code and the
dashboard filter consistently from one shared source.

### 9.5 Step 5 — Save the processed dataset

Redundant raw columns are dropped; outlier-detection metadata
(`Is_Outlier`, `MD_squared`) is deliberately **excluded from the saved
CSV** so it can never accidentally be used as a model input feature later.
Result: a processed dataset of 47 rows (outliers are *flagged*, not
yet removed, in this file) × 26 columns.

### 9.6 Step 6 — Model-specific feature selection

Each target's training code independently:

1. Reloads the processed dataset and filters out the flagged outlier rows
   → 35 rows.
2. Drops features irrelevant to *that* target for physical reasons (e.g.
   the RG path drops steering/UG-only features; the UG path drops
   roll-only features) — this is a deliberate physics-informed feature
   *pruning* step, not automatic feature selection.
3. Selects the final feature set via the canonical `RG_FEATURES` /
   `UG_FEATURES` constant imported from `src/feature_engineering.py` — the
   single edit point mentioned throughout this report.

### 9.7 Step 7 — Vehicle-grouped train/validation via nested CV

Call the shared [`run_nested_cv(X, y, groups, get_model_suite())`](src/training.py):

1. Build one shared `ColumnTransformer` preprocessor (`StandardScaler` +
   `OneHotEncoder`) from the feature dataframe's dtypes.
2. For each of the three candidate models, wrap `[preprocessor, model]` in
   an `sklearn.Pipeline`.
3. Run `cross_val_predict(GridSearchCV(pipeline, param_grid, cv=5), X, y,
   cv=LeaveOneGroupOut(), groups=vehicle_ids)` — this is the full nested
   loop described in Section 7.4: for each of the 10 held-out vehicles, a
   fresh inner 5-fold grid search re-tunes hyperparameters using only the
   other 9 vehicles' data, then predicts the held-out vehicle once.
4. Separately, `GridSearchCV(refit=True)` already refits the single best
   hyperparameter setting on **all 35 rows** as part of step 3 — this
   refit estimator (`search.best_estimator_`), **not** any of the 10
   per-fold models, is what gets saved and deployed.
5. Compute pooled RMSE/MAE/R² from the 35 honest out-of-fold predictions
   gathered in step 3, and rank all three models by LOOCV RMSE.

`n_jobs=1` is used throughout (a documented, deliberate choice — parallel
joblib workers on Windows caused memory-mapping errors that silently
corrupted hyperparameter selection during development).

### 9.8 Step 8 — Diagnostics before saving

Before persisting anything, the pipeline additionally computes:
per-vehicle fold-by-fold RMSE/R² (Section 6.4 shows how uneven this is),
three complementary feature-importance views (Ridge coefficients / RF Gini
importance / MLP permutation importance — Section 10.5), residual scatter
plots, and 5-fold learning curves (Section 7.8) — purely as verification
and reporting artefacts; none of this feeds back into model selection.

### 9.9 Step 9 — Save the deployment model and its provenance

[`save_model_with_metadata()`](src/training.py):

- `joblib.dump(best_model, "Models/rg_model.joblib")` (or `ug_model.joblib`)
  — the entire fitted `sklearn.Pipeline` (preprocessing **and** model
  together, so no separate encoder/scaler file ever needs to be tracked).
- A sidecar metadata JSON recording: training timestamp, row/vehicle
  counts, the training vehicle list, the excluded-outlier vehicle list, the
  exact feature list used, the winning model's name, and its LOOCV
  RMSE/MAE/R² — a full, human-readable provenance record for every
  deployed model.
- A full 3-model comparison CSV for the dashboard's "Model Performance"
  tab.

### 9.10 Step 10 — Regenerate diagnostic plots

Run once after (re)training. `generate_plots.py` and
`feature_contribution_plots.py` independently reload the saved `.joblib`
pipelines, recompute LOOCV predictions via
`cross_val_predict(clone(saved_model), X, y, cv=LeaveOneGroupOut(), ...)`,
and render every diagnostic chart. **Important nuance** (see Section 10.2):
because this recomputation clones the *already-tuned* final pipeline rather
than re-running the inner grid search per fold, its RMSE/R² numbers are a
slightly more conservative, independent sanity check — **not** a
discrepancy or a bug, but a different (and equally legitimate) way of
stress-testing the same deployed model.

### 9.11 Step 11 — Deploy

`streamlit run app.py` loads the two saved `.joblib` pipelines and their
metadata once (cached), and is then ready to answer live predictions —
detailed fully in Section 11.

---

## 10. Model Comparison

### 10.1 Headline comparison table (canonical nested-CV numbers)

**Roll Gradient**

| Model | LOOCV RMSE (deg/g) | LOOCV MAE (deg/g) | LOOCV R² |
|---|---:|---:|---:|
| **Ridge Regression (Tuned)** ★ | **1.458** | **1.120** | **0.762** |
| MLP Neural Network (Tuned) | 1.717 | 1.250 | 0.669 |
| Random Forest Regressor (Tuned) | 1.911 | 1.672 | 0.591 |

**Understeer Gradient**

| Model | LOOCV RMSE (deg/g) | LOOCV MAE (deg/g) | LOOCV R² |
|---|---:|---:|---:|
| **MLP Neural Network (Tuned)** ★ | **3.994** | **2.911** | **0.207** |
| Random Forest Regressor (Tuned) | 4.105 | 2.932 | 0.163 |
| Ridge Regression (Tuned) | 4.574 | 3.320 | −0.039 |

★ = the model actually saved and used by the dashboard.

### 10.2 A note on two different sets of numbers in this project

Careful readers who compare the numbers above against the dashboard's
actual-vs-predicted or residual charts will notice a second, slightly
different set of figures there: **RG RMSE ≈ 1.25, R² ≈ 0.82**, and **UG
RMSE ≈ 3.97, R² ≈ 0.22** (on the current 35-row/10-vehicle dataset — these
numbers move a little on every retrain and are not meant to be quoted
precisely). This is not an inconsistency to be alarmed by — it is two
different, both-legitimate evaluation procedures, explained fully in
Section 9.10:

- The table above (and the dashboard's headline metric cards) come from
  **`run_nested_cv`**, which re-tunes hyperparameters inside every one of
  the 10 outer folds (Section 7.4) — the more statistically rigorous
  number, and the one that should be quoted as "the model's performance."
- The scatter/residual/QQ/per-vehicle plots come from **`generate_plots.py`**
  re-evaluating the single already-tuned deployment model via
  `cross_val_predict` with *no* per-fold re-tuning — a quicker, fully
  independent sanity check that the deployed model behaves sensibly, useful
  precisely *because* it is computed differently. It can land on either side
  of the nested-CV number depending on fold variance — direction is not
  guaranteed, only that it is an independent cross-check.

Both procedures agree on the qualitative conclusion (RG is well-predicted,
UG only weakly so) — it is only the exact decimal values that differ, for
the reason above.

### 10.3 Interpreting the comparison

- **Roll Gradient:** Ridge wins by a clear margin (RMSE ~15% lower
  than the next-best MLP). This matches the physics discussion in Section
  3.4 and Section 8.1: RG's governing relationship is close to linear (after
  the Yeo–Johnson transform), so a linear model with the right target
  transform beats models that spend their extra flexibility fitting noise
  instead of signal on 35 rows.
- **Understeer Gradient:** all three models land in a **narrow band**
  (RMSE 3.99–4.57) with MLP nominally first — but the R² values (0.207,
  0.163, −0.039) show the linear model (Ridge) is qualitatively different
  (actively worse than the mean baseline) while MLP and RF are close
  cousins of each other. Given the bootstrap analysis in Section 7.11, the
  MLP-over-RF margin (0.207 vs. 0.163 R², a 0.044 R² gap) should be read as
  **statistically indistinguishable, not a confident win**.

### 10.4 Per-vehicle performance — why the fleet-average number hides a lot

RMSE computed separately for each of the 10 held-out vehicles (labelled
generically here since the underlying fleet is not published):

| Vehicle | RG RMSE (deg/g) | UG RMSE (deg/g) |
|---|---:|---:|
| Vehicle 1 | 0.97 | 1.05 |
| Vehicle 2 | 2.83 | 5.14 |
| Vehicle 3 | 0.51 | 2.58 |
| Vehicle 4 | 1.01 | 4.82 |
| Vehicle 5 | 0.87 | **8.76** |
| Vehicle 6 | 1.31 | 4.15 |
| Vehicle 7 | 0.54 | 1.24 |
| Vehicle 8 | 1.42 | **8.21** |
| Vehicle 9 | 1.39 | 2.27 |
| Vehicle 10 | 1.05 | 1.44 |

Roll Gradient's per-vehicle RMSE is consistently modest (0.5–2.8 deg/g)
across every single vehicle — no vehicle is a catastrophic failure case.
Understeer Gradient, by contrast, is well-predicted for some vehicles
(Vehicle 1: 1.05 deg/g) and dramatically mispredicted for others (Vehicle 5:
8.76 deg/g; Vehicle 8: 8.21 deg/g) — a roughly **8× spread** across the
same fleet. This uneven pattern, rather than a uniformly mediocre fit
everywhere, is exactly what one would expect if the *missing* variable
(tyre cornering stiffness, Section 4.4/4.6) happens to matter a lot for
some specific vehicle/tyre combinations and less for others — further
circumstantial support for the "missing variable, not a bad model"
diagnosis.

### 10.5 Feature importance — three complementary views

Because Ridge, Random Forest, and MLP each expose model importance
differently, all three were computed, and the dashboard shows the two most
relevant to the deployed models:

- **Ridge standardised coefficients** (RG, deployed model): discussed fully
  in Section 8.1 — `ARB_Present` and `ARB_Stiffness_Index` dominate, with
  the collinearity caveat explained there.
- **MLP permutation importance** (UG, deployed model): each feature is
  randomly shuffled 15 times and the resulting increase in RMSE is
  measured — a feature that matters a lot will hurt performance badly when
  scrambled; a feature that is noise may show near-zero or even *negative*
  importance (shuffling it can occasionally reduce RMSE purely by chance).
  The result: `Zcg_Wheelbase_Ratio` totally dominates (≈ 0.98 mean RMSE
  increase, far larger than any other feature), followed distantly by
  `Roll_Stiffness_Ratio` (≈ 0.16) and `Tire_Make` (≈ 0.14); `ARB_Present`,
  `Type`, and — notably — `Front_WD` all sit at or below zero, meaning the
  MLP has learned to rely on them **little or not at all**, despite
  `Front_WD` being one of UG's stronger *marginal* correlations in Section
  6.5. This is a genuinely useful, somewhat surprising finding: once
  `Zcg_Wheelbase_Ratio` is available, the model apparently finds it a more
  efficient summary of the pitch/load-transfer information than
  `Front_WD` alone.
- **Partial Dependence + ICE plots** (Section 6.10): show the *model's*
  learned response curve for each feature with the others held at their
  observed distribution (thick line = average across all 35 rows; thin
  lines = each individual vehicle's own response curve, ICE = "Individual
  Conditional Expectation"). For RG, all four plotted features show clean,
  monotonic, physically sensible trends that are consistent across nearly
  every individual vehicle (thin lines roughly parallel to the thick
  average line) — visual confirmation that Ridge has learned one consistent
  global relationship, not different rules for different vehicles. For UG,
  the `Zcg_Wheelbase_Ratio` panel shows a clear downward (oversteer-inducing)
  trend on average, but the individual per-vehicle ICE lines fan out much
  more, and the `Roll_Stiffness_Ratio`/`Front_WD` panels show *some*
  vehicles trending up while others trend down for the same feature — a
  visual signature of a model straining to fit a genuinely weaker, less
  consistent signal.

---

## 11. Streamlit Dashboard

### 11.1 Architecture

`app.py` is a single-file Streamlit application with **no hardcoded
knowledge of feature lists, category values, or metric numbers** — every
one of those is read live from the saved artefacts (`Models/*.joblib`,
`Models/*_metadata.json`, `Data/processed_dataset.csv`,
`Models/outlier_vehicles.json`). This means retraining a model (Section 9)
and simply restarting the dashboard is enough to pick up new features,
new categories, or new metrics — **no dashboard code change is ever
required** for a routine retrain.

Two Streamlit caching decorators keep the app responsive:

- `@st.cache_data` — caches the *data* returned by a function (training
  metadata ranges, JSON metadata files); Streamlit invalidates the cache if
  the function's code or arguments change.
- `@st.cache_resource` — caches the *loaded model objects* themselves
  (`joblib.load(...)`), which should only ever be loaded once per app
  session, not once per user interaction.

### 11.2 The five tabs

1. **🎯 Prediction** — the main interactive tool. The engineer enters
   vehicle & test details (type, test radius), mass/CG geometry, chassis
   geometry, tyre specification, tyre pressures, and ARB/suspension
   configuration, across three columns. Two large result cards immediately
   show predicted RG and UG in deg/g, each annotated with the model's own
   LOOCV RMSE as an honest "± uncertainty" figure, plus a plain-language
   label (e.g. "High roll tendency", "Understeer") and — specifically for
   UG when its LOOCV R² is below 0.3 — an explicit on-screen caution that
   the number should be read as directional only.
2. **📊 Model Performance** — reproduces the LOOCV summary cards, the full
   3-model comparison tables, the actual-vs-predicted scatter, residual
   analysis, Q-Q plots, and per-vehicle RMSE bar chart described in
   Sections 7 and 10, so any user of the dashboard (not just someone
   reading this report) can see exactly how trustworthy each prediction
   is likely to be.
3. **🔬 Feature Analysis** — shows the *live* engineered feature values for
   whatever inputs are currently entered in Tab 1 (so a user can see exactly
   which numbers the model is actually consuming), alongside the
   feature-importance, correlation, mutual-information, and engineered-vs-
   raw scatter plots from Section 6 and Section 10.5.
4. **📁 Dataset & Methodology** — dataset overview counts, target
   distributions, the feature correlation heatmap, both learning curves,
   the two physics interpretability indices (`Physics_RG_Index`,
   `UG_Physics_Index`, Sections 3.4/4.4, computed live for the current
   input), and a written summary of the cross-validation strategy and known
   limitations (Section 16).
5. **🛠️ Data & Retrain** — lets an engineer edit the raw dataset directly
   (add or remove rows), then run a full retrain preview through the same
   pipeline described in Section 9, into an isolated staging area that
   never touches the live model files. Before/after LOOCV RMSE and R² are
   shown per target with independent apply/discard controls, so improving
   one target never forces reverting the other. Promoting a staged run
   updates the processed dataset, both models' metadata and comparison
   tables (only for whichever target's checkbox is ticked), and regenerates
   every diagnostic plot.

### 11.3 The prediction pipeline, end to end

1. Raw widget inputs are collected into a single `raw_inputs` dict
   (`front_load`, `rear_load`, `zcg`, …).
2. [`validate_inputs()`](src/feature_engineering.py) checks basic
   physical sanity (no negative masses, no non-positive pressures, non-zero
   total mass, etc.) before anything else runs; any failure is shown to the
   user and prediction is halted.
3. [`build_inference_row()`](src/feature_engineering.py) maps
   `raw_inputs` onto the exact column names `add_engineered_features()`
   expects, then calls that **same function used during training** to
   compute every engineered feature for this one row — guaranteeing
   inference-time preprocessing is bit-for-bit identical to training-time
   preprocessing, which is the single most common source of silent bugs in
   deployed ML systems ("training/serving skew"), deliberately engineered
   away here.
4. [`get_model_input()`](src/feature_engineering.py) reads each saved
   pipeline's own `feature_names_in_` (recorded by scikit-learn's
   `ColumnTransformer` at fit time) and reindexes the engineered-feature row
   to exactly that column list and order — so the dashboard never needs its
   own hardcoded copy of `RG_FEATURES`/`UG_FEATURES` that could silently
   drift out of sync with the actual saved model.
5. `rg_model.predict(...)` / `ug_model.predict(...)` return the final
   numbers.

### 11.4 Safety nets built into the dashboard

- **Extrapolation warnings** — if the entered CG height, total mass, or
  track width falls outside the observed training range (Section 5.6), a
  warning banner appears, because a model — especially Ridge, and
  especially tree-based RF, which cannot extrapolate past its training
  leaves at all — should not be trusted far outside the data it learned
  from.
- **Unknown-category warnings** — a tyre brand or vehicle type never seen
  in training is explicitly flagged to the user, rather than silently
  defaulting to the reference category (Section 7.7) without the user's
  knowledge.
- **Hard failure over silent wrong answers** — if a saved model ever needs
  a feature that `add_engineered_features()` does not produce, `app.py`
  raises a visible error rather than guessing.
- **Low-reliability banner for UG** — whenever the deployed UG model's
  LOOCV R² is below 0.3 (currently true, at 0.21), the Prediction tab shows
  an explicit "Limited reliability" note next to the UG result, naming the
  missing tyre cornering-stiffness variable as the root cause (Section
  4.6) — the same honesty about the model's limits that this report aims
  to carry through every section.

### 11.5 Visual design

The dashboard uses a light, dark-blue automotive theme (primary blue
`#1565C0`, configured both in inline CSS within `app.py` and in
`.streamlit/config.toml`), with card-based metric displays, styled
tabs, and consistent color-coding (blue = Roll Gradient, amber = Understeer
Gradient) carried through every chart, so the same visual language appears
whether a chart is viewed inside the dashboard or this report.

---

## 12. Code Architecture

```
.
├── app.py                              The Streamlit dashboard (Section 11).
├── generate_plots.py                   Regenerates the main diagnostic plots (Section 9.10).
├── feature_contribution_plots.py       Regenerates the scatter + PDP/ICE feature-contribution plots.
├── requirements.txt                    Pinned dependencies (Python 3.10.2 — joblib/sklearn versions
│                                        are pinned together because a fitted Pipeline is not guaranteed
│                                        to reload correctly across scikit-learn minor versions).
├── src/
│   ├── feature_engineering.py          Single source of truth for every feature formula (Section 5, 9).
│   ├── training.py                     Single source of truth for the CV strategy and model suite (Section 8, 9).
│   └── data_pipeline.py                Full retrain pipeline: raw → processed → trained models (Section 9, 11.2).
└── README.md

The following directories are part of the project's runtime data flow but
are not published in this repository (they hold proprietary test data and
trained artifacts derived from it):
  Data/         raw + processed dataset
  Models/       trained .joblib pipelines, metadata, comparison CSVs
  Reports/      evaluation plots
```

### 12.1 Why the "single source of truth" pattern matters here

Two files — [`src/feature_engineering.py`](src/feature_engineering.py)
and [`src/training.py`](src/training.py) — are imported by *every* other
piece of code in the project (`app.py`, `generate_plots.py`,
`feature_contribution_plots.py`, `src/data_pipeline.py`). This is a
deliberate architectural choice, not an accident: it guarantees that a
feature formula, or a cross-validation strategy, is defined **exactly
once**. If it were instead copy-pasted separately into each entry point (a
common anti-pattern in ML projects), the training-time and inference-time
computations could silently drift apart over time as one copy was edited
and another was forgotten — a well-known and hard-to-detect class of
production ML bugs. Adding a new engineered feature or a new candidate
model is only ever a one-line change in one of these two files, and every
other entry point picks it up automatically on its next run/restart.

### 12.2 Data flow summary

```
raw dataset
        │  (feature engineering + outlier flagging)
        ▼
processed dataset  +  outlier list
        │
        ├──────────────► RG training ──► rg_model.joblib + rg_metadata.json + model_comparison_rg.csv
        │
        └──────────────► UG training ──► ug_model.joblib + ug_metadata.json + model_comparison_ug.csv
                                │
                                ▼
                  generate_plots.py / feature_contribution_plots.py
                                │
                                ▼
                          Reports/Plots/*.png
                                │
                                ▼
                             app.py  (reads models + metadata + plots + processed dataset live)
```

---

## 13. Mathematical Appendix

A single consolidated reference of every equation used in this report.

**Lateral acceleration**
$$a_y = \frac{V^2}{R}$$

**Roll Gradient (definition)**
$$RG = \frac{\Delta \phi}{\Delta a_y}$$

**Roll moment balance (governing physics of RG)**
$$M \cdot a_y \cdot Z_{cg} = K_\phi \cdot \phi \;\Rightarrow\; RG \propto \frac{M \cdot Z_{cg}}{K_\phi}$$

**Roll Gradient interpretability index (display-only, not a model input)**
$$\text{Physics\_RG\_Index} = \frac{M \cdot Z_{cg}}{\bar P \cdot W_{tire} \cdot T_W^2}$$

**Steady-state steering equation (bicycle model)**
$$\delta = 57.3\,\frac{L}{R} + UG \cdot a_y$$

**Understeer Gradient (definition, rearranged for measurement)**
$$UG = \frac{\delta - 57.3\,L/R}{a_y}$$

**Understeer Gradient (Olley's formula, governing physics)**
$$UG = \frac{W_f}{C_{\alpha f}} - \frac{W_r}{C_{\alpha r}}$$

**Understeer Gradient interpretability index (display-only, not a model input)**
$$\text{UG\_Physics\_Index} = \frac{1}{W_{tire}}\left(\frac{FWD}{P_f} - \frac{RWD}{P_r}\right)$$

**Tyre lateral force (small slip-angle approximation)**
$$F_y \approx C_\alpha \cdot \alpha$$

**Robust Mahalanobis distance (multivariate outlier detection)**
$$D_M(x) = \sqrt{(x-\hat\mu)^T \hat\Sigma^{-1}(x-\hat\mu)}, \qquad D_M^2 \sim \chi^2_{d},\quad \text{threshold} = \chi^2_{0.99,\,d=9} = 21.666$$

**Ridge Regression loss**
$$\hat\beta_{Ridge} = \arg\min_\beta \left[\sum_i (y_i - X_i\beta)^2 + \alpha\sum_j \beta_j^2\right]$$

**Variance Inflation Factor**
$$VIF_i = \frac{1}{1-R_i^2}$$

**Standardisation (feature scaling)**
$$z = \frac{x-\mu}{\sigma}$$

**MLP forward pass (one hidden layer)**
$$\hat y = W^{(2)\top} g\!\left(W^{(1)\top}x + b^{(1)}\right) + b^{(2)}$$

**Evaluation metrics**
$$MAE = \frac{1}{n}\sum_i \lvert y_i-\hat y_i\rvert \qquad RMSE = \sqrt{\frac{1}{n}\sum_i (y_i-\hat y_i)^2} \qquad R^2 = 1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}$$

**Pearson correlation**
$$r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2 \sum_i (y_i-\bar y)^2}}$$

**Spearman correlation**
$$\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}, \qquad d_i = \text{difference in ranks}$$

### Variable glossary

| Symbol | Meaning | Unit |
|---|---|---|
| $M$ | Total vehicle mass | kg |
| $Z_{cg}$ | CG height above ground | mm |
| $X_{cg}$ | CG longitudinal position from front axle | mm |
| $Y_{cg}$ | CG lateral offset | mm |
| $T_W$ | Track width | mm |
| $L$ | Wheelbase | mm / m |
| $W_{tire}$ | Tyre section width | mm |
| $D_{ARB}$ | Anti-roll bar solid bar diameter | mm |
| $P_f, P_r$ | Front / rear tyre pressure | psi |
| $W_f, W_r$ | Front / rear axle weight | kg / N |
| $FWD, RWD$ | Front / rear weight fraction | — |
| $C_\alpha$ | Tyre cornering stiffness | force/deg |
| $\alpha$ (tyre) | Tyre slip angle | deg |
| $\alpha$ (Ridge) | L2 regularisation strength | — |
| $\phi$ | Body roll angle | deg |
| $a_y$ | Lateral acceleration | g |
| $\delta$ | Road-wheel steering angle | deg |
| $R$ | Turn radius | m |
| $V$ | Vehicle speed | m/s |
| $K_\phi$ | Total roll stiffness | N·mm/deg |

---

## 14. Physics Appendix

- A vehicle's motion decomposes into **roll** (about the longitudinal axis),
  **pitch** (about the lateral axis), and **yaw** (about the vertical axis).
- **Lateral acceleration** $a_y = V^2/R$, conventionally expressed in g's.
- **Load transfer**: cornering/braking/accelerating shifts vertical load
  between wheels because inertial forces act at CG height while tyres react
  at ground level.
- **Roll Gradient (RG)**: body roll angle per g of lateral acceleration,
  driven by mass, CG height, and total roll stiffness (springs, ARB, tyre
  compliance) acting through the track width.
- **Understeer Gradient (UG)**: extra steering angle needed per g of
  lateral acceleration beyond pure geometric (Ackermann) steering, governed
  by the front/rear balance of axle load versus tyre cornering stiffness.
- **Tyre cornering stiffness ($C_\alpha$)** is the dominant physical driver
  of UG and is **not measured anywhere in this project's dataset** — the
  single biggest physical limitation carried through the whole project.
- **Anti-roll bars** add roll stiffness (scaling with diameter to the 4th
  power) without affecting ride comfort over symmetric bumps, because they
  only react to *opposite* left/right wheel motion.
- **Vehicle type** (3W vs. 4W) captures broad architectural differences
  (fewer wheels, different suspension kinematics, different load-carrying
  layout) not otherwise represented by the continuous geometric features.

---

## 15. ML Appendix

- **Small-sample regime**: 35 usable rows across 10 unique vehicles — every
  methodological choice in this project (grouped CV, nested CV, fixed/tiny
  model architectures, conservative feature-selection) exists specifically
  to make honest, non-leaky learning possible at this scale.
- **Vehicle-level `LeaveOneGroupOut`**: the outer validation loop, holding
  out an entire vehicle's runs per fold, to prevent optimistic leakage
  between runs of the same chassis.
- **Nested `GridSearchCV`**: the inner loop, re-tuning hyperparameters using
  only the current fold's training data, to prevent tuning leakage.
- **Standardisation**: numeric features rescaled to zero mean / unit
  variance, fit only on training folds.
- **One-Hot Encoding** with `drop='first'` and `handle_unknown='ignore'`:
  categorical features become 0/1 columns; unseen categories degrade
  gracefully to the reference category rather than crashing.
- **Ridge Regression**: linear model with an L2 penalty, additionally
  wrapped in a Yeo–Johnson target-power-transform for RG.
- **Random Forest**: bagged ensemble of shallow decision trees, used as a
  conservative non-linear cross-check.
- **MLP**: a deliberately tiny (4-hidden-unit), heavily L2-regularised
  neural network — the best model found for UG, by a narrow, noise-level
  margin over Random Forest.
- **MAE / RMSE / R²**: reported together because each tells a different
  part of the accuracy story; R² can be negative, and per-fold R² on
  2-sample folds is close to meaningless (hence the vehicle-level
  bootstrap check).
- **Learning curves**: confirm the model, not the algorithm choice, is the
  bottleneck for RG (more data would clearly help); for UG, a missing
  variable is an *additional*, data-volume-independent ceiling.
- **Permutation importance / Partial Dependence + ICE**: used, instead of
  relying solely on marginal correlations, to see what a *fitted model*
  actually depends on once other correlated features are accounted for.

---

## 16. Results

### 16.1 Final performance summary

| Metric | Roll Gradient | Understeer Gradient |
|---|---:|---:|
| Best model | Ridge Regression | MLP Neural Network |
| LOOCV RMSE | 1.46 deg/g | 3.99 deg/g |
| LOOCV MAE | 1.12 deg/g | 2.91 deg/g |
| LOOCV R² | 0.762 | 0.207 |
| Residual normality (Shapiro p) | 0.391 (normal)* | 0.010 (non-normal)* |
| Worst single-vehicle RMSE | 2.83 (Vehicle 2) | 8.76 (Vehicle 5) |

*Treat these two specific p-values as approximate; the qualitative
conclusion (RG residuals ~normal, UG residuals not) is robust to small
dataset revisions.

### 16.2 Strengths

- Roll Gradient is predicted with strong, honestly-validated accuracy
  (R² ≈ 0.76, RMSE ≈ 1.46 deg/g against a target range spanning roughly 11
  deg/g) — good enough for genuine early-stage design screening.
- Every reported number in this project is the product of **vehicle-level,
  nested** cross-validation, actively engineered to avoid the two most
  common ways small-data ML projects overstate their own accuracy (row-level
  leakage across repeated-vehicle runs, and hyperparameter-tuning leakage).
- The pipeline is fully reproducible and self-consistent: one feature
  formula file and one training-strategy file are shared, byte-for-byte,
  between offline training and the live dashboard, eliminating an entire
  class of training/serving mismatch bugs.
- The project is honest about its own limitations at every layer — code
  comments, the README, and the dashboard's own UI all independently state
  the same caveats about UG's low R² and its root cause, rather than only
  the strongest headline numbers being visible.

### 16.3 Limitations

- **Sample size.** 35 clean rows / 10 unique vehicles is a small sample by
  any standard; learning curves show RG performance is still meaningfully
  data-limited even now.
- **Missing tyre cornering stiffness.** The single largest limitation:
  Understeer Gradient's governing physics formula needs a variable
  ($C_\alpha$) that this dataset simply does not contain, capping UG R² at
  roughly 0.21 regardless of modelling sophistication.
- **Per-vehicle variance.** UG prediction quality varies roughly 8× across
  the 10 vehicles in the fleet — the aggregate R² masks vehicles that are
  predicted well and vehicles that are not.
- **Generalisation boundary.** Both models are only validated on, and
  should only be trusted for, vehicles broadly similar to the 10 in the
  training fleet (3W and 4W vehicles of comparable scale) — the
  dashboard's extrapolation warnings exist specifically to police this
  boundary.

### 16.4 Engineering conclusions

1. **Roll Gradient can already be used as a real early-design screening
   tool** for vehicles similar to this fleet — its physics is close to
   linear in the available variables, and the model captures it well.
2. **Understeer Gradient predictions should currently be read as
   directional guidance only** ("this design change will likely push the
   vehicle toward more understeer / more oversteer"), not as a precise
   target number — and the dashboard is explicit about this to its users.
3. The most valuable next investment for this project is **not** a fancier
   model — it is **collecting the missing tyre cornering-stiffness data**
   (Section 17), because no amount of algorithmic sophistication can
   recover a variable that was never measured.

---

## 17. Future Work

1. **Measure or source tyre cornering stiffness ($C_\alpha$) directly.**
   Either from tyre-manufacturer test-rig data for each tyre make/size
   combination, or from a dedicated flat-trac / drum-test rig. This is
   flagged everywhere in this project as the single highest-leverage
   improvement available for the Understeer Gradient model — it is a data
   problem, not a modelling problem. (Back-calculating $C_\alpha$ *from the
   same vehicle's own test run* would not help — that would be circular:
   the target UG value would leak directly into a feature used to predict
   it.)
2. **Collect more, and more diverse, vehicles.** The learning curves
   (Section 7.8) show Roll Gradient's validation error is still trending
   down at n = 35 — additional vehicles of the *same broad kind* would
   likely still improve RG directly, and would also make Understeer
   Gradient's cross-validation folds larger and less noisy (Section 7.11).
3. **Be cautious about further hand-engineered feature recombinations for
   UG.** Project history recorded in the codebase's own feature-list
   comments (`src/feature_engineering.py`) shows that removing two
   collinear features (`Pressure_Ratio`, `Tire_Stress_Difference`) *raised*
   UG R² from 0.096 to 0.160 — more engineered features are not
   automatically better at this sample size. Two further candidate
   interaction/proxy features explored during project review (a
   roll-couple interaction term, and a tyre sidewall-stiffness proxy) were
   each tested via the same nested-CV ablation procedure and did **not**
   improve UG performance — reinforcing that recombinations of the
   *existing* static specification variables are close to exhausted as a
   source of new signal; the productive path is genuinely new data
   (points 1–2 above), not further feature arithmetic on the same inputs.
4. **Physics-informed ML.** Rather than a purely data-driven MLP/RF for
   UG, a hybrid approach — e.g. fitting the classical Olley formula
   (Section 4.3) with a *learned* proxy for $C_\alpha$ as one model
   component, with a small ML correction term for whatever the physics
   formula still misses — could make better use of the strong domain
   knowledge already encoded in this project's physics indices
   (`UG_Physics_Index`) instead of leaving that formula purely as a
   dashboard display feature.
5. **Real-time / on-vehicle deployment.** The current dashboard is a
   design-time tool (static inputs → static prediction). A natural
   extension is estimating RG/UG in real time from an instrumented
   prototype's live IMU and steering-angle signals, enabling in-the-loop
   comparison between the *predicted* handling and the *measured* handling
   as a design is refined.
6. **Digital-twin integration.** Feeding this model's predictions into a
   broader vehicle digital-twin/simulation environment (e.g. as a fast
   surrogate for RG/UG inside a larger ride-and-handling simulation loop)
   would let design teams evaluate handling trade-offs alongside other
   simulated attributes (ride comfort, structural loads) in one place.
7. **Sensor integration for continuous data collection.** Standardising a
   lightweight, always-on logging rig (IMU + steering encoder) across
   future prototype builds would turn every future test run into
   additional, consistently-formatted training data automatically,
   directly addressing limitation #2 above without extra dedicated data-
   collection campaigns.
8. **Boosting was reconsidered and declined.** A diagnostic review proposed
   physics-informed residual boosting, extremely regularized gradient
   boosting (stumps, low learning rate), and CatBoost for the UG model. All
   three were evaluated against existing project history — plain Gradient
   Boosting was already removed from the suite as statistically
   indistinguishable from Random Forest (Section 8), and the two prior
   feature-engineering ablations (item 3 above) confirmed the bottleneck is
   the missing $C_\alpha$ signal rather than learner choice — and none were
   implemented. Physics-informed residual boosting remains the one variant
   worth revisiting, but only as part of item 4 above (first validating a
   trustworthy physics baseline), not as a standalone model swap.

---

## 18. References

**Vehicle dynamics**

- Milliken, W. F., & Milliken, D. L. — *Race Car Vehicle Dynamics*, SAE
  International. (Standard reference for the bicycle-model steering
  equation and the Olley understeer-gradient formula used in Section 4.)
- Gillespie, T. D. — *Fundamentals of Vehicle Dynamics*, SAE International.
  (Standard reference for roll-moment analysis, load transfer, and the SAE
  vehicle axis system used in Section 2–3.)
- SAE J266 — *Steady-State Directional Control Test Procedures for
  Passenger Cars and Light Trucks* (the constant-radius circular test
  methodology consistent with this project's `Test condition` column and
  the physical basis of both target variables).

**Statistics and outlier detection**

- Rousseeuw, P. J., & Van Driessen, K. (1999). "A Fast Algorithm for the
  Minimum Covariance Determinant Estimator." *Technometrics* — the robust
  covariance estimator (`sklearn.covariance.MinCovDet`) used for
  multivariate outlier detection in Section 6.4.
- Yeo, I.-K., & Johnson, R. A. (2000). "A New Family of Power
  Transformations to Improve Normality or Symmetry." *Biometrika* — the
  target transform (`sklearn.preprocessing.PowerTransformer`,
  `method='yeo-johnson'`) used inside the Ridge pipeline in Section 8.1.

**Machine learning**

- Hastie, T., Tibshirani, R., & Friedman, J. — *The Elements of Statistical
  Learning*. (Ridge regression, bias-variance trade-off, cross-validation —
  Sections 7–8.)
- Breiman, L. (2001). "Random Forests." *Machine Learning* 45(1) — the
  ensemble method used in Section 8.2.
- Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python."
  *Journal of Machine Learning Research* — the library implementing every
  model, preprocessing step, and cross-validation tool used throughout this
  project (`scikit-learn==1.7.2`, pinned in `requirements.txt`).

**Software**

- McKinney, W. — *pandas* (data manipulation).
- Streamlit Inc. — *Streamlit* (`streamlit==1.58.0`, the dashboard
  framework in `app.py`).
- Hunter, J. D. — *Matplotlib*; Waskom, M. — *seaborn* (all diagnostic
  plots, generated by `generate_plots.py` and `feature_contribution_plots.py`).
