# Dynamic Pricing in the Airline Industry: Adjusting for Fuel Price Volatility and Exchange Rate Risks

A time-series analytics project that models and forecasts airline ticket prices on the Delhi → Cochin route by incorporating external macroeconomic shocks — specifically Brent crude oil price volatility and USD/INR exchange rate fluctuations. The project delivers an end-to-end pipeline from raw data to an interactive Streamlit dashboard with what-if scenario simulation and real-time volatility risk scoring.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation & Applications](#motivation--applications)
- [Architecture & Data Pipeline](#architecture--data-pipeline)
- [Models](#models)
- [Dashboard Features](#dashboard-features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Key Results](#key-results)

---

## Project Overview

Airline ticket pricing is highly sensitive to fuel costs and currency exchange rates. This project quantifies those sensitivities using econometric and machine learning techniques:

- **VAR (Vector AutoRegression)** captures the multivariate dynamic interactions between daily ticket prices, Brent oil prices, and the USD/INR exchange rate.
- **Impulse Response Functions (IRF)** measure how a one-unit shock in oil or FX propagates through ticket prices over 15 future days.
- **XGBoost** incorporates the IRF-derived lag structure (15 days for both oil and FX) alongside temporal features to produce a more accurate forecast.
- **Volatility Risk Scoring** provides a weighted composite risk indicator based on rolling 14-day coefficient-of-variation for both macro drivers.

The dataset covers flights in the March–June 2019 period (Kaggle: Indian flight prices) merged with daily Brent oil and USD/INR data pulled from Yahoo Finance.

---

## Motivation & Applications

| Stakeholder | Application |
|---|---|
| **Airline Revenue Management** | Anticipate pricing adjustments ahead of fuel cost spikes |
| **Travel Aggregators / OTAs** | Surface risk-aware fare predictions to consumers |
| **Hedging & Procurement Teams** | Identify the 15-day lag window where fuel shocks peak in ticket prices |
| **Financial Analysts** | Quantify the relative impact of oil (38.4%) vs. FX (61.6%) on fare volatility |
| **Academic / Research** | Demonstrate Granger causality and IRF analysis in a real airline pricing context |

---

## Architecture & Data Pipeline

```
Raw Flight Data (Data_Train.xlsx)          Yahoo Finance API
        │                                        │
        ▼                                        ▼
  build_master.py                         clone_data.py
  ─ Filter: DEL → COK route               ─ Brent Oil (BZ=F)
  ─ Aggregate daily median price           ─ USD/INR (INR=X)
  ─ Merge on Date                          ─ Forward-fill weekends
        │                                        │
        └──────────────┬─────────────────────────┘
                       ▼
              master_data_merged.csv
                       │
                       ▼
          eda_and_testing.ipynb  (optional / exploratory)
          ─ ADF stationarity tests
          ─ Pearson correlation heatmap
          ─ First-difference & log-return feature creation
                       │
                       ▼
             modeling.ipynb  (optional / research)
          ─ Granger causality tests
          ─ VAR lag selection (AIC/BIC/HQIC)
          ─ IRF analysis (15 horizons)
          ─ XGBoost model development
          ─ Model comparison (VAR vs. XGBoost)
                       │
                       ▼
          dashboard/prepare_backend.py  (training script)
          ─ Trains VAR & XGBoost
          ─ Generates config.json, full_data.csv,
            irf_data.csv, var_model.pkl, xgb_model.pkl
                       │
                       ▼
          dashboard/app.py           dashboard/at_risk.py
          ─ Main interactive dashboard  ─ Volatility risk monitor
          ─ What-if oil shock simulator ─ Risk level classification
          ─ Model selector (VAR/XGBoost) ─ Price impact estimation
```

---

## Models

### VAR — Vector AutoRegression

- Operates on first-differenced (stationary) series: `Δ Price`, `Δ Brent Oil`, `Δ USD/INR`
- Optimal lag selected by **AIC criterion** over candidates 1–6 (selected: **lag 6**)
- Forecasting strategy: 1-step-ahead rolling forecast on the 20% held-out test set
- Impulse Response Functions extracted for 15 horizons to identify the peak transmission lag

| Metric | Value |
|---|---|
| RMSE | 5,194.71 INR |
| MAE | 4,671.78 INR |
| MAPE | 57.13% |

### XGBoost — Gradient Boosted Trees

- Uses the IRF-derived **15-day lag** for both oil and FX as input features
- Feature set: `Brent_lag_15`, `USD_INR_lag_15`, `day_of_week`, `month`, `day_of_month`, `is_weekend`
- Hyperparameters: `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`

| Metric | Value |
|---|---|
| RMSE | 3,248.82 INR |
| MAE | 2,113.62 INR |
| MAPE | 39.69% |

**XGBoost outperforms VAR by ~37% in RMSE.** The dominant feature is `month` (42.95% importance), confirming strong seasonal pricing patterns; `USD_INR_lag_15` ranks second (19.91%), followed by `is_weekend` (15.86%).

---

## Dashboard Features

### Main Dashboard (`app.py`)

- **3 KPI cards** — average ticket price, oil price change %, and model RMSE for the selected date range
- **Dual-axis time series chart** — ticket price and Brent oil price on the same timeline
- **Model selector** — switch between XGBoost and VAR predictions in real time
- **What-if oil shock simulator** — apply a custom % shock (−50% to +50%) on any date and see how XGBoost re-forecasts ticket prices
- **Feature importance bar chart** — XGBoost weight breakdown

Launch: `streamlit run times/dashboard/app.py`

### Risk Monitoring Dashboard (`at_risk.py`)

- **4 risk-level KPI cards** — count and percentage of days classified as Low / Medium / High / Critical
- **Risk score timeline** — composite score color-coded by threshold
- **Rolling volatility panel** — 14-day coefficient of variation for oil and FX separately
- **Price impact estimation** — estimated INR fare impact derived from macro deviations
- **Configurable sidebar** — tune the volatility window (7–30 days) and filter by risk level

Risk score formula:

```
risk_score = W_Oil × oil_CV(14d)  +  W_FX × fx_CV(14d)
           = 0.384 × oil_CV       +  0.616 × fx_CV
```

| Risk Level | Threshold |
|---|---|
| Low (green) | < 2.0% |
| Medium (orange) | 2.0% – 4.0% |
| High (red) | 4.0% – 6.0% |
| Critical (crimson) | ≥ 6.0% |

Launch: `streamlit run times/dashboard/at_risk.py`

---

## Project Structure

```
time-series-/
├── README.md
└── times/
    ├── clone_data.py              # Download Brent oil & USD/INR from Yahoo Finance
    ├── build_master.py            # Merge flight data with macro data
    ├── eda_and_testing.ipynb      # Exploratory analysis & stationarity tests
    ├── modeling.ipynb             # Model development, IRF, & comparison
    ├── data/
    │   ├── external_macro_data.csv    # Brent oil & USD/INR (Jan–Jul 2019)
    │   └── master_data_merged.csv     # Daily median prices merged with macro data
    └── dashboard/
        ├── app.py                 # Main Streamlit dashboard
        ├── at_risk.py             # Volatility risk monitoring dashboard
        ├── prepare_backend.py     # Model training & artifact generation
        └── artifacts/
            ├── config.json        # Lag values, metrics, feature columns
            ├── full_data.csv      # All data + VAR & XGBoost predictions
            ├── irf_data.csv       # Impulse response values (horizons 0–15)
            ├── var_model.pkl      # Serialized VAR model
            └── xgb_model.pkl      # Serialized XGBoost model
```

> **Note:** The raw flight data file (`Data_Train.xlsx`) is not included in the repository due to size/licensing. It is the [Flight Price Prediction dataset from Kaggle](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh). Place it at `time series/Data_Train.xlsx` relative to the project root before running `build_master.py`.

---

## Prerequisites

- Python **3.9+**
- pip

Required packages:

```
pandas
numpy
statsmodels
scikit-learn
xgboost
yfinance
streamlit
plotly
seaborn
matplotlib
openpyxl
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Dynamic-Pricing-in-the-Airline-Industry-Adjusting-for-Fuel-Price-Volatility-and-Exchange-Rate-Risks.git
cd Dynamic-Pricing-in-the-Airline-Industry-Adjusting-for-Fuel-Price-Volatility-and-Exchange-Rate-Risks
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy statsmodels scikit-learn xgboost yfinance streamlit plotly seaborn matplotlib openpyxl
```

---

## Running the Project

All commands are run from the repository root. The steps are listed in order; steps 1–4 only need to be run once to produce the artifacts. If the artifacts already exist (`dashboard/artifacts/`), you can skip directly to step 5.

### Step 1 — Download macroeconomic data

```bash
python times/clone_data.py
```

Downloads daily Brent crude oil prices and USD/INR exchange rates from Yahoo Finance and saves them to `times/data/external_macro_data.csv`.

### Step 2 — Build the master dataset

```bash
python times/build_master.py
```

Reads the raw flight data (`Data_Train.xlsx`), filters to the Delhi → Cochin route, computes daily median ticket prices, and merges with the macro data. Output: `times/data/master_data_merged.csv`.

### Step 3 — (Optional) Exploratory analysis

Open and run all cells in `times/eda_and_testing.ipynb` using Jupyter Lab or Jupyter Notebook. This notebook performs ADF stationarity tests, correlation analysis, and engineers differenced features.

```bash
jupyter lab times/eda_and_testing.ipynb
```

### Step 4 — Train models & generate artifacts

```bash
python times/dashboard/prepare_backend.py
```

Trains the VAR and XGBoost models, runs IRF analysis, and writes all artifacts to `times/dashboard/artifacts/`. This step must complete successfully before launching either dashboard.

### Step 5 — Launch the main dashboard

```bash
streamlit run times/dashboard/app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

### Step 6 — Launch the risk monitoring dashboard

```bash
streamlit run times/dashboard/at_risk.py
```

Opens at [http://localhost:8501](http://localhost:8501) (stop the previous instance first, or use `--server.port 8502`).

---

Or opens at [https://timeseriesfianl.streamlit.app/]
## Configuration

After `prepare_backend.py` runs, `times/dashboard/artifacts/config.json` is produced with the following structure:

```json
{
  "LAG_OIL": 15,
  "LAG_FX": 15,
  "OPTIMAL_VAR_LAG": 6,
  "FEATURE_COLS": [
    "Brent_lag_15", "USD_INR_lag_15",
    "day_of_week", "month", "day_of_month", "is_weekend"
  ],
  "TARGET_COL": "Median_Price",
  "SPLIT_RATIO": 0.8,
  "metrics": {
    "VAR":     { "RMSE": 5194.71, "MAE": 4671.78, "MAPE": 57.13 },
    "XGBoost": { "RMSE": 3248.82, "MAE": 2113.62, "MAPE": 39.69 }
  },
  "feature_importance": {
    "Brent_lag_15":   0.1243,
    "USD_INR_lag_15": 0.1991,
    "day_of_week":    0.0464,
    "month":          0.4295,
    "day_of_month":   0.0422,
    "is_weekend":     0.1586
  },
  "train_period": { "start": "2019-04-21", "end": "2019-06-12" },
  "test_period":  { "start": "2019-06-15", "end": "2019-06-27" }
}
```

---

## Key Results

| Finding | Detail |
|---|---|
| **Granger causality confirmed** | Both Brent oil and USD/INR statistically Granger-cause Delhi–Cochin ticket prices |
| **Transmission lag** | Peak price response to an oil or FX shock occurs at **day 15** (IRF) |
| **Seasonality dominates** | `month` accounts for 42.95% of XGBoost feature importance |
| **FX > Oil in volatility weight** | FX contributes 61.6% vs. oil's 38.4% to the composite risk score |
| **Best model** | XGBoost (RMSE 3,249 INR) outperforms VAR (RMSE 5,195 INR) by ~37% |
| **High-risk periods** | Critical risk days (CV ≥ 6%) concentrate around sharp INR depreciation events |
