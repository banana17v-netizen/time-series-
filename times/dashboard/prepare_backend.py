"""
prepare_backend.py
──────────────────
Chạy 1 lần để:
  1. Load & xử lý dữ liệu từ master_data_merged.csv
  2. Train mô hình VAR trên dữ liệu sai phân
  3. Trích xuất lag từ IRF
  4. Train mô hình XGBoost với lag features + time features
  5. Tạo dự báo (forecasts)
  6. Lưu toàn bộ artifacts → dashboard/artifacts/
"""

import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE    = os.path.join(BASE_DIR, 'data', 'master_data_merged.csv')
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print('=' * 60)
print('  PREPARE BACKEND — Dynamic Pricing Dashboard')
print('=' * 60)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print('\n[1/6] Loading data...')
df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f'  Loaded {len(df)} rows, columns: {df.columns.tolist()}')

# ═══════════════════════════════════════════════════════════════════
# 2. TRAIN VAR MODEL
# ═══════════════════════════════════════════════════════════════════
print('\n[2/6] Training VAR model...')

VARS_DIFF = ['Median_Price_diff', 'Brent_Oil_Price_diff', 'USD_INR_Exchange_diff']
var_data = df[['Date'] + VARS_DIFF].dropna().reset_index(drop=True)
var_data = var_data.rename(columns={
    'Median_Price_diff':     'dPrice',
    'Brent_Oil_Price_diff':  'dOil',
    'USD_INR_Exchange_diff': 'dFX'
})

FEATURES_VAR = ['dPrice', 'dOil', 'dFX']
SPLIT_RATIO = 0.80
n_var = len(var_data)
split_var = int(n_var * SPLIT_RATIO)

train_var_df = var_data.iloc[:split_var].copy()
test_var_df  = var_data.iloc[split_var:].copy()

train_var_vals = train_var_df[FEATURES_VAR].values

# Find optimal lag
n_obs, n_vars = train_var_vals.shape
safe_maxlags = max(1, int((n_obs - 1) / (n_vars + 1)))
MAX_LAGS_SEARCH = min(6, safe_maxlags)

var_model_obj = VAR(train_var_df[FEATURES_VAR])
best_aic_lag = 1
best_aic_val = float('inf')
for lag in range(1, MAX_LAGS_SEARCH + 1):
    try:
        res = var_model_obj.fit(maxlags=lag, ic=None)
        if res.aic < best_aic_val:
            best_aic_val = res.aic
            best_aic_lag = lag
    except Exception:
        pass

OPTIMAL_LAG = best_aic_lag if best_aic_lag > 0 else 1
var_fit = var_model_obj.fit(maxlags=OPTIMAL_LAG, ic=None)
print(f'  Optimal VAR lag (AIC): {OPTIMAL_LAG}')

# VAR forecast & metrics
n_test_var = len(test_var_df)
history = train_var_vals.tolist()
predictions_diff = []

for i in range(n_test_var):
    seed = np.array(history[-OPTIMAL_LAG:])
    fc = var_fit.forecast(seed, steps=1)[0]
    predictions_diff.append(fc)
    history.append(test_var_df[FEATURES_VAR].values[i])

pred_arr = np.array(predictions_diff)

# Reconstruct absolute prices
last_train_price_var = df.loc[df['Date'] == train_var_df['Date'].iloc[-1], 'Median_Price'].values
last_train_price_var = last_train_price_var[0] if len(last_train_price_var) > 0 else df['Median_Price'].iloc[split_var]

pred_price_var = np.cumsum(pred_arr[:, 0]) + last_train_price_var
actual_price_var = df.loc[df['Date'].isin(test_var_df['Date']), 'Median_Price'].values

min_len = min(len(pred_price_var), len(actual_price_var))
pred_price_var = pred_price_var[:min_len]
actual_price_var = actual_price_var[:min_len]
test_dates_var = test_var_df['Date'].values[:min_len]

rmse_var = np.sqrt(mean_squared_error(actual_price_var, pred_price_var))
mae_var  = mean_absolute_error(actual_price_var, pred_price_var)
mape_var = np.mean(np.abs((actual_price_var - pred_price_var) / actual_price_var)) * 100

print(f'  VAR RMSE: {rmse_var:,.2f} | MAE: {mae_var:,.2f} | MAPE: {mape_var:.2f}%')

# ═══════════════════════════════════════════════════════════════════
# 3. EXTRACT LAG FROM IRF
# ═══════════════════════════════════════════════════════════════════
print('\n[3/6] Extracting lag from IRF...')

PERIODS = 15
irf = var_fit.irf(periods=PERIODS)
irf_vals = irf.irfs

price_idx = FEATURES_VAR.index('dPrice')
oil_idx   = FEATURES_VAR.index('dOil')
fx_idx    = FEATURES_VAR.index('dFX')

response_oil = irf_vals[:, price_idx, oil_idx]
response_fx  = irf_vals[:, price_idx, fx_idx]

LAG_OIL = int(np.argmax(np.abs(response_oil[1:])) + 1)
LAG_FX  = int(np.argmax(np.abs(response_fx[1:])) + 1)

print(f'  LAG_OIL (Brent -> Price): {LAG_OIL} days')
print(f'  LAG_FX  (USD/INR -> Price): {LAG_FX} days')

# IRF data for dashboard
irf_data = pd.DataFrame({
    'horizon': np.arange(PERIODS + 1),
    'response_oil': response_oil,
    'response_fx': response_fx,
    'cumulative_oil': np.cumsum(response_oil),
    'cumulative_fx': np.cumsum(response_fx),
})

# ═══════════════════════════════════════════════════════════════════
# 4. TRAIN XGBOOST MODEL
# ═══════════════════════════════════════════════════════════════════
print('\n[4/6] Training XGBoost model...')

xgb_df = df[['Date', 'Median_Price', 'Brent_Oil_Price', 'USD_INR_Exchange']].copy()
xgb_df = xgb_df.sort_values('Date').reset_index(drop=True)

# Lag features
xgb_df[f'Brent_lag_{LAG_OIL}']  = xgb_df['Brent_Oil_Price'].shift(LAG_OIL)
xgb_df[f'USD_INR_lag_{LAG_FX}'] = xgb_df['USD_INR_Exchange'].shift(LAG_FX)

# Time features
xgb_df['day_of_week']  = xgb_df['Date'].dt.dayofweek
xgb_df['month']        = xgb_df['Date'].dt.month
xgb_df['day_of_month'] = xgb_df['Date'].dt.day
xgb_df['is_weekend']   = (xgb_df['day_of_week'] >= 5).astype(int)

# Drop NaN
xgb_df = xgb_df.dropna().reset_index(drop=True)

FEATURE_COLS = [f'Brent_lag_{LAG_OIL}', f'USD_INR_lag_{LAG_FX}',
                'day_of_week', 'month', 'day_of_month', 'is_weekend']
TARGET_COL = 'Median_Price'

# Train/Test split
n_xgb = len(xgb_df)
split_xgb = int(n_xgb * SPLIT_RATIO)

train_xgb = xgb_df.iloc[:split_xgb].copy()
test_xgb  = xgb_df.iloc[split_xgb:].copy()

X_train = train_xgb[FEATURE_COLS]
y_train = train_xgb[TARGET_COL]
X_test  = test_xgb[FEATURE_COLS]
y_test  = test_xgb[TARGET_COL]

xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# XGBoost metrics
y_pred_xgb = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = np.mean(np.abs((y_test.values - y_pred_xgb) / y_test.values)) * 100

print(f'  XGBoost RMSE: {rmse_xgb:,.2f} | MAE: {mae_xgb:,.2f} | MAPE: {mape_xgb:.2f}%')
print(f'  Train: {len(train_xgb)} obs | Test: {len(test_xgb)} obs')

# Feature importance
feat_imp = dict(zip(FEATURE_COLS, xgb_model.feature_importances_.tolist()))

# ═══════════════════════════════════════════════════════════════════
# 5. CREATE FORECAST DATA
# ═══════════════════════════════════════════════════════════════════
print('\n[5/6] Creating forecast datasets...')

# VAR forecast dataframe
var_forecast_df = pd.DataFrame({
    'Date':           pd.to_datetime(test_dates_var),
    'Actual_Price':   actual_price_var,
    'VAR_Predicted':  pred_price_var,
})

# XGBoost forecast dataframe
xgb_forecast_df = pd.DataFrame({
    'Date':               test_xgb['Date'].values,
    'Actual_Price':       y_test.values,
    'XGBoost_Predicted':  y_pred_xgb,
})

# Merge full data with all predictions
full_data = df[['Date', 'Median_Price', 'Brent_Oil_Price', 'USD_INR_Exchange']].copy()
full_data = full_data.merge(var_forecast_df[['Date', 'VAR_Predicted']], on='Date', how='left')
full_data = full_data.merge(xgb_forecast_df[['Date', 'XGBoost_Predicted']], on='Date', how='left')

print(f'  Full data: {len(full_data)} rows')
print(f'  VAR forecast points:     {var_forecast_df.shape[0]}')
print(f'  XGBoost forecast points: {xgb_forecast_df.shape[0]}')

# ═══════════════════════════════════════════════════════════════════
# 6. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
print('\n[6/6] Saving artifacts...')

# Save models
with open(os.path.join(ARTIFACT_DIR, 'var_model.pkl'), 'wb') as f:
    pickle.dump(var_fit, f)

with open(os.path.join(ARTIFACT_DIR, 'xgb_model.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)

# Save data
full_data.to_csv(os.path.join(ARTIFACT_DIR, 'full_data.csv'), index=False)
irf_data.to_csv(os.path.join(ARTIFACT_DIR, 'irf_data.csv'), index=False)

# Save config
config = {
    'LAG_OIL': LAG_OIL,
    'LAG_FX':  LAG_FX,
    'OPTIMAL_VAR_LAG': OPTIMAL_LAG,
    'FEATURE_COLS': FEATURE_COLS,
    'TARGET_COL': TARGET_COL,
    'SPLIT_RATIO': SPLIT_RATIO,
    'metrics': {
        'VAR':     {'RMSE': round(rmse_var, 2), 'MAE': round(mae_var, 2), 'MAPE': round(mape_var, 2)},
        'XGBoost': {'RMSE': round(rmse_xgb, 2), 'MAE': round(mae_xgb, 2), 'MAPE': round(mape_xgb, 2)},
    },
    'feature_importance': {k: round(v, 4) for k, v in feat_imp.items()},
    'train_period': {
        'start': str(train_xgb['Date'].min().date()),
        'end':   str(train_xgb['Date'].max().date()),
    },
    'test_period': {
        'start': str(test_xgb['Date'].min().date()),
        'end':   str(test_xgb['Date'].max().date()),
    },
}

with open(os.path.join(ARTIFACT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f'  Saved to: {ARTIFACT_DIR}')
for fname in os.listdir(ARTIFACT_DIR):
    fpath = os.path.join(ARTIFACT_DIR, fname)
    size = os.path.getsize(fpath)
    print(f'    {fname:30s} {size:>10,} bytes')

print('\n' + '=' * 60)
print('  ALL DONE! Run: streamlit run dashboard/app.py')
print('=' * 60)
