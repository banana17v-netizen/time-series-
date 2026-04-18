"""
app.py — Streamlit Dashboard: Dynamic Pricing in the Airline Industry
═════════════════════════════════════════════════════════════════════
Run:  streamlit run dashboard/app.py
"""
import os, json, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Dynamic Pricing Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONFIG ─────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(ARTIFACT_DIR, 'full_data.csv'), parse_dates=['Date'])
    irf = pd.read_csv(os.path.join(ARTIFACT_DIR, 'irf_data.csv'))
    with open(os.path.join(ARTIFACT_DIR, 'config.json'), 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return df, irf, cfg

@st.cache_resource
def load_models():
    with open(os.path.join(ARTIFACT_DIR, 'xgb_model.pkl'), 'rb') as f:
        xgb = pickle.load(f)
    return xgb

df, irf_data, config = load_data()
xgb_model = load_models()

LAG_OIL = config['LAG_OIL']
LAG_FX  = config['LAG_FX']
FEATURE_COLS = config['FEATURE_COLS']

# ─── CUSTOM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .kpi-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }
    .kpi-value {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 8px 0 4px;
    }
    .kpi-label { font-size: 0.85rem; color: rgba(255,255,255,0.55); text-transform: uppercase; letter-spacing: 1px; }
    .kpi-delta { font-size: 0.95rem; font-weight: 600; }
    .kpi-delta.up { color: #f87171; }
    .kpi-delta.down { color: #34d399; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #e0e7ff;
        border-left: 4px solid #6366f1; padding-left: 14px;
        margin: 32px 0 16px;
    }
    .strategy-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px; padding: 20px; margin: 10px 0;
    }
    .strategy-card h4 { color: #a78bfa; margin: 0 0 8px; }
    .strategy-card p  { color: rgba(255,255,255,0.8); line-height: 1.6; margin: 0; }
    h1 { color: #e0e7ff !important; }
    .stSlider label, .stRadio label, .stDateInput label { color: #c7d2fe !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ✈️ Control Panel")
    st.markdown("---")

    # Date range
    st.markdown("### 📅 Date Range")
    min_date, max_date = df['Date'].min().date(), df['Date'].max().date()
    date_range = st.date_input("Select date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.markdown("---")

    # Model selector
    st.markdown("### 🤖 Forecast Model")
    model_choice = st.radio("Select model", ["XGBoost", "VAR"], index=0,
                            help="Choose a model to display forecast results")

    st.markdown("---")

    # What-If
    st.markdown("### ⚡ What-If Scenario")
    whatif_enabled = st.toggle("Enable oil price shock simulation", value=False)

    oil_shock_pct = 0.0
    shock_date = max_date
    if whatif_enabled:
        if model_choice == "VAR":
            st.warning("⚠️ What-If is only available with XGBoost. Switching to XGBoost.")
            model_choice = "XGBoost"
        oil_shock_pct = st.slider("Oil price change (%)", -50, 50, 15, step=5,
                                  help="Positive = price increase, Negative = price decrease")
        shock_date = st.date_input("Shock start date", value=min_date,
                                   min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.markdown(f"**Oil Lag:** {LAG_OIL} days")
    st.markdown(f"**FX Lag:** {LAG_FX} days")

# ═══════════════════════════════════════════════════════════════════
# FILTER DATA
# ═══════════════════════════════════════════════════════════════════
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered = df[mask].copy()

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("# ✈️ Dynamic Pricing Dashboard")
st.markdown("*Analysis of Oil Price & Exchange Rate impact on Airfare — Route: Delhi → Cochin*")

# ═══════════════════════════════════════════════════════════════════
# KPI METRICS
# ═══════════════════════════════════════════════════════════════════
metrics = config['metrics'][model_choice]

avg_price = filtered['Median_Price'].mean()
oil_prices = filtered['Brent_Oil_Price'].dropna()
if len(oil_prices) >= 2:
    oil_change = ((oil_prices.iloc[-1] - oil_prices.iloc[0]) / oil_prices.iloc[0]) * 100
else:
    oil_change = 0.0
rmse_val = metrics['RMSE']

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg. Ticket Price</div>
        <div class="kpi-value">₹{avg_price:,.0f}</div>
        <div class="kpi-delta">Median Price (INR)</div>
    </div>""", unsafe_allow_html=True)

with col2:
    delta_class = "up" if oil_change > 0 else "down"
    arrow = "▲" if oil_change > 0 else "▼"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Oil Price Change</div>
        <div class="kpi-value">{abs(oil_change):.1f}%</div>
        <div class="kpi-delta {delta_class}">{arrow} In selected range</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model Error ({model_choice})</div>
        <div class="kpi-value">₹{rmse_val:,.0f}</div>
        <div class="kpi-delta">RMSE | MAE: ₹{metrics['MAE']:,.0f}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN CHART — Dual Y-Axis
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📊 Ticket Price & Oil Price Chart</div>', unsafe_allow_html=True)

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Actual price
fig.add_trace(go.Scatter(
    x=filtered['Date'], y=filtered['Median_Price'],
    name='Actual Price', mode='lines+markers',
    line=dict(color='#6366f1', width=2.5), marker=dict(size=5),
    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Ticket Price: ₹%{y:,.0f}<extra></extra>'
), secondary_y=False)

# Predicted price
pred_col = 'VAR_Predicted' if model_choice == 'VAR' else 'XGBoost_Predicted'
pred_data = filtered.dropna(subset=[pred_col])
if len(pred_data) > 0:
    fig.add_trace(go.Scatter(
        x=pred_data['Date'], y=pred_data[pred_col],
        name=f'{model_choice} Forecast', mode='lines+markers',
        line=dict(color='#f59e0b', width=2, dash='dash'), marker=dict(size=5, symbol='diamond'),
        hovertemplate=f'<b>%{{x|%d/%m/%Y}}</b><br>Forecast: ₹%{{y:,.0f}}<extra></extra>'
    ), secondary_y=False)

# Oil price (right axis)
fig.add_trace(go.Scatter(
    x=filtered['Date'], y=filtered['Brent_Oil_Price'],
    name='Brent Oil Price (USD)', mode='lines',
    line=dict(color='#ef4444', width=1.5, dash='dot'), opacity=0.7,
    hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Brent: $%{y:.2f}<extra></extra>'
), secondary_y=True)

# ── WHAT-IF SIMULATION ──
if whatif_enabled and oil_shock_pct != 0:
    sim_df = df[['Date', 'Median_Price', 'Brent_Oil_Price', 'USD_INR_Exchange']].copy()
    sim_df = sim_df.sort_values('Date').reset_index(drop=True)

    shock_mask = sim_df['Date'].dt.date >= shock_date
    sim_df.loc[shock_mask, 'Brent_Oil_Price'] *= (1 + oil_shock_pct / 100)

    sim_df[f'Brent_lag_{LAG_OIL}']  = sim_df['Brent_Oil_Price'].shift(LAG_OIL)
    sim_df[f'USD_INR_lag_{LAG_FX}'] = sim_df['USD_INR_Exchange'].shift(LAG_FX)
    sim_df['day_of_week']  = sim_df['Date'].dt.dayofweek
    sim_df['month']        = sim_df['Date'].dt.month
    sim_df['day_of_month'] = sim_df['Date'].dt.day
    sim_df['is_weekend']   = (sim_df['day_of_week'] >= 5).astype(int)

    sim_clean = sim_df.dropna(subset=FEATURE_COLS).copy()
    if len(sim_clean) > 0:
        sim_pred = xgb_model.predict(sim_clean[FEATURE_COLS])
        sim_clean['Simulated_Price'] = sim_pred

        sim_filtered = sim_clean[(sim_clean['Date'].dt.date >= start_date) &
                                 (sim_clean['Date'].dt.date <= end_date)]

        fig.add_trace(go.Scatter(
            x=sim_filtered['Date'], y=sim_filtered['Simulated_Price'],
            name=f'Simulation (Oil {oil_shock_pct:+d}%)', mode='lines+markers',
            line=dict(color='#f43f5e', width=2.5), marker=dict(size=5, symbol='star'),
            hovertemplate='<b>%{x|%d/%m/%Y}</b><br>Simulated: ₹%{y:,.0f}<extra></extra>'
        ), secondary_y=False)

        # Shock start line
        shock_ts_ms = int(pd.Timestamp(shock_date).timestamp() * 1000)
        fig.add_vline(x=shock_ts_ms, line_dash="dash",
                      line_color="#f43f5e", opacity=0.6,
                      annotation_text=f"Shock {int(oil_shock_pct):+d}%",
                      annotation_font_color="#f43f5e")

        # Lag effect annotation
        lag_effect_date = pd.Timestamp(shock_date) + pd.Timedelta(days=LAG_OIL)
        if lag_effect_date <= pd.Timestamp(end_date):
            lag_ts_ms = int(lag_effect_date.timestamp() * 1000)
            fig.add_vline(x=lag_ts_ms, line_dash="dot",
                          line_color="#34d399", opacity=0.6,
                          annotation_text=f"Lag Effect ({LAG_OIL} days)",
                          annotation_font_color="#34d399")

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    height=500, margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5,
                font=dict(size=11)),
    hovermode='x unified',
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
)
fig.update_yaxes(title_text="Ticket Price (INR)", secondary_y=False, color='#a5b4fc')
fig.update_yaxes(title_text="Brent Oil Price (USD)", secondary_y=True, color='#fca5a5')

st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE + IRF CHART (2 columns)
# ═══════════════════════════════════════════════════════════════════
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-title">🎯 Feature Importance (XGBoost)</div>', unsafe_allow_html=True)
    fi = config['feature_importance']
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))
    fig_fi = go.Figure(go.Bar(
        x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
        orientation='h', text=[f'{v*100:.1f}%' for v in fi_sorted.values()],
        textposition='outside', textfont=dict(color='#e0e7ff', size=12),
        marker=dict(color=list(fi_sorted.values()),
                    colorscale=[[0,'#312e81'],[0.5,'#6366f1'],[1,'#a78bfa']]),
    ))
    fig_fi.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=20, r=60, t=20, b=20),
        xaxis=dict(title='Importance Score', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

with col_right:
    st.markdown('<div class="section-title">📈 IRF — Impulse Response</div>', unsafe_allow_html=True)
    fig_irf = go.Figure()
    fig_irf.add_trace(go.Scatter(
        x=irf_data['horizon'], y=irf_data['cumulative_oil'],
        name='Oil → Ticket Price', mode='lines+markers',
        line=dict(color='#6366f1', width=2), marker=dict(size=5),
    ))
    fig_irf.add_trace(go.Scatter(
        x=irf_data['horizon'], y=irf_data['cumulative_fx'],
        name='FX Rate → Ticket Price', mode='lines+markers',
        line=dict(color='#f59e0b', width=2), marker=dict(size=5),
    ))
    fig_irf.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')
    fig_irf.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title='Days after shock', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Cumulative Δ Ticket Price', gridcolor='rgba(255,255,255,0.05)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
    )
    st.plotly_chart(fig_irf, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">⚖️ Model Comparison</div>', unsafe_allow_html=True)

comp_df = pd.DataFrame({
    'Metric':  ['RMSE (INR)', 'MAE (INR)', 'MAPE (%)'],
    'VAR':     [config['metrics']['VAR']['RMSE'], config['metrics']['VAR']['MAE'], config['metrics']['VAR']['MAPE']],
    'XGBoost': [config['metrics']['XGBoost']['RMSE'], config['metrics']['XGBoost']['MAE'], config['metrics']['XGBoost']['MAPE']],
})
comp_df['Better'] = comp_df.apply(
    lambda r: '✅ XGBoost' if r['XGBoost'] < r['VAR'] else '✅ VAR', axis=1)

st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# BUSINESS STRATEGY
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">💼 Business Strategy Recommendations</div>', unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown(f"""
    <div class="strategy-card">
        <h4>🎯 Dynamic Pricing Strategy</h4>
        <p>
        By identifying a <b>{LAG_OIL}-day lag</b> between oil price movements
        and ticket prices, the Sales team has a <b>golden time window of
        {LAG_OIL * 24} hours</b> to decide on new fare levels.
        <br><br>
        There is no need to raise prices immediately and shock customers.
        Instead, the airline can <b>gradually adjust fares</b> within this
        window to maximize revenue without losing passengers.
        </p>
    </div>""", unsafe_allow_html=True)

with col_s2:
    st.markdown(f"""
    <div class="strategy-card">
        <h4>🛡️ Hedging Strategy</h4>
        <p>
        The dependence on Brent oil price volatility indicates that the airline
        should increase its use of <b>Futures Contracts</b>
        to lock in fuel prices for at least the next month.
        <br><br>
        Feature Importance shows that the <b>{LAG_OIL}-day lagged oil price</b>
        contributes significantly to fare decisions. Hedging oil price risk
        will help <b>stabilize operating costs</b> and maintain profit margins.
        </p>
    </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown(
    "<center style='color:rgba(255,255,255,0.3);font-size:0.8rem;'>"
    "Dynamic Pricing Dashboard • Delhi → Cochin • Data: Mar–Jun 2019"
    "</center>", unsafe_allow_html=True
)
