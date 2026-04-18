"""
at_risk.py — At-Risk Analysis Page: Airline Dynamic Pricing
════════════════════════════════════════════════════════════
Run standalone:  streamlit run dashboard/at_risk.py
Or import and call render() from app.py.
"""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── CONFIG ─────────────────────────────────────────────────────────
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

_OIL_RAW = 0.1243
_FX_RAW  = 0.1991
_TOTAL   = _OIL_RAW + _FX_RAW
W_OIL = _OIL_RAW / _TOTAL
W_FX  = _FX_RAW  / _TOTAL

THRESHOLDS = {"Low": 2.0, "Medium": 4.0, "High": 6.0}

RISK_COLORS = {
    "Low":      "#34d399",
    "Medium":   "#f59e0b",
    "High":     "#f97316",
    "Critical": "#ef4444",
}

RISK_BG = {
    "Low":      "rgba(52,211,153,0.08)",
    "Medium":   "rgba(245,158,11,0.08)",
    "High":     "rgba(249,115,22,0.08)",
    "Critical": "rgba(239,68,68,0.10)",
}

ROLLING_WINDOW = 14

# ─── DATA ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df  = pd.read_csv(os.path.join(ARTIFACT_DIR, "full_data.csv"), parse_dates=["Date"])
    irf = pd.read_csv(os.path.join(ARTIFACT_DIR, "irf_data.csv"))
    with open(os.path.join(ARTIFACT_DIR, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return df.sort_values("Date").reset_index(drop=True), irf, cfg


# ─── RISK ENGINE ─────────────────────────────────────────────────────
def compute_risk(df: pd.DataFrame, lag_oil: int, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    d = df.copy()

    oil_mean = d["Brent_Oil_Price"].rolling(window, min_periods=3).mean()
    oil_std  = d["Brent_Oil_Price"].rolling(window, min_periods=3).std()
    fx_mean  = d["USD_INR_Exchange"].rolling(window, min_periods=3).mean()
    fx_std   = d["USD_INR_Exchange"].rolling(window, min_periods=3).std()

    d["oil_vol"] = (oil_std / oil_mean.replace(0, np.nan)) * 100
    d["fx_vol"]  = (fx_std  / fx_mean.replace(0, np.nan))  * 100
    d["risk_score"] = W_OIL * d["oil_vol"].fillna(0) + W_FX * d["fx_vol"].fillna(0)

    def classify(s):
        if s < THRESHOLDS["Low"]:    return "Low"
        if s < THRESHOLDS["Medium"]: return "Medium"
        if s < THRESHOLDS["High"]:   return "High"
        return "Critical"

    d["risk_level"] = d["risk_score"].apply(classify)

    lag_col = f"Brent_lag_{lag_oil}"
    if lag_col not in d.columns:
        d[lag_col] = d["Brent_Oil_Price"].shift(lag_oil)

    oil_pct_dev = (d["Brent_Oil_Price"] - oil_mean) / oil_mean.replace(0, np.nan) * 100
    avg_price   = d["Median_Price"].mean()
    d["price_impact_est"] = oil_pct_dev.fillna(0) * avg_price * 0.0025 * W_OIL

    return d


def get_risk_summary(d: pd.DataFrame) -> dict:
    counts = d["risk_level"].value_counts().to_dict()
    return {lvl: counts.get(lvl, 0) for lvl in ["Low", "Medium", "High", "Critical"]}


# ─── SHARED CHART LAYOUT ─────────────────────────────────────────────
_CHART_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,12,41,0.6)",
    font=dict(family="Inter, sans-serif", color="#c7d2fe"),
)


def _section(title: str, accent: str = "#6366f1") -> None:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin:32px 0 16px;">'
        f'<div style="width:4px;height:22px;background:{accent};border-radius:2px;flex-shrink:0;"></div>'
        f'<span style="font-size:1.05rem;font-weight:700;color:#e0e7ff;letter-spacing:0.3px;">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── RENDER ──────────────────────────────────────────────────────────
def render():
    df, irf_data, config = load_data()
    LAG_OIL = config["LAG_OIL"]

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<p style="color:#a5b4fc;font-size:0.78rem;text-transform:uppercase;'
            'letter-spacing:1.2px;font-weight:600;margin-bottom:8px;">AT-RISK SETTINGS</p>',
            unsafe_allow_html=True,
        )
        vol_window = st.slider(
            "Volatility window (days)", 7, 30, ROLLING_WINDOW, step=1,
            help="Rolling window used to compute Coefficient of Variation for oil & FX"
        )
        show_only_risk = st.checkbox("Show Medium+ risk days only", value=False)

        st.markdown("---")
        st.markdown(
            f'<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);'
            f'border-radius:8px;padding:12px 14px;">'
            f'<p style="color:#a5b4fc;font-size:0.75rem;margin:0 0 6px;text-transform:uppercase;letter-spacing:1px;">Risk Thresholds</p>'
            f'<p style="color:#e0e7ff;font-size:0.82rem;margin:2px 0;">&#x1F7E2; Low &nbsp;&nbsp;&lt; {THRESHOLDS["Low"]}%</p>'
            f'<p style="color:#e0e7ff;font-size:0.82rem;margin:2px 0;">&#x1F7E1; Medium &lt; {THRESHOLDS["Medium"]}%</p>'
            f'<p style="color:#e0e7ff;font-size:0.82rem;margin:2px 0;">&#x1F7E0; High &nbsp;&nbsp;&lt; {THRESHOLDS["High"]}%</p>'
            f'<p style="color:#e0e7ff;font-size:0.82rem;margin:2px 0;">&#x1F534; Critical &ge; {THRESHOLDS["High"]}%</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f'<div style="color:rgba(255,255,255,0.35);font-size:0.72rem;line-height:1.6;">'
            f'Weight Oil: <b style="color:#a5b4fc">{W_OIL:.2f}</b><br>'
            f'Weight FX: <b style="color:#a5b4fc">{W_FX:.2f}</b><br>'
            f'Oil lag: <b style="color:#a5b4fc">{LAG_OIL}d</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Compute risk ─────────────────────────────────────────────────
    df = compute_risk(df, LAG_OIL, window=vol_window)
    summary = get_risk_summary(df)
    total   = len(df)

    # ── Page header ──────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:4px;">'
        '<span style="font-size:2rem;font-weight:800;color:#e0e7ff;">&#x26A0;&#xFE0F; At-Risk Analysis</span>'
        '</div>'
        '<p style="color:rgba(255,255,255,0.45);font-size:0.9rem;margin-top:2px;margin-bottom:28px;">'
        'Daily price volatility risk monitor &nbsp;&middot;&nbsp; Route: Delhi &rarr; Cochin &nbsp;&middot;&nbsp; '
        'Composite score weighted by XGBoost feature importance'
        '</p>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════════
    # KPI CARDS
    # ════════════════════════════════════════════════════════════════
    c1, c2, c3, c4 = st.columns(4)

    RISK_ICONS = {"Low": "&#x1F7E2;", "Medium": "&#x1F7E1;", "High": "&#x1F7E0;", "Critical": "&#x1F534;"}
    RISK_DESC  = {
        "Low":      "Stable market conditions",
        "Medium":   "Moderate volatility",
        "High":     "Elevated risk — monitor closely",
        "Critical": "Extreme volatility — action needed",
    }

    for col, lvl in zip([c1, c2, c3, c4], ["Low", "Medium", "High", "Critical"]):
        cnt   = summary[lvl]
        pct   = cnt / total * 100 if total else 0
        color = RISK_COLORS[lvl]
        bg    = RISK_BG[lvl]
        col.markdown(
            f'<div style="background:{bg};border:1px solid {color}33;'
            f'border-top:3px solid {color};border-radius:12px;padding:20px 16px;text-align:center;">'
            f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.45);text-transform:uppercase;'
            f'letter-spacing:1.2px;font-weight:600;">{RISK_ICONS[lvl]} {lvl} RISK</div>'
            f'<div style="font-size:2.2rem;font-weight:800;color:{color};margin:8px 0 4px;">{cnt}</div>'
            f'<div style="font-size:0.82rem;color:rgba(255,255,255,0.55);">{pct:.1f}% of all days</div>'
            f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.3);margin-top:6px;">{RISK_DESC[lvl]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ════════════════════════════════════════════════════════════════
    # RISK SCORE TIMELINE
    # ════════════════════════════════════════════════════════════════
    _section("&#x1F4C8; Risk Score Timeline")

    plot_df = df.dropna(subset=["risk_score"])
    if show_only_risk:
        plot_df = plot_df[plot_df["risk_level"].isin(["Medium", "High", "Critical"])]

    fig_tl = go.Figure()

    for level, color in RISK_COLORS.items():
        sub = plot_df[plot_df["risk_level"] == level]
        if sub.empty:
            continue
        fig_tl.add_trace(go.Scatter(
            x=sub["Date"], y=sub["risk_score"],
            mode="markers", name=level,
            marker=dict(color=color, size=8, opacity=0.9,
                        line=dict(width=1, color="rgba(255,255,255,0.15)")),
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                f"Level: <b>{level}</b><br>"
                "Score: <b>%{y:.2f}%</b><extra></extra>"
            ),
        ))

    fig_tl.add_trace(go.Scatter(
        x=plot_df["Date"], y=plot_df["Median_Price"],
        name="Actual Fare (INR)", mode="lines",
        line=dict(color="#818cf8", width=1.5, dash="dot"), opacity=0.4,
        yaxis="y2",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Fare: Rs.%{y:,.0f}<extra></extra>",
    ))

    for label, threshold in THRESHOLDS.items():
        fig_tl.add_hline(
            y=threshold, line_dash="dash", line_width=1,
            line_color=RISK_COLORS[label], opacity=0.6,
            annotation_text=f"  {label}", annotation_position="right",
            annotation_font=dict(color=RISK_COLORS[label], size=11),
        )

    fig_tl.update_layout(
        **_CHART_BASE,
        height=440,
        margin=dict(l=10, r=80, t=36, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True),
        yaxis=dict(title="Risk Score (%)", gridcolor="rgba(255,255,255,0.04)",
                   color="#a5b4fc"),
        yaxis2=dict(title="Fare (INR)", overlaying="y", side="right",
                    color="#818cf8", showgrid=False),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # OIL VOL vs FX VOL  +  RISK SCORE DISTRIBUTION
    # ════════════════════════════════════════════════════════════════
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        _section("&#x1F6E2;&#xFE0F; Oil Volatility vs FX Volatility")
        scatter_df = df.dropna(subset=["oil_vol", "fx_vol"])
        fig_sc = go.Figure()
        for level, color in RISK_COLORS.items():
            sub = scatter_df[scatter_df["risk_level"] == level]
            if sub.empty:
                continue
            fig_sc.add_trace(go.Scatter(
                x=sub["oil_vol"], y=sub["fx_vol"],
                mode="markers", name=level,
                marker=dict(color=color, size=9, opacity=0.80,
                            line=dict(width=0.8, color="rgba(255,255,255,0.2)")),
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Oil Vol: %{x:.2f}%<br>"
                    "FX Vol: %{y:.2f}%<extra></extra>"
                ),
                customdata=sub["Date"].dt.strftime("%b %d, %Y"),
            ))
        fig_sc.update_layout(
            **_CHART_BASE,
            height=360,
            xaxis=dict(title="Oil Volatility (%)", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="FX Volatility (%)",  gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center", font=dict(size=11)),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        _section("&#x1F4CA; Risk Score Distribution")
        score_vals = df["risk_score"].dropna()
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=score_vals, nbinsx=28,
            marker=dict(
                color=score_vals,
                colorscale=[
                    [0.00, RISK_COLORS["Low"]],
                    [0.33, RISK_COLORS["Medium"]],
                    [0.66, RISK_COLORS["High"]],
                    [1.00, RISK_COLORS["Critical"]],
                ],
                showscale=False,
                line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
            ),
            hovertemplate="Score: %{x:.1f}%<br>Days: %{y}<extra></extra>",
        ))
        for label, threshold in THRESHOLDS.items():
            fig_hist.add_vline(
                x=threshold, line_dash="dash", line_width=1.2,
                line_color=RISK_COLORS[label], opacity=0.8,
                annotation_text=f" {label}",
                annotation_font=dict(color=RISK_COLORS[label], size=11),
            )
        fig_hist.update_layout(
            **_CHART_BASE,
            height=360,
            xaxis=dict(title="Risk Score (%)", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Number of Days",  gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # RISK HEATMAP
    # ════════════════════════════════════════════════════════════════
    _section("&#x1F321;&#xFE0F; Monthly Risk Heatmap")

    hm = df.dropna(subset=["risk_score"]).copy()
    hm["month"] = hm["Date"].dt.to_period("M").astype(str)
    hm["day"]   = hm["Date"].dt.day

    pivot = hm.pivot_table(index="month", columns="day",
                           values="risk_score", aggfunc="mean").fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, RISK_COLORS["Low"]],
            [0.25, RISK_COLORS["Medium"]],
            [0.55, RISK_COLORS["High"]],
            [1.00, RISK_COLORS["Critical"]],
        ],
        colorbar=dict(
            title=dict(text="Risk Score (%)", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=12,
        ),
        hovertemplate="Month: %{y}<br>Day: %{x}<br>Score: %{z:.2f}%<extra></extra>",
        zmin=0,
        zmax=max(df["risk_score"].quantile(0.99), THRESHOLDS["High"] + 1),
    ))
    fig_heat.update_layout(
        **_CHART_BASE,
        height=260,
        margin=dict(l=10, r=100, t=20, b=10),
        xaxis=dict(title="Day of Month", dtick=5, gridcolor="rgba(255,255,255,0.03)"),
        yaxis=dict(title="Month"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # AT-RISK DATES TABLE
    # ════════════════════════════════════════════════════════════════
    _section("&#x1F4CB; At-Risk Dates — Medium and Above")

    risk_table = (
        df[df["risk_level"].isin(["Medium", "High", "Critical"])]
        .copy()
        .sort_values("risk_score", ascending=False)
    )

    if risk_table.empty:
        st.info("No days with Medium or above risk in the current dataset.")
    else:
        display_cols = {
            "Date":             "Date",
            "Median_Price":     "Fare (INR)",
            "Brent_Oil_Price":  "Brent (USD)",
            "USD_INR_Exchange": "USD/INR",
            "oil_vol":          "Oil Vol (%)",
            "fx_vol":           "FX Vol (%)",
            "risk_score":       "Risk Score (%)",
            "risk_level":       "Risk Level",
        }
        tbl = risk_table[list(display_cols.keys())].rename(columns=display_cols).head(50)
        tbl["Date"] = tbl["Date"].dt.strftime("%b %d, %Y")

        def _color_risk(val):
            return f"color: {RISK_COLORS.get(val, '#fff')}; font-weight: 700;"

        styled = (
            tbl.style
            .map(_color_risk, subset=["Risk Level"])
            .format({
                "Fare (INR)":     "{:,.0f}",
                "Brent (USD)":    "{:.2f}",
                "USD/INR":        "{:.4f}",
                "Oil Vol (%)":    "{:.2f}",
                "FX Vol (%)":     "{:.2f}",
                "Risk Score (%)": "{:.2f}",
            })
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # TOP 5 ALERT CARDS
    # ════════════════════════════════════════════════════════════════
    alerts = (
        df[df["risk_level"].isin(["High", "Critical"])]
        .sort_values("risk_score", ascending=False)
        .head(5)
    )

    if not alerts.empty:
        _section("&#x1F6A8; Top 5 Highest-Risk Days", accent="#ef4444")

        for _, row in alerts.iterrows():
            lvl        = row["risk_level"]
            color      = RISK_COLORS[lvl]
            bg         = RISK_BG[lvl]
            price_chg  = row.get("price_impact_est", 0)
            arrow      = "▲" if price_chg > 0 else "▼"
            badge_bg   = "rgba(239,68,68,0.15)" if lvl == "Critical" else "rgba(249,115,22,0.15)"

            st.markdown(
                f'<div style="background:{bg};border:1px solid {color}30;'
                f'border-left:4px solid {color};border-radius:12px;'
                f'padding:18px 22px;margin:10px 0;">'

                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'flex-wrap:wrap;gap:8px;margin-bottom:14px;">'
                f'  <span style="font-size:1rem;font-weight:700;color:{color};">'
                f'    {row["Date"].strftime("%B %d, %Y")}'
                f'  </span>'
                f'  <span style="background:{badge_bg};border:1px solid {color}55;'
                f'  border-radius:20px;padding:3px 14px;font-size:0.78rem;'
                f'  font-weight:700;color:{color};">{lvl.upper()}</span>'
                f'</div>'

                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">'

                f'  <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 12px;">'
                f'    <div style="font-size:0.65rem;color:rgba(255,255,255,0.38);'
                f'    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Risk Score</div>'
                f'    <div style="font-size:1.2rem;font-weight:700;color:{color};">'
                f'    {row["risk_score"]:.2f}%</div>'
                f'  </div>'

                f'  <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 12px;">'
                f'    <div style="font-size:0.65rem;color:rgba(255,255,255,0.38);'
                f'    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Actual Fare</div>'
                f'    <div style="font-size:1.2rem;font-weight:700;color:#e0e7ff;">'
                f'    Rs.{row["Median_Price"]:,.0f}</div>'
                f'  </div>'

                f'  <div style="background:rgba(0,0,0,0.25);border-radius:8px;padding:10px 12px;">'
                f'    <div style="font-size:0.65rem;color:rgba(255,255,255,0.38);'
                f'    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">Est. Price Impact</div>'
                f'    <div style="font-size:1.2rem;font-weight:700;color:{color};">'
                f'    {arrow} Rs.{abs(price_chg):,.0f}</div>'
                f'  </div>'

                f'</div>'

                f'<div style="display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;">'
                f'  <span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">'
                f'    Brent: <b style="color:#fca5a5;">${row["Brent_Oil_Price"]:.2f}</b>'
                f'  </span>'
                f'  <span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">'
                f'    USD/INR: <b style="color:#fcd34d;">{row["USD_INR_Exchange"]:.4f}</b>'
                f'  </span>'
                f'  <span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">'
                f'    Oil Vol: <b style="color:{color};">{row["oil_vol"]:.2f}%</b>'
                f'  </span>'
                f'  <span style="font-size:0.8rem;color:rgba(255,255,255,0.45);">'
                f'    FX Vol: <b style="color:{color};">{row["fx_vol"]:.2f}%</b>'
                f'  </span>'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-top:48px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);'
        f'text-align:center;color:rgba(255,255,255,0.22);font-size:0.75rem;">'
        f'At-Risk Analysis &nbsp;&middot;&nbsp; Delhi &rarr; Cochin &nbsp;&middot;&nbsp; Data: Mar–Jun 2019'
        f' &nbsp;&middot;&nbsp; Window: {vol_window}d &nbsp;&middot;&nbsp;'
        f' Oil weight: {W_OIL:.2f} &nbsp;&middot;&nbsp; FX weight: {W_FX:.2f}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── STANDALONE ENTRY POINT ──────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.main {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #13132b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c7d2fe; }
.stSlider > label, .stCheckbox > label { color: #a5b4fc !important; font-size: 0.85rem !important; }
[data-testid="block-container"] { padding-top: 2rem; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 3px; }
h1, h2, h3 { color: #e0e7ff !important; }
</style>
"""

if __name__ == "__main__":
    st.set_page_config(
        page_title="At-Risk Analysis — Airline Pricing",
        page_icon="warning",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;padding:8px 0 4px;">'
            '<span style="font-size:1.4rem;">&#x2708;&#xFE0F;</span>'
            '<span style="font-size:1rem;font-weight:700;color:#e0e7ff;">Dashboard</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    render()