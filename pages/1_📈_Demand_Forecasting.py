"""
================================================================================
Haldiram's International Export Analytics Pipeline - Demand Forecasting Page
================================================================================
Author: Aanya
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (Analytics / Activity UI Reference)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.forecasting import DemandForecastingEngine
from src.utils import (
    apply_notion_crm_style,
    format_currency_inr,
    format_volume_kg,
    get_plotly_layout_dict,
    render_venture_topbar,
    render_metric_card,
    render_graph_note,
    render_section_header,
    VENTURE_THEME
)

st.set_page_config(page_title="Demand Forecasting | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

render_venture_topbar(
    title="Demand Forecasting",
    subtitle="Hierarchical FMCG Export Demand Modeling (SARIMAX vs. Prophet vs. XGBoost)"
)

def get_file_mtime(filepath: str) -> float:
    p = Path(filepath)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_data(_mtime: float):
    df = pd.read_parquet("data/haldirams_sales_consolidated.parquet")
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    return df

data_file = "data/haldirams_sales_consolidated.parquet"
df = load_data(get_file_mtime(data_file))

# Print Namkeens max invoice_date from inside the page code
namkeen_rows = df[df['category_name'] == 'Namkeens & Savouries']
namkeen_max_date = namkeen_rows['invoice_date'].max()
print(f"[PAGE FORECASTING] Namkeens max invoice_date: {namkeen_max_date.strftime('%B %Y')} ({namkeen_max_date.strftime('%Y-%m-%d')})", flush=True)

# -----------------------------------------------------------------------------
# FILTER CONTROLS (Venture style bar)
# -----------------------------------------------------------------------------
render_section_header("sliders-horizontal", "Time-Series Aggregation & Forecast Parameters")

fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)

with fc_c1:
    available_zones = ['All Zones'] + sorted(df['zone'].unique().tolist())
    sel_zone = st.selectbox("Export Zone", options=available_zones, index=0)

with fc_c2:
    available_countries = ['All Countries'] + sorted(df['country'].unique().tolist())
    sel_country = st.selectbox("Country Filter", options=available_countries, index=0)

with fc_c3:
    available_cats = ['All Categories'] + sorted(df['category_name'].dropna().unique().tolist())
    default_cat_idx = available_cats.index('Namkeens & Savouries') if 'Namkeens & Savouries' in available_cats else 0
    sel_cat = st.selectbox("Product Hierarchy Category", options=available_cats, index=default_cat_idx)

with fc_c4:
    horizon = st.slider("Forecast Horizon (Months Ahead)", min_value=3, max_value=12, value=6, step=1)

# One-line note under forecast filters indicating the end date for the selected slice
ts_filter_data = df.copy()
if sel_zone != 'All Zones':
    ts_filter_data = ts_filter_data[ts_filter_data['zone'] == sel_zone]
if sel_country != 'All Countries':
    ts_filter_data = ts_filter_data[ts_filter_data['country'] == sel_country]
if sel_cat != 'All Categories':
    ts_filter_data = ts_filter_data[ts_filter_data['category_name'] == sel_cat]

if not ts_filter_data.empty:
    filter_max_date = ts_filter_data['invoice_date'].max()
    filter_end_str = filter_max_date.strftime('%B %Y')
    st.caption(f"Data for this selection ends {filter_end_str}")

# -----------------------------------------------------------------------------
# RUN TIME SERIES ENGINE
# -----------------------------------------------------------------------------
with st.spinner("Training SARIMAX, Prophet, and XGBoost with 9-month time-based holdout evaluation..."):
    engine = DemandForecastingEngine(df)
    try:
        ts_data, forecasts_df, scorecard = engine.benchmark_all_models(
            zone=sel_zone,
            country=sel_country,
            category=sel_cat,
            forecast_horizon=horizon
        )
        has_forecast = True
        print(f"[PAGE FORECASTING] Final Scorecard:\n{scorecard[['Model', 'MAE', 'RMSE', 'MAPE (%)']].to_string(index=False)}", flush=True)
    except Exception as e:
        st.error(f"Error computing forecast: {e}")
        has_forecast = False

if has_forecast:
    best_model_row = scorecard.sort_values('MAPE (%)').iloc[0]
    best_model_name = best_model_row['Model']
    best_mape = best_model_row['MAPE (%)']
    best_rmse = best_model_row['RMSE']

    # -------------------------------------------------------------------------
    # 4 METRIC CARDS
    # -------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            icon_name="award",
            label="Optimal Model",
            value=best_model_name.split()[0],
            subtext=f"Lowest Error ({best_mape:.1f}% MAPE)",
            badge_text="Best Fit",
            badge_type="green"
        )
    with k2:
        render_metric_card(
            icon_name="chart-bar",
            label="Benchmark MAPE",
            value=f"{best_mape:.1f}%",
            subtext="FMCG Accuracy Metric",
            badge_text="Holdout Set",
            badge_type="blue"
        )
    with k3:
        render_metric_card(
            icon_name="target",
            label="RMSE Metric",
            value=f"{best_rmse:,.0f} kg",
            subtext="Variance Penalty Metric",
            badge_text="9M Holdout",
            badge_type="purple"
        )
    with k4:
        render_metric_card(
            icon_name="calendar",
            label="Horizon Output",
            value=f"+{horizon} Months",
            subtext="Future Production Plan",
            badge_text="Active MPS",
            badge_type="amber"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BENCHMARK SCORECARD
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Model Benchmark Scorecard (9-Month Time-Based Holdout)</span>
            <span class="venture-pill pill-blue">9-Month Holdout Split</span>
        </div>
        <div class="content-card-subtitle">Comparative performance metrics across classical time-series, Bayesian additive decomposition, and gradient boosting</div>
    </div>
    """, unsafe_allow_html=True)

    formatted_scorecard = scorecard.style.format({
        'MAE': '{:,.0f}',
        'RMSE': '{:,.0f}',
        'MAPE (%)': '{:.1f}%'
    }).highlight_min(subset=['RMSE', 'MAE', 'MAPE (%)'], color='#DCFCE7')
    st.dataframe(
        formatted_scorecard,
        use_container_width=True,
        hide_index=True
    )

    render_graph_note(
        title="9-Month Time-Based Holdout Scorecard",
        explanation="Models are trained strictly on past history and evaluated on the subsequent 9 holdout months. MAPE (%) measures percentage forecasting accuracy relative to actual orders, while RMSE penalizes large forecast variances."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE TIME SERIES VISUALIZATION
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Ground Truth vs. Out-of-Sample Predictions & Future Horizon</span>
        </div>
        <div class="content-card-subtitle">Historical export dispatches (solid black), 9-month time-based holdout validation (dashed), and future forecasts (solid colored lines)</div>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        show_sarimax = st.checkbox("Show SARIMAX (Statistical)", value=True)
    with col_t2:
        show_prophet = st.checkbox("Show Prophet (Additive Bayesian)", value=True)
    with col_t3:
        show_xgboost = st.checkbox("Show XGBoost (Autoregressive Lags)", value=True)

    fig_fc = go.Figure()

    # Ground Truth
    fig_fc.add_trace(go.Scatter(
        x=ts_data['ds'],
        y=ts_data['y'],
        mode='lines+markers',
        name='Historical Realized Demand (Kg)',
        line=dict(color='#000000', width=2.5),
        marker=dict(size=4)
    ))

    model_colors = {
        'SARIMAX': '#2563EB',
        'Prophet': '#7C3AED',
        'XGBoost (Lagged)': '#D97706'
    }

    for model_name, is_shown in [('SARIMAX', show_sarimax), ('Prophet', show_prophet), ('XGBoost (Lagged)', show_xgboost)]:
        if is_shown:
            sub = forecasts_df[forecasts_df['model'] == model_name]
            if not sub.empty:
                eval_p = sub[sub['type'] == 'Test_Evaluation']
                if not eval_p.empty:
                    fig_fc.add_trace(go.Scatter(
                        x=eval_p['ds'],
                        y=eval_p['y_pred'],
                        mode='lines',
                        name=f'{model_name} (Holdout Test)',
                        line=dict(color=model_colors.get(model_name, '#2563EB'), width=2, dash='dash')
                    ))
                fut_p = sub[sub['type'] == 'Future_Forecast']
                if not fut_p.empty:
                    connect_df = pd.concat([eval_p.tail(1), fut_p]) if not eval_p.empty else fut_p
                    fig_fc.add_trace(go.Scatter(
                        x=connect_df['ds'],
                        y=connect_df['y_pred'],
                        mode='lines+markers',
                        name=f'{model_name} ({horizon}M Future Horizon)',
                        line=dict(color=model_colors.get(model_name, '#2563EB'), width=3),
                        marker=dict(size=5, symbol='diamond')
                    ))

    # Add festival dispatch highlight bands (only up to last forecast month)
    max_forecast_date = forecasts_df['ds'].max()
    for diwali_date in ['2021-09-01', '2022-09-01', '2023-09-01', '2024-09-01', '2025-09-01', '2026-09-01']:
        if pd.to_datetime(diwali_date) <= max_forecast_date:
            fig_fc.add_vrect(
                x0=pd.to_datetime(diwali_date),
                x1=pd.to_datetime(diwali_date) + pd.DateOffset(months=2),
                fillcolor="#FEF3C7",
                opacity=0.35,
                layer="below",
                line_width=0,
                annotation_text="Diwali Surge",
                annotation_position="top left",
                annotation_font_size=9
            )

    fig_layout = get_plotly_layout_dict()
    fig_layout.update(
        height=450,
        yaxis=dict(title='Export Volume (Kg)', showgrid=True, gridcolor='#F3F4F6'),
        xaxis=dict(title='Timeline', showgrid=True, gridcolor='#F3F4F6'),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.85)',
            font=dict(size=10)
        )
    )
    fig_fc.update_layout(**fig_layout)
    st.plotly_chart(fig_fc, use_container_width=True)

    render_graph_note(
        title="Multi-Model Forecast Comparison & Festival Surge Banding",
        explanation="The chart plots historical monthly export volumes alongside model predictions. Yellow highlighted bands indicate pre-festive export shipment surge windows for Diwali (Sep-Oct)."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # NUMERICAL FUTURE DEMAND PLAN
    # -------------------------------------------------------------------------
    fut_slice = forecasts_df[forecasts_df['type'] == 'Future_Forecast']
    future_start_str = fut_slice['ds'].min().strftime('%b %Y') if not fut_slice.empty else "Next Month"

    st.markdown(f"""
    <div class="content-card">
        <div class="content-card-title">
            <span>Projected Monthly Demand Plan (Production Master Schedule)</span>
        </div>
        <div class="content-card-subtitle">Detailed monthly kilogram allocations for manufacturing plant replenishment ({future_start_str} onwards)</div>
    </div>
    """, unsafe_allow_html=True)

    future_table = forecasts_df[forecasts_df['type'] == 'Future_Forecast'].pivot(
        index='ds', columns='model', values='y_pred'
    ).reset_index()
    future_table['Month'] = future_table['ds'].dt.strftime('%B %Y')
    
    display_cols = ['Month'] + [c for c in future_table.columns if c not in ['ds', 'Month']]
    st.dataframe(
        future_table[display_cols].style.format({c: "{:,.0f} kg" for c in display_cols if c != 'Month'}),
        use_container_width=True,
        hide_index=True
    )
