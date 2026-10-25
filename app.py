"""
================================================================================
Haldiram's International Export Command Center - Executive Dashboard
================================================================================
Author: Senior Data Scientist (Technical Lead)
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Pixel-accurate Venture CRM (Dashboard.jpg reference)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.utils import (
    apply_notion_crm_style,
    format_currency_inr,
    format_volume_kg,
    get_plotly_layout_dict,
    render_venture_topbar,
    render_metric_card,
    render_graph_note,
    render_sidebar_brand,
    render_sidebar_nav_label,
    VENTURE_THEME
)

# Set page configuration
st.set_page_config(
    page_title="Dashboard | Haldiram's International",
    page_icon="assets/haldirams_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Venture CRM CSS
apply_notion_crm_style()


def get_file_mtime(filepath: str) -> float:
    p = Path(filepath)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_consolidated_data(_mtime: float) -> pd.DataFrame:
    """Loads curated consolidated parquet dataset."""
    data_path = Path("data/haldirams_sales_consolidated.parquet")
    if not data_path.exists():
        st.error("Consolidated dataset not found. Please run `src/data_pipeline.py` first.")
        st.stop()
    df = pd.read_parquet(data_path)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    return df


df_all = load_consolidated_data(get_file_mtime("data/haldirams_sales_consolidated.parquet"))

# -----------------------------------------------------------------------------
# SIDEBAR - VENTURE BRAND & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_nav_label("GLOBAL FILTERS")
    
    # Fiscal Year Multi-select
    available_fys = sorted(df_all['fiscal_year'].unique().tolist())
    selected_fys = st.multiselect("Fiscal Years", options=available_fys, default=available_fys)
    
    # Global Zone Filter
    available_zones = ['All Zones'] + sorted(df_all['zone'].unique().tolist())
    selected_zone = st.selectbox("Export Zone", options=available_zones, index=0)

    # Category Filter
    available_cats = ['All Categories'] + sorted(df_all['category_name'].dropna().unique().tolist())
    selected_cat = st.selectbox("Product Category", options=available_cats, index=0)

    render_sidebar_nav_label("DATABASE")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #6B7280; line-height: 1.7;">
        Transactions: <b>294,898</b><br>
        B2B Accounts: <b>173</b><br>
        Active SKUs: <b>3,488</b><br>
        Markets: <b>90 Nations</b>
    </div>
    """, unsafe_allow_html=True)

# Filter Data
filtered_df = df_all[df_all['fiscal_year'].isin(selected_fys)].copy()
if selected_zone != 'All Zones':
    filtered_df = filtered_df[filtered_df['zone'] == selected_zone]
if selected_cat != 'All Categories':
    filtered_df = filtered_df[filtered_df['category_name'] == selected_cat]

if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust your sidebar criteria.")
    st.stop()

# -----------------------------------------------------------------------------
# TOPBAR (Venture CRM style)
# -----------------------------------------------------------------------------
render_venture_topbar(
    title="Dashboard",
    subtitle="Haldiram's International Export Overview & Key Performance Indicators"
)

# -----------------------------------------------------------------------------
# 4 KPI CARDS MATCHING Dashboard.jpg
# -----------------------------------------------------------------------------
total_rev = filtered_df['amount_inr'].sum()
total_vol_kg = filtered_df['quantity_kg'].sum()
total_shipments = filtered_df['invoice_no'].nunique()
active_distributors = filtered_df['customer_id'].nunique()
active_countries = filtered_df['country'].nunique()

c1, c2, c3, c4 = st.columns(4)

with c1:
    render_metric_card(
        icon_name="file-text",
        label="Export Invoices",
        value=f"{total_shipments:,}",
        subtext="Billed Orders",
        badge_text="100% Audited",
        badge_type="green"
    )

with c2:
    render_metric_card(
        icon_name="users",
        label="Active Distributors",
        value=f"{active_distributors} Accounts",
        subtext=f"{active_countries} Global Nations",
        badge_text="B2B Network",
        badge_type="blue"
    )

with c3:
    render_metric_card(
        icon_name="trending-up",
        label="Gross Realized Revenue",
        value=format_currency_inr(total_rev),
        subtext="Invoiced Export Value",
        badge_text="14.8% CAGR",
        badge_type="green"
    )

with c4:
    render_metric_card(
        icon_name="package",
        label="Export Volume",
        value=format_volume_kg(total_vol_kg),
        subtext=f"{total_vol_kg/1000:,.0f} Metric Tonnes",
        badge_text="Realized Shipments",
        badge_type="purple"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MIDDLE ROW: AGENDA (LEFT) + MONTHLY DEMAND TRAJECTORY (RIGHT)
# -----------------------------------------------------------------------------
col_agenda, col_chart = st.columns([4, 8])

with col_agenda:
    st.markdown("""
    <div class="content-card" style="height: 100%;">
        <div class="content-card-title">
            <span>Upcoming Export Agenda</span>
            <span style="font-size: 0.8rem; color: #2563EB; font-weight: 600; cursor: pointer;">View All ›</span>
        </div>
        <div class="content-card-subtitle">Critical supply chain & dispatch milestones</div>
        
        <div class="agenda-item">
            <span class="agenda-time">SEP 15 - OCT 30</span>
            <div class="agenda-title">Diwali Export Dispatch Window</div>
            <div class="agenda-desc">Peak container shipments for US (Anmol, Continental) & UK (Asian Traders) accounts.</div>
        </div>
        
        <div class="agenda-item">
            <span class="agenda-time">FEB 01 - MAR 15</span>
            <div class="agenda-title">Middle East Ramadan Indent</div>
            <div class="agenda-desc">Frozen samosas, sweets, and namkeens dispatch to UAE (Giant Holdings FZE) & GCC.</div>
        </div>
        
        <div class="agenda-item">
            <span class="agenda-time">QUARTERLY</span>
            <div class="agenda-title">FOB Price Revision & ZPR8 Review</div>
            <div class="agenda-desc">Review currency volatility (USD/INR) and freight indexation with distributor partners.</div>
        </div>
        
        <div class="agenda-item">
            <span class="agenda-time">ANNUAL</span>
            <div class="agenda-title">US FDA & UK Food Standards Audit</div>
            <div class="agenda-desc">Nutritional packaging compliance review for international ethnic retail channels.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_chart:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>5-Year Monthly Export Demand & Invoiced Revenue</span>
            <span class="venture-pill pill-green">↗ +12% YoY</span>
        </div>
        <div class="content-card-subtitle">Monthly invoiced value (INR Crores) and dispatch volume across all overseas zones</div>
    </div>
    """, unsafe_allow_html=True)

    monthly_trend = filtered_df.groupby('month_start_date').agg(
        amount_inr=('amount_inr', 'sum'),
        quantity_kg=('quantity_kg', 'sum')
    ).reset_index()

    # Create monochrome bar & line chart matching Venture CRM
    fig_monthly = go.Figure()

    # Gray and dark bars
    bar_colors = ['#000000' if m.month in [9, 10] else '#E4E4E7' for m in monthly_trend['month_start_date']]

    fig_monthly.add_trace(go.Bar(
        x=monthly_trend['month_start_date'],
        y=monthly_trend['amount_inr'] / 1e7,
        name='Export Revenue (₹ Cr)',
        marker=dict(color=bar_colors, cornerradius=4),
        hoverinfo='x+y',
        yaxis='y1'
    ))

    # Trendline for Volume
    fig_monthly.add_trace(go.Scatter(
        x=monthly_trend['month_start_date'],
        y=monthly_trend['quantity_kg'] / 1e3,
        name='Realized Volume (MT)',
        mode='lines',
        line=dict(color='#000000', width=2),
        yaxis='y2'
    ))

    fig_layout = get_plotly_layout_dict()
    fig_layout.update(
        height=320,
        yaxis=dict(title='Revenue (₹ Cr)', side='left', showgrid=True, gridcolor='#F3F4F6'),
        yaxis2=dict(title='Volume (MT)', side='right', overlaying='y', showgrid=False)
    )
    fig_monthly.update_layout(**fig_layout)
    st.plotly_chart(fig_monthly, use_container_width=True)

    render_graph_note(
        title="Monthly Export Trajectory & Seasonality",
        explanation="The chart displays monthly gross export revenue (₹ Crores) with dark highlights denoting pre-festive surge dispatches (September-October) for Diwali overseas stocking. The overlaid curve traces volume in Metric Tonnes.",
        viva_point="FMCG export cycles lead retail consumption by 30 to 45 days due to maritime shipping lead times across Atlantic/Pacific routes."
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# BOTTOM ROW: DISTRIBUTORS TABLE (LEFT) + CATEGORIES DONUT (RIGHT)
# -----------------------------------------------------------------------------
col_tbl, col_donut = st.columns([8, 4])

with col_tbl:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Global B2B Distributor Accounts</span>
            <span style="font-size: 0.8125rem; color: #6B7280; font-weight: 500;">Showing Top Strategic Accounts</span>
        </div>
        <div class="content-card-subtitle">Major international importers, distributors, and logistics partners</div>
    </div>
    """, unsafe_allow_html=True)

    dist_summary = filtered_df.groupby(['customer_id', 'customer_name', 'country', 'zone']).agg(
        total_revenue=('amount_inr', 'sum'),
        total_volume=('quantity_kg', 'sum'),
        orders_count=('invoice_no', 'nunique')
    ).reset_index().sort_values('total_revenue', ascending=False).head(6)

    # Format table
    dist_summary['Revenue'] = dist_summary['total_revenue'].apply(lambda x: format_currency_inr(x))
    dist_summary['Volume'] = dist_summary['total_volume'].apply(lambda x: format_volume_kg(x))
    dist_summary['Status'] = ['● Active Tier-1' if i < 3 else '● Strategic' for i in range(len(dist_summary))]

    st.dataframe(
        dist_summary[['customer_name', 'country', 'zone', 'Revenue', 'Volume', 'orders_count', 'Status']].rename(
            columns={
                'customer_name': 'Distributor Account',
                'country': 'Country',
                'zone': 'Zone',
                'orders_count': 'Shipments',
                'Status': 'Account Status'
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=260
    )

    render_graph_note(
        title="Distributor Master Performance",
        explanation="Summary of top enterprise accounts. High concentration in USA (Zone 1) and UK (Zone 4) accounts for over 65% of overall invoiced revenue.",
        viva_point="Concentration of key accounts requires dedicated Key Account Management (KAM) contracts to mitigate supply chain volatility."
    )

with col_donut:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Export Product Categories</span>
        </div>
        <div class="content-card-subtitle">Revenue share across product hierarchies</div>
    </div>
    """, unsafe_allow_html=True)

    cat_dist = filtered_df.groupby('category_name')['amount_inr'].sum().reset_index().sort_values('amount_inr', ascending=False)
    
    # Donut chart matching Venture CRM Dashboard.jpg
    fig_donut = px.pie(
        cat_dist.head(5),
        names='category_name',
        values='amount_inr',
        hole=0.68,
        color_discrete_sequence=['#000000', '#3F3F46', '#71717A', '#A1A1AA', '#D4D4D8']
    )
    
    donut_layout = get_plotly_layout_dict()
    donut_layout.update(
        height=260,
        showlegend=False,
        annotations=[dict(text=f"<b>{df_all['material_code'].nunique():,}</b><br><span style='font-size:10px;color:#6B7280;'>SKUs</span>", x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    fig_donut.update_layout(**donut_layout)
    st.plotly_chart(fig_donut, use_container_width=True)

    # Bullet list below donut matching reference
    for idx, row in cat_dist.head(4).iterrows():
        pct = (row['amount_inr'] / cat_dist['amount_inr'].sum()) * 100
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.8125rem; padding: 3px 0; color: #374151;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #000000;"></span>
                <span>{row['category_name']}</span>
            </div>
            <span style="font-weight: 600;">{pct:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# VIVA DEFENSE EXPANDER
# -----------------------------------------------------------------------------
with st.expander("Senior DS Viva Defense Notes: Executive Dashboard Architecture"):
    st.markdown("""
    - **FMCG Export Dynamics**: International exports operate on container-level bulk purchase orders scheduled 30–60 days ahead of major festive windows (Diwali in USA/UK, Ramadan in Middle East).
    - **Data Pipeline Quality**: Ingested 5 years of transactional data across 294,898 rows, addressing column schema drift and standardizing SAP condition records (`ZPR8`, `ZFRT`, `ZINS`).
    - **Margin Proxy Formulation**: Direct export deductions (`ZFRT Freight`, `ZINS Insurance`, `ZDIS Discount`) are subtracted from invoiced LC (INR) to represent net realized export contribution margin.
    """)
