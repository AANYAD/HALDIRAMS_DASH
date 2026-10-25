"""
================================================================================
Haldiram's International Export Analytics Pipeline - Overview & EDA Page
================================================================================
Author: Aanya
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (Notion-style Analytics UI)
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.utils import (
    apply_notion_crm_style,
    format_currency_inr,
    get_plotly_layout_dict,
    render_venture_topbar,
    render_metric_card,
    render_graph_note,
    render_sidebar_nav_label
)

st.set_page_config(page_title="Overview & EDA | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

# -----------------------------------------------------------------------------
# DATA INGESTION
# -----------------------------------------------------------------------------
def get_file_mtime(filepath: str) -> float:
    p = Path(filepath)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_data(_mtime: float):
    df = pd.read_parquet("data/haldirams_sales_consolidated.parquet")
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    # Relabel zone '000001' as 'Unmapped'
    df['zone'] = df['zone'].replace({'000001': 'Unmapped'})
    return df

df = load_data(get_file_mtime("data/haldirams_sales_consolidated.parquet"))

# -----------------------------------------------------------------------------
# TOPBAR
# -----------------------------------------------------------------------------
render_venture_topbar(
    title="Overview & EDA",
    subtitle="Macro Export Sales Summary, Global Territory Distribution & Pareto Concentration"
)

# -----------------------------------------------------------------------------
# FISCAL YEAR MULTI-SELECT FILTER (ROW DIRECTLY ABOVE KPI CARDS)
# -----------------------------------------------------------------------------
available_fys = sorted(df['fiscal_year'].dropna().unique().tolist())
try:
    selected_fys = st.pills(
        "Fiscal Year Filter",
        options=available_fys,
        default=available_fys,
        selection_mode="multi",
        label_visibility="collapsed"
    )
except Exception:
    selected_fys = st.multiselect(
        "Fiscal Year Filter",
        options=available_fys,
        default=available_fys,
        label_visibility="collapsed"
    )

st.caption("No selection = all years | FY25-26 contains Apr-May 2025 only")

effective_fys = selected_fys if (selected_fys is not None and len(selected_fys) > 0) else available_fys
filtered_df = df if set(effective_fys) == set(available_fys) else df[df['fiscal_year'].isin(effective_fys)].copy()

# -----------------------------------------------------------------------------
# 1. KPI STRIP (4 CARDS)
# -----------------------------------------------------------------------------
total_rev = filtered_df['amount_inr'].sum()
total_txns = len(filtered_df)
total_cust = filtered_df['customer_id'].nunique()
min_month = filtered_df['invoice_date'].min().strftime('%b %Y')
max_month = filtered_df['invoice_date'].max().strftime('%b %Y')

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    render_metric_card(
        icon_name="trending-up",
        label="Total Export Revenue",
        value=format_currency_inr(total_rev),
        subtext="Invoiced gross value",
        badge_text="Invoiced Gross",
        badge_type="green"
    )

with kpi2:
    render_metric_card(
        icon_name="file-text",
        label="Invoice Line Items",
        value=f"{total_txns:,}",
        subtext=f"{filtered_df['invoice_no'].nunique():,} unique invoices",
        badge_text="Source: SAP",
        badge_type="blue"
    )

with kpi3:
    render_metric_card(
        icon_name="users",
        label="B2B Customers",
        value=f"{total_cust} Accounts",
        subtext=f"{filtered_df['country'].nunique()} Global Nations",
        badge_text="Active accounts",
        badge_type="purple"
    )

with kpi4:
    render_metric_card(
        icon_name="calendar",
        label="Invoice Date Range",
        value=f"{min_month} – {max_month}",
        subtext="4 years of data",
        badge_text="Multi-Year",
        badge_type="amber"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MONTHLY REVENUE TREND (WITH DIWALI BANDS)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="content-card">
    <div class="content-card-title">
        <span>Monthly Export Revenue Run-Rate Across Fiscal Years</span>
    </div>
    <div class="content-card-subtitle">Aggregated invoiced export value (₹ Crores) with pre-festive seasonal surge intervals</div>
</div>
""", unsafe_allow_html=True)

monthly_rev = filtered_df.groupby('month_start_date')['amount_inr'].sum().reset_index()

# Reindex across complete monthly grid to handle missing middle months like March 2025
min_m_date = monthly_rev['month_start_date'].min()
max_m_date = monthly_rev['month_start_date'].max()

# Check if selected years include FY24-25 ('2024-2025')
includes_fy2425 = any(y in ('2024-2025', 'FY24-25') for y in effective_fys)
if includes_fy2425:
    max_m_date = max(max_m_date, pd.to_datetime('2025-03-01'))

all_months = pd.date_range(min_m_date, max_m_date, freq='MS')
monthly_rev = monthly_rev.set_index('month_start_date').reindex(all_months).rename_axis('month_start_date').reset_index()

# Check if March 2025 was missing and needs interpolation
is_mar25 = (monthly_rev['month_start_date'] == '2025-03-01')
was_interpolated_mar25 = is_mar25 & (monthly_rev['amount_inr'].isna() | (monthly_rev['amount_inr'] == 0))

monthly_rev['amount_inr'] = monthly_rev['amount_inr'].interpolate(method='linear').bfill().ffill()
monthly_rev['amount_cr'] = monthly_rev['amount_inr'] / 1e7

fig_trend = go.Figure()
# Regular line and markers
fig_trend.add_trace(go.Scatter(
    x=monthly_rev['month_start_date'],
    y=monthly_rev['amount_cr'],
    mode='lines+markers',
    name='Invoiced Export Revenue (₹ Cr)',
    line=dict(color='#0F172A', width=2.5),
    marker=dict(size=5, color='#0F172A')
))

# Hollow marker for interpolated March 2025 (only when selected years include FY24-25)
show_mar25_marker = includes_fy2425 and was_interpolated_mar25.any() and is_mar25.any()
if show_mar25_marker:
    mar_pt = monthly_rev[is_mar25].iloc[0]
    fig_trend.add_trace(go.Scatter(
        x=[mar_pt['month_start_date']],
        y=[mar_pt['amount_cr']],
        mode='markers',
        name='March 2025 (Interpolated)',
        marker=dict(symbol='circle', size=10, color='white', line=dict(color='#0F172A', width=2.5)),
        hovertemplate='<b>March 2025 (Interpolated)</b><br>Revenue: ₹%{y:.2f} Cr<extra></extra>'
    ))

# Shaded Diwali months (Sep-Oct) matching forecasting page
for diwali_date in ['2021-09-01', '2022-09-01', '2023-09-01', '2024-09-01', '2025-09-01']:
    d_dt = pd.to_datetime(diwali_date)
    if min_m_date <= d_dt <= max_m_date + pd.DateOffset(months=1):
        fig_trend.add_vrect(
            x0=d_dt,
            x1=d_dt + pd.DateOffset(months=2),
            fillcolor="#FEF3C7",
            opacity=0.35,
            layer="below",
            line_width=0,
            annotation_text="Diwali Surge",
            annotation_position="top left",
            annotation_font_size=9
        )

trend_layout = get_plotly_layout_dict()
trend_layout.update(
    height=360,
    yaxis=dict(title='Export Revenue (₹ Cr)', showgrid=True, gridcolor='#F3F4F6'),
    xaxis=dict(title='Month', showgrid=True, gridcolor='#F3F4F6')
)
fig_trend.update_layout(**trend_layout)
st.plotly_chart(fig_trend, use_container_width=True)

if show_mar25_marker:
    st.caption("* Note: March 2025 (marked with a hollow circle ○) had no transaction logs in raw SAP export registers and has been linearly interpolated to maintain continuous time-series continuity.")

# Dynamic 1-2 sentence interpretation computed from data
monthly_rev['is_festive'] = monthly_rev['month_start_date'].dt.month.isin([9, 10])
diwali_avg = monthly_rev[monthly_rev['is_festive']]['amount_cr'].mean() if monthly_rev['is_festive'].any() else 0
baseline_avg = monthly_rev[~monthly_rev['is_festive']]['amount_cr'].mean() if (~monthly_rev['is_festive']).any() else 1
diwali_lift = ((diwali_avg - baseline_avg) / baseline_avg * 100.0) if baseline_avg > 0 else 0.0

render_graph_note(
    title="Monthly Export Trajectory & Festive Seasonality",
    explanation=f"Monthly export revenue averages ₹{diwali_avg:.1f} Cr during pre-Diwali dispatch windows (Sep–Oct), generating a {diwali_lift:+.1f}% demand surge over baseline non-festive months to accommodate 30–45 day ocean freight transit times."
)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. REVENUE BY ZONE & 5. REVENUE BY PRODUCT CATEGORY (2 COLUMNS)
# -----------------------------------------------------------------------------
col_z, col_c = st.columns(2)

with col_z:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Export Revenue by Trade Zone</span>
        </div>
        <div class="content-card-subtitle">Geographic market breakdown sorted descending by invoiced volume</div>
    </div>
    """, unsafe_allow_html=True)

    zone_df = filtered_df.groupby('zone')['amount_inr'].sum().reset_index()
    zone_df['amount_cr'] = zone_df['amount_inr'] / 1e7
    zone_df = zone_df.sort_values('amount_cr', ascending=True)

    fig_zone = px.bar(
        zone_df,
        x='amount_cr',
        y='zone',
        orientation='h',
        labels={'amount_cr': 'Invoiced Revenue (₹ Cr)', 'zone': 'Export Zone'},
        color_discrete_sequence=['#2563EB']
    )
    fig_zone.update_traces(marker=dict(cornerradius=4))
    z_layout = get_plotly_layout_dict()
    z_layout.update(height=340, yaxis=dict(showgrid=False))
    fig_zone.update_layout(**z_layout)
    st.plotly_chart(fig_zone, use_container_width=True)

    top_zone = zone_df.iloc[-1]['zone']
    top_zone_cr = zone_df.iloc[-1]['amount_cr']
    top_zone_pct = (zone_df.iloc[-1]['amount_inr'] / total_rev * 100) if total_rev > 0 else 0
    top_zone_desc = f"Zone {top_zone}" if not str(top_zone).startswith("Zone") else str(top_zone)

    render_graph_note(
        title="Territory Realization Breakdown",
        explanation=f"{top_zone_desc} represents the largest export market generating ₹{top_zone_cr:.1f} Cr ({top_zone_pct:.1f}% of total export revenue), computed directly from historical dispatch invoices."
    )

with col_c:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Revenue by Product Category</span>
        </div>
        <div class="content-card-subtitle">Top export categories ranked by cumulative invoiced turnover</div>
    </div>
    """, unsafe_allow_html=True)

    cat_df = filtered_df.groupby('category_name')['amount_inr'].sum().reset_index()
    cat_df['amount_cr'] = cat_df['amount_inr'] / 1e7
    cat_df = cat_df.sort_values('amount_cr', ascending=False).head(7)

    fig_cat = px.bar(
        cat_df,
        x='category_name',
        y='amount_cr',
        labels={'amount_cr': 'Invoiced Revenue (₹ Cr)', 'category_name': 'Category'},
        color_discrete_sequence=['#0F172A']
    )
    fig_cat.update_traces(marker=dict(cornerradius=4))
    c_layout = get_plotly_layout_dict()
    c_layout.update(height=340, xaxis=dict(tickangle=-25, showgrid=False))
    fig_cat.update_layout(**c_layout)
    st.plotly_chart(fig_cat, use_container_width=True)

    top_cat_name = cat_df.iloc[0]['category_name']
    top_cat_cr = cat_df.iloc[0]['amount_cr']
    top_cat_pct = (cat_df.iloc[0]['amount_inr'] / total_rev * 100) if total_rev > 0 else 0

    render_graph_note(
        title="Product Portfolio Dominance",
        explanation=f"{top_cat_name} constitutes the primary revenue engine delivering ₹{top_cat_cr:.1f} Cr ({top_cat_pct:.1f}% share), reflecting robust consumer demand for Haldiram's traditional packaged savouries."
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. CUSTOMER CONCENTRATION: PARETO CHART (TOP 10 CUSTOMERS)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="content-card">
    <div class="content-card-title">
        <span>B2B Distributor Account Concentration (Top 10 Pareto Analysis)</span>
    </div>
    <div class="content-card-subtitle">Individual invoiced revenue (bars) alongside cumulative turnover contribution (line)</div>
</div>
""", unsafe_allow_html=True)

# Group full data by raw account first and print top 12 diagnostic table
raw_cust = filtered_df.groupby(['customer_id', 'customer_name', 'country'])['amount_inr'].sum().reset_index()
raw_cust = raw_cust.sort_values('amount_inr', ascending=False).reset_index(drop=True)
top12_raw = raw_cust.head(12)
print("\n[PARETO OVERVIEW] Top 12 Customers (Pre-Merge):", flush=True)
for i, r in top12_raw.iterrows():
    print(f"  {i+1:2d}. Code: {r['customer_id']:<10} | Name: {r['customer_name'][:30]:<30} | Country: {r['country']:<12} | Rev: Rs. {r['amount_inr']/1e7:6.2f} Cr", flush=True)

# Consolidate multiple customer codes representing the same legal entity
def map_distributor_entity(row):
    cid = str(row['customer_id'])
    name = str(row['customer_name']).upper()
    if 'PREMIER FOOD' in name or cid in ['170024', '170038', '20000016', '20000028']:
        return 'PREMIER FOOD SUPPLIES LLC', 'USA', '170038/24'
    elif 'JMD LLC' in name or cid in ['170016', '170035', '20000011', '20000025']:
        return 'JMD LLC', 'USA', '170035/16'
    else:
        return row['customer_name'], row['country'], cid

entity_mapping = raw_cust.apply(map_distributor_entity, axis=1, result_type='expand')
raw_cust['entity_name'] = entity_mapping[0]
raw_cust['entity_country'] = entity_mapping[1]
raw_cust['entity_code'] = entity_mapping[2]

all_cust = raw_cust.groupby(['entity_name', 'entity_country', 'entity_code'])['amount_inr'].sum().reset_index()
all_cust = all_cust.sort_values('amount_inr', ascending=False).reset_index(drop=True)
all_cust['amount_cr'] = all_cust['amount_inr'] / 1e7
all_cust['cum_share_pct'] = (all_cust['amount_inr'].cumsum() / total_rev * 100.0) if total_rev > 0 else 0.0
all_cust['short_name'] = all_cust['entity_name'].apply(lambda x: (x[:15] + '...') if len(x) > 15 else x)

print("\n[PARETO OVERVIEW] Top 12 Customers (Post-Entity Merge):", flush=True)
for i, r in all_cust.head(12).iterrows():
    print(f"  {i+1:2d}. Code: {r['entity_code']:<10} | Name: {r['entity_name'][:30]:<30} | Country: {r['entity_country']:<12} | Rev: Rs. {r['amount_inr']/1e7:6.2f} Cr | Cum%: {r['cum_share_pct']:.2f}%", flush=True)

# Top 10 customer accounts
top10 = all_cust.head(10).copy()

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(
    x=top10['entity_name'],
    y=top10['amount_cr'],
    name='Invoiced Revenue (₹ Cr)',
    marker=dict(color='#0F172A', cornerradius=4),
    customdata=top10[['entity_name', 'entity_country', 'entity_code']].values,
    hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>Customer Key: %{customdata[2]}<br>Revenue: ₹%{y:.2f} Cr<extra></extra>',
    yaxis='y1'
))
fig_pareto.add_trace(go.Scatter(
    x=top10['entity_name'],
    y=top10['cum_share_pct'],
    name='Cumulative Revenue Share (%)',
    mode='lines+markers',
    line=dict(color='#D97706', width=2.5),
    marker=dict(size=7, color='#D97706'),
    customdata=top10[['entity_name', 'cum_share_pct']].values,
    hovertemplate='<b>%{customdata[0]}</b><br>Cumulative Share: %{customdata[1]:.1f}%<extra></extra>',
    yaxis='y2'
))

pareto_layout = get_plotly_layout_dict()
pareto_layout.update(
    height=420,
    xaxis=dict(
        title='Distributor Legal Entity',
        type='category',
        categoryorder='array',
        categoryarray=top10['entity_name'].tolist(),
        tickmode='array',
        tickvals=top10['entity_name'].tolist(),
        ticktext=top10['short_name'].tolist(),
        tickangle=-45,
        showgrid=False
    ),
    yaxis=dict(title='Revenue (₹ Cr)', showgrid=True, gridcolor='#F3F4F6', side='left'),
    yaxis2=dict(title='Cumulative Revenue %', overlaying='y', side='right', range=[0, 105], showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig_pareto.update_layout(**pareto_layout)
st.plotly_chart(fig_pareto, use_container_width=True)

top10_cum_pct = top10['cum_share_pct'].iloc[-1]
top1_cust_name = top10.iloc[0]['entity_name']
top1_cust_pct = (top10.iloc[0]['amount_inr'] / total_rev * 100.0) if total_rev > 0 else 0.0

render_graph_note(
    title="Key Account Concentration & Credit Exposure",
    explanation=f"The top 10 distributor accounts generate {top10_cum_pct:.1f}% of total export turnover, with {top1_cust_name} alone contributing {top1_cust_pct:.1f}%, underscoring the critical need for structured Key Account Management (KAM) to protect enterprise revenue."
)
