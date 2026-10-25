"""
================================================================================
Haldiram's International Export Analytics Pipeline - Segmentation Page
================================================================================
Author: Senior Data Scientist (Technical Lead)
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (Task Details / Analytics UI Reference)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.segmentation import B2BCustomerSegmentation
from src.utils import (
    apply_notion_crm_style,
    format_currency_inr,
    format_volume_kg,
    get_plotly_layout_dict,
    render_venture_topbar,
    render_metric_card,
    render_graph_note,
    VENTURE_THEME
)

st.set_page_config(page_title="Segmentation | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

render_venture_topbar(
    title="Customer & Product Segmentation",
    subtitle="Unsupervised K-Means Clustering & 2D PCA Dimensionality Decomposition"
)

def get_file_mtime(filepath: str) -> float:
    p = Path(filepath)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_data(_mtime: float):
    df = pd.read_parquet("data/haldirams_sales_consolidated.parquet")
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    return df

df = load_data(get_file_mtime("data/haldirams_sales_consolidated.parquet"))

# -----------------------------------------------------------------------------
# CLUSTERING PARAMETERS
# -----------------------------------------------------------------------------
seg_engine = B2BCustomerSegmentation(df)
sil_k3, sil_k4, rec_k = seg_engine.compute_k3_k4_silhouettes()
print(f"[PAGE SEGMENTATION] Silhouette Scores: K=3: {sil_k3:.4f} | K=4: {sil_k4:.4f}", flush=True)

col_k1, col_k2 = st.columns([4, 8])
with col_k1:
    n_clusters = st.slider("Select K (Number of Clusters)", min_value=2, max_value=6, value=3, step=1)
    st.caption(f"Cluster Separation Benchmark: **K=3 Silhouette = {sil_k3:.4f}** | **K=4 Silhouette = {sil_k4:.4f}** (Recommended: **K=3**)")
with col_k2:
    st.caption("Features Standardized via `StandardScaler`: Total Revenue, Volume (Kg), Order Frequency, AOV, YoY Growth Rate %, and Demand Volatility (CV).")

clustered_df, seg_metrics = seg_engine.fit_kmeans_clustering(n_clusters=n_clusters)
cluster_summary = seg_engine.get_cluster_summary_profile()

# Compute live benchmark inertia for K=3 and K=4
_, m_k3 = seg_engine.fit_kmeans_clustering(n_clusters=3)
_, m_k4 = seg_engine.fit_kmeans_clustering(n_clusters=4)
print(f"[PAGE SEGMENTATION] Live Inertia (WCSS): K=3: {m_k3['inertia']:,.2f} | K=4: {m_k4['inertia']:,.2f}", flush=True)
print(f"[PAGE SEGMENTATION] Selected K={n_clusters} Live Inertia (WCSS): {seg_metrics['inertia']:,.2f}", flush=True)

# -------------------------------------------------------------------------
# 4 METRIC CARDS
# -------------------------------------------------------------------------
sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
with sm_c1:
    render_metric_card(
        icon_name="building-2",
        label="B2B Distributors",
        value=f"{len(clustered_df)} Accounts",
        subtext="Global Network",
        badge_text="Active Corpus",
        badge_type="blue"
    )
with sm_c2:
    render_metric_card(
        icon_name="layout-grid",
        label="Silhouette Score",
        value=f"{seg_metrics['silhouette_score']:.3f}",
        subtext="Cluster Separation",
        badge_text="Cohesion Index",
        badge_type="green"
    )
with sm_c3:
    render_metric_card(
        icon_name="zap",
        label="Inertia (WCSS)",
        value=f"{seg_metrics['inertia']:,.0f}",
        subtext="Within-Cluster Distance",
        badge_text="Compactness",
        badge_type="purple"
    )
with sm_c4:
    render_metric_card(
        icon_name="globe",
        label="PCA Variance",
        value=f"{clustered_df['pca_explained_var_pct'].iloc[0]:.1f}%",
        subtext="2D Explained Variance",
        badge_text="Visual Fidelity",
        badge_type="amber"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2D PCA VISUALIZATION & PERSONA CARDS
# -------------------------------------------------------------------------
col_pca, col_desc = st.columns([7, 5])

with col_pca:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>2D PCA Distributor Cluster Space</span>
        </div>
        <div class="content-card-subtitle">Orthogonal projection of distributor accounts colored by assigned FMCG Persona</div>
    </div>
    """, unsafe_allow_html=True)
    
    PERSONA_COLORS = {
        'Mega Accounts': '#1E3A8A',                 # Navy
        'Core Strategic Partners': '#EA580C',       # Orange
        'High-Growth Strategic Partners': '#0D9488', # Teal
        'Long-Tail Accounts': '#C026D3',            # Magenta
        'Long-Tail Accounts (Tier 2)': '#C026D3',
        'Emerging Accounts': '#C026D3'
    }
    fig_pca = px.scatter(
        clustered_df,
        x='pca_dim_1',
        y='pca_dim_2',
        color='persona_name',
        hover_name='customer_name',
        hover_data={
            'country': True,
            'total_revenue_inr': ':.2f',
            'order_frequency': True,
            'demand_volatility_cv': ':.2f',
            'yoy_growth_pct': ':.1f',
            'pca_dim_1': False,
            'pca_dim_2': False,
            'persona_name': False
        },
        color_discrete_map=PERSONA_COLORS,
        labels={'pca_dim_1': 'Principal Component 1 (Scale & Volume)', 'pca_dim_2': 'Principal Component 2 (Volatility & Growth)'}
    )
    fig_pca.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color='#FFFFFF')))
    
    fig_layout = get_plotly_layout_dict()
    fig_layout.update(height=420, legend_title_text='')
    fig_pca.update_layout(**fig_layout)
    st.plotly_chart(fig_pca, use_container_width=True)

    render_graph_note(
        title="2D PCA Latent Feature Space",
        explanation="The scatter plot projects high-dimensional distributor characteristics (monetary volume, shipment cadence, and demand volatility) onto the first two principal components."
    )

with col_desc:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Commercial Action Playbooks</span>
        </div>
        <div class="content-card-subtitle">Key Account Management (KAM) strategies by Persona</div>
    </div>
    """, unsafe_allow_html=True)

    sorted_playbook = cluster_summary.sort_values('total_revenue_inr', ascending=False)
    for _, row in sorted_playbook.iterrows():
        badge_color = PERSONA_COLORS.get(row['persona_name'], row.get('tag_color', '#64748B'))
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid {badge_color}; border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.75rem;">
            <div style="font-weight: 700; color: #111827; font-size: 0.95rem;">
                {row['persona_name']}
                <span style="font-size: 0.75rem; font-weight: normal; color: #6B7280; margin-left: 0.5rem;">
                    ({row['distributor_count']} Accounts | {row['revenue_contribution_pct']:.1f}% Rev)
                </span>
            </div>
            <div style="font-size: 0.8125rem; color: #4B5563; margin-top: 0.35rem;">
                <b>Action Playbook:</b> {row['recommended_strategy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# CLUSTER CENTROID BENCHMARK TABLE
# -------------------------------------------------------------------------
st.markdown("""
<div class="content-card">
    <div class="content-card-title">
        <span>Cluster Centroid Benchmark Profile</span>
    </div>
    <div class="content-card-subtitle">Average numerical metrics across each identified B2B distributor segment</div>
</div>
""", unsafe_allow_html=True)

def format_yoy_clean(val: float) -> str:
    if pd.isna(val) or abs(val) < 0.05:
        return "0.0%"
    return f"{val:+.1f}%"

table_display = cluster_summary[['cluster_id', 'persona_name', 'distributor_count', 'total_revenue_inr', 'avg_order_freq', 'avg_volatility_cv', 'displayed_agg_growth', 'displayed_median_growth']].copy()
table_display['Total Revenue'] = table_display['total_revenue_inr'].apply(lambda x: format_currency_inr(x))
table_display['Agg Growth %'] = table_display['displayed_agg_growth'].apply(format_yoy_clean)
table_display['Median YoY %'] = table_display['displayed_median_growth'].apply(format_yoy_clean)
table_display = table_display.drop(columns=['total_revenue_inr', 'displayed_agg_growth', 'displayed_median_growth']).rename(columns={
    'cluster_id': 'Cluster ID',
    'persona_name': 'Assigned Persona',
    'distributor_count': 'Accounts',
    'avg_order_freq': 'Avg Orders',
    'avg_volatility_cv': 'Demand CV (Volatility)',
    'Agg Growth %': 'YoY Agg Growth %',
    'Median YoY %': 'YoY Median Growth %'
})
st.dataframe(table_display.style.format({
    'Avg Orders': '{:.1f}',
    'Demand CV (Volatility)': '{:.2f}'
}), use_container_width=True, hide_index=True)
