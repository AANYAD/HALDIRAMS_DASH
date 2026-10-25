"""
================================================================================
Haldiram's International Export Analytics Pipeline - Price Elasticity Page
================================================================================
Author: Aanya
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (Task Details / Pricing Studio UI Reference)
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.elasticity import PriceElasticityEngine
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

st.set_page_config(page_title="Price Elasticity | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

render_venture_topbar(
    title="Price Elasticity Studio",
    subtitle="Log-Log Econometric Demand Modeling & SAP ZPR8 Condition Cross-Validation"
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
engine = PriceElasticityEngine(df)

# -----------------------------------------------------------------------------
# TOP SKU SELECTION & PARAMETERS
# -----------------------------------------------------------------------------
top_skus_df = engine.get_top_export_skus(top_n=8)

st.markdown("""
<div class="content-card" style="padding: 1rem 1.25rem; margin-bottom: 1.25rem;">
    <div style="font-size: 0.8125rem; font-weight: 600; color: #374151; margin-bottom: 0.5rem; text-transform: uppercase;">
        🏷️ Export SKU & Geographic Market Scope
    </div>
</div>
""", unsafe_allow_html=True)

col_ctrl_1, col_ctrl_2, col_ctrl_3 = st.columns([5, 4, 3])

with col_ctrl_1:
    sku_options = {row['material_code']: f"{row['material_desc']} ({row['material_code']})" for _, row in top_skus_df.iterrows()}
    selected_sku_code = st.selectbox(
        "Select Top-Revenue Export SKU",
        options=list(sku_options.keys()),
        format_func=lambda x: sku_options[x]
    )

with col_ctrl_2:
    available_countries = ['All Countries'] + sorted(df[df['material_code'] == selected_sku_code]['country'].unique().tolist())
    selected_country = st.selectbox("Target Market Filter", options=available_countries, index=0)

with col_ctrl_3:
    sku_row = top_skus_df[top_skus_df['material_code'] == selected_sku_code].iloc[0]
    st.caption("Selected SKU Scope")
    st.caption(f"Lifetime Total: **{sku_row['total_volume_kg']/1e3:.1f}K kg** ({format_currency_inr(sku_row['total_revenue_inr'])})")

# -----------------------------------------------------------------------------
# RUN ECONOMETRIC OLS MODEL
# -----------------------------------------------------------------------------
regression_results = engine.fit_log_log_elasticity_model(
    material_code=selected_sku_code,
    country=selected_country
)
zpr8_validation = engine.validate_derived_vs_zpr8_price(selected_sku_code)

if regression_results['success']:
    beta = regression_results['elasticity_coefficient']
    p_val = regression_results['p_value']
    r2 = regression_results['r_squared']
    e_class = regression_results['elasticity_class']
    ci_low = regression_results.get('ci_lower', beta - 1.96 * 0.1)
    ci_high = regression_results.get('ci_upper', beta + 1.96 * 0.1)
    min_m = regression_results.get('min_month', 'April 2021')
    max_m = regression_results.get('max_month', 'May 2025')
    last_zpr8 = regression_results.get('last_zpr8_month', 'February 2025')
    n_obs = regression_results.get('sample_size', len(regression_results['regression_data']))
    
    scatter_min_m = regression_results['regression_data']['month_start_date'].min().strftime('%B %Y')
    scatter_max_m = regression_results['regression_data']['month_start_date'].max().strftime('%B %Y')
    price_min_m = zpr8_validation['comparison_df']['month_start_date'].min().strftime('%B %Y') if not zpr8_validation['comparison_df'].empty else 'N/A'
    price_max_m = zpr8_validation['comparison_df']['month_start_date'].max().strftime('%B %Y') if not zpr8_validation['comparison_df'].empty else 'N/A'

    scatter_pts = len(regression_results['regression_data'])
    print(f"[PRICE ELASTICITY] Observations (n): {n_obs}", flush=True)
    print(f"[PRICE ELASTICITY] Scatter chart points plotted: {scatter_pts} (equals n={n_obs}: {scatter_pts == n_obs})", flush=True)
    print(f"[PRICE ELASTICITY] Scatter chart window: {scatter_min_m} to {scatter_max_m}", flush=True)
    print(f"[PRICE ELASTICITY] Price chart window: {price_min_m} to {price_max_m}", flush=True)
    print(f"[PRICE ELASTICITY] Regression window: {min_m} to {max_m} | Last SAP ZPR8 month: {last_zpr8} | Beta: {beta:+.4f} | p-value: {p_val:.4f} | R2: {r2:.4f} | 95% CI: [{ci_low:+.4f}, {ci_high:+.4f}]", flush=True)

    if p_val > 0.05:
        pricing_rec = f"Indicative only: demand appears price-inelastic (|β| < 1), but the result is not statistically conclusive (p = {p_val:.4f}, R² = {r2:.3f}). Test any increase on a small scale before rolling it out."
    else:
        pricing_rec = regression_results['pricing_recommendation']

    # -------------------------------------------------------------------------
    # 4 METRIC CARDS
    # -------------------------------------------------------------------------
    el_c1, el_c2, el_c3, el_c4 = st.columns(4)

    with el_c1:
        render_metric_card(
            icon_name="tag",
            label="Price Elasticity (β)",
            value=f"{beta:+.3f}",
            subtext="Log-Log Slope",
            badge_text="Inelastic (<1.0)" if abs(beta) < 1.0 else "Elastic (>1.0)",
            badge_type="green" if abs(beta) < 1.0 else "amber"
        )
    with el_c2:
        render_metric_card(
            icon_name="circle-check",
            label="Statistical Significance",
            value=f"p = {p_val:.4f}",
            subtext="Hypothesis Test",
            badge_text="p < 0.05 (Sig)" if p_val < 0.05 else "Inconclusive",
            badge_type="green" if p_val < 0.05 else "amber"
        )
    with el_c3:
        render_metric_card(
            icon_name="trending-up",
            label="Model Fit (R²)",
            value=f"{r2:.3f}",
            subtext=f"n = {n_obs} monthly points",
            badge_text=f"n = {n_obs}",
            badge_type="blue"
        )
    
    z_corr = zpr8_validation.get('correlation', 0.0)
    if z_corr >= 0.8:
        z_badge_text = "Strong alignment"
        z_badge_type = "green"
    elif z_corr >= 0.5:
        z_badge_text = "Moderate alignment"
        z_badge_type = "blue"
    else:
        z_badge_text = "Weak alignment"
        z_badge_type = "amber"

    with el_c4:
        render_metric_card(
            icon_name="search",
            label="SAP ZPR8 Alignment",
            value=f"{z_corr:.2f} Corr",
            subtext="Tariff Condition Match",
            badge_text=z_badge_text,
            badge_type=z_badge_type
        )

    st.caption(f"OLS Double-Log Econometric Model: **n = {n_obs} monthly observations** ({min_m} – {max_m}, excluding March 2025) | **R² = {r2:.3f}** | **p = {p_val:.4f}**")

    if p_val > 0.05:
        st.markdown(f"""
        <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #D97706; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.5rem; margin-bottom: 1rem; font-size: 0.875rem; color: #92400E; font-weight: 600;">
            ⚠️ Result is indicative, not statistically conclusive (p = {p_val:.4f}, R² = {r2:.3f}, n = {n_obs})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # REGRESSION CURVE & ZPR8 VALIDATION CHARTS
    # -------------------------------------------------------------------------
    col_reg_chart, col_zpr8_chart = st.columns([7, 5])

    with col_reg_chart:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-title">
                <span>Log-Log Empirical Demand Curve</span>
            </div>
            <div class="content-card-subtitle">Natural log of realized price (INR/Kg) vs. natural log of monthly export volume (Kg)</div>
        </div>
        """, unsafe_allow_html=True)
        
        reg_df = regression_results['regression_data']

        fig_reg = px.scatter(
            reg_df,
            x='log_price',
            y='log_quantity',
            trendline='ols',
            trendline_color_override='#000000',
            hover_data={'q_var': ':.0f', 'p_var': ':.2f'},
            labels={'log_price': 'ln(Realized Unit Price INR/Kg)', 'log_quantity': 'ln(Monthly Export Quantity Kg)'}
        )
        fig_reg.update_traces(marker=dict(size=9, color='#000000', opacity=0.75))
        
        reg_layout = get_plotly_layout_dict()
        reg_layout.update(height=360)
        fig_reg.update_layout(**reg_layout)
        st.plotly_chart(fig_reg, use_container_width=True)

        render_graph_note(
            title="Log-Log Ordinary Least Squares (OLS) Demand Curve",
            explanation="In a double-logarithmic specification, the regression slope directly estimates constant price elasticity of demand (beta). Seasonal quarterly dummies control for festival shifts.",
            technical_note="Taking natural logarithms linearizes the multiplicative demand function Q = A * P^beta, allowing direct interpretation of beta as percentage change in demand per 1% change in price."
        )

    with col_zpr8_chart:
        st.markdown("""
        <div class="content-card">
            <div class="content-card-title">
                <span>SAP ZPR8 Master Price Validation</span>
            </div>
            <div class="content-card-subtitle">Comparing derived unit prices against SAP Condition Record ZPR8</div>
        </div>
        """, unsafe_allow_html=True)
        
        if zpr8_validation['has_zpr8_history'] and not zpr8_validation['comparison_df'].empty:
            z_df = zpr8_validation['comparison_df']
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(
                x=z_df['month_start_date'],
                y=z_df['realized_price_inr'],
                name='Realized Unit Price (INR/Kg)',
                mode='lines+markers',
                line=dict(color='#000000', width=2.5),
                marker=dict(size=4)
            ))
            z_valid = z_df[z_df['sap_zpr8_price'].notna()]
            if not z_valid.empty:
                fig_z.add_trace(go.Scatter(
                    x=z_valid['month_start_date'],
                    y=z_valid['sap_zpr8_price'] * 83.0,
                    name='SAP Master ZPR8 Price',
                    mode='lines+markers',
                    line=dict(color='#2563EB', width=2, dash='dot'),
                    marker=dict(size=4)
                ))
            
            z_layout = get_plotly_layout_dict()
            z_layout.update(height=360, yaxis=dict(title='Price (INR)', showgrid=True, gridcolor='#F3F4F6'))
            fig_z.update_layout(**z_layout)
            st.plotly_chart(fig_z, use_container_width=True)

            last_zpr8_m = zpr8_validation.get('last_zpr8_month', 'February 2025')
            st.caption(f"* Note: SAP ZPR8 price master available through {last_zpr8_m}")

            render_graph_note(
                title="SAP ZPR8 Cross-Validation",
                explanation="Validates that transaction-level realized export prices (amount_inr / quantity_kg) correspond to the baseline FOB condition sheet rates stored in SAP.",
                technical_note="This validation confirms that empirical price fluctuations reflect genuine contractual tariff adjustments rather than unit conversion artifacts."
            )
        else:
            st.info("ZPR8 pricing condition field is actively captured in recent fiscal years. Realized transaction price serves as primary index.")
            last_zpr8_m = zpr8_validation.get('last_zpr8_month', 'February 2025')
            st.caption(f"* Note: SAP ZPR8 price master available through {last_zpr8_m}")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE PRICING SIMULATION SLIDER
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Interactive Price Adjustment & Revenue Simulator</span>
        </div>
        <div class="content-card-subtitle">Simulate proposed FOB export tariff revisions and project volume/revenue impacts</div>
    </div>
    """, unsafe_allow_html=True)
    
    sim_col_l, sim_col_r = st.columns([5, 7])
    
    with sim_col_l:
        price_adj = st.slider("Proposed Price Revision (%)", min_value=-15.0, max_value=15.0, value=5.0, step=0.5)
        
        base_p = float(reg_df['p_var'].mean())
        base_q = float(reg_df['q_var'].mean())
        
        sim_res = engine.simulate_price_change_impact(
            base_price=base_p,
            base_volume=base_q,
            elasticity_coef=beta,
            price_change_pct=price_adj
        )

    with sim_col_r:
        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            st.metric(
                label="Simulated Unit Price",
                value=f"₹{sim_res['new_price']:.1f}/kg",
                delta=f"{price_adj:+.1f}%"
            )
        with sim_c2:
            st.metric(
                label="Volume / month",
                value=f"{sim_res['new_volume_kg']/1e3:.1f}K kg",
                delta=f"{sim_res['volume_change_pct']:+.1f}%"
            )
        with sim_c3:
            st.metric(
                label="Revenue / month",
                value=f"₹{sim_res['new_revenue_inr']/1e5:.1f}L" if sim_res['new_revenue_inr'] < 1e7 else f"₹{sim_res['new_revenue_inr']/1e7:.2f} Cr",
                delta=f"{sim_res['revenue_change_pct']:+.1f}%"
            )

    st.caption(f"* SKU Lifetime Export Total: **{format_volume_kg(sku_row['total_volume_kg'])}** ({format_currency_inr(sku_row['total_revenue_inr'])}) | Baseline monthly volume: **{base_q/1e3:.1f}K kg**")

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #000000; border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 1rem;">
        <div style="font-weight: 700; color: #111827; font-size: 1rem;">Strategic FMCG Pricing Recommendation</div>
        <div style="font-size: 0.85rem; color: #4B5563; margin-top: 0.35rem;">
            <b>Market Verdict:</b> {e_class}<br>
            <b>Guidance:</b> {pricing_rec}
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STRICT ECONOMETRIC DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="graph-note-box" style="border-left-color: #D97706; background: #FFFBEB;">
    <div class="graph-note-title" style="color: #92400E;">⚠️ Econometric &amp; Observational Data Disclaimer:</div>
    <div style="color: #78350F; font-size: 0.82rem;">
        The calculated price elasticity coefficients are derived from <b>observational transactional sales exhaust</b> using Ordinary Least Squares (OLS) with festive dummy controls. 
        This is an <i>associational</i> econometric model, <b>not a randomized causal A/B experiment</b>. International export demand is subject to endogeneity: 
        export prices and order volumes shift simultaneously due to international container freight shocks, exchange rate movements (USD/INR), 
        and long-term FOB distributor master agreements.
    </div>
</div>
""", unsafe_allow_html=True)
