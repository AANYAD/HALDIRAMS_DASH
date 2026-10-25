"""
================================================================================
Haldiram's International Export Analytics Pipeline - Notes & Briefings Page
================================================================================
Author: Aanya
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (Notes V1.jpg UI Reference)
================================================================================
"""

import streamlit as st
from src.utils import apply_notion_crm_style, render_venture_topbar, VENTURE_THEME

st.set_page_config(page_title="Executive Notes | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

render_venture_topbar(
    title="Notes & Executive Briefings",
    subtitle="Strategic FMCG Export Documentation, SAP Field Glossaries & Methodology Notes"
)

# -----------------------------------------------------------------------------
# TOPBAR CONTROLS (Sort & Filter buttons matching Notes V1.jpg)
# -----------------------------------------------------------------------------
col_n1, col_n2, col_n3 = st.columns([6, 3, 3])

with col_n1:
    search_term = st.text_input("Filter notes by topic...", placeholder="e.g. SARIMAX, Elasticity, Mega Accounts, Holdout...")

with col_n2:
    filter_category = st.selectbox("Category Filter", ["All Topics", "Forecasting", "Pricing Strategy", "Segmentation"])

with col_n3:
    sort_order = st.selectbox("Sort Order", ["Latest Updates", "Core Architecture"])

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# NOTES GRID (3 Columns matching Notes V1.jpg)
# -----------------------------------------------------------------------------
notes_data = [
    {
        "badge1": "Methodology", "b1_type": "blue",
        "badge2": "Forecasting", "b2_type": "green",
        "title": "Demand Forecasting Methodology & Benchmark",
        "category": "Forecasting",
        "bullets": [
            "Why monthly grain? Daily dispatch records have stochastic container shipping noise; monthly aggregation stabilizes variance for Master Production Scheduling.",
            "Why 9-month time-based holdout? Evaluates model generalization out-of-sample and prevents temporal look-ahead data leakage.",
            "SARIMAX incorporates pre-festive export shipment flags (Diwali, Eid); XGBoost models non-linear lag-1, lag-2, and rolling quarterly momentum."
        ],
        "author": "Aanya", "date": "24/06/2026"
    },
    {
        "badge1": "Econometrics", "b1_type": "purple",
        "badge2": "Pricing Strategy", "b2_type": "amber",
        "title": "Price Elasticity & Econometric OLS Modeling",
        "category": "Pricing Strategy",
        "bullets": [
            "Log-Log empirical OLS specification: ln(Q_t) = alpha + beta * ln(P_t) + gamma_quarter + delta_festive + epsilon_t.",
            "Classification derived from beta: |beta| < 1 is price-inelastic, |beta| approx 1 is unit elastic, and |beta| > 1 is price-elastic.",
            "Observational caveat: Results derived from transactional sales exhaust are indicative/correlational, not a randomized causal experiment."
        ],
        "author": "Aanya", "date": "08/07/2026"
    },
    {
        "badge1": "Machine Learning", "b1_type": "green",
        "badge2": "Segmentation", "b2_type": "purple",
        "title": "B2B Customer & Distributor Segmentation",
        "category": "Segmentation",
        "bullets": [
            "StandardScaler normalizes monetary revenue, volume (kg), order frequency, AOV, YoY growth %, and Demand CV (sigma/mu).",
            "Dynamically profiles clusters into FMCG personas, designating the cluster with the highest average revenue per account as 'Mega Accounts'.",
            "PCA (2 components) projects latent customer dimensions to visualize cluster separation and guide commercial account playbooks."
        ],
        "author": "Aanya", "date": "22/07/2026"
    }
]

# Filter notes
filtered_notes = [
    n for n in notes_data
    if (filter_category == "All Topics" or n["category"] == filter_category)
    and (not search_term or any(search_term.lower() in str(v).lower() for v in n.values()))
]

# Render 3-column grid
cols = st.columns(3)
for idx, note in enumerate(filtered_notes):
    col = cols[idx % 3]
    with col:
        b1_class = f"pill-{note['b1_type']}"
        b2_class = f"pill-{note['b2_type']}"
        
        bullets_html = "".join([f"<li style='margin-bottom: 8px; line-height: 1.45;'>{b}</li>" for b in note['bullets']])
        
        st.markdown(f"""
        <div class="content-card" style="min-height: 420px; display: flex; flex-direction: column; justify-content: space-between; margin-bottom: 1.25rem; box-sizing: border-box;">
            <div>
                <div style="display: flex; gap: 6px; margin-bottom: 0.6rem;">
                    <span class="venture-pill {b1_class}">{note['badge1']}</span>
                    <span class="venture-pill {b2_class}">{note['badge2']}</span>
                </div>
                <div style="font-size: 0.98rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
                    {note['title']}
                </div>
                <ul style="font-size: 0.8125rem; color: #4B5563; padding-left: 1.1rem; margin: 0;">
                    {bullets_html}
                </ul>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid #F3F4F6; padding-top: 0.6rem; font-size: 0.75rem; color: #9CA3AF;">
                <div style="display: flex; align-items: center; gap: 6px; color: #374151; font-weight: 600;">
                    <span>👤</span>
                    <span>{note['author']}</span>
                </div>
                <div>{note['date']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
