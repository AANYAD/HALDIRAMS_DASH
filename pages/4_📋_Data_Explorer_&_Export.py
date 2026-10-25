"""
================================================================================
Haldiram's International Export Analytics Pipeline - Data Explorer Page
================================================================================
Author: Senior Data Scientist (Technical Lead)
Project: Haldiram's Final-Year Export Sales & ML Analytics
Design System: Venture CRM (People / Companies UI Reference)
================================================================================
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from src.utils import (
    apply_notion_crm_style,
    render_venture_topbar,
    render_graph_note,
    VENTURE_THEME
)

st.set_page_config(page_title="Data Explorer | Haldiram's", page_icon="assets/haldirams_logo.png", layout="wide")
apply_notion_crm_style()

render_venture_topbar(
    title="Data Explorer & Master Tables",
    subtitle="Query, inspect, and export consolidated transactional records and extracted dimensional master tables"
)

def get_file_mtime(filepath: str) -> float:
    p = Path(filepath)
    return p.stat().st_mtime if p.exists() else 0.0

@st.cache_data
def load_all_tables(_mtime: float):
    data_dir = Path("data")
    sales_path = data_dir / "haldirams_sales_consolidated.parquet"
    cust_path = data_dir / "haldirams_customer_master.parquet"
    prod_path = data_dir / "haldirams_product_master.parquet"

    df_sales = pd.read_parquet(sales_path) if sales_path.exists() else pd.DataFrame()
    df_cust = pd.read_parquet(cust_path) if cust_path.exists() else pd.DataFrame()
    df_prod = pd.read_parquet(prod_path) if prod_path.exists() else pd.DataFrame()

    return df_sales, df_cust, df_prod

df_sales, df_cust, df_prod = load_all_tables(get_file_mtime("data/haldirams_sales_consolidated.parquet"))

tab_sales, tab_cust_m, tab_prod_m, tab_schema = st.tabs([
    "📑 Consolidated Sales Transactions", 
    "🏢 Customer Master (173 Accounts)", 
    "📦 Product Master (3,488 SKUs)",
    "⚙️ Schema & Data Quality Audit"
])

with tab_sales:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Filterable Transaction Exhaust (294,898 Records)</span>
        </div>
        <div class="content-card-subtitle">Search by SKU, Distributor, or Fiscal Year</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        available_fys = sorted(df_sales['fiscal_year'].dropna().unique())
        default_fy = [available_fys[-2]] if len(available_fys) >= 2 else available_fys
        sel_fy = st.multiselect("Fiscal Year Filter", options=available_fys, default=default_fy)
    with col_f2:
        available_cntries = sorted(df_sales['country'].dropna().unique())
        sel_cntry = st.multiselect("Country Filter", options=available_cntries)
    with col_f3:
        search_query = st.text_input("Search Material / SKU / Customer Name", placeholder="e.g. Bhujia, Samosa, Anmol...")

    display_sales = df_sales[df_sales['fiscal_year'].isin(sel_fy)] if sel_fy else df_sales
    if sel_cntry:
        display_sales = display_sales[display_sales['country'].isin(sel_cntry)]
    if search_query:
        mask = (
            display_sales['material_desc'].str.contains(search_query, case=False, na=False) |
            display_sales['customer_name'].str.contains(search_query, case=False, na=False) |
            display_sales['material_code'].str.contains(search_query, case=False, na=False)
        )
        display_sales = display_sales[mask]

    st.caption(f"Showing **{len(display_sales):,}** matching transactions (Rendering top 500 records).")
    
    key_cols = [
        'invoice_date', 'fiscal_year', 'odn_no', 'invoice_no', 'customer_name', 
        'country', 'zone', 'material_code', 'material_desc', 'category_name', 
        'quantity_kg', 'amount_inr', 'unit_price_inr_per_kg'
    ]
    available_key_cols = [c for c in key_cols if c in display_sales.columns]
    
    print(f"[DATA EXPLORER] Final displayed columns: {available_key_cols}", flush=True)

    # Format invoice_date as YYYY-MM-DD only
    table_render_df = display_sales[available_key_cols].head(500).copy()
    if 'invoice_date' in table_render_df.columns:
        table_render_df['invoice_date'] = pd.to_datetime(table_render_df['invoice_date']).dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        table_render_df,
        use_container_width=True,
        hide_index=True
    )

    csv_export_df = display_sales[available_key_cols].head(5000).copy()
    if 'invoice_date' in csv_export_df.columns:
        csv_export_df['invoice_date'] = pd.to_datetime(csv_export_df['invoice_date']).dt.strftime('%Y-%m-%d')
    csv_data = csv_export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="haldirams_sales_export_sample.csv",
        mime="text/csv"
    )

    render_graph_note(
        title="Harmonized Transaction Registry",
        explanation="Standardizes heterogeneous invoice records from 5 fiscal years, aligning currency fields, quantities in KG, and SAP product hierarchy codes into a unified analytical structure.",
        viva_point="Data harmonization eliminates schema drift across multi-year SAP table exports without losing line-item granularity."
    )

with tab_cust_m:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Dimensional Customer Master Record</span>
        </div>
        <div class="content-card-subtitle">Lifetime metrics for all 173 international B2B distributors</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_cust, use_container_width=True, hide_index=True)

with tab_prod_m:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>Dimensional Product Master Record</span>
        </div>
        <div class="content-card-subtitle">Aggregated metrics across 3,488 export materials and SKUs</div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_prod, use_container_width=True, hide_index=True)

with tab_schema:
    st.markdown("""
    <div class="content-card">
        <div class="content-card-title">
            <span>5-Year Schema Audit & Attribute Dictionary</span>
        </div>
        <div class="content-card-subtitle">Audit of data types, completeness, and non-null counts</div>
    </div>
    """, unsafe_allow_html=True)
    schema_info = pd.DataFrame({
        'Column Name': df_sales.columns,
        'Data Type': [str(dtype) for dtype in df_sales.dtypes],
        'Non-Null Count': [f"{count:,}" for count in df_sales.count()],
        'Sample Value': [str(df_sales[col].dropna().iloc[0]) if not df_sales[col].dropna().empty else 'None' for col in df_sales.columns]
    })
    st.dataframe(schema_info, use_container_width=True, hide_index=True)
