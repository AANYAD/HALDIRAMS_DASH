"""
================================================================================
Haldiram's International Export Analytics — UI & Visualization Utilities
================================================================================
Design System: Pixel-accurate Venture CRM (Notion-style) matching reference images.
Font: Inter (Google Fonts). Icons: Lucide SVG (python-lucide).
================================================================================
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from lucide import lucide_icon


# ── Brand Identity & Palette ────────────────────────────────────────────────
BRAND_RED = '#FE0002'  # Dominant red sampled directly from Haldiram's logo image

# ── Design Tokens (monochrome Venture CRM palette) ───────────────────────────
VENTURE_THEME = {
    'brand_red':        BRAND_RED,
    'brand_red_light':  '#FEF2F2',
    'bg_app':           '#FAFAFA',
    'bg_card':          '#FFFFFF',
    'bg_sidebar':       '#FBFBFB',
    'border_subtle':    '#E5E7EB',
    'border_light':     '#F3F4F6',
    'text_main':        '#111827',
    'text_secondary':   '#6B7280',
    'text_muted':       '#9CA3AF',
    'primary_black':    '#000000',
    'accent_dark':      '#18181B',
    'gray_bar':         '#E4E4E7',
    'badge_green_bg':   '#ECFDF5',
    'badge_green_text': '#059669',
    'badge_blue_bg':    '#EFF6FF',
    'badge_blue_text':  '#2563EB',
    'badge_purple_bg':  '#F5F3FF',
    'badge_purple_text':'#7C3AED',
    'badge_amber_bg':   '#FEF2F2',
    'badge_amber_text': BRAND_RED,
    'badge_rose_bg':    '#FFF1F2',
    'badge_rose_text':  '#E11D48',
}

# ── Dynamic Theme & Iconography Palettes ─────────────────────────────────────
METRIC_THEMES = {
    'green': {
        'bg': '#ECFDF5',
        'border': '#A7F3D0',
        'icon': '#059669',
        'pill_bg': '#ECFDF5',
        'pill_text': '#059669',
        'glow': 'rgba(5, 150, 105, 0.25)',
    },
    'blue': {
        'bg': '#EFF6FF',
        'border': '#BFDBFE',
        'icon': '#2563EB',
        'pill_bg': '#EFF6FF',
        'pill_text': '#2563EB',
        'glow': 'rgba(37, 99, 235, 0.25)',
    },
    'amber': {
        'bg': '#FEF2F2',
        'border': '#FECACA',
        'icon': BRAND_RED,
        'pill_bg': '#FEF2F2',
        'pill_text': BRAND_RED,
        'glow': 'rgba(254, 0, 2, 0.25)',
    },
    'purple': {
        'bg': '#F5F3FF',
        'border': '#DDD6FE',
        'icon': '#7C3AED',
        'pill_bg': '#F5F3FF',
        'pill_text': '#7C3AED',
        'glow': 'rgba(124, 58, 237, 0.25)',
    },
    'rose': {
        'bg': '#FFF1F2',
        'border': '#FECDD3',
        'icon': '#E11D48',
        'pill_bg': '#FFF1F2',
        'pill_text': '#E11D48',
        'glow': 'rgba(225, 29, 72, 0.25)',
    },
    'gray': {
        'bg': '#F8FAFC',
        'border': '#E2E8F0',
        'icon': '#1E293B',
        'pill_bg': '#F3F4F6',
        'pill_text': '#4B5563',
        'glow': 'rgba(100, 116, 139, 0.2)',
    }
}


def lc(name: str, size: int = 16, color: str = "#111827", stroke_width: float = 1.8) -> str:
    """
    Returns a Lucide icon as an inline SVG string.
    Usage: st.markdown(f"<span>{lc('bar-chart')}</span>", unsafe_allow_html=True)
    """
    return lucide_icon(
        name,
        width=size,
        height=size,
        stroke=color,
        stroke_width=stroke_width,
        fill="none",
    )


def apply_notion_crm_style():
    """
    Injects the full Venture CRM / Notion design CSS:
    - Inter font from Google Fonts
    - Clean white sidebar with lucide-icon nav links
    - KPI metric cards (icon box + chevron)
    - Pill badges, agenda items, graph note boxes
    - Streamlit element overrides
    """
    T = VENTURE_THEME
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Global reset ─────────────────────────────────────────── */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            font-size: 14.5px !important;
            color: {T['text_main']} !important;
            background-color: {T['bg_app']} !important;
            -webkit-font-smoothing: antialiased;
        }}

        header[data-testid="stHeader"] {{
            background-color: transparent !important;
            height: 1.5rem !important;
            min-height: 1.5rem !important;
        }}

        /* ── Main content area ────────────────────────────────────── */
        .main .block-container {{
            padding-top: 0.75rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1440px !important;
        }}

        /* ── Sidebar ──────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background-color: {T['bg_sidebar']} !important;
            border-right: 1px solid {T['border_subtle']} !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding: 1.25rem 0.875rem !important;
        }}

        /* ── Sidebar Logo & Header ────────────────────────────────── */
        [data-testid="stSidebarHeader"] {{
            padding-top: 18px !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            background-color: transparent !important;
            height: auto !important;
            min-height: auto !important;
            overflow: visible !important;
        }}
        [data-testid="stLogo"], [data-testid="stSidebarHeader"] img {{
            width: 130px !important;
            max-width: 130px !important;
            height: auto !important;
            display: block !important;
            margin: 0 !important;
            padding-top: 18px !important;
            overflow: visible !important;
        }}

        /* Sidebar Navigation items override */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {{
            content: "HALDIRAM'S EXPORT OPERATIONS";
            display: block;
            font-size: 0.7rem;
            font-weight: 800;
            color: {BRAND_RED};
            letter-spacing: 0.08em;
            padding: 0.6rem 0.75rem 0.5rem;
            border-bottom: 1px solid #E5E7EB;
            margin-bottom: 0.65rem;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            color: #374151 !important;
            border-radius: 8px !important;
            transition: all 0.15s ease !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 0.45rem 0.75rem !important;
            margin-bottom: 2px !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
            background-color: #F3F4F6 !important;
            color: #111827 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-weight: 700 !important;
            border-left: 3px solid {BRAND_RED} !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{
            color: inherit !important;
        }}
        /* Rename default 'app' in sidebar to 'Executive Command Center' */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child a span:last-child {{
            font-size: 0 !important;
            visibility: hidden;
            position: relative;
            display: inline-block;
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:first-child a span:last-child::after {{
            font-size: 0.875rem !important;
            visibility: visible;
            content: "Executive Command Center";
            font-weight: inherit;
            color: inherit;
            white-space: nowrap;
        }}

        /* ── Top-bar & Live Search Input ──────────────────────────── */
        div[data-testid="stTextInput"] {{
            margin-bottom: 0 !important;
        }}
        div[data-testid="stTextInput"] input {{
            border-radius: 9px !important;
            border: 1px solid #E5E7EB !important;
            background-color: #FFFFFF !important;
            font-size: 0.835rem !important;
            color: #111827 !important;
            padding: 7px 14px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }}
        div[data-testid="stTextInput"] input:focus {{
            border-color: {BRAND_RED} !important;
            box-shadow: 0 0 0 3px rgba(254, 0, 2, 0.15) !important;
        }}
        .topbar-right-badge {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #111827;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 6px 12px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            white-space: nowrap;
        }}

        /* ── Metric KPI Cards ─────────────────────────────────────── */
        .metric-card-venture {{
            background-color: #FFFFFF;
            border: 1px solid {T['border_subtle']};
            border-radius: 12px;
            padding: 1.25rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
            position: relative;
            min-height: 168px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }}
        .metric-card-venture:hover {{
            border-color: #CBD5E1;
            box-shadow: 0 6px 16px -2px rgba(0,0,0,0.06);
            transform: translateY(-1px);
        }}
        .metric-card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.875rem;
        }}
        .metric-icon-box {{
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #E2E8F0;
            transition: transform 0.15s ease;
        }}
        .metric-card-venture:hover .metric-icon-box {{
            transform: scale(1.05);
        }}
        .metric-icon-box svg {{ display: block; }}
        .card-status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8);
        }}
        .metric-label {{
            font-size: 0.875rem !important;
            color: {T['text_secondary']};
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        .metric-value {{
            font-size: 2.0rem !important;
            font-weight: 700 !important;
            color: {T['text_main']};
            line-height: 1.2;
            letter-spacing: -0.025em;
            margin-bottom: 0.35rem;
        }}
        .metric-value-compact {{
            font-size: 1.08rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            white-space: nowrap !important;
            line-height: 1.3 !important;
        }}
        .metric-subtext {{
            font-size: 0.8125rem !important;
            color: {T['text_muted']};
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}

        /* ── Content section cards ───────────────────────────────── */
        .content-card {{
            background-color: #FFFFFF;
            border: 1px solid {T['border_subtle']};
            border-radius: 12px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);
        }}
        .content-card-title {{
            font-size: 1.05rem !important;
            font-weight: 700;
            color: {T['text_main']};
            margin-bottom: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .content-card-subtitle {{
            font-size: 0.85rem !important;
            color: {T['text_secondary']};
            margin-bottom: 1rem;
        }}

        /* Section header with lucide icon (Analytics.jpg style) */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #374151;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }}
        .section-header svg {{ flex-shrink: 0; }}

        /* ── Pill Badges ─────────────────────────────────────────── */
        .venture-pill {{
            display: inline-flex;
            align-items: center;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 600;
            line-height: 1.5;
            white-space: nowrap;
        }}
        .pill-dot {{
            display: inline-block;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background-color: currentColor;
            margin-right: 5px;
        }}
        .pill-green  {{ background-color: {T['badge_green_bg']};  color: {T['badge_green_text']}; }}
        .pill-blue   {{ background-color: {T['badge_blue_bg']};   color: {T['badge_blue_text']}; }}
        .pill-purple {{ background-color: {T['badge_purple_bg']}; color: {T['badge_purple_text']}; }}
        .pill-amber  {{ background-color: {T['badge_amber_bg']};  color: {T['badge_amber_text']}; }}
        .pill-rose   {{ background-color: {T['badge_rose_bg']};   color: {T['badge_rose_text']}; }}
        .pill-gray   {{ background-color: #F3F4F6; color: #4B5563; }}
        .pill-black  {{ background-color: #111827; color: #FFFFFF; }}

        /* ── Agenda / Timeline items ─────────────────────────────── */
        .agenda-item {{
            padding: 0.65rem 0;
            border-bottom: 1px solid {T['border_light']};
        }}
        .agenda-item:last-child {{ border-bottom: none; }}
        .agenda-time {{
            font-size: 0.72rem;
            font-weight: 600;
            color: #2563EB;
            background: #EFF6FF;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 0.25rem;
        }}
        .agenda-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #111827;
        }}
        .agenda-desc {{
            font-size: 0.78rem;
            color: #6B7280;
            margin-top: 2px;
            line-height: 1.45;
        }}

        /* ── Graph Note Boxes ────────────────────────────────────── */
        .graph-note-box {{
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #000000;
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            font-size: 0.82rem;
            color: #334155;
            margin-top: 0.85rem;
            line-height: 1.55;
        }}
        .graph-note-title {{
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85rem;
        }}

        /* ── Notes cards (Notes V1.jpg) ──────────────────────────── */
        .notes-card {{
            background: #FFFFFF;
            border: 1px solid {T['border_subtle']};
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            transition: box-shadow 0.18s;
        }}
        .notes-card:hover {{
            box-shadow: 0 4px 12px -2px rgba(0,0,0,0.08);
        }}
        .notes-card-pills {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-bottom: 0.6rem;
        }}
        .notes-card-title {{
            font-size: 0.925rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.45rem;
        }}
        .notes-card-body {{
            font-size: 0.8rem;
            color: #374151;
            line-height: 1.55;
            margin-bottom: 0.75rem;
        }}
        .notes-card-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.75rem;
            color: #9CA3AF;
            border-top: 1px solid #F3F4F6;
            padding-top: 0.5rem;
        }}
        .notes-card-author {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
            color: #6B7280;
        }}
        .author-avatar {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #E5E7EB;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.65rem;
            font-weight: 700;
            color: #374151;
        }}

        /* ── Horizontal progress bars (Analytics.jpg) ────────────── */
        .progress-label-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #374151;
            margin-bottom: 4px;
            font-weight: 500;
        }}
        .progress-bg {{
            background-color: #E5E7EB;
            border-radius: 9999px;
            height: 7px;
            width: 100%;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        .progress-fill {{
            background-color: #000000;
            height: 100%;
            border-radius: 9999px;
        }}

        /* ── Streamlit widget overrides ──────────────────────────── */
        div[data-testid="stMetricValue"] {{
            font-size: 2.0rem !important;
            font-weight: 700 !important;
        }}
        .stSelectbox label, .stSlider label, .stMultiSelect label {{
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            color: #374151 !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.5rem;
            border-bottom: 1px solid #E5E7EB;
            background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 0.5rem 0 0.75rem;
            font-weight: 600;
            font-size: 0.875rem;
            color: #6B7280;
            background: transparent;
            border-bottom: 2px solid transparent;
        }}
        .stTabs [aria-selected="true"] {{
            color: #000000 !important;
            border-bottom-color: #000000 !important;
        }}
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #E5E7EB;
            font-size: 13.5px !important;
        }}
        div[data-testid="stDataFrame"] *, div[data-testid="stTable"] * {{
            font-size: 13.5px !important;
        }}
        .stExpander {{
            border: 1px solid #E5E7EB !important;
            border-radius: 10px !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    logo_path = Path("assets/haldirams_logo.png")
    if logo_path.exists():
        try:
            st.logo(str(logo_path), icon_image=str(logo_path))
        except Exception:
            pass

    render_sidebar_brand()


# ── Sidebar brand ─────────────────────────────────────────────────────────────

def render_sidebar_brand():
    """Renders a small muted text line at the bottom of the sidebar."""
    st.sidebar.markdown("""
    <div style="padding: 0.5rem 0.75rem 0.25rem; margin-top: 0.5rem; font-size: 0.75rem; color: #4B5563; font-weight: 600; letter-spacing: 0.01em;">
        International Export | Analytics Dashboard
        <div style="font-size: 0.7rem; color: #9CA3AF; margin-top: 3px; font-weight: 400;">
            Internal use | Confidential
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav_label(label: str):
    """Renders a small uppercase section divider label in sidebar."""
    st.markdown(f"""
    <div style="font-size:0.68rem;font-weight:700;color:#9CA3AF;letter-spacing:0.08em;
                text-transform:uppercase;margin:1rem 0 0.35rem;padding-top:0.25rem;
                border-top:1px solid #F3F4F6;">
        {label}
    </div>
    """, unsafe_allow_html=True)


# ── Global Master Index Caching & Search ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_search_master_indices():
    """Loads lightweight master tables for global fast lookup."""
    prod_path = Path("data/haldirams_product_master.parquet")
    cust_path = Path("data/haldirams_customer_master.parquet")
    prod = pd.read_parquet(prod_path) if prod_path.exists() else pd.DataFrame()
    cust = pd.read_parquet(cust_path) if cust_path.exists() else pd.DataFrame()
    return prod, cust


def _render_search_results_drawer(query: str):
    """Renders a market-ready search results drawer directly below the topbar."""
    prod_df, cust_df = _load_search_master_indices()
    q = query.strip().lower()

    # Search SKUs
    if not prod_df.empty:
        sku_matches = prod_df[
            prod_df['material_desc'].str.lower().str.contains(q, na=False) |
            prod_df['category_name'].str.lower().str.contains(q, na=False) |
            prod_df['material_code'].astype(str).str.lower().str.contains(q, na=False)
        ]
    else:
        sku_matches = pd.DataFrame()

    # Search Distributors
    if not cust_df.empty:
        cust_matches = cust_df[
            cust_df['customer_name'].str.lower().str.contains(q, na=False) |
            cust_df['country'].str.lower().str.contains(q, na=False) |
            cust_df['zone'].str.lower().str.contains(q, na=False)
        ]
    else:
        cust_matches = pd.DataFrame()

    n_skus = len(sku_matches)
    n_cust = len(cust_matches)

    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-left:4px solid {BRAND_RED}; border-radius:10px; padding:1.1rem 1.4rem; margin-bottom:1.5rem; box-shadow:0 4px 14px rgba(0,0,0,0.06);">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-weight:700; font-size:0.95rem; color:#111827;">
                    Global Search: <span style="color:{BRAND_RED};">"{query}"</span>
                </span>
                <span class="venture-pill pill-amber"><span class="pill-dot"></span>{n_skus:,} Matching SKUs</span>
                <span class="venture-pill pill-blue"><span class="pill-dot"></span>{n_cust:,} Distributors</span>
            </div>
            <div style="font-size:0.75rem; color:#6B7280;">Clear the search box above to close this view</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if n_skus == 0 and n_cust == 0:
        st.info(f"No SKUs or distributors found matching '{query}'. Try searching for terms like 'Bhujia', 'Samosa', 'Gulab', 'UAE', 'USA', or 'Frozen'.")
    else:
        tab_sku, tab_dist = st.tabs([f"📦 Matching Products & SKUs ({n_skus:,})", f"🏢 Matching International Distributors ({n_cust:,})"])

        with tab_sku:
            if n_skus > 0:
                cols_to_show = ['material_code', 'material_desc', 'category_name', 'is_frozen', 'avg_unit_price_inr_per_kg', 'total_volume_kg']
                avail_cols = [c for c in cols_to_show if c in sku_matches.columns]
                display_skus = sku_matches[avail_cols].head(20).copy()
                rename_map = {
                    'material_code': 'Material Code',
                    'material_desc': 'Description / SKU',
                    'category_name': 'Category',
                    'is_frozen': 'Cold Chain',
                    'avg_unit_price_inr_per_kg': 'Avg Realized Price (₹/kg)',
                    'total_volume_kg': 'Export Volume (kg)'
                }
                display_skus = display_skus.rename(columns=rename_map)
                if 'Avg Realized Price (₹/kg)' in display_skus.columns:
                    display_skus['Avg Realized Price (₹/kg)'] = display_skus['Avg Realized Price (₹/kg)'].apply(
                        lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "—"
                    )
                if 'Export Volume (kg)' in display_skus.columns:
                    display_skus['Export Volume (kg)'] = display_skus['Export Volume (kg)'].apply(
                        lambda x: f"{x:,.0f} kg" if pd.notnull(x) else "—"
                    )
                st.dataframe(display_skus, use_container_width=True, hide_index=True)
                if n_skus > 20:
                    st.caption(f"Showing top 20 of {n_skus:,} matching products.")
            else:
                st.write("No product matches found.")

        with tab_dist:
            if n_cust > 0:
                cols_to_show = ['customer_id', 'customer_name', 'country', 'zone', 'total_revenue_inr', 'total_volume_kg', 'total_orders']
                avail_cols = [c for c in cols_to_show if c in cust_matches.columns]
                display_cust = cust_matches[avail_cols].head(20).copy()
                rename_map = {
                    'customer_id': 'Customer ID',
                    'customer_name': 'Distributor Account',
                    'country': 'Country',
                    'zone': 'Export Zone',
                    'total_revenue_inr': 'Total Revenue (INR)',
                    'total_volume_kg': 'Export Volume (kg)',
                    'total_orders': 'Total Invoices'
                }
                display_cust = display_cust.rename(columns=rename_map)
                if 'Total Revenue (INR)' in display_cust.columns:
                    display_cust['Total Revenue (INR)'] = display_cust['Total Revenue (INR)'].apply(
                        lambda x: f"₹{x:,.0f}" if pd.notnull(x) else "—"
                    )
                if 'Export Volume (kg)' in display_cust.columns:
                    display_cust['Export Volume (kg)'] = display_cust['Export Volume (kg)'].apply(
                        lambda x: f"{x:,.0f} kg" if pd.notnull(x) else "—"
                    )
                st.dataframe(display_cust, use_container_width=True, hide_index=True)
                if n_cust > 20:
                    st.caption(f"Showing top 20 of {n_cust:,} matching distributors.")
            else:
                st.write("No distributor matches found.")


# ── Top bar ───────────────────────────────────────────────────────────────────

def render_venture_topbar(title: str = "Dashboard",
                           subtitle: str = "Haldiram's International Export Operations"):
    """
    Renders an executive topbar with title and subtitle.
    """
    st.markdown(f"""
    <div style="padding: 0;">
        <h2 style="margin:0;font-size:1.6rem;font-weight:800;color:#111827;letter-spacing:-0.03em;">{title}</h2>
        <div style="font-size:0.9rem;color:#6B7280;margin-top:2px;">{subtitle}</div>
    </div>
    <div style="border-bottom: 1px solid #E5E7EB; margin-top: 0.5rem; margin-bottom: 1rem;"></div>
    """, unsafe_allow_html=True)


# ── Metric KPI card ───────────────────────────────────────────────────────────

def render_metric_card(icon_name: str, label: str, value: str, subtext: str,
                        badge_text: str = "", badge_type: str = "green"):
    """
    Renders a Venture CRM-style KPI metric card with Lucide icon box + status glow.
    icon_name: lucide icon name e.g. 'package', 'trending-up', 'globe', 'receipt'
    """
    theme = METRIC_THEMES.get(badge_type, METRIC_THEMES['gray'])
    pill_class  = f"pill-{badge_type}"
    badge_html  = f'<span class="venture-pill {pill_class}"><span class="pill-dot"></span>{badge_text}</span>' if badge_text else ''
    icon_svg    = lc(icon_name, 19, theme['icon'], 2.0)

    val_str = str(value)
    val_class = "metric-value metric-value-compact" if len(val_str) > 13 else "metric-value"

    st.markdown(f"""
    <div class="metric-card-venture">
        <div class="metric-card-header">
            <div class="metric-icon-box" style="background-color: {theme['bg']}; border-color: {theme['border']};">
                {icon_svg}
            </div>
            <div class="card-status-dot" style="background-color: {theme['icon']}; box-shadow: 0 0 8px {theme['glow']};"></div>
        </div>
        <div>
            <div class="metric-label">{label}</div>
            <div class="{val_class}">{value}</div>
        </div>
        <div class="metric-subtext">{badge_html}<span>{subtext}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Section header with icon ──────────────────────────────────────────────────

def render_section_header(icon_name: str, title: str, color: str = "#374151"):
    """Renders an Analytics.jpg-style section header with lucide icon."""
    icon_svg = lc(icon_name, 14, color, 2.0)
    st.markdown(f"""
    <div class="section-header">
        {icon_svg}
        <span style="color:{color};">{title}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Graph note box ────────────────────────────────────────────────────────────

def render_graph_note(title: str, explanation: str, technical_note: Optional[str] = None, **kwargs):
    """Renders an explanatory note box below every chart."""
    pin_icon  = lc("map-pin", 15, "#0F172A", 2.0)
    st.markdown(f"""
    <div class="graph-note-box">
        <div class="graph-note-title">{pin_icon}<span>Graph Interpretation &amp; Decision Context: {title}</span></div>
        <div>{explanation}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Plotly theme dict ─────────────────────────────────────────────────────────

def get_plotly_layout_dict() -> Dict[str, Any]:
    """Returns Plotly layout dict matching the Venture monochrome chart aesthetic."""
    return dict(
        font=dict(family='Inter, -apple-system, sans-serif', color='#111827', size=11),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        margin=dict(l=35, r=20, t=30, b=30),
        xaxis=dict(
            gridcolor='#F3F4F6', showline=True, linecolor='#E5E7EB',
            zeroline=False, showgrid=True,
            tickfont=dict(size=10, color='#6B7280')
        ),
        yaxis=dict(
            gridcolor='#F3F4F6', showline=True, linecolor='#E5E7EB',
            zeroline=False, showgrid=True,
            tickfont=dict(size=10, color='#6B7280')
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor='rgba(255,255,255,0.85)',
            font=dict(size=10)
        )
    )


# ── Currency & volume formatters ──────────────────────────────────────────────

def format_currency_inr(val: float, compact: bool = True) -> str:
    """Formats values in Indian Currency (Crores / Lakhs)."""
    if compact:
        if val >= 1e7:   return f"₹{val/1e7:.2f} Cr"
        elif val >= 1e5: return f"₹{val/1e5:.2f} L"
        elif val >= 1e3: return f"₹{val/1e3:.1f} K"
        else:            return f"₹{val:,.0f}"
    return f"₹{val:,.2f}"


def format_volume_kg(val: float, compact: bool = True) -> str:
    """Formats volume in KG / Metric Tonnes."""
    if compact:
        if val >= 1e6:   return f"{val/1e6:.2f}M Kg ({val/1e3:,.0f} MT)"
        elif val >= 1e3: return f"{val/1e3:.1f}K Kg ({val/1e3:.1f} MT)"
        return f"{val:,.0f} Kg"
    return f"{val:,.1f} Kg"

