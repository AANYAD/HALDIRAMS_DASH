"""
================================================================================
Haldiram's International Export Analytics Pipeline - Data Engineering Module
================================================================================
Author: Senior Data Scientist (Technical Lead)
Project: Haldiram's Final-Year Export Sales & ML Analytics
Module: Data Ingestion, Schema Drift Resolution, Harmonization & Master Extraction

Viva Defense Notes:
- Dynamic Ingestion: Uses pathlib to scan directory trees for multi-year files.
- Memory-Efficient Markdown Streaming: Parses line-by-line markdown tables 
  instead of loading entire multi-megabyte string structures into RAM.
- Schema Drift Management: Employs a canonical schema mapping dictionary to 
  unify varying column nomenclatures across fiscal years (FY21-22 to FY25-26).
- Standard FMCG Metric Derivation: Computes realized export prices (INR/Kg),
  freight/insurance margin proxies, and standardizes currency to Local Currency (INR).
================================================================================
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


# Canonical schema dictionary mapping raw column names across various fiscal years
# to standardized, snake_case enterprise attributes.
CANONICAL_COLUMN_MAPPING: Dict[str, str] = {
    # Reference and Document keys
    'ODN No. - Reference Field': 'odn_no',
    'Invoice Number': 'invoice_no',
    'Invoice Date': 'invoice_date',
    'Document Date': 'document_date',
    'Sales Doc./Purchase Order': 'sales_order_no',
    'Sales Organization': 'sales_org',
    'Supplying Plant': 'supplying_plant',

    # Customer Attributes
    'Sold-To Party': 'customer_id',
    'Sold-to Party': 'customer_id',
    'Sold to Party Name': 'customer_name',
    'Ship-to party': 'ship_to_id',
    'Ship-to Party': 'ship_to_id',
    'Ship to Party Name': 'ship_to_name',

    # Geographic / Organizational Attributes
    'ZONE': 'zone',
    'Zone': 'zone',
    'Sales District': 'zone',
    'Country': 'country',
    'country': 'country',
    'Country Name': 'country',
    'Sales Group': 'sales_group',
    'Sales group description': 'sales_group_desc',

    # Product and Hierarchy Attributes
    'Material': 'material_code',
    'Material description': 'material_desc',
    'Material Description': 'material_desc',
    'Type of Material': 'material_type',
    'PH-1+PH-2': 'category_code',
    'PH1+PH2': 'category_code',
    'Product Hierarchy1': 'product_hierarchy_1',
    'Production Cat. Group': 'production_cat_group',
    'Design Code-L3': 'design_code_l3',
    'Design Code L-3': 'design_code_l3',
    'L2 - Description': 'category_desc',
    'L3 - Description': 'sub_category_desc',
    'Design Code-L4': 'design_code_l4',
    'Design Code L-4': 'design_code_l4',
    'L4 - Description': 'variant_desc',
    'Design Code': 'design_code',
    'Description 1': 'desc_1_market',
    'Description 2': 'desc_2_category',
    'Description 3': 'desc_3_subcategory',
    'Description 4': 'desc_4_type',
    'Description 5': 'desc_5_variant',
    'Description 6': 'desc_6_size_class',
    'Description 7': 'desc_7_pack_detail',

    # Volume and Quantity Attributes
    'Actual billed quantity': 'quantity_billed_cases',
    'Actual billed quantity in KG': 'quantity_kg',
    'Sales unit': 'sales_unit',
    'Sales Unit': 'sales_unit',
    'Alt. UOM': 'alt_uom',
    'Net Weight': 'net_weight',
    'Gross Weight': 'gross_weight',

    # Financial and Cost Attributes
    'Net Amount': 'net_amount_foreign',
    'Net Amount INR': 'net_amount_inr',
    'Gross amount': 'gross_amount_foreign',
    'Gross amount INR': 'gross_amount_inr',
    'Transaction Currency': 'currency',
    'Exchange Rate': 'exchange_rate',
    'Amount in LC(INR)': 'amount_inr',
    'ZBAS Export-Basic Value': 'basic_export_value',
    'ZDIS Discount (Exp)': 'discount_exp',
    'ZFRT Freight (Export)': 'freight_exp',
    'ZINS Insurance (Export)': 'insurance_exp',
    'ZK01 Export Discount-Amt': 'discount_k01',
    'ZPAC Packing Charges -EXP': 'packing_charges_exp',
    'ZPR8 Export Price': 'zpr8_price',
    'PR8 PRICE': 'pr8_price_alt',
    'FZN Purchase Price': 'fzn_purchase_price',
    'FZN Purchase Amount': 'fzn_purchase_amount',
    'FZN Sale Amount': 'fzn_sale_amount',
    'F.Y.': 'fiscal_year',
    'MONTH': 'month_str',
}


def clean_numeric(val: Union[str, float, int, None]) -> float:
    """
    Cleans and standardizes messy numeric strings containing commas, spaces, or currency symbols.
    
    Why: FMCG SAP export logs often store formatted strings like '1,088.64' or empty fields,
    which lead to parsing crashes in vanilla numeric converters.
    """
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else 0.0
    
    val_str = str(val).strip()
    if not val_str or val_str in ('-', 'NA', 'None', 'null'):
        return 0.0
    # Remove commas and clean
    cleaned = val_str.replace(',', '').replace(' ', '').replace('$', '').replace('£', '')
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_markdown_table_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    High-performance streaming parser for Haldiram's large Markdown table files.
    
    Why: Standard markdown-to-html parsers load entire 30MB strings into DOM trees.
    Streaming line-by-line splits on '|' guarantees O(1) memory complexity during line reads.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Markdown data file not found at: {file_path}")

    header_cols: Optional[List[str]] = None
    rows: List[List[str]] = []
    
    with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            # Identify Markdown table rows
            if not stripped.startswith('|'):
                continue
            
            # Split cells by pipe and trim
            raw_cells = [cell.strip() for cell in stripped.split('|')]
            # Markdown table lines start and end with '|', resulting in empty first & last items
            if len(raw_cells) >= 3:
                cells = raw_cells[1:-1]
            else:
                continue
            
            # Skip divider lines (e.g., | --- | --- | or |:---|---:|)
            if all(set(c).issubset({'-', ':', ' '}) for c in cells if c):
                continue
            
            if header_cols is None:
                # Capture header row
                header_cols = cells
            else:
                # Append data row
                if len(cells) == len(header_cols):
                    rows.append(cells)
                elif len(cells) < len(header_cols):
                    # Pad missing columns with empty string
                    rows.append(cells + [''] * (len(header_cols) - len(cells)))
                else:
                    # Truncate if spurious extra pipes exist
                    rows.append(cells[:len(header_cols)])

    if not header_cols:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=header_cols)
    df['source_file'] = path_obj.name
    return df


class HaldiramsDataPipeline:
    """
    Enterprise ETL & Data Harmonization Pipeline for Haldiram's International Export Data.
    """

    def __init__(self, raw_data_dir: Union[str, Path], processed_data_dir: Union[str, Path] = "data"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.consolidated_df: Optional[pd.DataFrame] = None
        self.customer_master: Optional[pd.DataFrame] = None
        self.product_master: Optional[pd.DataFrame] = None

    def load_and_harmonize_all_years(self) -> pd.DataFrame:
        """
        Iterates across all yearly Markdown sales registers, aligns schema drifts,
        standardizes date/numeric fields, and consolidates into a single analytic dataset.
        """
        md_files = sorted(list(self.raw_data_dir.glob("*.md")))
        if not md_files:
            raise FileNotFoundError(f"No markdown files found in directory: {self.raw_data_dir}")

        harmonized_dfs = []
        for file_path in md_files:
            print(f"[ETL Ingestion] Loading and parsing {file_path.name}...")
            raw_df = parse_markdown_table_file(file_path)
            print(f"  -> Extracted {len(raw_df):,} records with {len(raw_df.columns)} raw attributes.")
            
            # Map columns to canonical schema
            renamed_df = raw_df.rename(columns=CANONICAL_COLUMN_MAPPING)
            
            # Infer or standardize Fiscal Year from filename or existing column
            if 'fiscal_year' not in renamed_df.columns or renamed_df['fiscal_year'].str.strip().eq('').all():
                if '2021-2022' in file_path.name or '21-22' in file_path.name and '22-23' not in file_path.name:
                    renamed_df['fiscal_year'] = '2021-2022'
                elif '2022-2023' in file_path.name or '22-23' in file_path.name:
                    renamed_df['fiscal_year'] = '2022-2023'
                elif '2023-2024' in file_path.name or '23-24' in file_path.name:
                    renamed_df['fiscal_year'] = '2023-2024'
                elif '2024-2025' in file_path.name or '24-25' in file_path.name:
                    renamed_df['fiscal_year'] = '2024-2025'
                elif '25-26' in file_path.name or '2025-2026' in file_path.name:
                    renamed_df['fiscal_year'] = '2025-2026'
                else:
                    renamed_df['fiscal_year'] = 'Unknown'
            
            harmonized_dfs.append(renamed_df)

        print("[ETL Harmonization] Concatenating yearly dataframes...")
        consolidated = pd.concat(harmonized_dfs, ignore_index=True, sort=False)
        print(f"[ETL Harmonization] Raw Consolidated Rows: {len(consolidated):,}")

        # Post-Processing & Normalization
        self.consolidated_df = self._clean_and_enrich_data(consolidated)
        self._extract_master_tables()
        
        return self.consolidated_df

    def _clean_and_enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types, parses mixed-format dates, handles missing values,
        and calculates key FMCG derived attributes.
        """
        print("[ETL Transformation] Cleaning data types and deriving FMCG metrics...")
        clean_df = df.copy()

        # 1. Standardize text strings
        text_cols = ['odn_no', 'invoice_no', 'customer_id', 'customer_name', 
                     'ship_to_id', 'ship_to_name', 'zone', 'country', 
                     'material_code', 'material_desc', 'category_code', 
                     'category_desc', 'currency', 'fiscal_year']
        for col in text_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].fillna('').astype(str).str.strip()

        # Standardize Country nomenclature (e.g. 'Uk', 'UK' -> 'UK', 'USA' -> 'USA', 'UAE' -> 'UAE')
        country_fix = {
            'uk': 'UK', 'Uk': 'UK', 'united kingdom': 'UK', 'UNITED KINGDOM': 'UK',
            'usa': 'USA', 'Usa': 'USA', 'united states': 'USA', 'UNITED STATES': 'USA',
            'uae': 'UAE', 'Uae': 'UAE', 'united arab emirates': 'UAE',
            'australia': 'Australia', 'AUSTRALIA': 'Australia',
            'new zealand': 'New Zealand', 'NEW ZEALAND': 'New Zealand',
            'singapore': 'Singapore', 'SINGAPORE': 'Singapore',
            'canada': 'Canada', 'CANADA': 'Canada',
            'germany': 'Germany', 'GERMANY': 'Germany',
            'netherlands': 'Netherlands', 'NETHERLANDS': 'Netherlands'
        }
        if 'country' in clean_df.columns:
            clean_df['country'] = clean_df['country'].replace(country_fix)
            # Default empty countries to 'Other Export'
            clean_df['country'] = clean_df['country'].replace({'': 'Other Export', 'Unknown': 'Other Export'})

        # Standardize Zone nomenclature
        zone_fix = {
            'ZONE-1': 'Z1', 'ZONE-2': 'Z2', 'ZONE-3': 'Z3', 'ZONE-4': 'Z4', 
            'ZONE-5': 'Z5', 'ZONE-6': 'Z6', 'ZONE-7': 'Z7', 'Zone 1': 'Z1',
            'Zone 2': 'Z2', 'Zone 3': 'Z3', 'Zone 4': 'Z4', 'Zone 5': 'Z5'
        }
        if 'zone' in clean_df.columns:
            clean_df['zone'] = clean_df['zone'].replace(zone_fix)
            # Map missing zones using known Country associations if possible
            country_to_zone = {
                'USA': 'Z1', 'Canada': 'Z1', 'UAE': 'Z2', 'Saudi Arabia': 'Z2',
                'UK': 'Z4', 'Germany': 'Z4', 'Netherlands': 'Z4',
                'Australia': 'Z3', 'New Zealand': 'Z3', 'Singapore': 'Z5', 'Malaysia': 'Z5'
            }
            mask_empty_zone = clean_df['zone'].isin(['', 'Unknown', 'None', np.nan])
            clean_df.loc[mask_empty_zone, 'zone'] = clean_df.loc[mask_empty_zone, 'country'].map(country_to_zone).fillna('Z-Global')

        # 2. Date parsing (Handling mixed SAP formats: %m/%d/%Y, %d/%m/%Y, %Y-%m-%d)
        if 'invoice_date' in clean_df.columns:
            clean_df['invoice_date'] = pd.to_datetime(clean_df['invoice_date'], format='mixed', errors='coerce')
            # Extract time components for time series modeling
            clean_df['invoice_year'] = clean_df['invoice_date'].dt.year
            clean_df['invoice_month'] = clean_df['invoice_date'].dt.month
            clean_df['invoice_year_month'] = clean_df['invoice_date'].dt.to_period('M').astype(str)
            clean_df['month_start_date'] = clean_df['invoice_date'].dt.to_period('M').dt.to_timestamp()

            # Indian Fiscal Year (starts April 1st, ends March 31st)
            fy_start_year = np.where(clean_df['invoice_month'] >= 4, clean_df['invoice_year'], clean_df['invoice_year'] - 1)
            clean_df['fiscal_year'] = fy_start_year.astype(str) + '-' + (fy_start_year + 1).astype(str)

        # 3. Numeric Conversions
        numeric_cols = [
            'quantity_billed_cases', 'quantity_kg', 'net_weight', 'gross_weight',
            'net_amount_foreign', 'net_amount_inr', 'gross_amount_foreign', 
            'gross_amount_inr', 'exchange_rate', 'amount_inr',
            'discount_exp', 'freight_exp', 'insurance_exp', 'discount_k01',
            'packing_charges_exp', 'zpr8_price', 'pr8_price_alt'
        ]
        for col in numeric_cols:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].apply(clean_numeric)
            else:
                clean_df[col] = 0.0

        # Harmonize Amount in INR (If amount_inr is 0 but net_amount_inr is present, use net_amount_inr)
        clean_df['amount_inr'] = np.where(
            clean_df['amount_inr'] > 0, 
            clean_df['amount_inr'], 
            np.where(clean_df['net_amount_inr'] > 0, clean_df['net_amount_inr'], clean_df['net_amount_foreign'] * clean_df['exchange_rate'])
        )

        # 4. Standardize Product Category and Frozen/Dry Classification
        # Comprehensive harmonization mapping from cryptic SAP codes, synonyms, and subcategories
        category_mapping = {
            'EF-BREADS': 'Frozen Breads',
            'EF-CURRIES': 'Frozen Curries & Meals',
            'EF-SNACKS': 'Frozen Snacks',
            'EF-DAIRY': 'Frozen Dairy & Paneer',
            'EF-SWEETS': 'Frozen Sweets',
            'ED-NAMKEENS': 'Namkeens & Savouries',
            'ED-NAMKEEN': 'Namkeens & Savouries',
            'ED-SWEETS': 'Traditional Sweets',
            'ED-BAKERY': 'Bakery & Cookies',
            'ED-PAPAD': 'Papads & Wafers',
            'ED-SYRUPS': 'Syrups & Beverages',
            # Synonyms and raw subcategories from SAP registers
            'Namkeen': 'Namkeens & Savouries',
            'ALOO BHUJIA': 'Namkeens & Savouries',
            'BHUJIA': 'Namkeens & Savouries',
            'BOONDI': 'Namkeens & Savouries',
            'BHELPURI': 'Namkeens & Savouries',
            'REGULAR MIXTURE': 'Namkeens & Savouries',
            'PREMIUM MIXTURE': 'Namkeens & Savouries',
            'SEV NAMKEENS': 'Namkeens & Savouries',
            'DAL NAMKEENS': 'Namkeens & Savouries',
            'GUJRATI NAMKEENS': 'Namkeens & Savouries',
            'DAKSHIN NAMKEENS': 'Namkeens & Savouries',
            'TAKA-TAK': 'Namkeens & Savouries',
            'LACHHA': 'Namkeens & Savouries',
            'CHIPS': 'Namkeens & Savouries',
            'LITE SNACKS': 'Namkeens & Savouries',
            'SOYA SNACKS': 'Namkeens & Savouries',
            'TEA TIME SNACKS': 'Namkeens & Savouries',
            'Western Snacks': 'Namkeens & Savouries',
            'NUTS & PEANUTS': 'Nuts, Seeds, Berries',
            'MAKHANA': 'Nuts, Seeds, Berries',
            'DRY FRUIT': 'Nuts, Seeds, Berries',
            'Sweets': 'Traditional Sweets',
            'TIN SWEETS': 'Traditional Sweets',
            'SOAN PAPDI': 'Traditional Sweets',
            'MKTG-PATISHA': 'Traditional Sweets',
            'Bakery': 'Bakery & Cookies',
            'COOKIES': 'Bakery & Cookies',
            'KHARI': 'Bakery & Cookies',
            'RUSK': 'Bakery & Cookies',
            'FZN-BREADS': 'Frozen Breads',
            'FZN-PARATHA STUFFED': 'Frozen Breads',
            'FZN-PARATHA PLAIN': 'Frozen Breads',
            'FZN-CHAPATI': 'Frozen Breads',
            'FZN-KULCHA': 'Frozen Breads',
            'ATTA CHAPATI': 'Frozen Breads',
            'FZN-TANDOORI': 'Frozen Breads',
            'FZN-SAMOSA': 'Frozen Snacks',
            'FZN-TIKKI': 'Frozen Snacks',
            'FZN-PAKODA': 'Frozen Snacks',
            'FZN-CHINESE': 'Frozen Snacks',
            'FZN-KACHORI': 'Frozen Snacks',
            'FZN-KEBAB': 'Frozen Snacks',
            'FZN-DHOKLA': 'Frozen Snacks',
            'FZN-VADA': 'Frozen Snacks',
            'FZN-PATTIES': 'Frozen Snacks',
            'FZN-WRAPS': 'Frozen Snacks',
            'FZN-DAKSHIN': 'Frozen Snacks',
            'FZN-WESTERN SNACKS': 'Frozen Snacks',
            'FZN-COMBO': 'Frozen Curries & Meals',
            'FZN-CURRIES': 'Frozen Curries & Meals',
            'READY TO EAT-CURRIES': 'Frozen Curries & Meals',
            'READY TO EAT-FD': 'RTE',
            'READY TO EAT-INSTANT MIX': 'RTC & Condiments',
            'READY TO EAT-PANI PURI': 'RTE',
            'READY TO EAT-COMBO': 'RTE',
            'FZN-CHUTNEY': 'Frozen Snacks',
            'FZN-CUT VEGETABLES': 'Frozen Snacks',
            'FZN-CUT FRUITS': 'Frozen Snacks',
            'FZN-SWEETS': 'Frozen Sweets',
            'FZN-DAIRY': 'Frozen Dairy & Paneer',
            'Dairy': 'Frozen Dairy & Paneer',
            'DAIRY': 'Frozen Dairy & Paneer',
            'PAPAD': 'Papads & Wafers',
            'SYRUP': 'Syrups & Beverages',
            'Beverages': 'Syrups & Beverages',
            'GIFT PACK': 'Gift Pack',
            'Gift Pack': 'Gift Pack'
        }

        # Step 1: Map category_code
        cat_series = clean_df['category_code'].map(category_mapping) if 'category_code' in clean_df.columns else pd.Series(index=clean_df.index, dtype=object)

        # Step 2: Fallback to desc_2_category and category_desc mapped through category_mapping or directly
        if 'desc_2_category' in clean_df.columns:
            mapped_desc2 = clean_df['desc_2_category'].map(category_mapping).fillna(clean_df['desc_2_category'])
            cat_series = cat_series.fillna(mapped_desc2)
        if 'category_desc' in clean_df.columns:
            mapped_cdesc = clean_df['category_desc'].map(category_mapping).fillna(clean_df['category_desc'])
            cat_series = cat_series.fillna(mapped_cdesc)

        clean_df['category_name'] = cat_series.map(category_mapping).fillna(cat_series)

        # Step 3: Material code mapping from known standardized entries
        valid_standard_cats = [
            'Namkeens & Savouries', 'Frozen Snacks', 'Frozen Breads', 'Frozen Curries & Meals',
            'Traditional Sweets', 'Frozen Sweets', 'RTE', 'Bakery & Cookies',
            'Frozen Dairy & Paneer', 'RTC & Condiments', 'Nuts, Seeds, Berries',
            'Syrups & Beverages', 'Gift Pack', 'Papads & Wafers'
        ]
        clean_mask = clean_df['category_name'].isin(valid_standard_cats)
        if clean_mask.any():
            known_mat_map = clean_df[clean_mask].groupby('material_code')['category_name'].first().to_dict()
            missing_cat = clean_df['category_name'].isna() | (clean_df['category_name'].astype(str).str.strip().isin(['', 'Other FMCG', 'nan', 'None']))
            clean_df.loc[missing_cat, 'category_name'] = clean_df.loc[missing_cat, 'material_code'].map(known_mat_map)

        # Step 4: Fallback to Excel Product Master
        excel_path = self.raw_data_dir / 'product_data.xlsx'
        if not excel_path.exists():
            excel_path = Path('data/raw/product_data.xlsx')
        if excel_path.exists():
            try:
                df_prod_excel = pd.read_excel(excel_path)
                excel_cat_to_std = {
                    'Frozen Snacks': 'Frozen Snacks', 'Frozen Sweets': 'Frozen Sweets',
                    'Frozen Breads': 'Frozen Breads', 'Frozen Curries': 'Frozen Curries & Meals',
                    'Frozen Vegetables': 'Frozen Snacks', 'Frozen Meals': 'Frozen Curries & Meals',
                    'Frozen Chutney': 'Frozen Snacks', 'Frozen Paneer': 'Frozen Dairy & Paneer',
                    'Kulfi': 'Frozen Sweets', 'Frozen Chocolate': 'Frozen Sweets',
                    'Mixture': 'Namkeens & Savouries', 'Mixture-Premium': 'Namkeens & Savouries',
                    'Bhujia Sev': 'Namkeens & Savouries', 'Aloo Bhujia': 'Namkeens & Savouries',
                    'Besan Namkeen': 'Namkeens & Savouries', 'Namkeen Dal': 'Namkeens & Savouries',
                    'Sev Namkeen': 'Namkeens & Savouries', 'Murmura & Bhel': 'Namkeens & Savouries',
                    'Peanuts': 'Namkeens & Savouries', 'Tea Time Snacks': 'Namkeens & Savouries',
                    'Lite Namkeen': 'Namkeens & Savouries', 'Potato Wafers': 'Namkeens & Savouries',
                    'Banana Chips': 'Namkeens & Savouries', 'Punjabi Tadka': 'Namkeens & Savouries',
                    'Chana Jor Garam': 'Namkeens & Savouries', 'Matar': 'Namkeens & Savouries',
                    'Lachha': 'Namkeens & Savouries', 'Soya Snacks': 'Namkeens & Savouries',
                    'Falahari': 'Namkeens & Savouries', 'Pellets': 'Namkeens & Savouries',
                    'Bridges': 'Namkeens & Savouries', 'Healthy': 'Namkeens & Savouries',
                    'Soan Papdi': 'Traditional Sweets', 'Chenna': 'Traditional Sweets',
                    'Petha': 'Traditional Sweets', 'Others (Pulses & Cereal Based Sweets)': 'Traditional Sweets',
                    'Milk Based': 'Traditional Sweets', 'Dry Fruit & Nuts Based': 'Traditional Sweets',
                    'Dry Fruit': 'Nuts, Seeds, Berries', 'Cookies': 'Bakery & Cookies',
                    'Khari': 'Bakery & Cookies', 'Rusk/Toast': 'Bakery & Cookies',
                    'RTE Curries': 'Frozen Curries & Meals', 'RTE Combo': 'RTE',
                    'RTE Pani Puri': 'RTE', 'RTE Paneer': 'Frozen Curries & Meals',
                    'Freeze Dried': 'RTE', 'Instant Mix': 'RTC & Condiments',
                    'Papad': 'Papads & Wafers', 'Squash': 'Syrups & Beverages',
                    'Gift Pack': 'Gift Pack', 'Atta': 'RTC & Condiments', 'Vermicelli': 'RTC & Condiments'
                }
                excel_mat_map = {}
                for _, erow in df_prod_excel.iterrows():
                    em = str(erow.get('Material', '')).strip()
                    ec = str(erow.get('Category', '')).strip()
                    if ec in excel_cat_to_std:
                        excel_mat_map[em] = excel_cat_to_std[ec]
                missing_cat = clean_df['category_name'].isna() | (clean_df['category_name'].astype(str).str.strip().isin(['', 'Other FMCG', 'nan', 'None']))
                clean_df.loc[missing_cat, 'category_name'] = clean_df.loc[missing_cat, 'material_code'].map(excel_mat_map)
            except Exception as e:
                print(f"[ETL Category Harmonization] Excel load note: {e}")

        # Step 5: Material description keyword heuristics
        missing_cat = clean_df['category_name'].isna() | (clean_df['category_name'].astype(str).str.strip().isin(['', 'Other FMCG', 'nan', 'None']))
        m_desc = clean_df.loc[missing_cat, 'material_desc'].astype(str).str.lower()

        is_namkeen = m_desc.str.contains('bhujia|sev|namkeen|bhel|mixture|boondi|kachori|gathiya|mathri|chips|wafers|chana|peanut|moong dal|kurkure|makhana|murukku|shakarpara|para', na=False)
        is_sweet = m_desc.str.contains('laddu|ladoo|halwa|peda|barfi|burfi|rasgulla|gulab|soan papdi|soan papdi|cham cham|gujiya|sweet|soan|patisha', na=False)
        is_bread = m_desc.str.contains('naan|paratha|roti|kulcha|chapati|bread|puri|poori', na=False)
        is_fzn_snack = m_desc.str.contains('samosa|tikki|pakoda|vada|spring roll|paneer roll|cutlet|dhokla|kebab', na=False)
        is_fzn_meal = m_desc.str.contains('curry|curries|dal makhani|rajma|choley|paneer butter|subji|biryani', na=False)
        is_bakery = m_desc.str.contains('cookie|cookies|khari|rusk|toast|biscuit|bakery', na=False)

        clean_df.loc[missing_cat & is_namkeen, 'category_name'] = 'Namkeens & Savouries'
        clean_df.loc[missing_cat & is_sweet, 'category_name'] = 'Traditional Sweets'
        clean_df.loc[missing_cat & is_bread, 'category_name'] = 'Frozen Breads'
        clean_df.loc[missing_cat & is_fzn_snack, 'category_name'] = 'Frozen Snacks'
        clean_df.loc[missing_cat & is_fzn_meal, 'category_name'] = 'Frozen Curries & Meals'
        clean_df.loc[missing_cat & is_bakery, 'category_name'] = 'Bakery & Cookies'

        # Default any remaining empty to 'Other FMCG'
        missing_cat = clean_df['category_name'].isna() | (clean_df['category_name'].astype(str).str.strip().isin(['', 'nan', 'None']))
        clean_df.loc[missing_cat, 'category_name'] = 'Other FMCG'
        
        # Flag Dry vs Frozen goods
        clean_df['is_frozen'] = clean_df['material_code'].str.startswith('F') | clean_df['category_name'].str.contains('Frozen|EF-', case=False, na=False)

        # 5. Derived Realized Price (INR / Kg) & Margin Proxy
        # Avoid division by zero with safe numpy where
        clean_df['unit_price_inr_per_kg'] = np.where(
            clean_df['quantity_kg'] > 0,
            clean_df['amount_inr'] / clean_df['quantity_kg'],
            0.0
        )

        # Cost proxy: Total direct export deductions (freight + insurance + discount)
        clean_df['total_deductions_inr'] = (
            clean_df['freight_exp'] + clean_df['insurance_exp'] + clean_df['discount_exp'] + clean_df['discount_k01']
        )
        
        # Export Net Margin Proxy (Invoiced Amount in INR minus direct export logistics/discount deductions)
        clean_df['net_margin_proxy_inr'] = np.maximum(0.0, clean_df['amount_inr'] - clean_df['total_deductions_inr'])
        clean_df['margin_proxy_pct'] = np.where(
            clean_df['amount_inr'] > 0,
            (clean_df['net_margin_proxy_inr'] / clean_df['amount_inr']) * 100.0,
            100.0  # Default to 100% if no deductions are reported
        )

        # Filter out corrupted rows where quantity and amount are both zero
        valid_rows = (clean_df['quantity_kg'] > 0) | (clean_df['amount_inr'] > 0)
        clean_df = clean_df[valid_rows].reset_index(drop=True)

        print(f"[ETL Transformation] Cleaned Dataset Shape: {clean_df.shape[0]:,} rows x {clean_df.shape[1]} columns.")
        return clean_df

    def _extract_master_tables(self) -> None:
        """
        Derives dedicated Customer Master and Product Master dataframes from the cleaned transactional exhaust.
        """
        if self.consolidated_df is None:
            return

        print("[ETL Master Extraction] Building dimensional Customer Master & Product Master...")
        
        # 1. Customer Master
        cust_cols = ['customer_id', 'customer_name', 'country', 'zone']
        available_cust_cols = [c for c in cust_cols if c in self.consolidated_df.columns]
        self.customer_master = (
            self.consolidated_df.groupby('customer_id')[available_cust_cols]
            .first()
            .reset_index(drop=True)
        )
        
        # Aggregate Customer Lifetime Statistics
        cust_stats = self.consolidated_df.groupby('customer_id').agg(
            total_orders=('invoice_no', 'nunique'),
            total_revenue_inr=('amount_inr', 'sum'),
            total_volume_kg=('quantity_kg', 'sum'),
            first_order_date=('invoice_date', 'min'),
            last_order_date=('invoice_date', 'max'),
            active_years=('fiscal_year', 'nunique')
        ).reset_index()
        self.customer_master = self.customer_master.merge(cust_stats, on='customer_id', how='left')

        # 2. Product Master
        prod_cols = ['material_code', 'material_desc', 'category_name', 'is_frozen']
        available_prod_cols = [c for c in prod_cols if c in self.consolidated_df.columns]
        self.product_master = (
            self.consolidated_df.groupby('material_code')[available_prod_cols]
            .first()
            .reset_index(drop=True)
        )

        prod_stats = self.consolidated_df.groupby('material_code').agg(
            total_revenue_inr=('amount_inr', 'sum'),
            total_volume_kg=('quantity_kg', 'sum'),
            total_shipments=('invoice_no', 'nunique'),
            avg_unit_price_inr_per_kg=('unit_price_inr_per_kg', lambda x: x[x > 0].median() if (x > 0).any() else 0.0)
        ).reset_index()
        self.product_master = self.product_master.merge(prod_stats, on='material_code', how='left')

        print(f"  -> Extracted {len(self.customer_master):,} unique Customers and {len(self.product_master):,} unique SKUs.")

    def save_processed_datasets(self, format: str = "parquet") -> Dict[str, Path]:
        """
        Persists clean consolidated transactions, customer master, and product master
        to high-performance Parquet/CSV formats.
        """
        if self.consolidated_df is None:
            raise ValueError("No consolidated dataframe available to save. Run load_and_harmonize_all_years() first.")

        paths = {}
        if format == "parquet":
            main_path = self.processed_data_dir / "haldirams_sales_consolidated.parquet"
            cust_path = self.processed_data_dir / "haldirams_customer_master.parquet"
            prod_path = self.processed_data_dir / "haldirams_product_master.parquet"
            
            self.consolidated_df.to_parquet(main_path, index=False)
            if self.customer_master is not None:
                self.customer_master.to_parquet(cust_path, index=False)
            if self.product_master is not None:
                self.product_master.to_parquet(prod_path, index=False)
        else:
            main_path = self.processed_data_dir / "haldirams_sales_consolidated.csv"
            cust_path = self.processed_data_dir / "haldirams_customer_master.csv"
            prod_path = self.processed_data_dir / "haldirams_product_master.csv"
            
            self.consolidated_df.to_csv(main_path, index=False)
            if self.customer_master is not None:
                self.customer_master.to_csv(cust_path, index=False)
            if self.product_master is not None:
                self.product_master.to_csv(prod_path, index=False)

        paths['consolidated'] = main_path
        paths['customer_master'] = cust_path
        paths['product_master'] = prod_path
        print(f"[ETL Storage] Successfully saved datasets to {self.processed_data_dir} ({format.upper()})")
        return paths


if __name__ == "__main__":
    # Test harness for Data Engineering Module
    current_dir = Path(__file__).resolve().parent.parent
    raw_dir = current_dir / "md files"
    processed_dir = current_dir / "data"

    print("=" * 80)
    print("HALDIRAM'S INTERNATIONAL EXPORTS - DATA ENGINEERING & HARMONIZATION PIPELINE")
    print("=" * 80)
    pipeline = HaldiramsDataPipeline(raw_data_dir=raw_dir, processed_data_dir=processed_dir)
    clean_data = pipeline.load_and_harmonize_all_years()
    saved_files = pipeline.save_processed_datasets(format="parquet")
    print(f"\nConsolidated Data Summary:\n{clean_data.info()}")
    print("\nSample Records:")
    print(clean_data[['invoice_date', 'fiscal_year', 'country', 'zone', 'category_name', 'material_desc', 'quantity_kg', 'amount_inr']].head())
