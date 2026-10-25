# Haldiram's Export Analytics Data Schemas

This document defines the schema, column names, and data types for the expected datasets consumed by the Haldiram's International Export Sales Analytics platform.

All underlying commercial data files are stored in Apache Parquet format locally in the `data/` directory.

---

## 1. Consolidated Sales Dataset (`haldirams_sales_consolidated.parquet`)

Contains multi-year transaction line items harmonized from export sales registers across all active fiscal periods.

| Column Name | Data Type | Description |
|:---|:---|:---|
| `odn_no` | string | Official Document Number / SAP Billing reference |
| `invoice_no` | string | Commercial invoice number |
| `invoice_date` | timestamp[us] | Invoice billing date |
| `customer_id` | string | Sold-to party customer identifier |
| `customer_name` | string | Sold-to distributor account name |
| `ship_to_id` | string | Ship-to party account identifier |
| `ship_to_name` | string | Ship-to destination consignee name |
| `quantity_billed_cases` | double | Number of invoiced carton cases |
| `quantity_kg` | double | Net billed weight in kilograms |
| `sales_unit` | string | Billed unit of measurement |
| `net_weight` | double | Net consignment shipment weight |
| `gross_weight` | double | Gross consignment shipment weight |
| `net_amount_foreign` | double | Net line value in foreign billed currency |
| `gross_amount_foreign` | double | Gross line value in foreign billed currency |
| `currency` | string | Foreign transaction currency code (e.g., USD, AED, GBP) |
| `exchange_rate` | double | Conversion exchange rate applied |
| `amount_inr` | double | Realized gross transaction value in INR |
| `sales_order_no` | string | Upstream sales order reference number |
| `document_date` | string | SAP financial document posting date |
| `material_code` | string | Unique SKU material code |
| `material_desc` | string | SKU item specification and packaging description |
| `material_type` | string | SAP material type classification |
| `basic_export_value` | string | Export price base record |
| `discount_exp` | double | Export distributor trade discount deduction |
| `freight_exp` | double | Freight component deduction (`ZFRT`) |
| `insurance_exp` | double | Marine insurance component deduction (`ZINS`) |
| `discount_k01` | double | Secondary trade promotion discount (`ZK01`) |
| `packing_charges_exp` | double | Special export packing charge addition |
| `zone` | string | Global geographic commercial export territory |
| `sales_group` | string | Internal export sales division code |
| `production_cat_group` | string | Production facility product group code |
| `design_code_l3` | string | Packaging design level-3 hierarchy code |
| `design_code_l4` | string | Packaging design level-4 hierarchy code |
| `country` | string | Destination export country |
| `fiscal_year` | string | Reporting fiscal year (e.g., `FY 2023-2024`) |
| `category_code` | string | Internal product category code |
| `source_file` | string | Ingestion source audit file name |
| `sales_group_desc` | string | Sales group descriptive division |
| `Net Amount in INR` | string | Raw register net amount string |
| `zpr8_price` | double | SAP export condition standard price (`ZPR8`) |
| `pr8_price_alt` | double | Alternative standard pricing condition |
| `product_hierarchy_1` | string | Top-level product hierarchy descriptor |
| `design_code` | string | Primary packaging design classification |
| `category_desc` | string | Product category raw classification |
| `sub_category_desc` | string | Product sub-category classification |
| `variant_desc` | string | Product flavor and recipe variant |
| `fzn_purchase_price` | string | Frozen goods procurement condition price |
| `fzn_purchase_amount` | string | Frozen goods procurement line item amount |
| `fzn_sale_amount` | string | Frozen goods sales value |
| `month_str` | string | Invoiced month three-letter code |
| `sales_org` | string | SAP sales organization company code |
| `supplying_plant` | string | Manufacturing/dispatch plant code |
| `net_amount_inr` | double | Net invoiced line value in INR |
| `gross_amount_inr` | double | Gross export value before deductions in INR |
| `alt_uom` | string | Alternate unit of measurement |
| `desc_1_market` | string | Market segment classification |
| `desc_2_category` | string | Standardized category hierarchy level |
| `desc_3_subcategory` | string | Standardized subcategory level |
| `desc_4_type` | string | Product form or recipe type |
| `desc_5_variant` | string | Variant flavor descriptor |
| `desc_6_size_class` | string | Pack size classification |
| `desc_7_pack_detail` | string | Packaging type specification |
| `invoice_year` | int32 | Calendar year of invoice date |
| `invoice_month` | int32 | Calendar month of invoice date (1-12) |
| `invoice_year_month` | string | Period string format (`YYYY-MM`) |
| `month_start_date` | timestamp[us] | First day of invoicing month for time-series aggregation |
| `category_name` | string | Canonical product category (Namkeens & Savouries, Sweets, Frozen Foods, Bakery, Ready to Eat, Other FMCG) |
| `is_frozen` | bool | Boolean flag indicating cold-chain/frozen requirements |
| `unit_price_inr_per_kg` | double | Derived unit price realized (`amount_inr / quantity_kg`) |
| `total_deductions_inr` | double | Aggregate freight, insurance, and discount deductions |
| `net_margin_proxy_inr` | double | Estimated gross operational contribution margin (`amount_inr - total_deductions_inr`) |
| `margin_proxy_pct` | double | Estimated net margin proxy percentage |

---

## 2. Product Master (`haldirams_product_master.parquet`)

SKU master dataset containing categorized specifications and historical product-level aggregates.

| Column Name | Data Type | Description |
|:---|:---|:---|
| `material_code` | string | Unique SKU identifier code |
| `material_desc` | string | Full product description and pack size |
| `category_name` | string | Primary harmonized FMCG product category |
| `is_frozen` | bool | Temperature control classification |
| `total_revenue_inr` | double | Cumulative export billing revenue in INR |
| `total_volume_kg` | double | Cumulative shipped volume in kilograms |
| `total_shipments` | int64 | Total invoice line items recorded |
| `avg_unit_price_inr_per_kg` | double | Historical volume-weighted average price per kg |

---

## 3. Customer Master (`haldirams_customer_master.parquet`)

B2B distributor master dataset summarizing account dimensions and ordering behavior.

| Column Name | Data Type | Description |
|:---|:---|:---|
| `customer_id` | string | Unique B2B customer identifier |
| `customer_name` | string | Registered commercial distributor name |
| `country` | string | Primary destination country |
| `zone` | string | Assigned geographic export territory |
| `total_orders` | int64 | Total order shipments completed |
| `total_revenue_inr` | double | Cumulative lifetime invoiced export billing in INR |
| `total_volume_kg` | double | Cumulative lifetime volume shipped in kg |
| `first_order_date` | timestamp[us] | First recorded transaction billing date |
| `last_order_date` | timestamp[us] | Most recent transaction billing date |
| `active_years` | int64 | Count of distinct fiscal years with active ordering activity |
