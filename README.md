# Haldiram's Export Sales Analytics & ML Pipeline

An enterprise-grade FMCG export intelligence platform and machine learning decision suite built for Haldiram's international sales operations across multiple fiscal years. The system integrates data ingestion, multi-model demand forecasting, B2B customer and SKU segmentation, econometric price elasticity modeling, and interactive executive reporting into a unified, responsive dashboard architecture.

---

## 🧭 Dashboard Modules & Architecture

The application comprises five specialized analytics pages plus an executive briefing section:

1. **Overview & Exploratory Data Analysis (`0_📊_Overview_&_EDA.py` / `app.py`)**  
   High-level executive command center displaying core export KPIs, multi-fiscal-year temporal trends, geographic distribution across global export zones and destination countries, product category breakdowns, and entity-level B2B concentration analysis.

2. **Hierarchical Demand Forecasting (`1_📈_Demand_Forecasting.py`)**  
   Multi-model supply chain forecasting engine benchmarking SARIMAX (with seasonal festival regressors), Prophet (additive trend decomposition with holiday lead-times), and Lagged-Feature XGBoost across major product categories. Evaluates performance using walk-forward holdout validation on FMCG-standard MAPE (%) and RMSE metrics.

3. **B2B Customer & Product Segmentation (`2_🎯_Customer_&_Product_Segmentation.py`)**  
   Unsupervised clustering framework utilizing StandardScaler, K-Means clustering, and 2D Principal Component Analysis (PCA). Segments B2B distributor accounts and product SKUs across revenue volume, order frequency, average order value, demand volatility (CV), and net margin proxies.

4. **Econometric Price Elasticity Studio (`3_🏷️_Price_Elasticity.py`)**  
   Log-Log Ordinary Least Squares (OLS) econometric regression suite modeling price responsiveness for top-volume global export SKUs. Includes validation against export price conditions and interactive price-volume-revenue simulation scenarios.

5. **Data Explorer & Record Export (`4_📋_Data_Explorer_&_Export.py`)**  
   Dynamic tabular data exploration environment equipped with multi-dimensional filtering (by year, category, zone, and export metrics), interactive sorting, summary statistics, and secure CSV data export.

6. **Executive Briefings & Technical Notes (`5_📝_Notes_&_Executive_Briefings.py`)**  
   Comprehensive methodology documentation covering pipeline design, schema harmonization, econometric model specifications, clustering rationale, and viva defense references.

---

## 🛠️ Technology Stack

- **Application Framework**: Streamlit (with multi-page navigation and branded UI)
- **Data Engineering**: Pandas, PyArrow, NumPy, OpenPyXL
- **Visualization**: Plotly Express & Plotly Graph Objects, Python-Lucide icons
- **Statistical & Machine Learning**: Scikit-Learn, Statsmodels, Prophet, XGBoost, SciPy
- **Image Processing**: Pillow

---

## 📁 Repository Structure

```text
HALDIRAMS_DASH/
├── .streamlit/
│   └── config.toml             # Branded theme configuration (no credentials)
├── assets/
│   ├── haldirams_logo.png      # Haldiram's navigation brand asset
│   └── haldirams_logo_original.png
├── docs/
│   └── DATA_SCHEMA.md          # Parquet schema and column definitions
├── pages/
│   ├── 0_📊_Overview_&_EDA.py
│   ├── 1_📈_Demand_Forecasting.py
│   ├── 2_🎯_Customer_&_Product_Segmentation.py
│   ├── 3_🏷️_Price_Elasticity.py
│   ├── 4_📋_Data_Explorer_&_Export.py
│   └── 5_📝_Notes_&_Executive_Briefings.py
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py        # Ingestion, schema unification, and derivation pipeline
│   ├── elasticity.py           # OLS log-log elasticity modeling and simulation
│   ├── forecasting.py          # SARIMAX, Prophet, and XGBoost forecasting engine
│   ├── segmentation.py         # K-Means clustering and PCA dimensionality reduction
│   └── utils.py                # Shared sidebar styling, KPI cards, and formatting helpers
├── app.py                      # Application entry point and router
├── requirements.txt            # Python dependencies with pinned versions
├── .gitignore                  # Git exclusions for data, caches, and secrets
└── README.md                   # Project overview and run documentation
```

---

## 🔒 Confidentiality & Expected Data Assets

> **Confidentiality Notice**: The `data/` directory and all underlying transactional records, SAP export spreadsheets, and Parquet data files are strictly excluded from version control to protect proprietary commercial data. No client records or financial figures are committed to this repository.

To run the application locally, place the processed Parquet files in the local `data/` directory:
- `data/haldirams_sales_consolidated.parquet` (Consolidated multi-year export sales line items)
- `data/haldirams_product_master.parquet` (Product hierarchy, category mappings, and SKU master)
- `data/haldirams_customer_master.parquet` (Customer accounts, geographic tags, and order summaries)

Refer to [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) for full structural specifications, column names, and data types.

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AANYAD/HALDIRAMS_DASH.git
   cd HALDIRAMS_DASH
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```
