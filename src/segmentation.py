"""
================================================================================
Haldiram's International Export Analytics Pipeline - Segmentation Module
================================================================================
Author: Aanya
Project: Haldiram's Final-Year Export Sales & ML Analytics
Module: B2B Customer & Product Portfolio Clustering (K-Means + PCA + Persona Profiling)

Segmentation Methodology & Architecture Notes:
- Dimensional Feature Engineering:
  1. Monetary Value: Total export invoiced revenue in INR (lifetime export power).
  2. Order Frequency & Breadth: Total unique export shipments and active SKU count.
  3. YoY Volume Growth Rate: Compound annual growth rate across fiscal years.
  4. Demand Volatility (CV): Coefficient of Variation (std / mean of monthly order kg)
     measuring procurement regularity vs. opportunistic festive-only buying.
  5. Net Contribution Margin Proxy (%): Margin realization after factoring freight,
     insurance, and distributor export discounts.
- Standardization Rationale: Features span vastly different scales (Revenue in crores,
  CV between 0-3, Margin in percentage). `StandardScaler` prevents high-variance revenue
  from dominating Euclidean distance metrics in K-Means.
- Dimensionality Reduction: PCA (2 components) used strictly for 2D visual projection
  and orthogonal variance decomposition without loss of interpretability.
- Business Actionability: Translates mathematical cluster centroids into actionable FMCG
  commercial personas (Tier-1 Anchor, High-Growth Strategic, Volatile Seasonal, Margin Sensitive).
================================================================================
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class B2BCustomerSegmentation:
    """
    K-Means clustering and persona attribution for Haldiram's International B2B Distributors.
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()
        self.features_df: Optional[pd.DataFrame] = None
        self.scaled_features: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.kmeans_model: Optional[KMeans] = None
        self.pca_model = PCA(n_components=2, random_state=42)
        self.feature_columns = [
            'total_revenue_inr', 'total_volume_kg', 'order_frequency', 
            'avg_order_value_inr', 'yoy_growth_feature', 'demand_volatility_cv'
        ]

    def build_customer_feature_store(self) -> pd.DataFrame:
        """
        Aggregates raw invoice transactions into a rich B2B distributor feature store.
        """
        print("[Segmentation] Engineering B2B customer feature store...")
        
        # 1. Base aggregations per customer
        base_agg = self.df.groupby('customer_id').agg(
            customer_name=('customer_name', 'first'),
            country=('country', 'first'),
            zone=('zone', 'first'),
            total_revenue_inr=('amount_inr', 'sum'),
            total_volume_kg=('quantity_kg', 'sum'),
            order_frequency=('invoice_no', 'nunique'),
            active_skus_count=('material_code', 'nunique'),
            first_invoice_date=('invoice_date', 'min'),
            last_invoice_date=('invoice_date', 'max'),
            avg_margin_proxy_pct=('margin_proxy_pct', 'mean'),
            total_margin_inr=('net_margin_proxy_inr', 'sum')
        ).reset_index()

        # Derived Average Order Value (AOV)
        base_agg['avg_order_value_inr'] = np.where(
            base_agg['order_frequency'] > 0,
            base_agg['total_revenue_inr'] / base_agg['order_frequency'],
            0.0
        )
        base_agg['margin_proxy_pct'] = np.where(
            base_agg['total_revenue_inr'] > 0,
            (base_agg['total_margin_inr'] / base_agg['total_revenue_inr']) * 100.0,
            base_agg['avg_margin_proxy_pct']
        )

        # 2. Demand Volatility (Coefficient of Variation of monthly volume)
        monthly_cust_vol = (
            self.df.groupby(['customer_id', 'month_start_date'])['quantity_kg']
            .sum()
            .reset_index()
        )
        cv_stats = monthly_cust_vol.groupby('customer_id')['quantity_kg'].agg(
            mean_vol='mean',
            std_vol='std'
        ).reset_index()
        cv_stats['demand_volatility_cv'] = np.where(
            cv_stats['mean_vol'] > 0,
            cv_stats['std_vol'].fillna(0.0) / cv_stats['mean_vol'],
            0.0
        )
        base_agg = base_agg.merge(cv_stats[['customer_id', 'demand_volatility_cv']], on='customer_id', how='left')
        base_agg['demand_volatility_cv'] = base_agg['demand_volatility_cv'].fillna(0.0)

        # 3. YoY Volume Growth Rate (Same 11 months: Apr-Feb FY23-24 vs Apr-Feb FY24-25)
        df_11m_2324 = self.df[(self.df['invoice_date'] >= '2023-04-01') & (self.df['invoice_date'] <= '2024-02-29')]
        df_11m_2425 = self.df[(self.df['invoice_date'] >= '2024-04-01') & (self.df['invoice_date'] <= '2025-02-28')]
        
        v23_series = df_11m_2324.groupby('customer_id')['quantity_kg'].sum()
        v24_series = df_11m_2425.groupby('customer_id')['quantity_kg'].sum()
        
        vol_11m_df = pd.DataFrame({'vol_2324_11m': v23_series, 'vol_2425_11m': v24_series}).fillna(0.0)
        
        # Per account: set growth to NaN if FY23-24 volume is below 5% of median active account
        pos_v23 = vol_11m_df[vol_11m_df['vol_2324_11m'] > 0]['vol_2324_11m']
        median_v23 = pos_v23.median() if not pos_v23.empty else 1.0
        min_v23_thresh = 0.05 * median_v23
        
        def compute_account_growth(row: pd.Series) -> float:
            v_prev = row['vol_2324_11m']
            v_curr = row['vol_2425_11m']
            if v_prev < min_v23_thresh:
                return np.nan
            return ((v_curr - v_prev) / v_prev) * 100.0

        vol_11m_df['yoy_growth_pct'] = vol_11m_df.apply(compute_account_growth, axis=1)
        base_agg = base_agg.merge(vol_11m_df.reset_index(), on='customer_id', how='left')
        base_agg['vol_2324_11m'] = base_agg['vol_2324_11m'].fillna(0.0)
        base_agg['vol_2425_11m'] = base_agg['vol_2425_11m'].fillna(0.0)
        # Numerical feature for KMeans: fill NaN with 0 and clip
        base_agg['yoy_growth_feature'] = base_agg['yoy_growth_pct'].fillna(0.0).clip(-100.0, 200.0)

        # Filter out invalid entries with zero orders
        self.features_df = base_agg[base_agg['order_frequency'] > 0].reset_index(drop=True)
        print(f"[Segmentation] Constructed features for {len(self.features_df):,} B2B distributors.", flush=True)
        return self.features_df

    def fit_kmeans_clustering(self, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Executes standard feature scaling, K-Means clustering, PCA projection,
        outlier cluster folding (no <=2 account clusters), and business persona assignment.
        """
        if self.features_df is None:
            self.build_customer_feature_store()

        df_feat = self.features_df.copy()
        
        # Extract numerical feature matrix
        X = df_feat[self.feature_columns].values
        self.scaled_features = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=300,
            random_state=random_state
        )
        cluster_labels = self.kmeans_model.fit_predict(self.scaled_features)

        # Fold any small clusters (<= 2 accounts) into the nearest cluster
        cluster_counts = pd.Series(cluster_labels).value_counts()
        small_clusters = cluster_counts[cluster_counts <= 2].index.tolist()
        if small_clusters:
            centroids = self.kmeans_model.cluster_centers_
            other_clusters = [c for c in range(n_clusters) if c not in small_clusters]
            if other_clusters:
                for sc in small_clusters:
                    sc_indices = np.where(cluster_labels == sc)[0]
                    for idx in sc_indices:
                        dists = [np.linalg.norm(self.scaled_features[idx] - centroids[oc]) for oc in other_clusters]
                        nearest_cluster = other_clusters[int(np.argmin(dists))]
                        cluster_labels[idx] = nearest_cluster

        df_feat['cluster_id'] = cluster_labels

        # Calculate cluster evaluation metrics
        n_unique_clusters = len(np.unique(cluster_labels))
        silhouette = float(silhouette_score(self.scaled_features, cluster_labels)) if n_unique_clusters > 1 else 0.0
        inertia = float(self.kmeans_model.inertia_)
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': round(silhouette, 3),
            'inertia': round(inertia, 2)
        }

        # 2D PCA for visual mapping
        pca_coords = self.pca_model.fit_transform(self.scaled_features)
        df_feat['pca_dim_1'] = pca_coords[:, 0]
        df_feat['pca_dim_2'] = pca_coords[:, 1]
        df_feat['pca_explained_var_pct'] = round(float(np.sum(self.pca_model.explained_variance_ratio_) * 100.0), 1)

        # Assign Business Personas based on Cluster Centroid profiles
        df_feat = self._assign_business_personas(df_feat)
        self.features_df = df_feat

        return df_feat, metrics

    def compute_k3_k4_silhouettes(self) -> Tuple[float, float, int]:
        """
        Computes the silhouette score for K=3 and K=4 and prints both.
        Returns (sil_k3, sil_k4, recommended_k=3).
        """
        if self.features_df is None:
            self.build_customer_feature_store()
        X = self.features_df[self.feature_columns].values
        scaled = self.scaler.fit_transform(X)

        km3 = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=300, random_state=42)
        l3 = km3.fit_predict(scaled)
        sil_k3 = float(silhouette_score(scaled, l3))

        km4 = KMeans(n_clusters=4, init='k-means++', n_init=15, max_iter=300, random_state=42)
        l4 = km4.fit_predict(scaled)
        sil_k4 = float(silhouette_score(scaled, l4))

        print(f"[Segmentation Silhouette] K=3: {sil_k3:.4f} | K=4: {sil_k4:.4f}", flush=True)
        rec_k = 3
        return round(sil_k3, 4), round(sil_k4, 4), rec_k

    def _assign_business_personas(self, df_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Translates cluster IDs into FMCG distribution personas based on cluster statistics:
        - Highest avg revenue per account -> 'Mega Accounts' (Navy #1E3A8A)
        - If aggregate growth > 5% and revenue share > 10% -> 'High-Growth Strategic Partners' (Teal #0D9488)
        - Highest remaining revenue -> 'Core Strategic Partners' (Orange #EA580C)
        - The rest -> 'Long-Tail Accounts' (Magenta #C026D3)
        No 'Outlier Accounts' persona.
        """
        tot_rev = df_clustered['total_revenue_inr'].sum()
        cluster_grp = df_clustered.groupby('cluster_id')

        cluster_info = {}
        for cid, g in cluster_grp:
            v23 = g['vol_2324_11m'].sum() if 'vol_2324_11m' in g.columns else 0.0
            v24 = g['vol_2425_11m'].sum() if 'vol_2425_11m' in g.columns else 0.0
            agg_growth = ((v24 - v23) / v23 * 100.0) if v23 > 0 else 0.0
            median_growth = g['yoy_growth_pct'].median()
            disp_agg_growth = min(agg_growth, 100.0)
            disp_median_growth = min(median_growth, 100.0) if pd.notna(median_growth) else 0.0

            # Print per cluster: FY23-24 volume, FY24-25 volume, aggregate growth, median per-account growth, accounts
            print(f"[Segmentation Cluster {cid}] Accounts: {len(g):>3} | FY23-24 (11m): {v23:>10,.0f} kg | FY24-25 (11m): {v24:>10,.0f} kg | Agg Growth: {agg_growth:>+6.2f}% (Capped: {disp_agg_growth:>+6.2f}%) | Median YoY: {median_growth:>+6.2f}%", flush=True)

            cluster_info[cid] = {
                'count': len(g),
                'rev_share': (g['total_revenue_inr'].sum() / tot_rev * 100.0) if tot_rev > 0 else 0.0,
                'mean_rev': g['total_revenue_inr'].mean(),
                'total_rev': g['total_revenue_inr'].sum(),
                'vol_2324_11m': v23,
                'vol_2425_11m': v24,
                'agg_growth': agg_growth,
                'disp_agg_growth': disp_agg_growth,
                'median_growth': median_growth,
                'disp_median_growth': disp_median_growth,
                'mean_vol': g['total_volume_kg'].mean(),
                'mean_freq': g['order_frequency'].mean(),
                'mean_cv': g['demand_volatility_cv'].mean()
            }

        persona_map = {}
        assigned_names = set()

        # 1. Identify Mega Accounts: Cluster with highest average revenue per account
        mega_cid = max(cluster_info.keys(), key=lambda c: cluster_info[c]['mean_rev'])
        persona_map[mega_cid] = {
            'persona_name': 'Mega Accounts',
            'recommended_strategy': 'Executive C-Suite Sponsorship, Dedicated Supply Line & Customized Volume Rebates',
            'tag_color': '#1E3A8A'  # Navy
        }
        assigned_names.add('Mega Accounts')

        # 2. Check if any remaining cluster qualifies for "High-Growth"
        # Rule: Use "High-Growth" only if the aggregate growth is above 5 percent and revenue share is above 10 percent.
        rem_cids = [c for c in cluster_info if c not in persona_map]
        hg_candidates = [c for c in rem_cids if cluster_info[c]['agg_growth'] > 5.0 and cluster_info[c]['rev_share'] > 10.0]
        if hg_candidates:
            hg_cid = max(hg_candidates, key=lambda c: cluster_info[c]['total_rev'])
            persona_map[hg_cid] = {
                'persona_name': 'High-Growth Strategic Partners',
                'recommended_strategy': 'SKU Range Expansion (Bakery & Frozen), Co-Marketing & Flexible Credit Lines',
                'tag_color': '#0D9488'  # Teal
            }
            assigned_names.add('High-Growth Strategic Partners')

        # 3. Core Strategic Partners: Cluster with highest remaining revenue
        rem_cids = [c for c in cluster_info if c not in persona_map]
        if rem_cids:
            core_cid = max(rem_cids, key=lambda c: cluster_info[c]['total_rev'])
            persona_map[core_cid] = {
                'persona_name': 'Core Strategic Partners',
                'recommended_strategy': 'Key Account Management, Annual Volume Rebates & Direct Container Routing',
                'tag_color': '#EA580C'  # Orange
            }
            assigned_names.add('Core Strategic Partners')

        # 4. The rest: "Long-Tail Accounts"
        rem_cids = [c for c in cluster_info if c not in persona_map]
        for idx, c in enumerate(rem_cids):
            p_name = 'Long-Tail Accounts' if idx == 0 else f'Long-Tail Accounts (Tier {idx+1})'
            persona_map[c] = {
                'persona_name': p_name,
                'recommended_strategy': 'Automated Re-ordering, Minimum Order Quantity (MOQ) Discipline & Freight Pooling',
                'tag_color': '#C026D3'  # Magenta
            }
            assigned_names.add(p_name)

        df_clustered['persona_name'] = df_clustered['cluster_id'].map(lambda x: persona_map[x]['persona_name'])
        df_clustered['recommended_strategy'] = df_clustered['cluster_id'].map(lambda x: persona_map[x]['recommended_strategy'])
        df_clustered['tag_color'] = df_clustered['cluster_id'].map(lambda x: persona_map[x]['tag_color'])

        return df_clustered

    def get_cluster_summary_profile(self) -> pd.DataFrame:
        """
        Returns a formatted executive summary table of all customer personas.
        """
        if self.features_df is None or 'persona_name' not in self.features_df.columns:
            raise ValueError("Run fit_kmeans_clustering() first.")

        summary = self.features_df.groupby(['cluster_id', 'persona_name', 'recommended_strategy', 'tag_color']).agg(
            distributor_count=('customer_id', 'count'),
            total_revenue_inr=('total_revenue_inr', 'sum'),
            avg_revenue_inr=('total_revenue_inr', 'mean'),
            avg_volume_kg=('total_volume_kg', 'mean'),
            vol_2324_11m=('vol_2324_11m', 'sum'),
            vol_2425_11m=('vol_2425_11m', 'sum'),
            avg_order_freq=('order_frequency', 'mean'),
            avg_volatility_cv=('demand_volatility_cv', 'mean'),
            avg_yoy_growth_pct=('yoy_growth_pct', 'mean'),
            median_yoy_growth_pct=('yoy_growth_pct', 'median')
        ).reset_index()

        summary['agg_yoy_growth_pct'] = np.where(
            summary['vol_2324_11m'] > 0,
            ((summary['vol_2425_11m'] - summary['vol_2324_11m']) / summary['vol_2324_11m']) * 100.0,
            0.0
        )
        summary['displayed_agg_growth'] = summary['agg_yoy_growth_pct'].clip(-100.0, 100.0)
        summary['displayed_median_growth'] = summary['median_yoy_growth_pct'].clip(-100.0, 100.0)

        summary['revenue_contribution_pct'] = (
            summary['total_revenue_inr'] / summary['total_revenue_inr'].sum()
        ) * 100.0

        return summary


class ProductPortfolioSegmentation:
    """
    SKU-level Portfolio Segmentation categorizing products into Strategic FMCG Quadrants
    (Stars, Cash Cows, Growth Bets, Long-Tail Niche).
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()

    def build_sku_quadrant_analysis(self) -> pd.DataFrame:
        """
        Aggregates product performance into Volume, Revenue, Pricing, and Growth metrics.
        """
        sku_agg = self.df.groupby(['material_code', 'material_desc', 'category_name', 'is_frozen']).agg(
            total_volume_kg=('quantity_kg', 'sum'),
            total_revenue_inr=('amount_inr', 'sum'),
            total_shipments=('invoice_no', 'nunique'),
            export_countries_count=('country', 'nunique'),
            avg_price_inr_per_kg=('unit_price_inr_per_kg', lambda x: x[x > 0].median() if (x > 0).any() else 0.0),
            avg_margin_proxy_pct=('margin_proxy_pct', 'mean')
        ).reset_index()

        # Categorize into BCG/FMCG Quadrant
        med_vol = sku_agg['total_volume_kg'].median()
        med_rev = sku_agg['total_revenue_inr'].median()

        def assign_sku_class(row: pd.Series) -> str:
            if row['total_volume_kg'] >= med_vol and row['total_revenue_inr'] >= med_rev:
                return "Core Export Pillars (High Vol / High Rev)"
            elif row['total_volume_kg'] < med_vol and row['total_revenue_inr'] >= med_rev:
                return "Premium High-Margin SKUs (Low Vol / High Rev)"
            elif row['total_volume_kg'] >= med_vol and row['total_revenue_inr'] < med_rev:
                return "Volume Drivers / Entry Packs (High Vol / Low Rev)"
            else:
                return "Long-Tail Regional SKUs (Low Vol / Low Rev)"

        sku_agg['portfolio_quadrant'] = sku_agg.apply(assign_sku_class, axis=1)
        return sku_agg


if __name__ == "__main__":
    from pathlib import Path
    data_file = Path("data/haldirams_sales_consolidated.parquet")
    if data_file.exists():
        print("Testing Customer & Product Segmentation Module...")
        df = pd.read_parquet(data_file)
        seg = B2BCustomerSegmentation(df)
        clustered_df, metrics = seg.fit_kmeans_clustering(n_clusters=4)
        summary = seg.get_cluster_summary_profile()
        print("\n--- CLUSTERING PERFORMANCE ---")
        print(f"Silhouette Score: {metrics['silhouette_score']} | Inertia: {metrics['inertia']}")
        print("\n--- DISTRIBUTOR PERSONA SUMMARY ---")
        print(summary[['cluster_id', 'persona_name', 'distributor_count', 'revenue_contribution_pct', 'avg_volatility_cv']].to_string(index=False))
    else:
        print("Data file not found. Ingest data first.")
