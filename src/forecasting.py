"""
================================================================================
Haldiram's International Export Analytics Pipeline - Demand Forecasting Module
================================================================================
Author: Senior Data Scientist (Technical Lead)
Project: Haldiram's Final-Year Export Sales & ML Analytics
Module: Hierarchical Monthly Demand Forecasting (SARIMAX vs. Prophet vs. XGBoost)

Viva Defense Notes:
- Aggregation Grain: Monthly demand (Quantity in KG and Invoiced Amount in INR)
  at (Zone x Category) and (Country x Category) grains. Monthly aggregation cancels out
  daily dispatch noise and aligns with FMCG supply chain factory replenishment cycles.
- Walk-Forward Split: Respects time ordering (e.g. 27-36 months train, remaining 9-12 months test).
- Festival Exogenous Regressors: Explicit holiday lead-time flags (Diwali in Oct/Nov,
  Ramadan/Eid, New Year) to model international FMCG surge shipments sent 4-6 weeks in advance.
- Tri-Model Benchmark:
  1. SARIMAX: Classical autoregressive integrated moving average with seasonal order (12).
  2. Prophet: Decomposable additive model robust to missing months and trend inflection.
  3. Lagged Feature XGBoost: Gradient boosted decision trees utilizing lag-1, lag-2, lag-12,
     rolling 3M/6M means, and calendar cyclic features.
- Metric Rigor: Reports RMSE (penalizes stockout catastrophic errors) and MAPE (FMCG industry standard).
================================================================================
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

import xgboost as xgb

warnings.filterwarnings("ignore")


# Indian and Global Festive Calendar Windows (Key demand drivers for Haldiram's overseas shipments)
# Note: FMCG international export shipments occur ~30-45 days PRIOR to festival dates.
FESTIVAL_CALENDAR = {
    # Diwali dispatch surge months (usually Sep - Oct)
    'diwali_dispatch': ['2021-09', '2021-10', '2022-09', '2022-10', '2023-09', '2023-10', '2024-09', '2024-10', '2025-09', '2025-10'],
    # Ramadan / Eid dispatch surge months (Middle East & UK surge)
    'eid_dispatch': ['2021-03', '2021-04', '2022-03', '2022-04', '2023-02', '2023-03', '2024-02', '2024-03', '2025-02', '2025-03'],
    # Summer / Holiday demand surge (May - Jun)
    'summer_dispatch': ['2021-05', '2021-06', '2022-05', '2022-06', '2023-05', '2023-06', '2024-05', '2024-06', '2025-05', '2025-06']
}


def calculate_forecast_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Computes standard regression & time-series evaluation metrics.
    
    Why: FMCG supply chains evaluate MAPE for inventory turnover planning and
    RMSE to quantify variance risks that cause factory warehouse stockouts.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)
    # Clip negative predictions to zero (physical shipments cannot be negative)
    y_p = np.maximum(0.0, y_p)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    
    # Safe MAPE calculation avoiding zero-division
    mask = y_t > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)
    else:
        mape = 0.0

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape, 2)
    }


class DemandForecastingEngine:
    """
    Production-grade multi-model forecasting engine for FMCG Export time-series.
    """

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()
        if 'month_start_date' not in self.df.columns or self.df['month_start_date'].isna().all():
            self.df['month_start_date'] = pd.to_datetime(self.df['invoice_date']).dt.to_period('M').dt.to_timestamp()
        
    def aggregate_time_series(self, 
                              zone: Optional[str] = None, 
                              country: Optional[str] = None, 
                              category: Optional[str] = None) -> pd.DataFrame:
        """
        Filters and aggregates transactional data into a continuous monthly time series.
        Fills missing monthly periods with zeros to maintain temporal continuity.
        """
        filtered = self.df.copy()
        if zone and zone != 'All Zones':
            filtered = filtered[filtered['zone'] == zone]
        if country and country != 'All Countries':
            filtered = filtered[filtered['country'] == country]
        if category and category != 'All Categories':
            filtered = filtered[filtered['category_name'] == category]

        # Group by month
        monthly = filtered.groupby('month_start_date').agg(
            quantity_kg=('quantity_kg', 'sum'),
            amount_inr=('amount_inr', 'sum'),
            shipments_count=('invoice_no', 'nunique')
        ).reset_index()

        if monthly.empty:
            return pd.DataFrame(columns=['ds', 'y', 'amount_inr', 'month_str', 'is_diwali', 'is_eid', 'is_summer'])

        # Find the last month with real invoice data and print it
        min_date = monthly['month_start_date'].min()
        max_date = monthly['month_start_date'].max()
        print(f"[DemandForecasting] Last real data month: {max_date.strftime('%b %Y')}")

        # Strictly span between first and last real data month (never zero-fill after it)
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        
        full_ts = pd.DataFrame({'month_start_date': full_date_range})
        merged = full_ts.merge(monthly, on='month_start_date', how='left')
        
        # If middle months have missing data, use linear interpolation (do not zero fill)
        merged['quantity_kg'] = merged['quantity_kg'].interpolate(method='linear').bfill()
        merged['amount_inr'] = merged['amount_inr'].interpolate(method='linear').bfill()
        merged['shipments_count'] = merged['shipments_count'].interpolate(method='linear').bfill()

        # Format for forecasting
        merged['ds'] = merged['month_start_date']
        merged['y'] = merged['quantity_kg']
        merged['month_str'] = merged['ds'].dt.strftime('%Y-%m')
        
        # Add festival and seasonal indicator features
        merged['is_diwali'] = merged['month_str'].isin(FESTIVAL_CALENDAR['diwali_dispatch']).astype(int)
        merged['is_eid'] = merged['month_str'].isin(FESTIVAL_CALENDAR['eid_dispatch']).astype(int)
        merged['is_summer'] = merged['month_str'].isin(FESTIVAL_CALENDAR['summer_dispatch']).astype(int)

        return merged

    def train_test_split(self, ts_data: pd.DataFrame, test_months: int = 9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a 9-month time-based holdout split to prevent look-ahead bias.
        """
        if len(ts_data) <= test_months + 6:
            # Fallback if time series is short: 80-20 split
            split_idx = int(len(ts_data) * 0.8)
        else:
            split_idx = len(ts_data) - test_months
            
        train = ts_data.iloc[:split_idx].copy()
        test = ts_data.iloc[split_idx:].copy()
        return train, test

    def fit_predict_sarima(self, train: pd.DataFrame, test: pd.DataFrame, 
                           forecast_horizon: int = 6) -> Tuple[pd.DataFrame, Dict[str, float], Optional[SARIMAX]]:
        """
        Fits a seasonal SARIMAX model with exogenous festival flags.
        Order: (1, 1, 1) x (1, 0, 1, 12) with exog=[is_diwali, is_eid].
        """
        exog_cols = ['is_diwali', 'is_eid']
        y_train = train['y']
        exog_train = train[exog_cols]
        exog_test = test[exog_cols]

        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 0, 12) if len(train) >= 24 else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=200)
            
            # Predict on test set
            test_forecast_res = fitted_model.get_forecast(steps=len(test), exog=exog_test)
            pred_test = np.maximum(0.0, test_forecast_res.predicted_mean.values)
            metrics = calculate_forecast_metrics(test['y'].values, pred_test)

            # Refit on full series (train + test) for future forecast
            full_exog = pd.concat([exog_train, exog_test], ignore_index=True)
            full_y = pd.concat([y_train, test['y']], ignore_index=True)
            
            last_date = test['ds'].max() if not test.empty else train['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
            future_df = pd.DataFrame({'ds': future_dates})
            future_df['month_str'] = future_df['ds'].dt.strftime('%Y-%m')
            future_df['is_diwali'] = future_df['month_str'].isin(FESTIVAL_CALENDAR['diwali_dispatch']).astype(int)
            future_df['is_eid'] = future_df['month_str'].isin(FESTIVAL_CALENDAR['eid_dispatch']).astype(int)
            exog_future = future_df[exog_cols]

            model_full = SARIMAX(
                full_y,
                exog=full_exog,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 0, 12) if len(full_y) >= 24 else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_full = model_full.fit(disp=False, maxiter=150)
            future_forecast_res = fitted_full.get_forecast(steps=forecast_horizon, exog=exog_future)
            pred_future = np.maximum(0.0, future_forecast_res.predicted_mean.values)

            # Compile predictions dataframe
            results_df = pd.DataFrame({
                'ds': list(test['ds']) + list(future_dates),
                'y_pred': np.concatenate([pred_test, pred_future]),
                'type': ['Test_Evaluation'] * len(test) + ['Future_Forecast'] * forecast_horizon,
                'model': 'SARIMAX'
            })
            return results_df, metrics, fitted_model

        except Exception as e:
            print(f"[SARIMAX Fallback] Convergence notice: {e}. Using mean persistence.")
            # Fallback mean persistence model
            mean_val = float(train['y'].mean())
            pred_test = np.full(len(test), mean_val)
            pred_future = np.full(forecast_horizon, mean_val)
            metrics = calculate_forecast_metrics(test['y'].values, pred_test)
            
            last_date = test['ds'].max() if not test.empty else train['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
            results_df = pd.DataFrame({
                'ds': list(test['ds']) + list(future_dates),
                'y_pred': np.concatenate([pred_test, pred_future]),
                'type': ['Test_Evaluation'] * len(test) + ['Future_Forecast'] * forecast_horizon,
                'model': 'SARIMAX (Baseline)'
            })
            return results_df, metrics, None

    def fit_predict_prophet(self, train: pd.DataFrame, test: pd.DataFrame, 
                            forecast_horizon: int = 6) -> Tuple[pd.DataFrame, Dict[str, float], Optional[object]]:
        """
        Fits Meta Prophet additive model with holidays and yearly seasonality.
        """
        if not HAS_PROPHET:
            # Fallback polynomial-fourier model if prophet is absent
            return self._prophet_fallback(train, test, forecast_horizon)

        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative' if (train['y'] > 0).all() else 'additive'
            )
            m.add_regressor('is_diwali')
            m.add_regressor('is_eid')

            prophet_train = train[['ds', 'y', 'is_diwali', 'is_eid']].copy()
            m.fit(prophet_train)

            # Build future dataframe encompassing test + horizon
            total_periods = len(test) + forecast_horizon
            future = m.make_future_dataframe(periods=total_periods, freq='MS')
            future['month_str'] = future['ds'].dt.strftime('%Y-%m')
            future['is_diwali'] = future['month_str'].isin(FESTIVAL_CALENDAR['diwali_dispatch']).astype(int)
            future['is_eid'] = future['month_str'].isin(FESTIVAL_CALENDAR['eid_dispatch']).astype(int)

            forecast = m.predict(future)
            test_preds = forecast.iloc[len(train):len(train) + len(test)]['yhat'].values
            metrics = calculate_forecast_metrics(test['y'].values, test_preds)

            eval_and_future = forecast.iloc[len(train):].copy()
            results_df = pd.DataFrame({
                'ds': eval_and_future['ds'].values,
                'y_pred': np.maximum(0.0, eval_and_future['yhat'].values),
                'y_lower': np.maximum(0.0, eval_and_future['yhat_lower'].values),
                'y_upper': np.maximum(0.0, eval_and_future['yhat_upper'].values),
                'type': ['Test_Evaluation'] * len(test) + ['Future_Forecast'] * forecast_horizon,
                'model': 'Prophet'
            })
            return results_df, metrics, m

        except Exception as e:
            print(f"[Prophet Fallback] Notice: {e}")
            return self._prophet_fallback(train, test, forecast_horizon)

    def _prophet_fallback(self, train: pd.DataFrame, test: pd.DataFrame, forecast_horizon: int) -> Tuple[pd.DataFrame, Dict[str, float], None]:
        """Simple seasonal decomposition fallback if Prophet has C++ stan compile issues."""
        month_avg = train.groupby(train['ds'].dt.month)['y'].mean().to_dict()
        overall_avg = train['y'].mean()
        
        last_date = test['ds'].max() if not test.empty else train['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
        all_dates = list(test['ds']) + list(future_dates)
        
        preds = [month_avg.get(d.month, overall_avg) for d in all_dates]
        test_preds = preds[:len(test)]
        metrics = calculate_forecast_metrics(test['y'].values, test_preds)

        results_df = pd.DataFrame({
            'ds': all_dates,
            'y_pred': np.maximum(0.0, preds),
            'type': ['Test_Evaluation'] * len(test) + ['Future_Forecast'] * forecast_horizon,
            'model': 'Seasonal Decomposition'
        })
        return results_df, metrics, None

    def fit_predict_xgboost(self, ts_data: pd.DataFrame, test_months: int = 9, 
                            forecast_horizon: int = 6) -> Tuple[pd.DataFrame, Dict[str, float], Optional[xgb.XGBRegressor]]:
        """
        Fits XGBoost Regressor with autoregressive lag and rolling statistics features.
        
        Why: Tree ensembles capture non-linear interactions between promotional festival months
        and moving averages without stationarity transformations.
        """
        df_feat = ts_data.copy()
        
        # Engineer Lags
        df_feat['lag_1'] = df_feat['y'].shift(1)
        df_feat['lag_2'] = df_feat['y'].shift(2)
        df_feat['lag_3'] = df_feat['y'].shift(3)
        df_feat['lag_12'] = df_feat['y'].shift(12)

        # Rolling statistics
        df_feat['rolling_mean_3'] = df_feat['y'].shift(1).rolling(window=3, min_periods=1).mean()
        df_feat['rolling_std_3'] = df_feat['y'].shift(1).rolling(window=3, min_periods=1).std().fillna(0.0)
        df_feat['rolling_mean_6'] = df_feat['y'].shift(1).rolling(window=6, min_periods=1).mean()

        # Calendar features
        df_feat['month'] = df_feat['ds'].dt.month
        df_feat['quarter'] = df_feat['ds'].dt.quarter
        df_feat['sin_month'] = np.sin(2 * np.pi * df_feat['month'] / 12)
        df_feat['cos_month'] = np.cos(2 * np.pi * df_feat['month'] / 12)

        # Drop initial NaN lag rows
        feature_cols = ['lag_1', 'lag_2', 'lag_3', 'lag_12', 'rolling_mean_3', 
                        'rolling_std_3', 'rolling_mean_6', 'is_diwali', 'is_eid', 
                        'is_summer', 'sin_month', 'cos_month']
        
        # Fill missing early lags with backfill/median
        df_feat[feature_cols] = df_feat[feature_cols].bfill().fillna(0.0)

        # Split
        train_idx = len(df_feat) - test_months
        train_df = df_feat.iloc[:train_idx]
        test_df = df_feat.iloc[train_idx:]

        X_train, y_train = train_df[feature_cols], train_df['y']
        X_test, y_test = test_df[feature_cols], test_df['y']

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)

        pred_test = model.predict(X_test)
        metrics = calculate_forecast_metrics(y_test.values, pred_test)

        # Recursive Autoregressive Forecasting for Future Horizon
        last_history = df_feat.copy()
        future_preds = []
        last_date = ts_data['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')

        for f_date in future_dates:
            m_str = f_date.strftime('%Y-%m')
            # Extract lagged values from updated series
            recent_y = last_history['y'].values
            lag1 = recent_y[-1] if len(recent_y) >= 1 else 0.0
            lag2 = recent_y[-2] if len(recent_y) >= 2 else lag1
            lag3 = recent_y[-3] if len(recent_y) >= 3 else lag2
            lag12 = recent_y[-12] if len(recent_y) >= 12 else np.mean(recent_y)

            roll3 = np.mean(recent_y[-3:]) if len(recent_y) >= 3 else lag1
            roll_std3 = np.std(recent_y[-3:]) if len(recent_y) >= 3 else 0.0
            roll6 = np.mean(recent_y[-6:]) if len(recent_y) >= 6 else roll3

            cur_features = pd.DataFrame([{
                'lag_1': lag1,
                'lag_2': lag2,
                'lag_3': lag3,
                'lag_12': lag12,
                'rolling_mean_3': roll3,
                'rolling_std_3': roll_std3,
                'rolling_mean_6': roll6,
                'is_diwali': int(m_str in FESTIVAL_CALENDAR['diwali_dispatch']),
                'is_eid': int(m_str in FESTIVAL_CALENDAR['eid_dispatch']),
                'is_summer': int(m_str in FESTIVAL_CALENDAR['summer_dispatch']),
                'sin_month': np.sin(2 * np.pi * f_date.month / 12),
                'cos_month': np.cos(2 * np.pi * f_date.month / 12)
            }])[feature_cols]

            next_pred = float(model.predict(cur_features)[0])
            next_pred = max(0.0, next_pred)
            future_preds.append(next_pred)

            # Append synthetic row to last_history for next step lag
            new_row = pd.DataFrame({'ds': [f_date], 'y': [next_pred]})
            last_history = pd.concat([last_history, new_row], ignore_index=True)

        results_df = pd.DataFrame({
            'ds': list(test_df['ds']) + list(future_dates),
            'y_pred': np.maximum(0.0, np.concatenate([pred_test, future_preds])),
            'type': ['Test_Evaluation'] * len(test_df) + ['Future_Forecast'] * forecast_horizon,
            'model': 'XGBoost (Lagged)'
        })
        return results_df, metrics, model

    def benchmark_all_models(self, zone: Optional[str] = None, 
                            country: Optional[str] = None, 
                            category: Optional[str] = None,
                            forecast_horizon: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Runs comparative benchmarking across SARIMAX, Prophet, and XGBoost.
        Returns:
          1. Historical ground truth dataframe
          2. Combined forecast predictions dataframe
          3. Metric comparison scorecard table
        """
        ts_data = self.aggregate_time_series(zone=zone, country=country, category=category)
        if len(ts_data) < 12:
            raise ValueError(f"Insufficient time-series history ({len(ts_data)} months). Need at least 12 months.")

        train, test = self.train_test_split(ts_data, test_months=9)

        # 1. SARIMAX
        sarima_df, sarima_metrics, _ = self.fit_predict_sarima(train, test, forecast_horizon)
        # 2. Prophet
        prophet_df, prophet_metrics, _ = self.fit_predict_prophet(train, test, forecast_horizon)
        # 3. XGBoost
        xgb_df, xgb_metrics, _ = self.fit_predict_xgboost(ts_data, test_months=9, forecast_horizon=forecast_horizon)

        # Combine predictions
        all_forecasts = pd.concat([sarima_df, prophet_df, xgb_df], ignore_index=True)

        # Benchmark Scorecard
        metrics_summary = pd.DataFrame([
            {'Model': 'SARIMAX (1,1,1)x(1,0,0)12', **sarima_metrics, 'Type': 'Statistical / Econometric'},
            {'Model': 'Prophet (Additive Holidays)', **prophet_metrics, 'Type': 'Additive Decomposition'},
            {'Model': 'XGBoost (Lags + Rolling)', **xgb_metrics, 'Type': 'Gradient Boosted Trees'}
        ])

        return ts_data, all_forecasts, metrics_summary


if __name__ == "__main__":
    from pathlib import Path
    data_file = Path("data/haldirams_sales_consolidated.parquet")
    if data_file.exists():
        print("Testing Demand Forecasting Engine on Consolidated Export Data...")
        df = pd.read_parquet(data_file)
        engine = DemandForecastingEngine(df)
        ts, forecasts, scorecard = engine.benchmark_all_models(category='Namkeens & Savouries')
        print("\n--- MODEL BENCHMARK SCORECARD ---")
        print(scorecard.to_string(index=False))
        print("\nSample Forecasts:")
        print(forecasts.head(10))
    else:
        print("Consolidated data file not yet ready. Run data_pipeline.py first.")
