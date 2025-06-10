import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# 1) LOAD & RENAME (same as original)
usecols = [
    'LSOA code','Ward code','Date','Burglary Count','covid_period',
    'LSOA Area Size (HA)',
    'Overall Ranking - IMD','Housing rank','Health rank',
    'Living environment rank','Crime rank','Education rank',
    'Income rank','Employment rank'
]
dtypes = {
    'LSOA code':'category','Ward code':'category',
    'Burglary Count':'int16','covid_period':'int8',
    'LSOA Area Size (HA)':'float32',
    'Overall Ranking - IMD':'float32','Housing rank':'float32','Health rank':'float32',
    'Living environment rank':'float32','Crime rank':'float32','Education rank':'float32',
    'Income rank':'float32','Employment rank':'float32'
}
df = pd.read_csv(
    r"C:/Users/20232553/Downloads/data_aggregated_covid.csv",
    usecols=usecols, dtype=dtypes, parse_dates=['Date']
)
df.rename(columns={
    'LSOA code':'lsoa_code','Ward code':'ward_code','Burglary Count':'burglary_count',
    'Date':'date','LSOA Area Size (HA)':'area_ha',
    'Overall Ranking - IMD':'imd_overall','Housing rank':'imd_housing','Health rank':'imd_health',
    'Living environment rank':'imd_living_env','Crime rank':'imd_crime',
    'Education rank':'imd_education','Income rank':'imd_income','Employment rank':'imd_employment'
}, inplace=True)

print(f"Data loaded: {df.shape}")

# 2) FILTER & SORT
df = df[df['covid_period']==0].copy()
df.sort_values(['lsoa_code','date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 3) BASIC TEMPORAL FEATURES
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['time_idx'] = (df['year'] - df['year'].min())*12 + (df['month'] - df['month'].min())

# MULTI-HARMONIC SEASONALITY FEATURES
print("Creating multi-harmonic seasonality features...")

# 1) Monthly patterns (12-month cycle) - multiple harmonics
for harmonic in [1, 2, 3]:  # 1st, 2nd, 3rd harmonics
    df[f'sin_12h{harmonic}'] = np.sin(2 * np.pi * harmonic * df['month'] / 12)
    df[f'cos_12h{harmonic}'] = np.cos(2 * np.pi * harmonic * df['month'] / 12)

# 2) Quarterly patterns (3-month cycle)
for harmonic in [1, 2]:
    df[f'sin_3h{harmonic}'] = np.sin(2 * np.pi * harmonic * df['month'] / 3)
    df[f'cos_3h{harmonic}'] = np.cos(2 * np.pi * harmonic * df['month'] / 3)

# 3) Bi-annual patterns (6-month cycle)
for harmonic in [1, 2]:
    df[f'sin_6h{harmonic}'] = np.sin(2 * np.pi * harmonic * df['month'] / 6)
    df[f'cos_6h{harmonic}'] = np.cos(2 * np.pi * harmonic * df['month'] / 6)

# 4) Seasonal transition patterns (4-month cycle)
for harmonic in [1, 2]:
    df[f'sin_4h{harmonic}'] = np.sin(2 * np.pi * harmonic * df['month'] / 4)
    df[f'cos_4h{harmonic}'] = np.cos(2 * np.pi * harmonic * df['month'] / 4)

# 5) Combined harmonic interactions
df['season_interaction_1'] = df['sin_12h1'] * df['sin_6h1']
df['season_interaction_2'] = df['cos_12h1'] * df['cos_6h1']
df['season_interaction_3'] = df['sin_12h1'] * df['cos_6h1']

# 6) Area-specific seasonality
df['crime_seasonal_1'] = df['sin_12h1'] * df['imd_crime']
df['crime_seasonal_2'] = df['cos_12h1'] * df['imd_crime']
df['crime_seasonal_3'] = df['sin_6h1'] * df['imd_crime']

# 7) Time-varying seasonality
df['evolving_season_1'] = df['sin_12h1'] * df['time_idx']
df['evolving_season_2'] = df['cos_12h1'] * df['time_idx']

# Keep season indicators (these are still useful)
df['is_winter'] = (df['month'].isin([12,1,2])).astype(int)
df['is_summer'] = (df['month'].isin([6,7,8])).astype(int)

print("Multi-harmonic seasonality features created")



# 4) IMD FEATURES
imd_feats = ['imd_overall','imd_housing','imd_health','imd_living_env',
             'imd_crime','imd_education','imd_income','imd_employment']

# Impute missing values
imd_imp = SimpleImputer(strategy='median')
df[imd_feats] = imd_imp.fit_transform(df[imd_feats])

# Key IMD interactions
df['crime_x_income'] = df['imd_crime'] * df['imd_income']
df['crime_x_education'] = df['imd_crime'] * df['imd_education']

# Single clustering
km = KMeans(n_clusters=8, random_state=42, n_init=10)
df['imd_cluster'] = km.fit_predict(df[imd_feats])

print("IMD features created")

# 5) KEY LAG FEATURES
print("Creating lag features...")
grp = df.groupby('lsoa_code')['burglary_count']

# Essential lags only
key_lags = [1, 2, 3, 6, 12]
for lag in key_lags:
    df[f'lag_{lag}m'] = grp.shift(lag)
    print(f"  lag_{lag}m created")

# ADVANCED LAG FEATURES (add right after the basic lag features)
print("Adding advanced lag features...")

# Seasonal lags (same month different years) - VERY POWERFUL FOR CRIME
df['lag_12m_seasonal'] = grp.shift(12)  # Same month last year
df['lag_24m_seasonal'] = grp.shift(24)  # Same month 2 years ago
print("  Seasonal lags created")

# Year-over-year changes (THE MOST PREDICTIVE FEATURE FOR CRIME)
df['yoy_change'] = (df['lag_1m'] - df['lag_12m_seasonal']) / (df['lag_12m_seasonal'] + 1e-6)
df['yoy_change_2y'] = (df['lag_1m'] - df['lag_24m_seasonal']) / (df['lag_24m_seasonal'] + 1e-6)
print("  Year-over-year changes created")

# Multi-step trends (better than simple differences)
df['trend_2m'] = grp.shift(1) - grp.shift(3)  # 2-month change
df['trend_6m'] = grp.shift(1) - grp.shift(7)  # 6-month change
print("  Multi-step trends created")

# Seasonal momentum (current vs same time last year)
df['seasonal_momentum'] = df['lag_1m'] / (df['lag_12m_seasonal'] + 1e-6)
df['seasonal_momentum_2y'] = df['lag_1m'] / (df['lag_24m_seasonal'] + 1e-6)
print("  Seasonal momentum created")

print("✅ Advanced lag features complete!")

# SIMPLE BUT EFFECTIVE MONTHLY FEATURES
print("Adding simple but effective monthly features...")

# Better area size effects (monthly appropriate)
df['area_crime_ratio'] = np.log1p(df['lag_1m']) / np.log1p(df['area_ha'])
df['area_size_category'] = pd.qcut(df['area_ha'], q=3, labels=[0,1,2], duplicates='drop')

# Simple ratios using only existing lag features
df['lag1_vs_seasonal'] = df['lag_1m'] / (df['lag_12m_seasonal'] + 1e-6)
df['lag3_vs_seasonal'] = df['lag_3m'] / (df['lag_12m_seasonal'] + 1e-6)
df['lag1_vs_lag6'] = df['lag_1m'] / (df['lag_6m'] + 1e-6)

# Better crime intensity (monthly)
df['crime_per_area_log'] = np.log1p(df['lag_1m']) / np.log1p(df['area_ha'])

# Monthly economic patterns
df['benefit_month'] = ((df['month'] == 1) | (df['month'] == 4) | (df['month'] == 7) | (df['month'] == 10)).astype(int)

# School term effects
df['school_term'] = (~df['month'].isin([7, 8, 12])).astype(int)

# Better momentum using only existing features
df['yoy_momentum'] = np.abs(df['yoy_change'])  # Strength of year-over-year change
df['seasonal_stability'] = 1 / (np.abs(df['yoy_change']) + 1e-6)  # More stable = higher value

print("Simple effective monthly features created")

# 6) ENHANCED ROLLING FEATURES
print("Creating rolling features...")

# Key rolling windows
df['roll3_mean'] = grp.shift(1).rolling(3).mean()
df['roll6_mean'] = grp.shift(1).rolling(6).mean()
df['roll12_mean'] = grp.shift(1).rolling(12).mean()

df['roll3_std'] = grp.shift(1).rolling(3).std()
df['roll6_std'] = grp.shift(1).rolling(6).std()

# High-value percentiles (capture extremes better than std)
df['roll6_p75'] = grp.shift(1).rolling(6).quantile(0.75)
df['roll6_p25'] = grp.shift(1).rolling(6).quantile(0.25)
df['roll6_iqr'] = df['roll6_p75'] - df['roll6_p25']

# Exponentially weighted (recent months matter more)
df['ewm_3'] = grp.shift(1).ewm(alpha=0.4).mean()  # ~3 month half-life
df['ewm_6'] = grp.shift(1).ewm(alpha=0.2).mean()  # ~6 month half-life

print("Enhanced rolling features created")

# 7) ENHANCED TREND FEATURES
print("Creating trend features...")

# Key differences (lagged to prevent leakage)
df['diff_1m'] = grp.shift(1) - grp.shift(2)
df['diff_3m'] = grp.shift(1) - grp.shift(4)
df['diff_12m'] = grp.shift(1) - grp.shift(13)

# Percentage changes
df['pct_change_3m'] = df['diff_3m'] / (grp.shift(4) + 1e-6)
df['pct_change_12m'] = df['diff_12m'] / (grp.shift(13) + 1e-6)

# Volatility
df['volatility_3m'] = grp.shift(1).rolling(3).std()
df['volatility_6m'] = grp.shift(1).rolling(6).std()

# Momentum indicators
df['momentum_3m'] = df['roll3_mean'] / (df['roll6_mean'] + 1e-6)  # Recent vs longer term
df['momentum_6m'] = df['roll6_mean'] / (df['roll12_mean'] + 1e-6)

# Trend direction (simple but effective)
df['trend_up_3m'] = (df['lag_1m'] > df['roll3_mean']).astype(int)
df['trend_up_6m'] = (df['lag_1m'] > df['roll6_mean']).astype(int)

print("Enhanced trend features created")

# 8) WARD-LEVEL FEATURES
print("Creating ward features...")
wgrp = df.groupby('ward_code')['burglary_count']

df['ward_lag1'] = wgrp.shift(1)
df['ward_lag12'] = wgrp.shift(12)
df['ward_roll6_mean'] = wgrp.shift(1).rolling(6).mean()

print("Ward features created")

# 9) ENHANCED TARGET ENCODING
print("Creating target encoding...")

def fast_target_encode(df_input, cat_col, target_col, smoothing=10):
    """Fast target encoding with smoothing"""
    # Calculate global mean
    global_mean = df_input[target_col].mean()
    
    # Calculate category means and counts
    agg = df_input.groupby(cat_col)[target_col].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
    
    # Map to original data
    return df_input[cat_col].map(smoothed).fillna(global_mean)

# Apply target encoding to key categories
df['lsoa_target_enc'] = fast_target_encode(df, 'lsoa_code', 'lag_1m')
df['ward_target_enc'] = fast_target_encode(df, 'ward_code', 'lag_1m')
df['month_target_enc'] = fast_target_encode(df, 'month', 'lag_1m')
df['cluster_target_enc'] = fast_target_encode(df, 'imd_cluster', 'lag_1m')

# Create numeric combination instead of string
df['lsoa_ward_numeric'] = df['lsoa_code'].cat.codes * 10000 + df['ward_code'].cat.codes
df['lsoa_ward_target_enc'] = fast_target_encode(df, 'lsoa_ward_numeric', 'lag_1m')

print("Enhanced target encoding created")

# 10) ENHANCED INTERACTIONS
print("Creating interactions...")

# Core interactions
df['area_x_crime'] = df['area_ha'] * df['imd_crime']
df['lag1_x_seasonal'] = df['lag_1m'] * df['sin_12h1']
df['lag1_x_volatility'] = df['lag_1m'] * df['volatility_6m']

# Performance interactions
df['lag1_x_ward_avg'] = df['lag_1m'] * df['ward_target_enc']
df['lag1_x_lsoa_avg'] = df['lag_1m'] * df['lsoa_target_enc']

# Seasonal interactions with area characteristics
df['winter_x_crime'] = df['is_winter'] * df['imd_crime']
df['summer_x_crime'] = df['is_summer'] * df['imd_crime']

# Rolling vs lag interactions (momentum)
df['roll3_vs_lag1'] = df['roll3_mean'] / (df['lag_1m'] + 1e-6)
df['roll6_vs_lag1'] = df['roll6_mean'] / (df['lag_1m'] + 1e-6)

# Time interactions
df['time_x_crime'] = df['time_idx'] * df['imd_crime']  # Crime trends over time

print("Enhanced interactions created")

# NEW: DENSITY & SPATIAL FEATURES
print("Creating density features...")

# Crime density (crimes per area unit)
df['crime_density'] = df['lag_1m'] / (df['area_ha'] + 1e-6)

# Area size categories (numeric, not categorical)
df['area_quintile'] = pd.qcut(df['area_ha'], q=5, labels=False, duplicates='drop')
df['area_cat_target_enc'] = fast_target_encode(df, 'area_quintile', 'lag_1m')

# Area size vs crime relationship
df['area_crime_efficiency'] = df['lag_1m'] / (np.log1p(df['area_ha']))

print("Density features created")

# NEW: STABILITY FEATURES
print("Creating stability features...")

# Coefficient of variation (volatility relative to mean)
df['cv_6m'] = df['roll6_std'] / (df['roll6_mean'] + 1e-6)

# Count of months above/below average (fix the comparison)
df['months_above_avg'] = (grp.shift(1).rolling(6).apply(lambda x: (x > x.mean()).sum() if len(x) > 0 else 0))
df['months_below_avg'] = (grp.shift(1).rolling(6).apply(lambda x: (x < x.mean()).sum() if len(x) > 0 else 0))

# Streak features (consecutive up/down months) - simplified
df['is_increasing'] = (grp.shift(1) > grp.shift(2)).astype(int)

# Simple trend consistency (how often recent trend continues)
df['trend_consistency'] = grp.shift(1).rolling(3).apply(lambda x: (x.diff() > 0).sum() if len(x) > 1 else 0)

print("Stability features created")

# 12) PREPARE FEATURES (EXCLUDE TEMP COLUMNS)
exclude_cols = ['burglary_count', 'lsoa_code', 'ward_code', 'date', 'covid_period', 
                'lsoa_ward_combo', 'area_category']  # Exclude temp/categorical columns
features = [col for col in df.columns if col not in exclude_cols]

# Remove any remaining non-numeric columns
numeric_features = []
for col in features:
    if df[col].dtype in ['object', 'category']:
        print(f"Skipping non-numeric column: {col}")
    else:
        numeric_features.append(col)

features = numeric_features
print(f"Total numeric features: {len(features)}")

# 13) DROP NaNs
print("Handling missing values...")
important_features = [f'lag_{l}m' for l in key_lags] + ['roll3_mean', 'ward_lag1']
df.dropna(subset=important_features, inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Final dataset: {df.shape}")

# 14) PREPARE X, y
X = df[features]
y = df['burglary_count']

# 15) SPLIT
split = int(len(df)*0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
df_test = df.iloc[split:].copy()

# 16) SIMPLIFIED IMPUTATION (NUMERIC ONLY)
print("Handling missing values...")

# Only use numeric features - no categorical handling needed
imp = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imp.fit_transform(X_train), columns=features, index=X_train.index)
X_test = pd.DataFrame(imp.transform(X_test), columns=features, index=X_test.index)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# 17) LOAD PARAMS AND TRAIN
params = json.load(open(r"MODELS/best_params_combined.json"))

# Quick CV
print("Running CV...")
tscv = TimeSeriesSplit(n_splits=3)  # Reduced folds for speed
cv_rows = []
for i, (tr, va) in enumerate(tscv.split(X_train), start=1):
    model = xgb.XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1)
    model.fit(X_train.iloc[tr], y_train.iloc[tr])
    preds = model.predict(X_train.iloc[va])
    cv_rows.append({
        'fold': i,
        'R2': r2_score(y_train.iloc[va], preds),
        'MAE': mean_absolute_error(y_train.iloc[va], preds),
        'RMSE': np.sqrt(mean_squared_error(y_train.iloc[va], preds))
    })
cv_df = pd.DataFrame(cv_rows)

# 18) FINAL MODEL
print("Training final model...")
final = xgb.XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1)
final.fit(X_train, y_train)
df_test['pred'] = final.predict(X_test)

# 19) RESULTS AND MONTHLY-LEVEL EVALUATION
overall_r2 = r2_score(df_test['burglary_count'], df_test['pred'])
overall_mae = mean_absolute_error(df_test['burglary_count'], df_test['pred'])
overall_rmse = np.sqrt(mean_squared_error(df_test['burglary_count'], df_test['pred']))

print(f"\n=== FAST MODEL RESULTS ===")
print(f"Overall R²: {overall_r2:.4f}")
print(f"Overall MAE: {overall_mae:.4f}")
print(f"Overall RMSE: {overall_rmse:.4f}")
print(f"\nCV Results:")
print(cv_df)

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(importance_df.head(15))

# MONTHLY-LEVEL EVALUATION BY GEOGRAPHIC UNIT
print(f"\n=== MONTHLY-LEVEL EVALUATION BY GEOGRAPHIC UNIT ===")

# Clean up categories first
df_test['ward_code'] = df_test['ward_code'].cat.remove_unused_categories()
df_test['lsoa_code'] = df_test['lsoa_code'].cat.remove_unused_categories()

# 1) WARD-MONTH LEVEL EVALUATION
print("Calculating ward-month level performance...")

# Each row is already a ward-month combination, so we group by ward and evaluate all their monthly predictions
ward_data_all = []
for ward in df_test['ward_code'].unique():
    ward_monthly_data = df_test[df_test['ward_code'] == ward]
    
    if len(ward_monthly_data) >= 3:  # Need minimum observations
        ward_data_all.extend(ward_monthly_data[['burglary_count', 'pred']].values)

ward_actual = [x[0] for x in ward_data_all]
ward_predicted = [x[1] for x in ward_data_all]

ward_monthly_r2 = r2_score(ward_actual, ward_predicted)
ward_monthly_mae = mean_absolute_error(ward_actual, ward_predicted)
ward_monthly_rmse = np.sqrt(mean_squared_error(ward_actual, ward_predicted))

# 2) LSOA-MONTH LEVEL EVALUATION  
print("Calculating LSOA-month level performance...")

# Each row is already an LSOA-month combination
lsoa_data_all = []
for lsoa in df_test['lsoa_code'].unique():
    lsoa_monthly_data = df_test[df_test['lsoa_code'] == lsoa]
    
    if len(lsoa_monthly_data) >= 3:  # Need minimum observations
        lsoa_data_all.extend(lsoa_monthly_data[['burglary_count', 'pred']].values)

lsoa_actual = [x[0] for x in lsoa_data_all]
lsoa_predicted = [x[1] for x in lsoa_data_all]

lsoa_monthly_r2 = r2_score(lsoa_actual, lsoa_predicted)
lsoa_monthly_mae = mean_absolute_error(lsoa_actual, lsoa_predicted)
lsoa_monthly_rmse = np.sqrt(mean_squared_error(lsoa_actual, lsoa_predicted))

# 3) ALTERNATIVE: DIRECT SUBSET EVALUATION
print("Calculating direct subset evaluation...")

# Ward subset: all observations that belong to wards
ward_mask = df_test['ward_code'].notna()
ward_subset = df_test[ward_mask]

ward_subset_r2 = r2_score(ward_subset['burglary_count'], ward_subset['pred'])
ward_subset_mae = mean_absolute_error(ward_subset['burglary_count'], ward_subset['pred'])
ward_subset_rmse = np.sqrt(mean_squared_error(ward_subset['burglary_count'], ward_subset['pred']))

# LSOA subset: all observations that belong to LSOAs  
lsoa_mask = df_test['lsoa_code'].notna()
lsoa_subset = df_test[lsoa_mask]

lsoa_subset_r2 = r2_score(lsoa_subset['burglary_count'], lsoa_subset['pred'])
lsoa_subset_mae = mean_absolute_error(lsoa_subset['burglary_count'], lsoa_subset['pred'])
lsoa_subset_rmse = np.sqrt(mean_squared_error(lsoa_subset['burglary_count'], lsoa_subset['pred']))

# 4) COMPARISON TABLE
monthly_comparison = pd.DataFrame([
    {
        'evaluation_type': 'all_monthly_observations',
        'description': 'All monthly observations (original)',
        'n_observations': len(df_test),
        'n_areas': 'all',
        'R2': overall_r2,
        'MAE': overall_mae,
        'RMSE': overall_rmse
    },
    {
        'evaluation_type': 'ward_monthly_subset',
        'description': 'Ward-month observations only',
        'n_observations': len(ward_subset),
        'n_areas': f"{len(df_test['ward_code'].unique())} wards",
        'R2': ward_subset_r2,
        'MAE': ward_subset_mae,
        'RMSE': ward_subset_rmse
    },
    {
        'evaluation_type': 'lsoa_monthly_subset',
        'description': 'LSOA-month observations only',
        'n_observations': len(lsoa_subset),
        'n_areas': f"{len(df_test['lsoa_code'].unique())} LSOAs",
        'R2': lsoa_subset_r2,
        'MAE': lsoa_subset_mae,
        'RMSE': lsoa_subset_rmse
    }
])

print(f"\n=== MONTHLY-LEVEL PERFORMANCE COMPARISON ===")
print(monthly_comparison[['evaluation_type', 'n_observations', 'n_areas', 'R2', 'MAE', 'RMSE']].round(4))

# 5) DETAILED RESULTS
print(f"\n=== DETAILED RESULTS ===")
print(f"1. All monthly observations:")
print(f"   n={len(df_test):,} observations")
print(f"   R²: {overall_r2:.4f}, MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}")

print(f"\n2. Ward-month level:")
print(f"   n={len(ward_subset):,} ward-month observations from {len(df_test['ward_code'].unique())} wards")
print(f"   R²: {ward_subset_r2:.4f}, MAE: {ward_subset_mae:.2f}, RMSE: {ward_subset_rmse:.2f}")

print(f"\n3. LSOA-month level:")
print(f"   n={len(lsoa_subset):,} LSOA-month observations from {len(df_test['lsoa_code'].unique())} LSOAs")
print(f"   R²: {lsoa_subset_r2:.4f}, MAE: {lsoa_subset_mae:.2f}, RMSE: {lsoa_subset_rmse:.2f}")

# 6) SAMPLE DATA BREAKDOWN
print(f"\n=== DATA BREAKDOWN ===")
print(f"Ward coverage: {len(ward_subset) / len(df_test) * 100:.1f}% of observations")
print(f"LSOA coverage: {len(lsoa_subset) / len(df_test) * 100:.1f}% of observations")

# Show sample of data structure
print(f"\nSample ward-month data:")
sample_ward = df_test[['ward_code', 'date', 'burglary_count', 'pred']].head(5)
print(sample_ward)

print(f"\nSample LSOA-month data:")
sample_lsoa = df_test[['lsoa_code', 'date', 'burglary_count', 'pred']].head(5)
print(sample_lsoa)

# 7) MONTHLY PERFORMANCE BY AREA TYPE
print(f"\n=== MONTHLY PERFORMANCE BY AREA TYPE ===")

# Ward monthly statistics
ward_monthly_stats = ward_subset.groupby('ward_code').agg({
    'burglary_count': ['count', 'mean', 'sum'],
    'pred': ['mean', 'sum']
}).round(2)

print(f"Ward monthly summary (showing first 5):")
print(ward_monthly_stats.head())

# LSOA monthly statistics  
lsoa_monthly_stats = lsoa_subset.groupby('lsoa_code').agg({
    'burglary_count': ['count', 'mean', 'sum'],
    'pred': ['mean', 'sum']
}).round(2)

print(f"\nLSOA monthly summary (showing first 5):")
print(lsoa_monthly_stats.head())

# 8) SAVE RESULTS
print(f"\n=== SAVING RESULTS ===")

# Save comparison table
monthly_comparison.to_csv('monthly_level_performance_comparison.csv', index=False)
print("✅ Saved: monthly_level_performance_comparison.csv")

# Save ward monthly data
ward_subset.to_csv('ward_monthly_predictions.csv', index=False)
print("✅ Saved: ward_monthly_predictions.csv")

# Save LSOA monthly data
lsoa_subset.to_csv('lsoa_monthly_predictions.csv', index=False)
print("✅ Saved: lsoa_monthly_predictions.csv")

# Save ward monthly statistics
ward_monthly_stats.to_csv('ward_monthly_statistics.csv')
print("✅ Saved: ward_monthly_statistics.csv")

# Save LSOA monthly statistics
lsoa_monthly_stats.to_csv('lsoa_monthly_statistics.csv')
print("✅ Saved: lsoa_monthly_statistics.csv")

# Save feature importance
importance_df.to_csv('feature_importance.csv', index=False)
print("✅ Saved: feature_importance.csv")

print(f"\n✅ MONTHLY-LEVEL EVALUATION COMPLETE!")
print(f"📊 Overall R²: {overall_r2:.4f}")
print(f"🏛️  Ward-month R²: {ward_subset_r2:.4f} (n={len(ward_subset):,})")
print(f"🏠 LSOA-month R²: {lsoa_subset_r2:.4f} (n={len(lsoa_subset):,})")
print(f"💾 All monthly-level results saved to CSV files")

print(f"\nTop 15 Most Important Features:")
print(importance_df.head(15))

# POPULATION DISTRIBUTION ANALYSIS
print(f"\n=== POPULATION DISTRIBUTION ANALYSIS ===")

# Clean up categories first
df_test['ward_code'] = df_test['ward_code'].cat.remove_unused_categories()
df_test['lsoa_code'] = df_test['lsoa_code'].cat.remove_unused_categories()

# 1) WARD-LEVEL INDIVIDUAL R² DISTRIBUTION
print("Calculating individual ward R² scores...")
ward_r2_scores = []

for ward in df_test['ward_code'].unique():
    ward_data = df_test[df_test['ward_code'] == ward]
    
    if len(ward_data) >= 5:  # Minimum observations for reliable R²
        try:
            ward_r2 = r2_score(ward_data['burglary_count'], ward_data['pred'])
            ward_mae = mean_absolute_error(ward_data['burglary_count'], ward_data['pred'])
            ward_rmse = np.sqrt(mean_squared_error(ward_data['burglary_count'], ward_data['pred']))
            
            ward_r2_scores.append({
                'ward_code': str(ward),
                'n_observations': len(ward_data),
                'R2': ward_r2,
                'MAE': ward_mae,
                'RMSE': ward_rmse,
                'total_actual': ward_data['burglary_count'].sum(),
                'total_predicted': ward_data['pred'].sum(),
                'mean_actual': ward_data['burglary_count'].mean(),
                'mean_predicted': ward_data['pred'].mean()
            })
        except:
            pass

ward_df = pd.DataFrame(ward_r2_scores)

# 2) LSOA-LEVEL INDIVIDUAL R² DISTRIBUTION
print("Calculating individual LSOA R² scores...")
lsoa_r2_scores = []

for lsoa in df_test['lsoa_code'].unique():
    lsoa_data = df_test[df_test['lsoa_code'] == lsoa]
    
    if len(lsoa_data) >= 3:  # Minimum observations for LSOA R²
        try:
            lsoa_r2 = r2_score(lsoa_data['burglary_count'], lsoa_data['pred'])
            lsoa_mae = mean_absolute_error(lsoa_data['burglary_count'], lsoa_data['pred'])
            lsoa_rmse = np.sqrt(mean_squared_error(lsoa_data['burglary_count'], lsoa_data['pred']))
            
            lsoa_r2_scores.append({
                'lsoa_code': str(lsoa),
                'n_observations': len(lsoa_data),
                'R2': lsoa_r2,
                'MAE': lsoa_mae,
                'RMSE': lsoa_rmse,
                'total_actual': lsoa_data['burglary_count'].sum(),
                'total_predicted': lsoa_data['pred'].sum(),
                'mean_actual': lsoa_data['burglary_count'].mean(),
                'mean_predicted': lsoa_data['pred'].mean()
            })
        except:
            pass

lsoa_df = pd.DataFrame(lsoa_r2_scores)

# 3) POPULATION DISTRIBUTION STATISTICS
print(f"\n=== R² DISTRIBUTION STATISTICS ===")

# Ward-level distribution
ward_stats = {
    'level': 'ward',
    'n_entities': len(ward_df),
    'mean_R2': ward_df['R2'].mean(),
    'median_R2': ward_df['R2'].median(),
    'std_R2': ward_df['R2'].std(),
    'min_R2': ward_df['R2'].min(),
    'max_R2': ward_df['R2'].max(),
    'q25_R2': ward_df['R2'].quantile(0.25),
    'q75_R2': ward_df['R2'].quantile(0.75),
    'pct_positive_R2': (ward_df['R2'] > 0).mean() * 100,
    'pct_good_R2': (ward_df['R2'] > 0.3).mean() * 100,
    'pct_excellent_R2': (ward_df['R2'] > 0.5).mean() * 100
}

# LSOA-level distribution  
lsoa_stats = {
    'level': 'lsoa',
    'n_entities': len(lsoa_df),
    'mean_R2': lsoa_df['R2'].mean(),
    'median_R2': lsoa_df['R2'].median(),
    'std_R2': lsoa_df['R2'].std(),
    'min_R2': lsoa_df['R2'].min(),
    'max_R2': lsoa_df['R2'].max(),
    'q25_R2': lsoa_df['R2'].quantile(0.25),
    'q75_R2': lsoa_df['R2'].quantile(0.75),
    'pct_positive_R2': (lsoa_df['R2'] > 0).mean() * 100,
    'pct_good_R2': (lsoa_df['R2'] > 0.3).mean() * 100,
    'pct_excellent_R2': (lsoa_df['R2'] > 0.5).mean() * 100
}

distribution_stats = pd.DataFrame([ward_stats, lsoa_stats])

print("DISTRIBUTION SUMMARY:")
print(distribution_stats[['level', 'n_entities', 'mean_R2', 'median_R2', 'std_R2', 'min_R2', 'max_R2']].round(4))

print(f"\nPERFORMANCE PERCENTAGES:")
print(distribution_stats[['level', 'pct_positive_R2', 'pct_good_R2', 'pct_excellent_R2']].round(1))

# 4) R² DISTRIBUTION BINS
print(f"\n=== R² DISTRIBUTION BINS ===")

# Define R² bins
r2_bins = [-np.inf, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf]
r2_labels = ['<0', '0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '>0.8']

# Ward R² distribution
ward_bins = pd.cut(ward_df['R2'], bins=r2_bins, labels=r2_labels).value_counts().sort_index()
ward_pct = (ward_bins / len(ward_df) * 100).round(1)

# LSOA R² distribution
lsoa_bins = pd.cut(lsoa_df['R2'], bins=r2_bins, labels=r2_labels).value_counts().sort_index()
lsoa_pct = (lsoa_bins / len(lsoa_df) * 100).round(1)

print(f"Ward R² Distribution (n={len(ward_df)}):")
for i, (bin_label, count, pct) in enumerate(zip(r2_labels, ward_bins, ward_pct)):
    print(f"  {bin_label}: {count:3d} wards ({pct:5.1f}%)")

print(f"\nLSOA R² Distribution (n={len(lsoa_df)}):")
for i, (bin_label, count, pct) in enumerate(zip(r2_labels, lsoa_bins, lsoa_pct)):
    print(f"  {bin_label}: {count:4d} LSOAs ({pct:5.1f}%)")

# 5) BEST AND WORST PERFORMERS
print(f"\n=== BEST/WORST INDIVIDUAL PERFORMERS ===")

print("TOP 10 WARDS (by R²):")
top_wards = ward_df.nlargest(10, 'R2')[['ward_code', 'R2', 'MAE', 'n_observations']]
print(top_wards.round(4))

print("\nWORST 10 WARDS (by R²):")
worst_wards = ward_df.nsmallest(10, 'R2')[['ward_code', 'R2', 'MAE', 'n_observations']]
print(worst_wards.round(4))

print("\nTOP 10 LSOAs (by R²):")
top_lsoas = lsoa_df.nlargest(10, 'R2')[['lsoa_code', 'R2', 'MAE', 'n_observations']]
print(top_lsoas.round(4))

print("\nWORST 10 LSOAs (by R²):")
worst_lsoas = lsoa_df.nsmallest(10, 'R2')[['lsoa_code', 'R2', 'MAE', 'n_observations']]
print(worst_lsoas.round(4))

# 6) SAVE POPULATION DISTRIBUTION RESULTS
print(f"\n=== SAVING POPULATION DISTRIBUTION RESULTS ===")

# Save distribution statistics
distribution_stats.to_csv('r2_distribution_statistics.csv', index=False)
print("✅ Saved: r2_distribution_statistics.csv")

# Save individual ward R² scores
ward_df.to_csv('individual_ward_r2_scores.csv', index=False)
print("✅ Saved: individual_ward_r2_scores.csv")

# Save individual LSOA R² scores
lsoa_df.to_csv('individual_lsoa_r2_scores.csv', index=False)
print("✅ Saved: individual_lsoa_r2_scores.csv")

# Create and save R² bin distribution
r2_distribution = pd.DataFrame({
    'R2_bin': r2_labels,
    'ward_count': ward_bins.values,
    'ward_percentage': ward_pct.values,
    'lsoa_count': lsoa_bins.values,
    'lsoa_percentage': lsoa_pct.values
})
r2_distribution.to_csv('r2_distribution_bins.csv', index=False)
print("✅ Saved: r2_distribution_bins.csv")

# Save feature importance
importance_df.to_csv('feature_importance.csv', index=False)
print("✅ Saved: feature_importance.csv")

print(f"\n✅ POPULATION DISTRIBUTION ANALYSIS COMPLETE!")
print(f"📊 Ward-level: {len(ward_df)} wards analyzed")
print(f"   Mean R²: {ward_df['R2'].mean():.4f}, Median R²: {ward_df['R2'].median():.4f}")
print(f"   {(ward_df['R2'] > 0.3).sum()} wards ({(ward_df['R2'] > 0.3).mean()*100:.1f}%) have R² > 0.3")

print(f"🏠 LSOA-level: {len(lsoa_df)} LSOAs analyzed") 
print(f"   Mean R²: {lsoa_df['R2'].mean():.4f}, Median R²: {lsoa_df['R2'].median():.4f}")
print(f"   {(lsoa_df['R2'] > 0.3).sum()} LSOAs ({(lsoa_df['R2'] > 0.3).mean()*100:.1f}%) have R² > 0.3")
print(f"💾 All distribution analysis saved to CSV files")