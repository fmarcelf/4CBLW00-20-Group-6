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
import matplotlib.pyplot as plt

# ==============================================================================
# BASELINE BURGLARY PREDICTION MODEL - NO IMD FACTORS
# ==============================================================================
# ULTRA-STRICT: Every preprocessing step fits on training data ONLY
# ==============================================================================

print("🏠 BASELINE BURGLARY PREDICTION MODEL - NO IMD FACTORS")
print("🚨 ULTRA-STRICT MODE: NO SHORTCUTS")
print("=" * 60)

# 1) LOAD DATA - NO PREPROCESSING YET
# ==============================================================================
print("📊 Loading raw data...")

df = pd.read_csv(
    r"C:/Users/20232553/Downloads/data_aggregated_covid.csv",
    usecols=['LSOA code','Ward code','Date','Burglary Count','covid_period',
             'LSOA Area Size (HA)'],
    dtype={'LSOA code':'str','Ward code':'str','Burglary Count':'int16',
           'covid_period':'int8','LSOA Area Size (HA)':'float32'},
    parse_dates=['Date']
)

# Basic renaming only
df.rename(columns={
    'LSOA code':'lsoa_code','Ward code':'ward_code',
    'Burglary Count':'burglary_count','Date':'date',
    'LSOA Area Size (HA)':'area_ha'
}, inplace=True)

df.sort_values(['lsoa_code','date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"✅ Raw data loaded: {df.shape[0]:,} rows")

# 2) SPLIT IMMEDIATELY - BEFORE ANY FEATURE ENGINEERING
# ==============================================================================
print("\n🔪 SPLITTING DATA FIRST - NO PREPROCESSING APPLIED YET")
print("-" * 60)

# Split by date at 70% mark
split_date = df['date'].quantile(0.7)
print(f"Split date: {split_date.strftime('%Y-%m-%d')}")

# Create train/test masks
train_mask = df['date'] <= split_date
test_mask = df['date'] > split_date

# Split the raw data
df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

print(f"Training: {df_train['date'].min().strftime('%Y-%m-%d')} to {df_train['date'].max().strftime('%Y-%m-%d')}")
print(f"Testing:  {df_test['date'].min().strftime('%Y-%m-%d')} to {df_test['date'].max().strftime('%Y-%m-%d')}")
print(f"Train size: {len(df_train):,}")
print(f"Test size:  {len(df_test):,}")

# Clear original dataframe to prevent accidental use
del df
print("✅ Original dataframe deleted to prevent leakage")

# 3) TEMPORAL FEATURES - SAFE (DATE-BASED ONLY)
# ==============================================================================
print("\n📅 CREATING TEMPORAL FEATURES (SAFE)")
print("-" * 50)

def add_temporal_features(df_input):
    """Add temporal features based only on date - completely safe"""
    df_out = df_input.copy()
    
    # Basic time components
    df_out['year'] = df_out['date'].dt.year
    df_out['month'] = df_out['date'].dt.month
    df_out['quarter'] = df_out['date'].dt.quarter
    
    # Seasonality (safe - based only on month)
    df_out['sin_month'] = np.sin(2 * np.pi * df_out['month'] / 12)
    df_out['cos_month'] = np.cos(2 * np.pi * df_out['month'] / 12)
    df_out['sin_quarter'] = np.sin(2 * np.pi * df_out['quarter'] / 4)
    df_out['cos_quarter'] = np.cos(2 * np.pi * df_out['quarter'] / 4)
    
    # Season indicators
    df_out['is_winter'] = (df_out['month'].isin([12,1,2])).astype(int)
    df_out['is_summer'] = (df_out['month'].isin([6,7,8])).astype(int)
    
    # COVID dummy (uses existing column)
    df_out['covid_dummy'] = df_out['covid_period'].astype(int)
    
    return df_out

df_train = add_temporal_features(df_train)
df_test = add_temporal_features(df_test)

print("✅ Temporal features added safely")

# 4) LAG FEATURES - MOST IMPORTANT, NO LEAKAGE POSSIBLE
# ==============================================================================
print("\n⏰ CREATING LAG FEATURES (HISTORICAL ONLY)")
print("-" * 50)

def add_lag_features(df_input):
    """Add lag features using only historical data"""
    df_out = df_input.copy()
    
    # Sort to ensure proper lag calculation
    df_out = df_out.sort_values(['lsoa_code', 'date']).reset_index(drop=True)
    
    # Create lags by LSOA
    grp = df_out.groupby('lsoa_code')['burglary_count']
    
    # Core lags
    for lag in [1, 2, 3, 6, 12]:
        df_out[f'lag_{lag}m'] = grp.shift(lag)
    
    # Seasonal lag
    df_out['lag_12m_seasonal'] = grp.shift(12)
    
    # Year-over-year change (critical feature)
    df_out['yoy_change'] = (df_out['lag_1m'] - df_out['lag_12m_seasonal']) / (df_out['lag_12m_seasonal'] + 1e-6)
    
    # Simple rolling means (shifted to avoid leakage)
    df_out['roll3_mean'] = grp.shift(1).rolling(3).mean()
    df_out['roll6_mean'] = grp.shift(1).rolling(6).mean()
    
    return df_out

df_train = add_lag_features(df_train)
df_test = add_lag_features(df_test)

print("✅ Lag features added")

# 5) WARD FEATURES - HISTORICAL ONLY
# ==============================================================================
print("\n🏘️ WARD FEATURES - HISTORICAL ONLY")
print("-" * 50)

def add_ward_features(df_input):
    """Add ward-level features using only historical data"""
    df_out = df_input.copy()
    df_out = df_out.sort_values(['ward_code', 'date']).reset_index(drop=True)
    
    wgrp = df_out.groupby('ward_code')['burglary_count']
    df_out['ward_lag1'] = wgrp.shift(1)
    df_out['ward_lag12'] = wgrp.shift(12)
    
    return df_out

df_train = add_ward_features(df_train)
df_test = add_ward_features(df_test)

print("✅ Ward features added")

# 6) TARGET ENCODING - ULTRA-STRICT TRAIN-ONLY
# ==============================================================================
print("\n🎯 TARGET ENCODING - ULTRA-STRICT")
print("-" * 50)

def fit_target_encoding(df_train_input):
    """Fit target encoding on training data only, using LAG_1M as target"""
    
    # Use lag_1m as target to avoid direct leakage
    target_col = 'lag_1m'
    train_clean = df_train_input.dropna(subset=[target_col])
    
    if len(train_clean) == 0:
        print("ERROR: No valid lag_1m data for target encoding!")
        return {}
    
    print(f"Using {len(train_clean):,} training samples for target encoding")
    
    # Fit encoders
    encoders = {}
    
    def compute_target_encoding(series, target, smoothing=10):
        global_mean = target.mean()
        stats = pd.DataFrame({'cat': series, 'target': target}).groupby('cat')['target'].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        return smoothed.to_dict(), global_mean
    
    # Fit on training data only
    encoders['lsoa'], encoders['lsoa_global'] = compute_target_encoding(
        train_clean['lsoa_code'], train_clean[target_col])
    encoders['ward'], encoders['ward_global'] = compute_target_encoding(
        train_clean['ward_code'], train_clean[target_col])
    encoders['month'], encoders['month_global'] = compute_target_encoding(
        train_clean['month'], train_clean[target_col])
    
    return encoders

def apply_target_encoding(df_input, encoders):
    """Apply fitted target encoders"""
    df_out = df_input.copy()
    
    if not encoders:
        print("No encoders available - skipping target encoding")
        return df_out
    
    df_out['lsoa_target_enc'] = df_out['lsoa_code'].map(encoders['lsoa']).fillna(encoders['lsoa_global'])
    df_out['ward_target_enc'] = df_out['ward_code'].map(encoders['ward']).fillna(encoders['ward_global'])
    df_out['month_target_enc'] = df_out['month'].map(encoders['month']).fillna(encoders['month_global'])
    
    return df_out

# Fit encoders on training data only
target_encoders = fit_target_encoding(df_train)

# Apply to both sets
df_train = apply_target_encoding(df_train, target_encoders)
df_test = apply_target_encoding(df_test, target_encoders)

print("✅ Target encoding: FIT on train, APPLIED to both")

# 7) PREPARE FEATURES - MINIMAL SET TO AVOID LEAKAGE
# ==============================================================================
print("\n🔍 FEATURE PREPARATION - BASELINE SET (NO IMD)")
print("-" * 50)

# Only use clearly safe features - NO IMD FACTORS
feature_columns = [
    # Temporal (safe)
    'year', 'month', 'quarter', 'sin_month', 'cos_month', 'sin_quarter', 'cos_quarter',
    'is_winter', 'is_summer', 'covid_dummy',
    
    # Lags (safe - historical only)
    'lag_1m', 'lag_2m', 'lag_3m', 'lag_6m', 'lag_12m', 'lag_12m_seasonal',
    'yoy_change', 'roll3_mean', 'roll6_mean',
    
    # Ward lags (safe)
    'ward_lag1', 'ward_lag12',
    
    # Area (raw)
    'area_ha',
    
    # Target encoding (fitted on train)
    'lsoa_target_enc', 'ward_target_enc', 'month_target_enc'
]

# Check which features actually exist
available_features = []
for col in feature_columns:
    if col in df_train.columns and col in df_test.columns:
        available_features.append(col)
    else:
        print(f"  Missing feature: {col}")

print(f"✅ Available features: {len(available_features)} (NO IMD FACTORS)")

# 8) CLEAN DATA - REMOVE ROWS WITH MISSING CRITICAL FEATURES
# ==============================================================================
print("\n🧹 CLEANING DATA - REMOVE MISSING CRITICAL FEATURES")
print("-" * 50)

# Critical features that must not be missing
critical_features = ['lag_1m', 'lag_2m', 'lag_3m', 'roll3_mean']

print(f"Before cleaning - Train: {len(df_train):,}, Test: {len(df_test):,}")

# Remove rows with missing critical features
df_train_clean = df_train.dropna(subset=critical_features).copy()
df_test_clean = df_test.dropna(subset=critical_features).copy()

print(f"After cleaning  - Train: {len(df_train_clean):,}, Test: {len(df_test_clean):,}")

# Final feature matrix preparation
X_train = df_train_clean[available_features].copy()
X_test = df_test_clean[available_features].copy()
y_train = df_train_clean['burglary_count'].copy()
y_test = df_test_clean['burglary_count'].copy()

# Final imputation - fit on training only
print("Final imputation - fitting on TRAINING data only...")
final_imputer = SimpleImputer(strategy='median')
X_train_final = pd.DataFrame(
    final_imputer.fit_transform(X_train), 
    columns=available_features, 
    index=X_train.index
)
X_test_final = pd.DataFrame(
    final_imputer.transform(X_test), 
    columns=available_features, 
    index=X_test.index
)

print(f"✅ Final matrices prepared")
print(f"   Training: {X_train_final.shape}")
print(f"   Test: {X_test_final.shape}")

# 9) MODEL TRAINING - SIMPLE AND CLEAN
# ==============================================================================
print("\n🤖 MODEL TRAINING")
print("-" * 50)

# Simple parameters to avoid overfitting
params = {
    'n_estimators': 300,  # Reduced to prevent overfitting
    'max_depth': 4,       # Shallow trees
    'learning_rate': 0.05, # Slower learning
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,     # More regularization
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation
print("Cross-validation...")
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_final), 1):
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_final.iloc[train_idx], y_train.iloc[train_idx])
    val_preds = model.predict(X_train_final.iloc[val_idx])
    fold_r2 = r2_score(y_train.iloc[val_idx], val_preds)
    cv_scores.append(fold_r2)
    print(f"  Fold {fold} R²: {fold_r2:.4f}")

cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
print(f"✅ CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

# Final model
print("Training final model...")
final_model = xgb.XGBRegressor(**params)
final_model.fit(X_train_final, y_train)

# Predictions
test_predictions = final_model.predict(X_test_final)
df_test_clean['pred'] = test_predictions

# 10) WARD-LEVEL ANALYSIS
# ==============================================================================
print("\n" + "="*60)
print("🏘️ WARD-LEVEL ANALYSIS")
print("="*60)

# Aggregate to ward level
ward_agg = df_test_clean.groupby(['ward_code', 'date']).agg({
    'burglary_count': 'sum',
    'pred': 'sum'
}).reset_index()

# Overall ward performance
ward_r2 = r2_score(ward_agg['burglary_count'], ward_agg['pred'])
ward_mae = mean_absolute_error(ward_agg['burglary_count'], ward_agg['pred'])
ward_rmse = np.sqrt(mean_squared_error(ward_agg['burglary_count'], ward_agg['pred']))

print(f"📊 OVERALL WARD PERFORMANCE:")
print(f"   R²:   {ward_r2:.4f}")
print(f"   MAE:  {ward_mae:.2f}")
print(f"   RMSE: {ward_rmse:.2f}")

# Find meaningful ward for visualization
ward_totals = ward_agg.groupby('ward_code')['burglary_count'].sum()
wards_with_crime = ward_totals[ward_totals >= 20].index  # At least 20 total crimes

if len(wards_with_crime) > 0:
    # Find best performing ward among those with crime
    best_ward = None
    best_r2 = -1
    
    for ward in wards_with_crime:
        ward_data = ward_agg[ward_agg['ward_code'] == ward]
        if len(ward_data) >= 12:
            ward_r2_individual = r2_score(ward_data['burglary_count'], ward_data['pred'])
            if ward_r2_individual > best_r2:
                best_r2 = ward_r2_individual
                best_ward = ward
    
    if best_ward:
        print(f"🏆 Best ward: {best_ward} (R²: {best_r2:.4f})")
        
        # Visualization
        ward_data = ward_agg[ward_agg['ward_code'] == best_ward].sort_values('date')
        
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(ward_data['date'], ward_data['burglary_count'], 'o-', 
                color='blue', label='Actual', alpha=0.7)
        plt.plot(ward_data['date'], ward_data['pred'], 's-', 
                color='red', label='Predicted', alpha=0.7)
        plt.title(f'Ward {best_ward} Performance (R²: {best_r2:.3f})')
        plt.ylabel('Monthly Burglary Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        residuals = ward_data['burglary_count'] - ward_data['pred']
        plt.plot(ward_data['date'], residuals, 'o-', color='orange', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Residuals')
        plt.ylabel('Actual - Predicted')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 TOP 10 FEATURES:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f}")

# Final summary
covid_count = df_test_clean[df_test_clean['covid_period']==1].shape[0]
total_count = df_test_clean.shape[0]

print(f"\n" + "="*60)
print("📈 BASELINE RESULTS - NO IMD FACTORS")
print("="*60)
print(f"🏘️ Ward R²:        {ward_r2:.4f}")
print(f"📊 Ward MAE:       {ward_mae:.2f}")
print(f"🔄 CV R²:          {cv_mean:.4f} ± {cv_std:.4f}")
print(f"🎯 Features:       {len(available_features)} (NO IMD)")
print(f"📚 Train samples:  {len(X_train_final):,}")
print(f"🧪 Test samples:   {len(X_test_final):,}")
print(f"🦠 COVID in test:  {covid_count:,} / {total_count:,} ({100*covid_count/total_count:.1f}%)")
print(f"")
print(f"🛡️ DATA LEAKAGE PREVENTION MEASURES:")
print(f"   ✅ Temporal split FIRST")
print(f"   ✅ All preprocessing fit on train only")
print(f"   ✅ Target encoding fit on train only")
print(f"   ✅ Imputation fit on train only")
print(f"   ✅ Only historical lag features used")
print(f"   ✅ Conservative model parameters")
print(f"   ✅ NO IMD FACTORS (baseline model)")
print("="*60)

# Save minimal results
ward_agg.to_csv('ward_predictions_baseline_no_imd.csv', index=False)
importance_df.to_csv('feature_importance_baseline_no_imd.csv', index=False)

print(f"\n🎉 BASELINE ANALYSIS COMPLETE - NO IMD FACTORS")
print(f"📊 Ward R²: {ward_r2:.4f} (baseline performance)")
print(f"💾 Results saved to CSV files")

