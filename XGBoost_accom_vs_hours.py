import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# === Configuration ===

# (1) Baseline CSV: already monthly-aggregated, one row per LSOA-Date,
#     with a column named “Burglary Count” (the total burglaries that month).
DATA_BASELINE_PATH = r"C:/Users/20232553/Downloads/data_aggregated_covid.csv"

# (2) Augmented CSV: same rows as baseline, but with 11 extra columns:
#     7 accommodation-type proportions (prop_…) and 4 hours-worked proportions (prop_hrs_…)
DATA_AUGMENTED_PATH = r"C:/Users/20232553/Downloads/burglaries_with_accom_and_hours_props.csv"

# Base filename for saving models; we will append "_accom" or "_hours"
MODEL_BASE = "models/xgb_burglary_model.json"

# Tuning & CV settings
TRIALS       = 50     # Number of Optuna trials per study
TIMEOUT      = 1800   # Seconds per Optuna study (30 minutes)
N_SPLITS     = 5      # 5-fold time-series CV
RANDOM_STATE = 42


def load_baseline_df(path):
    """
    Loads the already-aggregated burglary CSV for baseline usage.
    - Expects columns: ['LSOA code', 'Date', 'Burglary Count', … + optional IMD ranks …]
    - Drops pandemic years (2020, 2021) and 'Crime rank' if present.
    - Renames 'Burglary Count' → 'Count'.
    - Creates inverse-rank “<rank>_score” columns for any IMD rank columns.
    Returns a DataFrame with columns:
      ['LSOA code', 'Date', 'Count', <rank_score cols>].
    """
    try:
        print(f"Loading baseline data from: {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        print(f"  → {len(df)} rows loaded")
    except FileNotFoundError:
        cwd = os.getcwd()
        print(f"ERROR: Could not find baseline CSV at: {path}")
        print(f"Current working directory: {cwd}")
        print("Files in current directory:")
        for f in os.listdir(cwd):
            print(f"  - {f}")
        raise FileNotFoundError(f"Please check DATA_BASELINE_PATH; file not found: {path}")

    # 1) Drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()

    # 2) Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    # 3) Rename 'Burglary Count' → 'Count'
    if "Burglary Count" in df.columns:
        df = df.rename(columns={"Burglary Count": "Count"})
    else:
        raise KeyError("Expected column 'Burglary Count' in baseline CSV.")

    # 4) Ensure Count is float
    df["Count"] = df["Count"].astype(float)

    # 5) Create inverse-rank “<rank>_score” columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # 6) Keep only: ['LSOA code', 'Date', 'Count'] + any '<rank>_score' columns
    keep_cols = ["LSOA code", "Date", "Count"] + [c for c in df.columns if c.endswith("_score")]
    df = df[keep_cols].copy()

    return df


def load_accom_df(path):
    """
    Loads the “augmented” CSV but keeps only:
      - ['LSOA code', 'Date', 'Count', <rank_score cols>]
      - + the 7 accommodation-type proportions (columns starting with 'prop_' but NOT 'prop_hrs_')
    Drops pandemic years, drops 'Crime rank', renames 'Burglary Count' → 'Count', creates rank_scores.
    Returns a DataFrame with exactly:
      ['LSOA code', 'Date', 'Count', <rank_score cols>, <7 prop_… columns>].
    """
    try:
        print(f"Loading accommodation-only data from: {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        print(f"  → {len(df)} rows loaded")
    except FileNotFoundError:
        cwd = os.getcwd()
        print(f"ERROR: Could not find augmented CSV at: {path}")
        print(f"Current working directory: {cwd}")
        print("Files in current directory:")
        for f in os.listdir(cwd):
            print(f"  - {f}")
        raise FileNotFoundError(f"Please check DATA_AUGMENTED_PATH; file not found: {path}")

    # 1) Drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()

    # 2) Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    # 3) Rename 'Burglary Count' → 'Count'
    if "Burglary Count" in df.columns:
        df = df.rename(columns={"Burglary Count": "Count"})
    else:
        raise KeyError("Expected column 'Burglary Count' in augmented CSV.")

    # 4) Ensure Count is float
    df["Count"] = df["Count"].astype(float)

    # 5) Create inverse-rank “<rank>_score” columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # 6) Identify exactly the 7 accommodation-type props:
    #    columns that start with 'prop_' but NOT 'prop_hrs_'.
    accom_cols = [c for c in df.columns if c.startswith("prop_") and not c.startswith("prop_hrs_")]
    if len(accom_cols) < 7:
        raise KeyError(f"Expected at least 7 accommodation prop_… columns in augmented CSV, found {len(accom_cols)}.")

    # 7) Keep only ['LSOA code', 'Date', 'Count', <rank_score cols>, <7 accom_cols>]
    keep_cols = ["LSOA code", "Date", "Count"] \
                + [c for c in df.columns if c.endswith("_score")] \
                + accom_cols
    df = df[keep_cols].copy()

    return df


def load_hours_df(path):
    """
    Loads the “augmented” CSV but keeps only:
      - ['LSOA code', 'Date', 'Count', <rank_score cols>]
      - + the 4 hours-worked proportions (columns starting with 'prop_hrs_')
    Drops pandemic years, drops 'Crime rank', renames 'Burglary Count' → 'Count', creates rank_scores.
    Returns a DataFrame with exactly:
      ['LSOA code', 'Date', 'Count', <rank_score cols>, <4 prop_hrs_… columns>].
    """
    try:
        print(f"Loading hours-worked-only data from: {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        print(f"  → {len(df)} rows loaded")
    except FileNotFoundError:
        cwd = os.getcwd()
        print(f"ERROR: Could not find augmented CSV at: {path}")
        print(f"Current working directory: {cwd}")
        print("Files in current directory:")
        for f in os.listdir(cwd):
            print(f"  - {f}")
        raise FileNotFoundError(f"Please check DATA_AUGMENTED_PATH; file not found: {path}")

    # 1) Drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()

    # 2) Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    # 3) Rename 'Burglary Count' → 'Count'
    if "Burglary Count" in df.columns:
        df = df.rename(columns={"Burglary Count": "Count"})
    else:
        raise KeyError("Expected column 'Burglary Count' in augmented CSV.")

    # 4) Ensure Count is float
    df["Count"] = df["Count"].astype(float)

    # 5) Create inverse-rank “<rank>_score” columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # 6) Identify exactly the 4 hours-worked props: columns starting with 'prop_hrs_'
    hours_cols = [c for c in df.columns if c.startswith("prop_hrs_")]
    if len(hours_cols) < 4:
        raise KeyError(f"Expected at least 4 prop_hrs_… columns in augmented CSV, found {len(hours_cols)}.")

    # 7) Keep only ['LSOA code', 'Date', 'Count', <rank_score cols>, <4 hours_cols>]
    keep_cols = ["LSOA code", "Date", "Count"] \
                + [c for c in df.columns if c.endswith("_score")] \
                + hours_cols
    df = df[keep_cols].copy()

    return df


def create_time_aware_features(df, historical_stats=None):
    """
    Adds:
      1) month, year, month_sin, month_cos
      2) season dummy variables (Spring, Summer, Autumn; Winter as baseline)
      3) is_summer_month, is_winter_month, is_school_holiday_month
      4) lag_1, lag_2, lag_3, lag_6, lag_12 (per LSOA code)
      5) roll3_mean, roll3_std, roll6_mean, roll6_std, roll12_mean, roll12_std
         (computed on Count.shift(1) to avoid leakage)
      6) lsoa_historical_mean, lsoa_historical_std, lsoa_historical_count
         (expanding + shift(1))
      7) diff_1 = Count - lag_1 ; yoy_diff = Count - lag_12
      8) imd_lag1 = (first rank_score) * lag_1 ; imd_historical = (first rank_score) * lsoa_historical_mean
    Finally, fills any NaNs with 0 and returns the enriched DataFrame.
    """
    df = df.sort_values(["LSOA code", "Date"]).copy()

    # 1) Basic temporal features
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 2) Season dummy encoding
    conditions = [
        df["month"].between(3, 5),   # Spring
        df["month"].between(6, 8),   # Summer
        df["month"].between(9, 11),  # Autumn
    ]
    choices = ["Spring", "Summer", "Autumn"]
    df["season"] = np.select(conditions, choices, default="Winter")
    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    # Now we have 'season_Spring', 'season_Summer', 'season_Autumn'; Winter is implicit.

    # 3) Holiday / school indicators
    df["is_summer_month"] = df["month"].between(6, 8).astype(int)            # June–August
    df["is_winter_month"] = ((df["month"] == 12) | (df["month"].between(1, 2))).astype(int)
    df["is_school_holiday_month"] = df["month"].isin([7, 8, 12]).astype(int)  # July, August, December

    # 4) Lag features: previous 1, 2, 3, 6, 12 months (per LSOA)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)

    # 5) Rolling statistics (based on Count.shift(1))
    for window in [3, 6, 12]:
        df[f"roll{window}_mean"] = (
            df.groupby("LSOA code")["Count"]
              .shift(1)
              .rolling(window, min_periods=1)
              .mean()
        )
        df[f"roll{window}_std"] = (
            df.groupby("LSOA code")["Count"]
              .shift(1)
              .rolling(window, min_periods=2)
              .std()
        )

    # 6) Historical (expanding) stats
    if historical_stats is not None:
        df = df.merge(historical_stats, on="LSOA code", how="left")
    else:
        df["lsoa_historical_mean"] = (
            df.groupby("LSOA code")["Count"]
              .expanding()
              .mean()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
        df["lsoa_historical_std"] = (
            df.groupby("LSOA code")["Count"]
              .expanding()
              .std()
              .shift(1)
              .reset_index(level=0, drop=True)
        )
        df["lsoa_historical_count"] = (
            df.groupby("LSOA code")["Count"]
              .expanding()
              .count()
              .shift(1)
              .reset_index(level=0, drop=True)
        )

    # 7) Trend features
    df["diff_1"]  = df["Count"] - df["lag_1"]
    df["yoy_diff"] = df["Count"] - df["lag_12"]

    # 8) IMD interactions (use first rank_score column if any)
    rank_score_cols = [c for c in df.columns if c.endswith("_score")]
    if rank_score_cols:
        rcol = rank_score_cols[0]
        df["imd_lag1"]       = df[rcol] * df["lag_1"]
        df["imd_historical"] = df[rcol] * df["lsoa_historical_mean"]

    # 9) Fill NaNs
    df = df.fillna(0)
    return df


def time_aware_preprocess(X_train, y_train, X_val, y_val):
    """
    Fits a StandardScaler on X_train only, transforms both X_train & X_val.
    Does NOT scale dummy/binary features (season_*, is_*, month).
    Returns: (X_train_scaled, X_val_scaled, fitted_scaler).
    """
    cols_to_scale = [
        c for c in X_train.columns
        if not (c.startswith("season_") or c.startswith("is_") or c == "month")
    ]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    return X_train_scaled, X_val_scaled, scaler


def objective(trial, df_full):
    """
    Optuna objective: returns average MAE over N_SPLITS time-series folds.
    Includes debug prints so you can see fold sizes.
    """
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "eta":              trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "alpha":            trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "lambda":           trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds = trial.suggest_int("n_estimators", 100, 500)

    # Debug: show number of rows and sample Count values
    print(f"[DEBUG] Starting objective: df_full has {len(df_full)} rows")
    print(f"[DEBUG]   Sample Count values: {np.unique(df_full['Count'])[:5]}")

    dates = np.sort(df_full["Date"].unique())
    print(f"[DEBUG]   Unique dates count: {len(dates)}")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    date_indices = np.arange(len(dates))

    fold_maes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(date_indices)):
        train_dates = dates[train_idx]
        val_dates   = dates[val_idx]

        df_train = df_full[df_full["Date"].isin(train_dates)].copy()
        df_val   = df_full[df_full["Date"].isin(val_dates)].copy()

        # Debug: sizes
        print(f"[DEBUG] Fold {fold} → train rows = {len(df_train)}, val rows = {len(df_val)}")
        if len(df_val) == 0:
            print(f"[ERROR] Fold {fold} validation set is empty → returning inf")
            return float("inf")

        # 1) Feature engineering on training set
        df_train = create_time_aware_features(df_train)
        hist_stats = (
            df_train.groupby("LSOA code")["Count"]
                    .agg(['mean', 'std', 'count'])
                    .reset_index()
        )
        hist_stats.columns = [
            "LSOA code",
            "lsoa_historical_mean",
            "lsoa_historical_std",
            "lsoa_historical_count"
        ]

        # 2) Feature engineering on validation set using training stats (no leakage)
        df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

        feature_cols = [
            c for c in df_train.columns
            if c not in ["LSOA code", "Date", "Count"]
        ]
        X_tr, y_tr = df_train[feature_cols], df_train["Count"]
        X_vl, y_vl = df_val[feature_cols],   df_val["Count"]

        # 3) Scale
        X_tr_s, X_vl_s, _ = time_aware_preprocess(X_tr, y_tr, X_vl, y_vl)

        # 4) Train XGBoost on this fold
        dtr = xgb.DMatrix(X_tr_s, label=y_tr)
        dvl = xgb.DMatrix(X_vl_s, label=y_vl)
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            evals=[(dvl, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        preds = bst.predict(dvl)
        mae_fold = mean_absolute_error(y_vl, preds)
        print(f"[DEBUG] Fold {fold} MAE = {mae_fold:.4f}")
        fold_maes.append(mae_fold)

    mean_mae = float(np.mean(fold_maes))
    print(f"[DEBUG] Returning mean MAE = {mean_mae:.4f}")
    return mean_mae


def train_and_save():
    """
    Trains and saves two XGBoost models:
      1) Accommodation-only model (IMD ranks + accommodation proportions)
      2) Hours-worked-only model (IMD ranks + hours-worked proportions)
    Each uses the same TimeSeriesSplit/Optuna tuning structure as before.
    Saves:
      - models/xgb_burglary_model_accom.json
      - models/xgb_burglary_model_hours.json
    plus scalers, feature lists, and historical stats for each.
    """
    # Ensure the "models" directory exists
    os.makedirs(os.path.dirname(MODEL_BASE), exist_ok=True)

    # ─────────────────────────────
    # 1) ACCOMMODATION-ONLY MODEL
    # ─────────────────────────────
    print("\n=== TUNING & TRAINING ACCOMMODATION-ONLY MODEL ===")
    df_accom = load_accom_df(DATA_AUGMENTED_PATH)

    study_accom = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_accom.optimize(lambda t: objective(t, df_accom),
                         n_trials=TRIALS,
                         timeout=TIMEOUT)
    best_accom = study_accom.best_params
    print("Best accommodation-only params:", best_accom)

    # Feature engineering on full accom DataFrame
    df_full_accom = create_time_aware_features(df_accom)
    feature_cols_accom = [
        c for c in df_full_accom.columns 
        if c not in ["LSOA code", "Date", "Count"]
    ]
    X_a = df_full_accom[feature_cols_accom]
    y_a = df_full_accom["Count"]

    # Scale features
    cols_to_scale_a = [
        c for c in feature_cols_accom
        if not (c.startswith("season_") or c.startswith("is_") or c == "month")
    ]
    scaler_a = StandardScaler()
    X_a_scaled = X_a.copy()
    X_a_scaled[cols_to_scale_a] = scaler_a.fit_transform(X_a[cols_to_scale_a])

    params_a = {
        "max_depth":        best_accom["max_depth"],
        "eta":              best_accom["learning_rate"],
        "subsample":        best_accom["subsample"],
        "colsample_bytree": best_accom["colsample_bytree"],
        "min_child_weight": best_accom["min_child_weight"],
        "alpha":            best_accom["reg_alpha"],
        "lambda":           best_accom["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds_a = best_accom["n_estimators"]

    dmat_a = xgb.DMatrix(X_a_scaled, label=y_a)
    bst_a = xgb.train(params_a, dmat_a, num_boost_round=n_rounds_a, verbose_eval=False)

    # Save accommodation-only model + artifacts
    accom_path = MODEL_BASE.replace(".json", "_accom.json")
    bst_a.save_model(accom_path)
    print(f"Accommodation-only model saved to {accom_path}")

    joblib.dump(scaler_a, "models/scaler_accom.pkl")
    joblib.dump(feature_cols_accom, "models/feature_cols_accom.pkl")
    historical_stats_a = (
        df_full_accom.groupby("LSOA code")["Count"]
                     .agg(['mean','std','count'])
                     .reset_index()
    )
    historical_stats_a.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    joblib.dump(historical_stats_a, "models/historical_stats_accom.pkl")


    # ───────────────────────
    # 2) HOURS-WORKED-ONLY MODEL
    # ───────────────────────
    print("\n=== TUNING & TRAINING HOURS-WORKED-ONLY MODEL ===")
    df_hours = load_hours_df(DATA_AUGMENTED_PATH)

    study_hours = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_hours.optimize(lambda t: objective(t, df_hours),
                         n_trials=TRIALS,
                         timeout=TIMEOUT)
    best_hours = study_hours.best_params
    print("Best hours-worked-only params:", best_hours)

    # Feature engineering on full hours DataFrame
    df_full_hours = create_time_aware_features(df_hours)
    feature_cols_hours = [
        c for c in df_full_hours.columns 
        if c not in ["LSOA code", "Date", "Count"]
    ]
    X_h = df_full_hours[feature_cols_hours]
    y_h = df_full_hours["Count"]

    # Scale features
    cols_to_scale_h = [
        c for c in feature_cols_hours
        if not (c.startswith("season_") or c.startswith("is_") or c == "month")
    ]
    scaler_h = StandardScaler()
    X_h_scaled = X_h.copy()
    X_h_scaled[cols_to_scale_h] = scaler_h.fit_transform(X_h[cols_to_scale_h])

    params_h = {
        "max_depth":        best_hours["max_depth"],
        "eta":              best_hours["learning_rate"],
        "subsample":        best_hours["subsample"],
        "colsample_bytree": best_hours["colsample_bytree"],
        "min_child_weight": best_hours["min_child_weight"],
        "alpha":            best_hours["reg_alpha"],
        "lambda":           best_hours["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds_h = best_hours["n_estimators"]

    dmat_h = xgb.DMatrix(X_h_scaled, label=y_h)
    bst_h = xgb.train(params_h, dmat_h, num_boost_round=n_rounds_h, verbose_eval=False)

    # Save hours-worked-only model + artifacts
    hours_path = MODEL_BASE.replace(".json", "_hours.json")
    bst_h.save_model(hours_path)
    print(f"Hours-worked-only model saved to {hours_path}")

    joblib.dump(scaler_h, "models/scaler_hours.pkl")
    joblib.dump(feature_cols_hours, "models/feature_cols_hours.pkl")
    historical_stats_h = (
        df_full_hours.groupby("LSOA code")["Count"]
                      .agg(['mean','std','count'])
                      .reset_index()
    )
    historical_stats_h.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    joblib.dump(historical_stats_h, "models/historical_stats_hours.pkl")


def cross_val_metrics(df_full):
    """
    Runs N_SPLITS-fold time-series CV on df_full and returns:
      - maes, rmses, r2s (lists of length N_SPLITS)
      - fold_results: list of (fold_number, mae, rmse, r2)
    """
    dates = np.sort(df_full["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    date_indices = np.arange(len(dates))

    maes, rmses, r2s = [], [], []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(date_indices)):
        train_dates = dates[train_idx]
        val_dates   = dates[val_idx]

        print(f"Fold {fold+1}: Train until {pd.to_datetime(train_dates[-1]).strftime('%Y-%m')}, "
              f"Test {pd.to_datetime(val_dates[0]).strftime('%Y-%m')} to {pd.to_datetime(val_dates[-1]).strftime('%Y-%m')}")

        df_train = df_full[df_full["Date"].isin(train_dates)].copy()
        df_val   = df_full[df_full["Date"].isin(val_dates)].copy()

        # Feature engineering
        df_train = create_time_aware_features(df_train)
        hist_stats = (
            df_train.groupby("LSOA code")["Count"]
                    .agg(['mean','std','count'])
                    .reset_index()
        )
        hist_stats.columns = [
            "LSOA code",
            "lsoa_historical_mean",
            "lsoa_historical_std",
            "lsoa_historical_count"
        ]
        df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

        feature_cols = [
            c for c in df_train.columns 
            if c not in ["LSOA code", "Date", "Count"]
        ]
        X_train, y_train = df_train[feature_cols], df_train["Count"]
        X_val,   y_val   = df_val[feature_cols],   df_val["Count"]

        X_train_s, X_val_s, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
        dtr = xgb.DMatrix(X_train_s, label=y_train)
        dval = xgb.DMatrix(X_val_s,   label=y_val)

        # Use fixed “evaluation” hyperparameters to compare across models
        params = {
            "max_depth":        6,
            "eta":              0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "alpha":            0.1,
            "lambda":           1.0,
            "objective":        "reg:squarederror",
            "eval_metric":      "mae",
            "seed":             RANDOM_STATE,
        }

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=200,
            evals=[(dval, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        preds = bst.predict(dval)
        mae_fold  = mean_absolute_error(y_val, preds)
        rmse_fold = np.sqrt(mean_squared_error(y_val, preds))
        r2_fold   = r2_score(y_val, preds)

        maes.append(mae_fold)
        rmses.append(rmse_fold)
        r2s.append(r2_fold)
        fold_results.append((fold+1, mae_fold, rmse_fold, r2_fold))

    return maes, rmses, r2s, fold_results


def evaluate():
    """
    Evaluates both “accom-only” and “hours-only” models via time-series CV.
    Prints CV MAE, RMSE, R² (mean ± std) and fold-by-fold results for each.
    """
    # 1) Accommodation-only DataFrame
    df_full_accom = load_accom_df(DATA_AUGMENTED_PATH)
    print("\n=== EVALUATING ACCOMMODATION-ONLY MODEL ===")
    mae_a, rmse_a, r2_a, folds_a = cross_val_metrics(df_full_accom)

    # 2) Hours-worked-only DataFrame
    df_full_hours = load_hours_df(DATA_AUGMENTED_PATH)
    print("\n=== EVALUATING HOURS-WORKED-ONLY MODEL ===")
    mae_h, rmse_h, r2_h, folds_h = cross_val_metrics(df_full_hours)

    # 3) Compare
    print("\n--- CROSS-VALIDATION COMPARISON ---")
    print(f"Accom-only MAE   : {np.mean(mae_a):.4f} ± {np.std(mae_a):.4f}")
    print(f"Hours-only MAE   : {np.mean(mae_h):.4f} ± {np.std(mae_h):.4f}")
    print(f"Accom-only RMSE  : {np.mean(rmse_a):.4f} ± {np.std(rmse_a):.4f}")
    print(f"Hours-only RMSE  : {np.mean(rmse_h):.4f} ± {np.std(rmse_h):.4f}")
    print(f"Accom-only R²    : {np.mean(r2_a):.4f} ± {np.std(r2_a):.4f}")
    print(f"Hours-only R²    : {np.mean(r2_h):.4f} ± {np.std(r2_h):.4f}")

    print("\nAccommodation-only fold-by-fold results:")
    for fnum, m, r, r2 in folds_a:
        print(f"  Fold {fnum}: MAE={m:.4f}, RMSE={r:.4f}, R²={r2:.4f}")

    print("\nHours-worked-only fold-by-fold results:")
    for fnum, m, r, r2 in folds_h:
        print(f"  Fold {fnum}: MAE={m:.4f}, RMSE={r:.4f}, R²={r2:.4f}")


def plot_cv_performance_accom():
    """
    Plots citywide aggregated Actual vs. Predicted for the ACCOMMODATION-ONLY model's last fold.
    Saves figure to “models/cv_performance_accom.png”.
    """
    df = load_accom_df(DATA_AUGMENTED_PATH)
    dates = np.sort(df["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    date_indices = np.arange(len(dates))

    # Last CV fold
    train_idx, val_idx = list(tscv.split(date_indices))[-1]
    train_dates = dates[train_idx]
    val_dates   = dates[val_idx]

    df_train = df[df["Date"].isin(train_dates)].copy()
    df_val   = df[df["Date"].isin(val_dates)].copy()

    df_train = create_time_aware_features(df_train)
    hist_stats = (
        df_train.groupby("LSOA code")["Count"]
                .agg(['mean','std','count'])
                .reset_index()
    )
    hist_stats.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

    feature_cols = [
        c for c in df_train.columns
        if c not in ["LSOA code", "Date", "Count"]
    ]
    X_train, y_train = df_train[feature_cols], df_train["Count"]
    X_val,   y_val   = df_val[feature_cols],   df_val["Count"]

    X_train_s, X_val_s, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
    dtr = xgb.DMatrix(X_train_s, label=y_train)
    dval = xgb.DMatrix(X_val_s,   label=y_val)

    # Fixed params for plotting
    params = {
        "max_depth":        6,
        "eta":              0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "alpha":            0.1,
        "lambda":           1.0,
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=200,
        evals=[(dval, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    df_train["Pred"] = bst.predict(dtr)
    df_val["Pred"]   = bst.predict(dval)
    df_train["Pred"] = np.maximum(0, df_train["Pred"])
    df_val["Pred"]   = np.maximum(0, df_val["Pred"])

    train_agg = (
        df_train.groupby("Date")
                .agg(Actual=("Count", "sum"), Predicted=("Pred", "sum"))
                .sort_index()
    )
    val_agg = (
        df_val.groupby("Date")
              .agg(Actual=("Count", "sum"), Predicted=("Pred", "sum"))
              .sort_index()
    )

    train_r2  = r2_score(train_agg["Actual"], train_agg["Predicted"])
    train_mae = mean_absolute_error(train_agg["Actual"], train_agg["Predicted"])
    train_rmse= np.sqrt(mean_squared_error(train_agg["Actual"], train_agg["Predicted"]))
    val_r2    = r2_score(val_agg["Actual"], val_agg["Predicted"])
    val_mae   = mean_absolute_error(val_agg["Actual"], val_agg["Predicted"])
    val_rmse  = np.sqrt(mean_squared_error(val_agg["Actual"], val_agg["Predicted"]))

    print(f"ACCOM Last-fold Train → R²={train_r2:.3f}, MAE={train_mae:.1f}, RMSE={train_rmse:.1f}")
    print(f"ACCOM Last-fold Val   → R²={val_r2:.3f}, MAE={val_mae:.1f}, RMSE={val_rmse:.1f}")

    plt.figure(figsize=(14, 6))
    plt.plot(train_agg.index, train_agg["Actual"], label="Train Actual", color='blue', linewidth=2)
    plt.plot(train_agg.index, train_agg["Predicted"], "--", label="Train Predicted", color='orange', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Actual"],   label="Val Actual", color='green', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Predicted"], ":", label="Val Predicted", color='red', linewidth=2)

    split_date = pd.to_datetime(val_dates[0])
    plt.axvline(x=split_date, color='black', linestyle='-', alpha=0.5, linewidth=2)

    plt.title(f"ACCOMMODATION-ONLY MODEL (Last Fold) → Train R²={train_r2:.3f}, Val R²={val_r2:.3f}")
    plt.xlabel("Date")
    plt.ylabel("Citywide Monthly Burglaries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance_accom.png", dpi=300, bbox_inches='tight')
    print("Plot saved to models/cv_performance_accom.png")


def plot_cv_performance_hours():
    """
    Plots citywide aggregated Actual vs. Predicted for the HOURS-WORKED-ONLY model's last fold.
    Saves figure to “models/cv_performance_hours.png”.
    """
    df = load_hours_df(DATA_AUGMENTED_PATH)
    dates = np.sort(df["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    date_indices = np.arange(len(dates))

    train_idx, val_idx = list(tscv.split(date_indices))[-1]
    train_dates = dates[train_idx]
    val_dates   = dates[val_idx]

    df_train = df[df["Date"].isin(train_dates)].copy()
    df_val   = df[df["Date"].isin(val_dates)].copy()

    df_train = create_time_aware_features(df_train)
    hist_stats = (
        df_train.groupby("LSOA code")["Count"]
                .agg(['mean','std','count'])
                .reset_index()
    )
    hist_stats.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

    feature_cols = [
        c for c in df_train.columns
        if c not in ["LSOA code", "Date", "Count"]
    ]
    X_train, y_train = df_train[feature_cols], df_train["Count"]
    X_val,   y_val   = df_val[feature_cols],   df_val["Count"]

    X_train_s, X_val_s, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
    dtr = xgb.DMatrix(X_train_s, label=y_train)
    dval = xgb.DMatrix(X_val_s,   label=y_val)

    params = {
        "max_depth":        6,
        "eta":              0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "alpha":            0.1,
        "lambda":           1.0,
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=200,
        evals=[(dval, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    df_train["Pred"] = bst.predict(dtr)
    df_val["Pred"]   = bst.predict(dval)
    df_train["Pred"] = np.maximum(0, df_train["Pred"])
    df_val["Pred"]   = np.maximum(0, df_val["Pred"])

    train_agg = (
        df_train.groupby("Date")
                .agg(Actual=("Count", "sum"), Predicted=("Pred", "sum"))
                .sort_index()
    )
    val_agg = (
        df_val.groupby("Date")
              .agg(Actual=("Count", "sum"), Predicted=("Pred", "sum"))
              .sort_index()
    )

    train_r2  = r2_score(train_agg["Actual"], train_agg["Predicted"])
    train_mae = mean_absolute_error(train_agg["Actual"], train_agg["Predicted"])
    train_rmse= np.sqrt(mean_squared_error(train_agg["Actual"], train_agg["Predicted"]))
    val_r2    = r2_score(val_agg["Actual"], val_agg["Predicted"])
    val_mae   = mean_absolute_error(val_agg["Actual"], val_agg["Predicted"])
    val_rmse  = np.sqrt(mean_squared_error(val_agg["Actual"], val_agg["Predicted"]))

    print(f"HOURS Last-fold Train → R²={train_r2:.3f}, MAE={train_mae:.1f}, RMSE={train_rmse:.1f}")
    print(f"HOURS Last-fold Val   → R²={val_r2:.3f}, MAE={val_mae:.1f}, RMSE={val_rmse:.1f}")

    plt.figure(figsize=(14, 6))
    plt.plot(train_agg.index, train_agg["Actual"], label="Train Actual", color='blue', linewidth=2)
    plt.plot(train_agg.index, train_agg["Predicted"], "--", label="Train Predicted", color='orange', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Actual"],   label="Val Actual", color='green', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Predicted"], ":", label="Val Predicted", color='red', linewidth=2)

    split_date = pd.to_datetime(val_dates[0])
    plt.axvline(x=split_date, color='black', linestyle='-', alpha=0.5, linewidth=2)

    plt.title(f"HOURS-WORKED-ONLY MODEL (Last Fold) → Train R²={train_r2:.3f}, Val R²={val_r2:.3f}")
    plt.xlabel("Date")
    plt.ylabel("Citywide Monthly Burglaries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance_hours.png", dpi=300, bbox_inches='tight')
    print("Plot saved to models/cv_performance_hours.png")


if __name__ == "__main__":
    print("Starting burglary prediction (accommodation-only vs. hours-worked-only)…\n")

    # 1) Train & save both new models
    train_and_save()

    # 2) Evaluate both via time-series CV
    print("\n2. Evaluating accommodation-only vs. hours-worked-only via time-series CV…")
    evaluate()

    # 3) Plot performance for each model’s last fold
    print("\n3. Plotting CV performance for accommodation-only model…")
    plot_cv_performance_accom()

    print("\n4. Plotting CV performance for hours-worked-only model…")
    plot_cv_performance_hours()

    print("\nAll steps complete.")
