import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# === Configuration ===

# (1) Baseline CSV: already monthly‐aggregated, one row per LSOA‐Date, with a column
#     named “Burglary Count” (the total number of burglaries that month for that LSOA).
# ADD YOUR PATH
DATA_BASELINE_PATH  = 'data_aggregated_covid.csv'

# (2) Augmented CSV: exactly the same rows/columns as the baseline CSV, but with
#     11 extra columns appended: 7 accommodation‐type proportions (prop_…)
#     and 4 hours‐worked proportions (prop_hrs_…).
# ADD YOUR PATH
DATA_AUGMENTED_PATH = 'burglaries_with_accom_and_hours_props.csv'

# Base filename for saving models; we will append "_baseline" or "_augmented"
MODEL_BASE = "models/xgb_burglary_model.json"

# Tuning & cross‐validation settings
TRIALS       = 50     # Number of Optuna trials per study (reduce to speed up)
TIMEOUT      = 1800   # Seconds per Optuna study (30 minutes)
N_SPLITS     = 5      # 5‐fold time‐series CV
RANDOM_STATE = 42


def load_baseline_df(path):
    """
    Loads the already‐aggregated burglary CSV for the baseline model.
    - The CSV must have columns: ['LSOA code', 'Date', 'Burglary Count', ... + optional IMD ranks ...]
    - Drops pandemic years (2020, 2021) and any 'Crime rank' column if present.
    - Renames 'Burglary Count' → 'Count' (float).
    - Creates inverse‐rank “rank_score” columns for any IMD rank columns.
    Returns: DataFrame with ['LSOA code', 'Date', 'Count', <rank_score cols>].
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

    # Drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()

    # Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    # Rename 'Burglary Count' → 'Count'
    if "Burglary Count" in df.columns:
        df = df.rename(columns={"Burglary Count": "Count"})
    else:
        raise KeyError("Expected column 'Burglary Count' in baseline CSV.")

    # Ensure Count is float
    df["Count"] = df["Count"].astype(float)

    # Create inverse‐rank “rank_score” columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # We only need: 'LSOA code', 'Date', 'Count', and any '..._score' columns
    keep_cols = ["LSOA code", "Date", "Count"] + [c for c in df.columns if c.endswith("_score")]
    df = df[keep_cols].copy()

    return df


def load_augmented_df(path):
    """
    Loads the augmented CSV, which is identical to the baseline CSV plus 11 extra columns:
        - 7 accommodation‐type proportions: prop_Detached, prop_Semi_detached, prop_Terraced, etc.
        - 4 hours‐worked proportions: prop_hrs_15_or_less, prop_hrs_16_30, ...
    - Drops pandemic years (2020, 2021) and any 'Crime rank' column.
    - Renames 'Burglary Count' → 'Count'.
    - Creates inverse‐rank "rank_score" columns for any IMD rank columns.
    Returns: DataFrame with:
      ['LSOA code', 'Date', 'Count', <rank_score cols>, <11 prop_… cols> ]
    """
    try:
        print(f"Loading augmented data from: {path}")
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

    # Drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()

    # Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    # Rename 'Burglary Count' → 'Count'
    if "Burglary Count" in df.columns:
        df = df.rename(columns={"Burglary Count": "Count"})
    else:
        raise KeyError("Expected column 'Burglary Count' in augmented CSV.")

    # Ensure Count is float
    df["Count"] = df["Count"].astype(float)

    # Create inverse‐rank “rank_score” columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # Identify the 11 prop_… columns (anything starting with "prop_")
    prop_cols = [c for c in df.columns if c.startswith("prop_")]
    if len(prop_cols) < 11:
        raise KeyError(f"Expected at least 11 prop_... columns in augmented CSV, found {len(prop_cols)}.")

    # We want: 'LSOA code', 'Date', 'Count', all '*_score' columns, and all 'prop_...' columns
    keep_cols = ["LSOA code", "Date", "Count"] + [c for c in df.columns if c.endswith("_score")] + prop_cols
    df = df[keep_cols].copy()

    return df


def create_time_aware_features(df, historical_stats=None):
    """
    Adds:
      1) month, year, month_sin, month_cos
      2) season dummy variables (Spring, Summer, Autumn; Winter is baseline)
      3) is_summer_month, is_winter_month, is_school_holiday_month
      4) lag_1, lag_2, lag_3, lag_6, lag_12 (per LSOA code)
      5) roll3_mean, roll3_std, roll6_mean, roll6_std, roll12_mean, roll12_std
         (all computed on Count.shift(1))
      6) lsoa_historical_mean, lsoa_historical_std, lsoa_historical_count
         (expanding + shift(1))
      7) diff_1 = Count - lag_1 ; yoy_diff = Count - lag_12
      8) imd_lag1 = (first rank_score) * lag_1 ; imd_historical = (first rank_score) * lsoa_historical_mean
    Fills any remaining NaNs with 0. Returns the enriched DataFrame.
    """
    df = df.sort_values(["LSOA code", "Date"]).copy()

    # 1) Temporal features
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # 2) Season dummy
    conditions = [
        (df["month"].between(3, 5)),   # Spring
        (df["month"].between(6, 8)),   # Summer
        (df["month"].between(9, 11)),  # Autumn
    ]
    choices = ["Spring", "Summer", "Autumn"]
    df["season"] = np.select(conditions, choices, default="Winter")
    df = pd.get_dummies(df, columns=["season"], drop_first=True)

    # 3) Holiday indicators
    df["is_summer_month"] = df["month"].between(6, 8).astype(int)
    df["is_winter_month"] = ((df["month"] == 12) | (df["month"].between(1, 2))).astype(int)
    df["is_school_holiday_month"] = df["month"].isin([7, 8, 12]).astype(int)

    # 4) Lag features
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

    # 8) IMD interactions (use the first “_score” column, if any exist)
    rank_score_cols = [c for c in df.columns if c.endswith("_score")]
    if rank_score_cols:
        rcol = rank_score_cols[0]
        df["imd_lag1"]       = df[rcol] * df["lag_1"]
        df["imd_historical"] = df[rcol] * df["lsoa_historical_mean"]

    # Fill any remaining NaNs
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
    Optuna objective: returns average MAE over N_SPLITS time‐series folds.
    Includes debug prints so that you can see fold sizes (no more silent 0.0).
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

    # Unique sorted dates
    dates = np.sort(df_full["Date"].unique())
    print(f"[DEBUG]   Unique dates count: {len(dates)}")

    tscv        = TimeSeriesSplit(n_splits=N_SPLITS)
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

        # Feature engineering
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
        df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

        feature_cols = [
            c for c in df_train.columns
            if c not in ["LSOA code", "Date", "Count"]
        ]
        X_tr, y_tr = df_train[feature_cols], df_train["Count"]
        X_vl, y_vl = df_val[feature_cols],   df_val["Count"]

        # Preprocess/scale
        X_tr_s, X_vl_s, _ = time_aware_preprocess(X_tr, y_tr, X_vl, y_vl)

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
    Trains two XGBoost models:
      1) Baseline (no prop_ columns) on DATA_BASELINE_PATH
      2) Augmented (with prop_ columns) on DATA_AUGMENTED_PATH
    Saves each model plus its scaler, feature‐list, and historical stats to distinct files.
    """
    # Ensure "models" directory exists
    os.makedirs(os.path.dirname(MODEL_BASE), exist_ok=True)

    # ────────────────
    # 1) BASELINE MODEL
    # ────────────────
    print("\n=== TUNING & TRAINING BASELINE MODEL (no prop_ columns) ===")
    df_base = load_baseline_df(DATA_BASELINE_PATH)

    study_base = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_base.optimize(lambda t: objective(t, df_base),
                        n_trials=TRIALS,
                        timeout=TIMEOUT)
    best_base = study_base.best_params
    print("Best baseline params:", best_base)

    # Build features on the entire baseline DataFrame
    df_full_base = create_time_aware_features(df_base)
    feature_cols_base = [
        c for c in df_full_base.columns
        if c not in ["LSOA code", "Date", "Count"]
    ]
    Xb = df_full_base[feature_cols_base]
    yb = df_full_base["Count"]

    # Scale
    cols_to_scale_b = [
        c for c in feature_cols_base
        if not (c.startswith("season_") or c.startswith("is_") or c == "month")
    ]
    scaler_b = StandardScaler()
    Xb_scaled = Xb.copy()
    Xb_scaled[cols_to_scale_b] = scaler_b.fit_transform(Xb[cols_to_scale_b])

    params_b = {
        "max_depth":        best_base["max_depth"],
        "eta":              best_base["learning_rate"],
        "subsample":        best_base["subsample"],
        "colsample_bytree": best_base["colsample_bytree"],
        "min_child_weight": best_base["min_child_weight"],
        "alpha":            best_base["reg_alpha"],
        "lambda":           best_base["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds_b = best_base["n_estimators"]

    dmat_b = xgb.DMatrix(Xb_scaled, label=yb)
    bst_b  = xgb.train(params_b, dmat_b, num_boost_round=n_rounds_b, verbose_eval=False)

    # Save baseline model + artifacts
    baseline_path = MODEL_BASE.replace(".json", "_baseline.json")
    bst_b.save_model(baseline_path)
    print(f"Baseline model saved to {baseline_path}")

    joblib.dump(scaler_b, "models/scaler_baseline.pkl")
    joblib.dump(feature_cols_base, "models/feature_cols_baseline.pkl")
    historical_stats_b = (
        df_full_base.groupby("LSOA code")["Count"]
                     .agg(['mean', 'std', 'count'])
                     .reset_index()
    )
    historical_stats_b.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    joblib.dump(historical_stats_b, "models/historical_stats_baseline.pkl")


    # ────────────────
    # 2) AUGMENTED MODEL
    # ────────────────
    print("\n=== TUNING & TRAINING AUGMENTED MODEL (with prop_ columns) ===")
    df_aug = load_augmented_df(DATA_AUGMENTED_PATH)

    study_aug = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study_aug.optimize(lambda t: objective(t, df_aug),
                       n_trials=TRIALS,
                       timeout=TIMEOUT)
    best_aug = study_aug.best_params
    print("Best augmented params:", best_aug)

    df_full_aug = create_time_aware_features(df_aug)
    feature_cols_aug = [
        c for c in df_full_aug.columns
        if c not in ["LSOA code", "Date", "Count"]
    ]
    Xa = df_full_aug[feature_cols_aug]
    ya = df_full_aug["Count"]

    cols_to_scale_a = [
        c for c in feature_cols_aug
        if not (c.startswith("season_") or c.startswith("is_") or c == "month")
    ]
    scaler_a = StandardScaler()
    Xa_scaled = Xa.copy()
    Xa_scaled[cols_to_scale_a] = scaler_a.fit_transform(Xa[cols_to_scale_a])

    params_a = {
        "max_depth":        best_aug["max_depth"],
        "eta":              best_aug["learning_rate"],
        "subsample":        best_aug["subsample"],
        "colsample_bytree": best_aug["colsample_bytree"],
        "min_child_weight": best_aug["min_child_weight"],
        "alpha":            best_aug["reg_alpha"],
        "lambda":           best_aug["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds_a = best_aug["n_estimators"]

    dmat_a = xgb.DMatrix(Xa_scaled, label=ya)
    bst_a  = xgb.train(params_a, dmat_a, num_boost_round=n_rounds_a, verbose_eval=False)

    # Save augmented model + artifacts
    augmented_path = MODEL_BASE.replace(".json", "_augmented.json")
    bst_a.save_model(augmented_path)
    print(f"Augmented model saved to {augmented_path}")

    joblib.dump(scaler_a, "models/scaler_augmented.pkl")
    joblib.dump(feature_cols_aug, "models/feature_cols_augmented.pkl")
    historical_stats_a = (
        df_full_aug.groupby("LSOA code")["Count"]
                     .agg(['mean', 'std', 'count'])
                     .reset_index()
    )
    historical_stats_a.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    joblib.dump(historical_stats_a, "models/historical_stats_augmented.pkl")


def cross_val_metrics(df_full):
    """
    Runs N_SPLITS-fold time-series CV on df_full and returns:
      maes, rmses, r2s (lists of length N_SPLITS),
      and fold_by_fold list of (fold_number, mae, rmse, r2).
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
        df_val = create_time_aware_features(df_val, historical_stats=hist_stats)

        feature_cols = [
            c for c in df_train.columns
            if c not in ["LSOA code", "Date", "Count"]
        ]
        X_train, y_train = df_train[feature_cols], df_train["Count"]
        X_val,   y_val   = df_val[feature_cols],   df_val["Count"]

        X_train_scaled, X_val_scaled, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
        dtr = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled,   label=y_val)

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
    Evaluates both baseline and augmented models via time‐series CV.
    Prints CV MAE, RMSE, R² (mean ± std) and fold-by-fold results.
    """
    df_full_base = load_baseline_df(DATA_BASELINE_PATH)
    df_full_aug  = load_augmented_df(DATA_AUGMENTED_PATH)

    print("\n=== EVALUATING BASELINE MODEL (no prop_ columns) ===")
    mae_b, rmse_b, r2_b, folds_b = cross_val_metrics(df_full_base)

    print("\n=== EVALUATING AUGMENTED MODEL (with prop_ columns) ===")
    mae_a, rmse_a, r2_a, folds_a = cross_val_metrics(df_full_aug)

    print("\n--- CROSS-VALIDATION COMPARISON ---")
    print(f"Baseline MAE   : {np.mean(mae_b):.4f} ± {np.std(mae_b):.4f}")
    print(f"Augmented MAE  : {np.mean(mae_a):.4f} ± {np.std(mae_a):.4f}")
    print(f"Baseline RMSE  : {np.mean(rmse_b):.4f} ± {np.std(rmse_b):.4f}")
    print(f"Augmented RMSE : {np.mean(rmse_a):.4f} ± {np.std(rmse_a):.4f}")
    print(f"Baseline R²    : {np.mean(r2_b):.4f} ± {np.std(r2_b):.4f}")
    print(f"Augmented R²   : {np.mean(r2_a):.4f} ± {np.std(r2_a):.4f}")

    print("\nBaseline fold-by-fold results:")
    for fnum, m, r, r2 in folds_b:
        print(f"  Fold {fnum}: MAE={m:.4f}, RMSE={r:.4f}, R²={r2:.4f}")

    print("\nAugmented fold-by-fold results:")
    for fnum, m, r, r2 in folds_a:
        print(f"  Fold {fnum}: MAE={m:.4f}, RMSE={r:.4f}, R²={r2:.4f}")


def plot_cv_performance():
    """
    Plots citywide aggregated Actual vs. Predicted for the AUGMENTED model's last fold.
    Saves figure to “models/cv_performance_augmented.png”.
    """
    df = load_augmented_df(DATA_AUGMENTED_PATH)
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
                .agg(['mean', 'std', 'count'])
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

    print(f"Last-fold Train → R²={train_r2:.3f}, MAE={train_mae:.1f}, RMSE={train_rmse:.1f}")
    print(f"Last-fold Val   → R²={val_r2:.3f}, MAE={val_mae:.1f}, RMSE={val_rmse:.1f}")

    plt.figure(figsize=(14, 6))
    plt.plot(train_agg.index, train_agg["Actual"], label="Training Actual", color='blue', linewidth=2)
    plt.plot(train_agg.index, train_agg["Predicted"], "--", label="Training Predicted", color='orange', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Actual"],   label="Validation Actual", color='green', linewidth=2)
    plt.plot(val_agg.index,   val_agg["Predicted"], ":", label="Validation Predicted", color='red', linewidth=2)

    split_date = pd.to_datetime(val_dates[0])
    plt.axvline(x=split_date, color='black', linestyle='-', alpha=0.5, linewidth=2)

    plt.title(f"AUGMENTED MODEL (Last Fold) → Train R²={train_r2:.3f}, Val R²={val_r2:.3f}")
    plt.xlabel("Date")
    plt.ylabel("Citywide Monthly Burglaries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance_augmented.png", dpi=300, bbox_inches='tight')
    print("Plot saved to models/cv_performance_augmented.png")


if __name__ == "__main__":
    print("Starting burglary prediction (baseline vs. augmented)…\n")

    # 1) Train & save both models
    train_and_save()

    # 2) Evaluate both via time-series CV
    print("\n2. Evaluating baseline vs. augmented via time-series CV…")
    evaluate()

    # 3) Plot performance for the augmented model’s last fold
    print("\n3. Plotting CV performance for augmented model…")
    plot_cv_performance()

    print("\nAll steps complete.")

