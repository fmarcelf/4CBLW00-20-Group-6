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
import json

warnings.filterwarnings('ignore')

# === Configuration ===

# (1) Baseline CSV: already monthly-aggregated, one row per LSOA-Date,
#     with a column named "Burglary Count" (the total burglaries that month).
DATA_BASELINE_PATH = r"C:/Users/20232553/Downloads/data_aggregated_covid.csv"

# (2) Augmented CSV: same rows as baseline, but with 11 extra columns:
#     7 accommodation-type proportions (prop_…) and 4 hours-worked proportions (prop_hrs_…)
DATA_AUGMENTED_PATH = r"C:/Users/20232553/Downloads/burglaries_with_accom_and_hours_props.csv"

# Base filename for saving the combined model and Optuna params
MODEL_BASE = "models/xgb_burglary_model.json"
OPTUNA_PARAMS_PATH = "models/best_params_combined.json"

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
    - Creates inverse-rank "<rank>_score" columns for any IMD rank columns.
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

    # 5) Create inverse-rank "<rank>_score" columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # 6) Keep only: ['LSOA code', 'Date', 'Count'] + any '<rank>_score' columns
    keep_cols = ["LSOA code", "Date", "Count"] + [c for c in df.columns if c.endswith("_score")]
    df = df[keep_cols].copy()

    return df


def load_combined_df(path):
    """
    Loads the "augmented" CSV and keeps:
      - ['LSOA code', 'Ward', 'Date', 'Count', <rank_score cols>]
      - + all accommodation-type proportions (prop_… but not prop_hrs_…)
      - + all hours-worked proportions (prop_hrs_…)
    Drops pandemic years, drops 'Crime rank', renames 'Burglary Count' → 'Count', creates rank_scores.
    Returns a DataFrame with exactly:
      ['LSOA code', 'Ward', 'Date', 'Count', <rank_score cols>, <prop_… columns>, <prop_hrs_… columns>].
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

    # 5) Create inverse-rank "<rank>_score" columns for any IMD rank columns
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c not in ["Burglary Count", "Count"]]
    for col in rank_cols:
        df[f"{col}_score"] = 1.0 / (df[col] + 1.0)

    # 6) Identify all accommodation-type props (prop_… but not prop_hrs_…) and hours-worked props (prop_hrs_…)
    accom_cols = [c for c in df.columns if c.startswith("prop_") and not c.startswith("prop_hrs_")]
    hours_cols = [c for c in df.columns if c.startswith("prop_hrs_")]

    if len(accom_cols) < 7:
        raise KeyError(f"Expected at least 7 accommodation prop_… columns, found {len(accom_cols)}.")
    if len(hours_cols) < 4:
        raise KeyError(f"Expected at least 4 prop_hrs_… columns, found {len(hours_cols)}.")

    # 7) Keep only ['LSOA code', 'Ward', 'Date', 'Count', <rank_score cols>, <accom_cols>, <hours_cols>]
    keep_cols = ["LSOA code"]
    if "Ward" in df.columns:
        keep_cols.append("Ward")
    keep_cols += ["Date", "Count"] + [c for c in df.columns if c.endswith("_score")] + accom_cols + hours_cols
    df = df[keep_cols].copy()

    return df


def create_time_aware_features(df, historical_stats=None, cutoff_date=None):
    """
    FIXED VERSION: Adds time-aware features without data leakage.
    Args:
        df: Input DataFrame
        historical_stats: Pre-computed historical statistics (for validation sets)
        cutoff_date: Date before which to compute historical stats (prevents leakage)

    Features added:
      1) month, year, month_sin, month_cos
      2) season dummy variables (Spring, Summer, Autumn; Winter as baseline)
      3) is_summer_month, is_winter_month, is_school_holiday_month
      4) lag_1, lag_2, lag_3, lag_6, lag_12 (per LSOA code)
      5) roll3_mean, roll3_std, roll6_mean, roll6_std, roll12_mean, roll12_std
         (computed on Count.shift(1) to avoid leakage)
      6) lsoa_historical_mean, lsoa_historical_std, lsoa_historical_count
         (only computed from data before cutoff_date if provided)
      7) diff_1 = lag_1 - lag_2 ; yoy_diff = lag_1 - lag_12
      8) imd_lag1 = (first rank_score) * lag_1 (removed imd_historical to avoid leakage)
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

    # 3) Holiday / school indicators
    df["is_summer_month"] = df["month"].between(6, 8).astype(int)
    df["is_winter_month"] = ((df["month"] == 12) | (df["month"].between(1, 2))).astype(int)
    df["is_school_holiday_month"] = df["month"].isin([7, 8, 12]).astype(int)

    # 4) Lag features: previous 1, 2, 3, 6, 12 months (per LSOA)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)

    # 5) Rolling statistics (Count.shift(1))
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

    # 6) Historical stats
    if historical_stats is not None:
        df = df.merge(historical_stats, on="LSOA code", how="left")
    else:
        if cutoff_date is not None:
            historical_data = df[df["Date"] < cutoff_date].copy()
            if len(historical_data) > 0:
                hist_stats = (
                    historical_data.groupby("LSOA code")["Count"]
                                  .agg(['mean', 'std', 'count'])
                                  .reset_index()
                )
                hist_stats.columns = [
                    "LSOA code",
                    "lsoa_historical_mean",
                    "lsoa_historical_std",
                    "lsoa_historical_count"
                ]
                df = df.merge(hist_stats, on="LSOA code", how="left")
            else:
                df["lsoa_historical_mean"] = 0
                df["lsoa_historical_std"] = 0
                df["lsoa_historical_count"] = 0
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

    # 7) Difference features
    df["diff_1"] = df["lag_1"] - df["lag_2"]
    df["yoy_diff"] = df["lag_1"] - df["lag_12"]

    # 8) IMD interaction
    rank_score_cols = [c for c in df.columns if c.endswith("_score")]
    if rank_score_cols:
        df["imd_lag1"] = df[rank_score_cols[0]] * df["lag_1"]

    # 9) Fill NaNs
    return df.fillna(0)


def time_aware_preprocess(X_train, y_train, X_val, y_val):
    """
    Fits a StandardScaler on X_train only, transforms both X_train & X_val.
    Does NOT scale dummy/binary features.
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
    FIXED VERSION: Optuna objective without data leakage.
    Returns average MAE over N_SPLITS folds.
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

    dates = np.sort(df_full["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    date_indices = np.arange(len(dates))

    fold_maes = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(date_indices)):
        train_dates = dates[train_idx]
        val_dates   = dates[val_idx]
        cutoff_date = val_dates[0]

        df_train = df_full[df_full["Date"].isin(train_dates)].copy()
        df_val   = df_full[df_full["Date"].isin(val_dates)].copy()

        # historical stats
        historical_data = df_full[df_full["Date"] < cutoff_date]
        if len(historical_data) > 0:
            hist_stats = (
                historical_data.groupby("LSOA code")["Count"]
                                .agg(['mean','std','count'])
                                .reset_index()
            )
            hist_stats.columns = [
                "LSOA code",
                "lsoa_historical_mean",
                "lsoa_historical_std",
                "lsoa_historical_count"
            ]
        else:
            unique_lsoas = df_full["LSOA code"].unique()
            hist_stats = pd.DataFrame({
                "LSOA code": unique_lsoas,
                "lsoa_historical_mean": 0,
                "lsoa_historical_std": 0,
                "lsoa_historical_count": 0
            })

        df_train = create_time_aware_features(df_train, historical_stats=hist_stats)
        df_val   = create_time_aware_features(df_val,   historical_stats=hist_stats)

        feature_cols = [c for c in df_train.columns if c not in ["LSOA code","Date","Count"]]
        X_tr, y_tr = df_train[feature_cols], df_train["Count"]
        X_vl, y_vl = df_val[feature_cols],   df_val["Count"]
        X_tr_s, X_vl_s, _ = time_aware_preprocess(X_tr, y_tr, X_vl, y_vl)

        dtr = xgb.DMatrix(X_tr_s, label=y_tr)
        dvl = xgb.DMatrix(X_vl_s, label=y_vl)
        bst = xgb.train(params, dtr, num_boost_round=n_rounds,
                        evals=[(dvl, "valid")], early_stopping_rounds=30, verbose_eval=False)

        preds = bst.predict(dvl)
        fold_maes.append(mean_absolute_error(y_vl, preds))

    return float(np.mean(fold_maes))


def train_and_save():
    """
    FIXED VERSION: Trains and saves XGBoost model without data leakage.
    """
    os.makedirs(os.path.dirname(MODEL_BASE), exist_ok=True)
    combined_path = MODEL_BASE.replace(".json", "_combined.json")
    params_path = OPTUNA_PARAMS_PATH

    print("\n=== TRAINING OR LOADING COMBINED MODEL (ACCOM + HOURS) ===")
    df_combined = load_combined_df(DATA_AUGMENTED_PATH)

    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            best_combined = json.load(f)
        print(f"Loaded Optuna parameters from {params_path}:")
        print(best_combined)
    else:
        study_combined = optuna.create_study(direction="minimize",
                                            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study_combined.optimize(lambda t: objective(t, df_combined),
                                n_trials=TRIALS, timeout=TIMEOUT)
        best_combined = study_combined.best_params
        with open(params_path, "w") as f:
            json.dump(best_combined, f, indent=4)
        print(f"Saved Optuna parameters to {params_path}")

    df_full_combined = create_time_aware_features(df_combined)
    feature_cols_combined = [c for c in df_full_combined.columns if c not in ["LSOA code","Date","Count"]]
    X_c = df_full_combined[feature_cols_combined]
    y_c = df_full_combined["Count"]

    cols_to_scale = [c for c in feature_cols_combined if not c.startswith("season_") and not c.startswith("is_") and c!="month"]
    scaler_c = StandardScaler()
    X_c[cols_to_scale] = scaler_c.fit_transform(X_c[cols_to_scale])

    params_c = {
        "max_depth":        best_combined["max_depth"],
        "eta":              best_combined["learning_rate"],
        "subsample":        best_combined["subsample"],
        "colsample_bytree": best_combined["colsample_bytree"],
        "min_child_weight": best_combined["min_child_weight"],
        "alpha":            best_combined["reg_alpha"],
        "lambda":           best_combined["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds_c = best_combined["n_estimators"]
    dmat_c = xgb.DMatrix(X_c, label=y_c)
    bst_c = xgb.train(params_c, dmat_c, num_boost_round=n_rounds_c, verbose_eval=False)
    bst_c.save_model(combined_path)
    print(f"Combined model saved to {combined_path}")

    joblib.dump(scaler_c, "models/scaler_combined.pkl")
    joblib.dump(feature_cols_combined, "models/feature_cols_combined.pkl")
    historical_stats_c = (
        df_full_combined.groupby("LSOA code")["Count"]
                        .agg(['mean','std','count'])
                        .reset_index()
    )
    historical_stats_c.columns = [
        "LSOA code",
        "lsoa_historical_mean",
        "lsoa_historical_std",
        "lsoa_historical_count"
    ]
    joblib.dump(historical_stats_c, "models/historical_stats_combined.pkl")


def cross_val_metrics(df_full):
    """
    FIXED VERSION: Runs N_SPLITS-fold time-series CV without data leakage.
    """
    dates = np.sort(df_full["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    idx = np.arange(len(dates))
    maes, rmses, r2s, fold_results = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(idx)):
        train_dates = dates[train_idx]
        val_dates   = dates[val_idx]
        cutoff_date = val_dates[0]

        df_train = df_full[df_full["Date"].isin(train_dates)].copy()
        df_val   = df_full[df_full["Date"].isin(val_dates)].copy()

        historical_data = df_full[df_full["Date"] < cutoff_date]
        if not historical_data.empty:
            hist_stats = (
                historical_data.groupby("LSOA code")["Count"]
                               .agg(['mean','std','count'])
                               .reset_index()
            )
            hist_stats.columns = [
                "LSOA code",
                "lsoa_historical_mean",
                "lsoa_historical_std",
                "lsoa_historical_count"
            ]
        else:
            unique_lsoas = df_full["LSOA code"].unique()
            hist_stats = pd.DataFrame({
                "LSOA code": unique_lsoas,
                "lsoa_historical_mean": 0,
                "lsoa_historical_std": 0,
                "lsoa_historical_count": 0
            })

        df_train = create_time_aware_features(df_train, historical_stats=hist_stats)
        df_val   = create_time_aware_features(df_val,   historical_stats=hist_stats)

        feat = [c for c in df_train.columns if c not in ["LSOA code","Date","Count"]]
        X_tr, y_tr = df_train[feat], df_train["Count"]
        X_vl, y_vl = df_val[feat],   df_val["Count"]
        X_tr_s, X_vl_s, _ = time_aware_preprocess(X_tr, y_tr, X_vl, y_vl)

        params = {
            "max_depth":6, "eta":0.1,
            "subsample":0.8, "colsample_bytree":0.8,
            "min_child_weight":3, "alpha":0.1, "lambda":1.0,
            "objective":"reg:squarederror", "eval_metric":"mae",
            "seed":RANDOM_STATE
        }
        bst = xgb.train(params, xgb.DMatrix(X_tr_s, label=y_tr),
                        num_boost_round=200,
                        evals=[(xgb.DMatrix(X_vl_s,label=y_vl),"valid")],
                        early_stopping_rounds=30, verbose_eval=False)

        preds = bst.predict(xgb.DMatrix(X_vl_s))
        maes.append(mean_absolute_error(y_vl, preds))
        rmses.append(np.sqrt(mean_squared_error(y_vl, preds)))
        r2s.append(r2_score(y_vl, preds))
        fold_results.append((fold+1, maes[-1], rmses[-1], r2s[-1]))

    return maes, rmses, r2s, fold_results


def evaluate():
    """
    Evaluates the combined model via time-series CV.
    Prints CV MAE, RMSE, R² (mean ± std) and fold-by-fold results.
    """
    df_full_combined = load_combined_df(DATA_AUGMENTED_PATH)
    print("\n=== EVALUATING COMBINED MODEL (ACCOM + HOURS) ===")
    mae_c, rmse_c, r2_c, folds_c = cross_val_metrics(df_full_combined)

    print("\n--- CROSS-VALIDATION RESULTS FOR COMBINED MODEL ---")
    print(f"Combined MAE   : {np.mean(mae_c):.4f} ± {np.std(mae_c):.4f}")
    print(f"Combined RMSE  : {np.mean(rmse_c):.4f} ± {np.std(rmse_c):.4f}")
    print(f"Combined R²    : {np.mean(r2_c):.4f} ± {np.std(r2_c):.4f}")

    print("\nCombined fold-by-fold results:")
    for fnum, m, r, r2 in folds_c:
        print(f"  Fold {fnum}: MAE={m:.4f}, RMSE={r:.4f}, R²={r2:.4f}")


def plot_cv_performance_combined():
    """
    FIXED VERSION: Plots citywide aggregated Actual vs. Predicted for the COMBINED model's last fold.
    Saves figure to "models/cv_performance_combined.png" and prints per-ward metrics.
    """
    df = load_combined_df(DATA_AUGMENTED_PATH)
    dates = np.sort(df["Date"].unique())
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    idx = np.arange(len(dates))

    train_idx, val_idx = list(tscv.split(idx))[-1]
    train_dates, val_dates = dates[train_idx], dates[val_idx]
    cutoff_date = val_dates[0]

    df_train = df[df["Date"].isin(train_dates)].copy()
    df_val   = df[df["Date"].isin(val_dates)].copy()

    historical_data = df[df["Date"] < cutoff_date]
    if not historical_data.empty:
        hist_stats = (
            historical_data.groupby("LSOA code")["Count"]
                           .agg(['mean','std','count'])
                           .reset_index()
        )
        hist_stats.columns = [
            "LSOA code",
            "lsoa_historical_mean",
            "lsoa_historical_std",
            "lsoa_historical_count"
        ]
    else:
        unique_lsoas = df["LSOA code"].unique()
        hist_stats = pd.DataFrame({
            "LSOA code": unique_lsoas,
            "lsoa_historical_mean": 0,
            "lsoa_historical_std": 0,
            "lsoa_historical_count": 0
        })

    df_train = create_time_aware_features(df_train, historical_stats=hist_stats)
    df_val   = create_time_aware_features(df_val,   historical_stats=hist_stats)

    feat = [c for c in df_train.columns if c not in ["LSOA code","Ward","Date","Count"]]
    X_tr, y_tr = df_train[feat], df_train["Count"]
    X_vl, y_vl = df_val[feat],   df_val["Count"]
    X_tr_s, X_vl_s, _ = time_aware_preprocess(X_tr, y_tr, X_vl, y_vl)

    params = {
        "max_depth":6, "eta":0.1,
        "subsample":0.8, "colsample_bytree":0.8,
        "min_child_weight":3, "alpha":0.1, "lambda":1.0,
        "objective":"reg:squarederror", "eval_metric":"mae",
        "seed":RANDOM_STATE
    }
    bst = xgb.train(params, xgb.DMatrix(X_tr_s, label=y_tr), num_boost_round=200)

    df_train["Pred"] = bst.predict(xgb.DMatrix(X_tr_s))
    df_val["Pred"]   = bst.predict(xgb.DMatrix(X_vl_s))

    # Citywide plot
    train_agg = df_train.groupby("Date").agg(Actual=("Count","sum"), Predicted=("Pred","sum")).sort_index()
    val_agg   = df_val.groupby("Date").agg(Actual=("Count","sum"), Predicted=("Pred","sum")).sort_index()

    plt.figure(figsize=(14,6))
    plt.plot(train_agg.index, train_agg["Actual"],    label="Train Actual")
    plt.plot(train_agg.index, train_agg["Predicted"],"--", label="Train Pred")
    plt.plot(val_agg.index,   val_agg["Actual"],     label="Val Actual")
    plt.plot(val_agg.index,   val_agg["Predicted"],  ":", label="Val Pred")
    plt.axvline(x=val_dates[0], color='k', linestyle='-')
    plt.title(f"Last Fold → Train R²={r2_score(train_agg['Actual'], train_agg['Predicted']):.3f}, "
              f"Val R²={r2_score(val_agg['Actual'], val_agg['Predicted']):.3f}")
    plt.xlabel("Date")
    plt.ylabel("Burglaries")
    plt.legend()
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance_combined.png", dpi=300)
    print("Plot saved to models/cv_performance_combined.png")

    # ---- Per-Ward Performance ----
    if "Ward" in df_val.columns:
        ward_perf = df_val.groupby("Ward").apply(
            lambda d: pd.Series({
                "R2"  : r2_score(d["Count"], d["Pred"]),
                "MAE" : mean_absolute_error(d["Count"], d["Pred"]),
                "RMSE": np.sqrt(mean_squared_error(d["Count"], d["Pred"]))
            })
        ).reset_index()
        print("\nPer-Ward Performance (last fold):")
        print(ward_perf.to_string(index=False))
        ward_perf.to_csv("models/ward_performance_last_fold.csv", index=False)
        print("Saved per-ward performance to models/ward_performance_last_fold.csv")


if __name__ == "__main__":
    print("Starting burglary prediction (combined accommodation + hours-worked) - FIXED VERSION…\n")
    train_and_save()
    print("\n2. Evaluating combined model via time-series CV…")
    evaluate()
    print("\n3. Plotting CV performance for combined model…")
    plot_cv_performance_combined()
    print("\nAll steps complete. Data leakage has been fixed!")
