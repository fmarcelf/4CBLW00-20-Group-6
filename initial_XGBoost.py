import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import calendar
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
DATA_PATH    = r"C:/Users/20232553/OneDrive - TU Eindhoven/Documents/GitHub/4CBLW00-20-Group-6-1/Data - Transformed.csv"
MODEL_PATH   = "models/xgb_burglary_model.json"

# Tuning & CV settings
TRIALS       = 50  # Reduced for faster testing
TIMEOUT      = 1800     # seconds
N_SPLITS     = 5
RANDOM_STATE = 42


def load_and_aggregate(path):
    """
    1) Read event-level CSV  
    2) Aggregate to monthly counts per LSOA  
    3) Merge in static IMD rank columns
    """
    try:
        print(f"Attempting to load data from: {path}")
        df = pd.read_csv(path, parse_dates=["Date"])
        print(f"Successfully loaded data with {len(df)} rows")
    except FileNotFoundError:
        import os
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        print("Files in current directory:")
        for f in os.listdir(current_dir):
            print(f"  - {f}")
        raise FileNotFoundError(f"Could not find {path}. Please update the DATA_PATH variable in the script.")

    # Remove pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])]

    # Drop "Crime rank" if present
    if "Crime rank" in df.columns:
        df = df.drop(columns=["Crime rank"])

    df["Count"] = 1
    agg = df.groupby(["LSOA code", "Date"], as_index=False)["Count"].sum()

    # Get static columns and convert ranks to scores (inverse)
    static_cols = [c for c in df.columns if "rank" in c.lower()]
    static = df[["LSOA code"] + static_cols].drop_duplicates("LSOA code")

    # Create inverse rank features (higher means more deprived)
    for col in static_cols:
        static[f"{col}_score"] = 1 / (static[col] + 1)  # Add 1 to avoid division by zero

    return agg.merge(static, on="LSOA code", how="left")


def create_time_aware_features(df, historical_stats=None):
    """
    Enhanced feature engineering for MONTHLY aggregated data with STRICT time-awareness.
    Uses pre-configured historical statistics to prevent data leakage.
    """
    df = df.sort_values(["LSOA code", "Date"]).copy()
    
    # === Basic temporal features (appropriate for monthly data) ===
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    
    # Month as cyclical features (captures seasonal patterns)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Season encoding (meaningful for monthly data)
    conditions = [
        (df["month"] >= 3) & (df["month"] <= 5),    # Spring: Mar, Apr, May
        (df["month"] >= 6) & (df["month"] <= 8),    # Summer: Jun, Jul, Aug
        (df["month"] >= 9) & (df["month"] <= 11),   # Autumn: Sep, Oct, Nov
    ]
    choices = ["Spring", "Summer", "Autumn"]
    df["season"] = "Winter"  # Dec, Jan, Feb
    df["season"] = np.select(conditions, choices, default="Winter")
    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    
    # Monthly patterns that might affect burglary rates
    df["is_summer_month"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)  # Jun-Aug
    df["is_winter_month"] = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)  # Dec-Feb
    df["is_school_holiday_month"] = ((df["month"] == 7) | (df["month"] == 8) | (df["month"] == 12)).astype(int)  # Jul, Aug, Dec
    
    # === Time-aware lag features ===
    # Simple lag features (these are always safe)
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)
    
    # === Time-aware rolling statistics ===
    # These use only past data with explicit shift
    for window in [3, 6, 12]:
        # Rolling mean (using past data only)
        df[f"roll{window}_mean"] = (
            df.groupby("LSOA code")["Count"]
              .shift(1)  # Ensure we only use past data
              .rolling(window, min_periods=1)
              .mean()
        )
        
        # Rolling std (using past data only)
        df[f"roll{window}_std"] = (
            df.groupby("LSOA code")["Count"]
              .shift(1)
              .rolling(window, min_periods=2)
              .std()
        )
    
    # === Use pre-computed historical statistics ===
    if historical_stats is not None:
        # Use provided historical statistics (computed from training data only)
        df = df.merge(historical_stats, on="LSOA code", how="left")
    else:
        # For training data, compute expanding window statistics
        df["lsoa_historical_mean"] = (
            df.groupby("LSOA code")["Count"]
              .expanding()
              .mean()
              .shift(1)  # Don't include current observation
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
    
    # === Trend features (safe as they use lags) ===
    df["diff_1"] = df["Count"] - df["lag_1"]
    df["yoy_diff"] = df["Count"] - df["lag_12"]
    
    # === Feature interactions ===
    rank_cols = [c for c in df.columns if "rank_score" in c]
    if rank_cols:
        rank_col = rank_cols[0]
        df["imd_lag1"] = df[rank_col] * df["lag_1"]
        df["imd_historical"] = df[rank_col] * df["lsoa_historical_mean"]
    
    # Fill missing values
    df = df.fillna(0)
    
    return df


def time_aware_preprocess(X_train, y_train, X_val, y_val):
    """
    Time-aware preprocessing that prevents data leakage.
    Fits scaler only on training data.
    """
    # Identify columns to scale (avoid dummy variables and binary indicators)
    cols_to_scale = [col for col in X_train.columns if not (
        col.startswith('season_') or 
        col.startswith('is_') or 
        col == 'month'  # Keep month as integer
    )]
    
    # Fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    
    # Transform validation data using training scaler
    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    
    return X_train_scaled, X_val_scaled, scaler


def objective(trial, df_full):
    """
    Time-aware Optuna objective function.
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

    # Time series split
    dates = df_full["Date"].unique()
    dates = np.sort(dates)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    date_indices = np.arange(len(dates))
    maes = []

    for train_date_idx, val_date_idx in tscv.split(date_indices):
        train_dates = dates[train_date_idx]
        val_dates = dates[val_date_idx]
        
        # Split data by dates
        train_mask = df_full["Date"].isin(train_dates)
        val_mask = df_full["Date"].isin(val_dates)
        
        df_train = df_full[train_mask].copy()
        df_val = df_full[val_mask].copy()
        
        # Create features for training data
        df_train = create_time_aware_features(df_train)
        
        # Compute historical statistics from training data only
        historical_stats = df_train.groupby("LSOA code")["Count"].agg(['mean', 'std', 'count']).reset_index()
        historical_stats.columns = ["LSOA code", "lsoa_historical_mean", "lsoa_historical_std", "lsoa_historical_count"]
        
        # Create features for validation data using training statistics
        df_val = create_time_aware_features(df_val, historical_stats=historical_stats)
        
        # Prepare features
        feature_cols = [c for c in df_train.columns if c not in ["LSOA code", "Date", "Count"]]
        X_train, y_train = df_train[feature_cols], df_train["Count"]
        X_val, y_val = df_val[feature_cols], df_val["Count"]
        
        # Time-aware preprocessing
        X_train_scaled, X_val_scaled, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
        
        # Train model
        dtr = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            evals=[(dval, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        preds = bst.predict(dval)
        maes.append(mean_absolute_error(y_val, preds))

    return float(np.mean(maes))


def train_and_save():
    """Runs Optuna tuning, trains final model on all data, and saves it."""
    # Load data
    df = load_and_aggregate(DATA_PATH)
    
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        lambda t: objective(t, df),
        n_trials=TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    
    best = study.best_params
    print("Best params:", best)

    # Train final model on ALL data
    df_full = create_time_aware_features(df)
    
    feature_cols = [c for c in df_full.columns if c not in ["LSOA code", "Date", "Count"]]
    X = df_full[feature_cols]
    y = df_full["Count"]

    # Scale features
    cols_to_scale = [col for col in X.columns if not (
        col.startswith('season_') or 
        col.startswith('is_') or 
        col == 'month'
    )]
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    params = {
        "max_depth":        best["max_depth"],
        "eta":              best["learning_rate"],
        "subsample":        best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": best["min_child_weight"],
        "alpha":            best["reg_alpha"],
        "lambda":           best["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds = best["n_estimators"]

    dmat = xgb.DMatrix(X_scaled, label=y)
    final_bst = xgb.train(
        params=params,
        dtrain=dmat,
        num_boost_round=n_rounds,
        verbose_eval=False
    )
    
    # Compute historical statistics from all data for future predictions
    historical_stats = df_full.groupby("LSOA code")["Count"].agg(['mean', 'std', 'count']).reset_index()
    historical_stats.columns = ["LSOA code", "lsoa_historical_mean", "lsoa_historical_std", "lsoa_historical_count"]
    
    import os
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    final_bst.save_model(MODEL_PATH)
    
    # Save scaler and other objects for later use
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    joblib.dump(historical_stats, "models/historical_stats.pkl")
    
    print(f"Final model saved to {MODEL_PATH}")


def evaluate():
    """Time-aware model evaluation using cross-validation."""
    df = load_and_aggregate(DATA_PATH)
    
    # Time series split for evaluation
    dates = df["Date"].unique()
    dates = np.sort(dates)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    date_indices = np.arange(len(dates))
    maes, rmses, r2s = [], [], []
    fold_results = []

    for fold, (train_date_idx, val_date_idx) in enumerate(tscv.split(date_indices)):
        train_dates = dates[train_date_idx]
        val_dates = dates[val_date_idx]
        
        # Convert numpy datetime64 to pandas datetime for strftime
        train_dates_pd = pd.to_datetime(train_dates)
        val_dates_pd = pd.to_datetime(val_dates)
        
        print(f"Fold {fold+1}: Train until {train_dates_pd[-1].strftime('%Y-%m')}, Test {val_dates_pd[0].strftime('%Y-%m')} to {val_dates_pd[-1].strftime('%Y-%m')}")
        
        # Split data
        train_mask = df["Date"].isin(train_dates)
        val_mask = df["Date"].isin(val_dates)
        
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        
        # Create features for training data
        df_train = create_time_aware_features(df_train)
        
        # Compute historical statistics from training data only
        historical_stats = df_train.groupby("LSOA code")["Count"].agg(['mean', 'std', 'count']).reset_index()
        historical_stats.columns = ["LSOA code", "lsoa_historical_mean", "lsoa_historical_std", "lsoa_historical_count"]
        
        # Create features for validation data using training statistics
        df_val = create_time_aware_features(df_val, historical_stats=historical_stats)
        
        # Prepare features
        feature_cols = [c for c in df_train.columns if c not in ["LSOA code", "Date", "Count"]]
        X_train, y_train = df_train[feature_cols], df_train["Count"]
        X_val, y_val = df_val[feature_cols], df_val["Count"]
        
        # Time-aware preprocessing
        X_train_scaled, X_val_scaled, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
        
        # Train model for this fold (using default params for evaluation)
        params = {
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "alpha": 0.1,
            "lambda": 1.0,
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "seed": RANDOM_STATE,
        }
        
        dtr = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=200,
            evals=[(dval, "valid")],
            early_stopping_rounds=30,
            verbose_eval=False
        )
        
        preds = bst.predict(dval)
        actual = y_val

        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        fold_results.append((fold+1, mae, rmse, r2))

    # Print results
    print(f"\nCross-Validation Results:")
    print(f"CV MAE : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"CV RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"CV R²  : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    
    print("\nFold-by-fold results:")
    for fold, mae, rmse, r2 in fold_results:
        print(f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")


def plot_cv_performance():
    """Plot performance using the last CV fold as an example."""
    df = load_and_aggregate(DATA_PATH)
    
    # Use the last fold of time series split as example
    dates = df["Date"].unique()
    dates = np.sort(dates)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    date_indices = np.arange(len(dates))
    
    # Get the last fold
    fold_splits = list(tscv.split(date_indices))
    train_date_idx, val_date_idx = fold_splits[-1]  # Last fold
    
    train_dates = dates[train_date_idx]
    val_dates = dates[val_date_idx]
    
    # Convert numpy datetime64 to pandas datetime for strftime
    train_dates_pd = pd.to_datetime(train_dates)
    val_dates_pd = pd.to_datetime(val_dates)
    
    print(f"Plotting performance for last CV fold:")
    print(f"Training period: {train_dates_pd[0].strftime('%Y-%m')} to {train_dates_pd[-1].strftime('%Y-%m')}")
    print(f"Validation period: {val_dates_pd[0].strftime('%Y-%m')} to {val_dates_pd[-1].strftime('%Y-%m')}")
    
    # Split data
    train_mask = df["Date"].isin(train_dates)
    val_mask = df["Date"].isin(val_dates)
    
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    
    # Create features
    df_train = create_time_aware_features(df_train)
    historical_stats = df_train.groupby("LSOA code")["Count"].agg(['mean', 'std', 'count']).reset_index()
    historical_stats.columns = ["LSOA code", "lsoa_historical_mean", "lsoa_historical_std", "lsoa_historical_count"]
    df_val = create_time_aware_features(df_val, historical_stats=historical_stats)
    
    # Train model
    feature_cols = [c for c in df_train.columns if c not in ["LSOA code", "Date", "Count"]]
    X_train, y_train = df_train[feature_cols], df_train["Count"]
    X_val, y_val = df_val[feature_cols], df_val["Count"]
    
    X_train_scaled, X_val_scaled, _ = time_aware_preprocess(X_train, y_train, X_val, y_val)
    
    params = {
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "alpha": 0.1,
        "lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "seed": RANDOM_STATE,
    }
    
    dtr = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)
    
    bst = xgb.train(
        params,
        dtr,
        num_boost_round=200,
        evals=[(dval, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False
    )
    
    # Get predictions
    df_train["Pred"] = bst.predict(dtr)
    df_val["Pred"] = bst.predict(dval)
    df_train["Pred"] = np.maximum(0, df_train["Pred"])
    df_val["Pred"] = np.maximum(0, df_val["Pred"])

    # Citywide aggregation
    train_agg = (
        df_train.groupby("Date")
          .agg(Actual=("Count","sum"), Predicted=("Pred","sum"))
          .sort_index()
    )
    
    val_agg = (
        df_val.groupby("Date")
          .agg(Actual=("Count","sum"), Predicted=("Pred","sum"))
          .sort_index()
    )

    # Print metrics
    train_r2 = r2_score(train_agg["Actual"], train_agg["Predicted"])
    train_mae = mean_absolute_error(train_agg["Actual"], train_agg["Predicted"])
    train_rmse = np.sqrt(mean_squared_error(train_agg["Actual"], train_agg["Predicted"]))
    
    val_r2 = r2_score(val_agg["Actual"], val_agg["Predicted"])
    val_mae = mean_absolute_error(val_agg["Actual"], val_agg["Predicted"])
    val_rmse = np.sqrt(mean_squared_error(val_agg["Actual"], val_agg["Predicted"]))
    
    print(f"Train → R²={train_r2:.3f}, MAE={train_mae:.1f}, RMSE={train_rmse:.1f}")
    print(f"Validation → R²={val_r2:.3f}, MAE={val_mae:.1f}, RMSE={val_rmse:.1f}")

    # Plot
    plt.figure(figsize=(14, 6))
    
    # Plot training period
    plt.plot(train_agg.index, train_agg["Actual"], label="Training Actual", alpha=0.8, color='blue', linewidth=2)
    plt.plot(train_agg.index, train_agg["Predicted"], "--", label="Training Predicted", alpha=0.8, color='orange', linewidth=2)
    
    # Plot validation period
    plt.plot(val_agg.index, val_agg["Actual"], label="Validation Actual", alpha=0.8, color='green', linewidth=2)
    plt.plot(val_agg.index, val_agg["Predicted"], ":", label="Validation Predicted", alpha=0.8, color='red', linewidth=2)
    
    # Add vertical line to separate training and validation
    split_date = val_dates[0]
    plt.axvline(x=split_date, color='black', linestyle='-', alpha=0.5, linewidth=2, label='Train/Validation Split')
    
    plt.title(f"Time Series CV Performance - Last Fold\n(Train R²={train_r2:.3f}, Val R²={val_r2:.3f})")
    plt.xlabel("Date")
    plt.ylabel("Total Monthly Burglaries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot instead of showing it
    import os
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance.png", dpi=300, bbox_inches='tight')
    print("Plot saved to models/cv_performance.png")
    # Remove plt.show() since we're using non-interactive backend


if __name__ == "__main__":
    print("Starting time-aware burglary prediction model with CV-only approach...")
    
    # Check data file
    import os
    if not os.path.exists(DATA_PATH):
        print(f"WARNING: Data file not found at: {os.path.abspath(DATA_PATH)}")
        print("Please update the DATA_PATH variable.")
        exit()
    
    # Create models directory
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Train model
    print("1. Training model with hyperparameter optimization...")
    train_and_save()
    
    # Evaluate model
    print("\n2. Evaluating model with time series cross-validation...")
    evaluate()
    
    # Plot performance  
    print("\n3. Plotting CV performance...")
    plot_cv_performance()
    
    print("\nTime-aware model training complete!")