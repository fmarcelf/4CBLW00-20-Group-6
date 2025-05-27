import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Configuration ===
# Replace DATA_PATH with the location of your Data-Transformed.csv - make sure the data is in the same folder as this file
DATA_PATH    = r"C:/path/to/Data-Transformed.csv"

# Replace MODEL_PATH if you want to save/load the model somewhere else - you can keep it as it is
MODEL_PATH   = "models/xgb_burglary_model.json"

# Tuning & CV settings (you probably don’t need to change these)
TRIALS       = 150
TIMEOUT      = 3600     # seconds
N_SPLITS     = 5
RANDOM_STATE = 42

# Date at which you want to split train vs. test for plotting
DATE_SPLIT   = pd.Timestamp("2023-01-01")


def load_and_aggregate(path):
    """
    1) Read event-level CSV  
    2) Aggregate to monthly counts per LSOA  
    3) Merge in static IMD rank columns
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Count"] = 1
    agg = df.groupby(["LSOA code", "Date"], as_index=False)["Count"].sum()
    static_cols = [c for c in df.columns if "rank" in c.lower()]
    static      = df[["LSOA code"] + static_cols].drop_duplicates("LSOA code")
    return agg.merge(static, on="LSOA code", how="left")


def create_features(df):
    """
    Add:
      • month, year  
      • lag_1, lag_2, lag_3  
      • 3‐month rolling mean (roll3)
    """
    df = df.sort_values(["LSOA code", "Date"]).copy()
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year

    # autoregressive lags
    for lag in (1, 2, 3):
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)

    # 3-month rolling average
    df["roll3"] = (
        df.groupby("LSOA code")["Count"]
          .shift(1)
          .rolling(3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    return df.fillna(0)


def objective(trial, X, y):
    """
    Optuna objective: for each trial sample hyperparams,
    run 5-fold TimeSeriesSplit CV, return avg MAE.
    """
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "eta":              trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha":            trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "lambda":           trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds = trial.suggest_int("n_estimators", 100, 500)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes = []

    for tr_idx, val_idx in tscv.split(X):
        dtr  = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
        dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            evals=[(dval, "valid")],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        preds = bst.predict(dval)
        maes.append(mean_absolute_error(y.iloc[val_idx], preds))

    return float(np.mean(maes))


def train_and_save():
    """Runs Optuna tuning, trains final model on all data, and saves it."""
    # ---- Load & feature-engineer ----
    df = load_and_aggregate(DATA_PATH)
    df = create_features(df)
    X  = df.drop(columns=["LSOA code", "Date", "Count"])
    y  = df["Count"]

    # ---- Hyperparameter search ----
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=5
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=pruner
    )
    study.optimize(
        lambda t: objective(t, X, y),
        n_trials=TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    best = study.best_params
    print("Best params:", best)

    # ---- Train final model ----
    params = {
        "max_depth":        best["max_depth"],
        "eta":              best["learning_rate"],
        "subsample":        best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "alpha":            best["reg_alpha"],
        "lambda":           best["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE,
    }
    n_rounds = best["n_estimators"]

    dmat = xgb.DMatrix(X, label=y)
    final_bst = xgb.train(
        params=params,
        dtrain=dmat,
        num_boost_round=n_rounds,
        verbose_eval=False
    )
    import os

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    final_bst.save_model(MODEL_PATH)
    print(f"Final model saved to {MODEL_PATH}")


def evaluate():
    """Performs 5-fold TS CV on the saved model and prints CV MAE/RMSE."""
    df = load_and_aggregate(DATA_PATH)
    df = create_features(df)
    X  = df.drop(columns=["LSOA code", "Date", "Count"])
    y  = df["Count"]

    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes, rmses = [], []

    for _, val_idx in tscv.split(X):
        preds = bst.predict(xgb.DMatrix(X.iloc[val_idx]))
        actual = y.iloc[val_idx]

        maes.append(mean_absolute_error(actual, preds))
        rmses.append(np.sqrt(mean_squared_error(actual, preds)))  # RMSE

    print(f"CV MAE : {np.mean(maes):.4f}")
    print(f"CV RMSE: {np.mean(rmses):.4f}")


def plot_performance():
    """
    Aggregates LSOA-month preds to city totals, 
    splits at DATE_SPLIT, prints R²/MAE/RMSE, and plots results.
    """
    df = load_and_aggregate(DATA_PATH)
    df = create_features(df)

    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    X_all = df.drop(columns=["LSOA code", "Date", "Count"])
    df["Pred"] = bst.predict(xgb.DMatrix(X_all))

    monthly = (
        df.groupby("Date")
          .agg(Actual=("Count","sum"), Predicted=("Pred","sum"))
          .sort_index()
    )
    train_agg = monthly.loc[:DATE_SPLIT - pd.Timedelta(days=1)]
    test_agg  = monthly.loc[DATE_SPLIT:]

    # Print metrics
    for label, data in [("Train", train_agg), ("Test", test_agg)]:
        r2   = r2_score(data["Actual"], data["Predicted"])
        mae  = mean_absolute_error(data["Actual"], data["Predicted"])
        rmse = np.sqrt(mean_squared_error(data["Actual"], data["Predicted"]))
        print(f"{label} → R²={r2:.3f}, MAE={mae:.1f}, RMSE={rmse:.1f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(monthly.index,   monthly["Actual"],    label="Actual")
    plt.plot(train_agg.index, train_agg["Predicted"], "--", label="Train Pred")
    plt.plot(test_agg.index,  test_agg["Predicted"],  ":", label="Test Pred")
    plt.title("Citywide Performance\n(Train < 2023 | Test ≥ 2023)")
    plt.xlabel("Month")
    plt.ylabel("Total Burglaries")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === Workflow ===
    train_and_save()
    evaluate()
    plot_performance()