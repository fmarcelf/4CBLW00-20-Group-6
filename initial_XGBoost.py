import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === Configuration ===
DATA_PATH    = r"C:/Users/20232553/Downloads/data - transformed.xlsx"  # .xlsx or .csv
MODEL_PATH   = "models/xgb_burglary_model.json"

TRIALS       = 150
TIMEOUT      = 3600
N_SPLITS     = 5
RANDOM_STATE = 42

# Cut‐off between train vs. test
DATE_SPLIT   = pd.Timestamp("2023-01-01")


def load_and_aggregate(path):
    # 1) Read Excel (header on second row) or CSV
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=1, engine="openpyxl")
    else:
        df = pd.read_csv(path)

    # 2) Drop empty “Unnamed:” columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # 3) Force Date → datetime
    if "Date" not in df.columns:
        raise KeyError(f"'Date' column not found! cols: {df.columns.tolist()}")
    df["Date"] = pd.to_datetime(df["Date"])

    # 4) Count each event
    df["Count"] = 1
    agg = (
        df
        .groupby(["Ward code","Ward name","LSOA code","Date"], as_index=False)["Count"]
        .sum()
    )

    # 5) Bring in any static “rank” cols (if present)
    static_cols = [c for c in df.columns if "rank" in c.lower()]
    if static_cols:
        static = df[["LSOA code"] + static_cols].drop_duplicates("LSOA code")
        agg = agg.merge(static, on="LSOA code", how="left")

    return agg


def create_features(df):
    df = df.sort_values(["LSOA code","Date"]).copy()

    # Date parts
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year
    # one-hot months (drop_first avoids collinearity)
    month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    # autoregressive lags
    for lag in (1,2,3):
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)

    # 3-month rolling average
    df["roll3"] = (
        df.groupby("LSOA code")["Count"]
          .shift(1)
          .rolling(3, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # fill NaNs (i.e. first few rows per LSOA)
    return df.fillna(0)


def objective(trial, X_tr, y_tr):
    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "eta":              trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha":            trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "lambda":           trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE
    }
    n_rounds = trial.suggest_int("n_estimators", 100, 500)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    maes = []

    for tr_idx, val_idx in tscv.split(X_tr):
        dtr  = xgb.DMatrix(X_tr.iloc[tr_idx], label=y_tr.iloc[tr_idx])
        dval = xgb.DMatrix(X_tr.iloc[val_idx], label=y_tr.iloc[val_idx])

        bst = xgb.train(
            params, dtr,
            num_boost_round=n_rounds,
            evals=[(dval, "valid")],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        pred = bst.predict(dval)
        maes.append(mean_absolute_error(y_tr.iloc[val_idx], pred))

    return float(np.mean(maes))


def train_and_save():
    # 1) Load & features
    df = load_and_aggregate(DATA_PATH)
    df = create_features(df)

    # 2) Split train/test by DATE_SPLIT
    train = df[df["Date"] < DATE_SPLIT].copy()
    test  = df[df["Date"] >= DATE_SPLIT].copy()

    X_tr = train.drop(columns=["Ward code","Ward name","LSOA code","Date","Count"])
    y_tr = np.log1p(train["Count"])    # <-- log1p transform

    # 3) Optuna on train only
    pruner = optuna.pruners.MedianPruner(5,10,5)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=pruner
    )
    study.optimize(lambda t: objective(t, X_tr, y_tr),
                   n_trials=TRIALS,
                   timeout=TIMEOUT,
                   show_progress_bar=True)

    best = study.best_params
    print("Best params:", best)

    # 4) Final train on full train set
    params = {
        "max_depth":        best["max_depth"],
        "eta":              best["learning_rate"],
        "subsample":        best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "alpha":            best["reg_alpha"],
        "lambda":           best["reg_lambda"],
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "seed":             RANDOM_STATE
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    final_bst = xgb.train(params, dtrain,
                          num_boost_round=best["n_estimators"],
                          verbose_eval=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    final_bst.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def evaluate_and_plot():
    # Reload & re-feature
    df = load_and_aggregate(DATA_PATH)
    df = create_features(df)

    # Split again
    train = df[df["Date"] < DATE_SPLIT].copy()
    test  = df[df["Date"] >= DATE_SPLIT].copy()

    # Load model
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)

    # — City-wide hold-out —
    X_te = test.drop(columns=["Ward code","Ward name","LSOA code","Date","Count"])
    y_te = test["Count"]
    pred_log = bst.predict(xgb.DMatrix(X_te))
    preds    = np.expm1(pred_log)  # invert log1p

    r2_city = r2_score(y_te, preds)
    mae_city = mean_absolute_error(y_te, preds)
    rmse_city = np.sqrt(mean_squared_error(y_te, preds))
    print(f"\nCity-wide Test Hold-out → R²={r2_city:.3f}, MAE={mae_city:.2f}, RMSE={rmse_city:.2f}")

    # — Ward-level metrics —
    test = test.assign(Pred=preds)
    ward_metrics = []
    for (wcode, wname), grp in test.groupby(["Ward code","Ward name"]):
        a = grp["Count"]
        p = grp["Pred"]
        ward_metrics.append({
            "Ward code": wcode,
            "Ward name": wname,
            "R2": r2_score(a,p),
            "MAE": mean_absolute_error(a,p),
            "RMSE": np.sqrt(mean_squared_error(a,p))
        })
    ward_df = pd.DataFrame(ward_metrics).sort_values("R2", ascending=False)

    print("\nPer-Ward Test Metrics:")
    print(ward_df.to_string(index=False, float_format='%.3f'))

    # — Optional: Plot city aggregate over time —
    df["Pred"] = np.expm1(bst.predict(xgb.DMatrix(
        df.drop(columns=["Ward code","Ward name","LSOA code","Date","Count"])
    )))
    monthly = (
        df.groupby("Date")
          .agg(Actual=("Count","sum"), Pred=("Pred","sum"))
          .sort_index()
    )
    plt.figure(figsize=(10,5))
    plt.plot(monthly.index, monthly["Actual"], label="Actual")
    plt.plot(monthly.index, monthly["Pred"],   "--", label="Predicted")
    plt.title(f"Citywide: Actual vs Pred (Train < {DATE_SPLIT.date()})")
    plt.xlabel("Month")
    plt.ylabel("Total Burglaries")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_save()
    evaluate_and_plot()

