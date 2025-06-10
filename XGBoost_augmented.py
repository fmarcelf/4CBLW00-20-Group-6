import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# === Configuration ===
DATA_BASELINE_PATH  = r"C:/Users/20232553/Downloads/data_aggregated_covid (2).csv"
DATA_AUGMENTED_PATH = r"C:/Users/20232553/Downloads/burglaries_with_accom_and_hours_props.csv"
MODEL_BASE          = "models/xgb_burglary_model.json"
TRIALS              = 30       # 30 Optuna trials
TIMEOUT             = 960      # 16 minutes = 960 seconds
N_SPLITS            = 3        # 3-fold CV
N_JOBS              = 4        # parallel Optuna trials
RANDOM_STATE        = 42

# === Data loading functions ===

def load_baseline_df(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    # drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()
    if "Crime rank" in df:
        df.drop(columns=["Crime rank"], inplace=True)
    if "Burglary Count" in df:
        df.rename(columns={"Burglary Count": "Count"}, inplace=True)
    else:
        raise KeyError("Expected 'Burglary Count'")
    df["Count"] = df["Count"].astype(float)
    # IMD rank → score
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c != "Count"]
    for c in rank_cols:
        df[f"{c}_score"] = 1.0 / (df[c] + 1.0)
    keep = ["LSOA code", "Date", "Count"] + [c for c in df if c.endswith("_score")]
    return df[keep].copy()

def load_augmented_df(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    # drop pandemic years
    df = df[~df["Date"].dt.year.isin([2020, 2021])].copy()
    if "Crime rank" in df:
        df.drop(columns=["Crime rank"], inplace=True)
    if "Burglary Count" in df:
        df.rename(columns={"Burglary Count": "Count"}, inplace=True)
    else:
        raise KeyError("Expected 'Burglary Count'")
    df["Count"] = df["Count"].astype(float)
    # IMD rank → score
    rank_cols = [c for c in df.columns if "rank" in c.lower() and c != "Count"]
    for c in rank_cols:
        df[f"{c}_score"] = 1.0 / (df[c] + 1.0)
    # accommodation & hours props
    prop_cols = [c for c in df.columns if c.startswith("prop_")]
    if len(prop_cols) < 11:
        raise KeyError(f"Expected ≥11 prop_, got {len(prop_cols)}")
    keep = ["LSOA code", "Date", "Count"] + [c for c in df if c.endswith("_score")] + prop_cols
    return df[keep].copy()

# === Anti-leakage stats & feature engineering ===

def compute_global_historical_stats(df, cutoff_date):
    hist = df[df["Date"] < cutoff_date]
    if hist.empty:
        return pd.DataFrame(
            columns=["LSOA code","lsoa_historical_mean","lsoa_historical_std","lsoa_historical_count"]
        )
    stats = hist.groupby("LSOA code")["Count"].agg(['mean','std','count']).reset_index()
    stats.columns = ["LSOA code","lsoa_historical_mean","lsoa_historical_std","lsoa_historical_count"]
    stats["lsoa_historical_std"].fillna(0, inplace=True)
    return stats

def create_time_aware_features(df, historical_stats=None, validation_mode=False):
    df = df.sort_values(["LSOA code","Date"]).copy()
    df["month"] = df["Date"].dt.month
    df["year"]  = df["Date"].dt.year
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    # seasons
    conds = [
        df["month"].between(3,5),
        df["month"].between(6,8),
        df["month"].between(9,11)
    ]
    choices = ["Spring","Summer","Autumn"]
    df["season"] = np.select(conds, choices, default="Winter")
    df = pd.get_dummies(df, columns=["season"], drop_first=True)
    # holiday flags
    df["is_summer_month"]        = df["month"].between(6,8).astype(int)
    df["is_winter_month"]        = df["month"].isin([12,1,2]).astype(int)
    df["is_school_holiday_month"] = df["month"].isin([7,8,12]).astype(int)
    # lags
    for lag in [1,2,3,6,12,24]:
        df[f"lag_{lag}"] = df.groupby("LSOA code")["Count"].shift(lag)
    # rolling on shifted
    shifted = df.groupby("LSOA code")["Count"].shift(1)
    for w in [3,6,12]:
        df[f"roll{w}_mean"] = shifted.rolling(w, min_periods=1).mean()
        df[f"roll{w}_std"]  = shifted.rolling(w, min_periods=2).std()
    # historical stats
    if historical_stats is not None and validation_mode:
        df = df.merge(historical_stats, on="LSOA code", how="left")
        df["lsoa_historical_mean"].fillna(df["lag_1"], inplace=True)
        df["lsoa_historical_std"].fillna(0, inplace=True)
        df["lsoa_historical_count"].fillna(1, inplace=True)
    else:
        df["lsoa_historical_mean"]  = df.groupby("LSOA code")["Count"].expanding().mean().shift(1).reset_index(level=0, drop=True)
        df["lsoa_historical_std"]   = df.groupby("LSOA code")["Count"].expanding().std().shift(1).reset_index(level=0, drop=True)
        df["lsoa_historical_count"] = df.groupby("LSOA code")["Count"].expanding().count().shift(1).reset_index(level=0, drop=True)
    # diffs & trends
    df["diff_1"]   = df["lag_1"] - df["lag_2"]
    df["yoy_diff"] = df["lag_12"] - df["lag_24"]
    df["trend_3m"] = (df["lag_1"] + df["lag_2"] + df["lag_3"]) / 3 - df["roll6_mean"]
    df["volatility"] = df["roll3_std"] / (df["roll3_mean"] + 1e-8)
    # IMD interactions
    rank_scores = [c for c in df.columns if c.endswith("_score")]
    if rank_scores:
        r = rank_scores[0]
        df["imd_lag1"]       = df[r] * df["lag_1"]
        df["imd_historical"] = df[r] * df["lsoa_historical_mean"]
        df["imd_trend"]      = df[r] * df["diff_1"]
    df["lag_12_seasonal"] = df["lag_12"] / (df["lsoa_historical_mean"] + 1e-8)
    df.fillna(0, inplace=True)
    return df

# === Scaling ===

def time_aware_preprocess(X_train, y_train, X_val, y_val):
    cols = [c for c in X_train.columns if not (c.startswith("season_") or c.startswith("is_") or c=="month")]
    scaler = StandardScaler()
    X_tr, X_va = X_train.copy(), X_val.copy()
    X_tr[cols] = scaler.fit_transform(X_tr[cols])
    X_va[cols] = scaler.transform(X_va[cols])
    return X_tr, X_va, scaler

# === Optuna objective (maximize mean R²) ===

def objective(trial, df_full):
    booster = trial.suggest_categorical("booster", ["gbtree","dart"])
    params = {
        "booster":          booster,
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "eta":              trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "alpha":            trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "lambda":           trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "grow_policy":      trial.suggest_categorical("grow_policy", ["depthwise","lossguide"]),
        "objective":        "reg:squarederror",
        "eval_metric":      "rmse",
        "tree_method":      "hist",
        "n_jobs":           1,
        "seed":             RANDOM_STATE,
    }
    if booster == "dart":
        params["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.3)
        params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.3)
    n_rounds = trial.suggest_int("n_estimators", 50, 500)

    dates = np.sort(df_full["Date"].unique())
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    idx   = np.arange(len(dates))
    r2s   = []

    for tr_idx, vl_idx in tscv.split(idx):
        tr_dates, vl_dates = dates[tr_idx], dates[vl_idx]
        df_tr = df_full[df_full["Date"].isin(tr_dates)].copy()
        df_va = df_full[df_full["Date"].isin(vl_dates)].copy()

        hist_stats = compute_global_historical_stats(df_tr, vl_dates[0])
        df_tr = create_time_aware_features(df_tr, validation_mode=False)
        df_va = create_time_aware_features(df_va, historical_stats=hist_stats, validation_mode=True)

        feats = [c for c in df_tr.columns if c not in ["LSOA code","Date","Count"]]
        X_tr, y_tr = df_tr[feats], df_tr["Count"]
        X_va, y_va = df_va[feats], df_va["Count"]
        common = X_tr.columns.intersection(X_va.columns)
        X_tr, X_va = X_tr[common], X_va[common]

        X_tr_s, X_va_s, _ = time_aware_preprocess(X_tr, y_tr, X_va, y_va)
        dtr = xgb.DMatrix(X_tr_s, label=y_tr)
        dva = xgb.DMatrix(X_va_s, label=y_va)

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            evals=[(dva, "valid")],
            early_stopping_rounds=20,
            verbose_eval=False,
            callbacks=[XGBoostPruningCallback(trial, "valid-rmse")]
        )

        preds = np.maximum(0, bst.predict(dva))
        r2s.append(r2_score(y_va, preds))

    return float(np.mean(r2s))

# === Train & save ===

def train_and_save():
    os.makedirs(os.path.dirname(MODEL_BASE), exist_ok=True)

    # Baseline
    df_b = load_baseline_df(DATA_BASELINE_PATH)
    study_b = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_b.optimize(lambda t: objective(t, df_b), n_trials=TRIALS, timeout=TIMEOUT, n_jobs=N_JOBS)
    best_b = study_b.best_params
    print("Best baseline params:", best_b)

    df_fb   = create_time_aware_features(df_b, validation_mode=False)
    feats_b = [c for c in df_fb.columns if c not in ["LSOA code","Date","Count"]]
    Xb, yb  = df_fb[feats_b], df_fb["Count"]
    sb      = StandardScaler()
    cols_b  = [c for c in feats_b if not (c.startswith("season_") or c.startswith("is_") or c=="month")]
    Xb[cols_b] = sb.fit_transform(Xb[cols_b])

    params_b = {k: best_b[k] for k in ["booster","max_depth","subsample","colsample_bytree","min_child_weight","gamma","grow_policy"]}
    params_b.update({
        "eta": best_b["learning_rate"],
        "alpha": best_b["reg_alpha"],
        "lambda": best_b["reg_lambda"],
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": 1,
        "seed": RANDOM_STATE,
    })
    if best_b["booster"] == "dart":
        params_b["rate_drop"] = best_b["rate_drop"]
        params_b["skip_drop"] = best_b["skip_drop"]

    db    = xgb.DMatrix(Xb, label=yb)
    bst_b = xgb.train(params_b, db, num_boost_round=best_b["n_estimators"], verbose_eval=False)
    bst_b.save_model(MODEL_BASE.replace(".json", "_baseline.json"))
    joblib.dump(sb, "models/scaler_baseline.pkl")
    joblib.dump(feats_b, "models/feature_cols_baseline.pkl")
    hist_stats_b = compute_global_historical_stats(df_fb, df_fb["Date"].max() + pd.Timedelta(days=1))
    joblib.dump(hist_stats_b, "models/historical_stats_baseline.pkl")

    # Augmented
    df_a = load_augmented_df(DATA_AUGMENTED_PATH)
    study_a = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_a.optimize(lambda t: objective(t, df_a), n_trials=TRIALS, timeout=TIMEOUT, n_jobs=N_JOBS)
    best_a = study_a.best_params
    print("Best augmented params:", best_a)

    df_fa   = create_time_aware_features(df_a, validation_mode=False)
    feats_a = [c for c in df_fa.columns if c not in ["LSOA code","Date","Count"]]
    Xa, ya  = df_fa[feats_a], df_fa["Count"]
    sa      = StandardScaler()
    cols_a  = [c for c in feats_a if not (c.startswith("season_") or c.startswith("is_") or c=="month")]
    Xa[cols_a] = sa.fit_transform(Xa[cols_a])

    params_a = {k: best_a[k] for k in ["booster","max_depth","subsample","colsample_bytree","min_child_weight","gamma","grow_policy"]}
    params_a.update({
        "eta": best_a["learning_rate"],
        "alpha": best_a["reg_alpha"],
        "lambda": best_a["reg_lambda"],
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": 1,
        "seed": RANDOM_STATE,
    })
    if best_a["booster"] == "dart":
        params_a["rate_drop"] = best_a["rate_drop"]
        params_a["skip_drop"] = best_a["skip_drop"]

    da    = xgb.DMatrix(Xa, label=ya)
    bst_a = xgb.train(params_a, da, num_boost_round=best_a["n_estimators"], verbose_eval=False)
    bst_a.save_model(MODEL_BASE.replace(".json", "_augmented.json"))
    joblib.dump(sa, "models/scaler_augmented.pkl")
    joblib.dump(feats_a, "models/feature_cols_augmented.pkl")
    hist_stats_a = compute_global_historical_stats(df_fa, df_fa["Date"].max() + pd.Timedelta(days=1))
    joblib.dump(hist_stats_a, "models/historical_stats_augmented.pkl")

# === Evaluation, plotting, feature importance ===

def cross_val_metrics(df_full):
    dates = np.sort(df_full["Date"].unique())
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    idx   = np.arange(len(dates))
    maes, rmses, r2s, results = [], [], [], []

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(idx), 1):
        tr_dates, vl_dates = dates[tr_idx], dates[vl_idx]
        df_tr = df_full[df_full["Date"].isin(tr_dates)].copy()
        df_va = df_full[df_full["Date"].isin(vl_dates)].copy()
        hist_stats = compute_global_historical_stats(df_tr, vl_dates[0])
        df_tr = create_time_aware_features(df_tr, validation_mode=False)
        df_va = create_time_aware_features(df_va, historical_stats=hist_stats, validation_mode=True)

        feats = [c for c in df_tr.columns if c not in ["LSOA code","Date","Count"]]
        X_tr, y_tr = df_tr[feats], df_tr["Count"]
        X_va, y_va = df_va[feats], df_va["Count"]
        common = X_tr.columns.intersection(X_va.columns)
        X_tr, X_va = X_tr[common], X_va[common]

        X_tr_s, X_va_s, _ = time_aware_preprocess(X_tr, y_tr, X_va, y_va)
        dtr = xgb.DMatrix(X_tr_s, label=y_tr)
        dva = xgb.DMatrix(X_va_s, label=y_va)

        params = {
            "max_depth":        6,
            "eta":              0.1,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "alpha":            0.1,
            "lambda":           1.0,
            "gamma":            0.5,
            "objective":        "reg:squarederror",
            "eval_metric":      "mae",
            "seed":             RANDOM_STATE,
        }

        bst = xgb.train(
            params,
            dtr,
            num_boost_round=300,
            evals=[(dva, "valid")],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        preds = np.maximum(0, bst.predict(dva))
        maes.append(mean_absolute_error(y_va, preds))
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        r2s.append(r2_score(y_va, preds))
        results.append((fold, maes[-1], rmses[-1], r2s[-1]))

    return maes, rmses, r2s, results

def evaluate():
    df_b = load_baseline_df(DATA_BASELINE_PATH)
    df_a = load_augmented_df(DATA_AUGMENTED_PATH)

    print("\n=== BASELINE CV ===")
    _, _, r2b, _ = cross_val_metrics(df_b)
    print(f"Baseline R²: {np.mean(r2b):.4f} ± {np.std(r2b):.4f}")

    print("\n=== AUGMENTED CV ===")
    _, _, r2a, _ = cross_val_metrics(df_a)
    print(f"Augmented R²: {np.mean(r2a):.4f} ± {np.std(r2a):.4f}")

def plot_cv_performance():
    df = load_augmented_df(DATA_AUGMENTED_PATH)
    dates = np.sort(df["Date"].unique())
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    tr_idx, vl_idx = list(tscv.split(np.arange(len(dates))))[-1]
    tr_dates, vl_dates = dates[tr_idx], dates[vl_idx]

    df_tr = df[df["Date"].isin(tr_dates)].copy()
    df_va = df[df["Date"].isin(vl_dates)].copy()
    hist_stats = compute_global_historical_stats(df_tr, vl_dates[0])
    df_tr = create_time_aware_features(df_tr, validation_mode=False)
    df_va = create_time_aware_features(df_va, historical_stats=hist_stats, validation_mode=True)

    feats = [c for c in df_tr.columns if c not in ["LSOA code","Date","Count"]]
    X_tr, y_tr = df_tr[feats], df_tr["Count"]
    X_va, y_va = df_va[feats], df_va["Count"]
    common = X_tr.columns.intersection(X_va.columns)
    X_tr, X_va = X_tr[common], X_va[common]

    X_tr_s, X_va_s, _ = time_aware_preprocess(X_tr, y_tr, X_va, y_va)
    dtr = xgb.DMatrix(X_tr_s, label=y_tr)
    dva = xgb.DMatrix(X_va_s, label=y_va)

    params = {
        "max_depth":6, "eta":0.1, "subsample":0.8,
        "colsample_bytree":0.8, "min_child_weight":3,
        "alpha":0.1, "lambda":1.0, "gamma":0.5,
        "objective":"reg:squarederror","eval_metric":"mae",
        "seed":RANDOM_STATE
    }

    bst = xgb.train(
        params,
        dtr,
        num_boost_round=300,
        evals=[(dva,"valid")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    df_tr["Pred"] = np.maximum(0, bst.predict(dtr))
    df_va["Pred"] = np.maximum(0, bst.predict(dva))

    agg_tr = df_tr.groupby("Date").agg(Actual=("Count","sum"), Pred=("Pred","sum"))
    agg_va = df_va.groupby("Date").agg(Actual=("Count","sum"), Pred=("Pred","sum"))

    plt.figure(figsize=(14,6))
    plt.plot(agg_tr.index, agg_tr["Actual"], label="Train Actual")
    plt.plot(agg_tr.index, agg_tr["Pred"], "--", label="Train Pred")
    plt.plot(agg_va.index, agg_va["Actual"], label="Val Actual")
    plt.plot(agg_va.index, agg_va["Pred"], ":", label="Val Pred")
    plt.axvline(x=vl_dates[0], color='k', linestyle='--')
    plt.legend()
    plt.title("Augmented Model Last-Fold")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/cv_performance_augmented_fixed.png", dpi=300)
    print("Plot saved to models/cv_performance_augmented_fixed.png")

def feature_importance_analysis():
    print("\n=== FEATURE IMPORTANCE ===")
    df_b = load_baseline_df(DATA_BASELINE_PATH)
    df_a = load_augmented_df(DATA_AUGMENTED_PATH)

    df_fb = create_time_aware_features(df_b, validation_mode=False)
    df_fa = create_time_aware_features(df_a, validation_mode=False)

    feats_b = [c for c in df_fb.columns if c not in ["LSOA code","Date","Count"]]
    feats_a = [c for c in df_fa.columns if c not in ["LSOA code","Date","Count"]]

    Xb, yb = df_fb[feats_b].fillna(0), df_fb["Count"]
    Xa, ya = df_fa[feats_a].fillna(0), df_fa["Count"]

    sb, sa = StandardScaler(), StandardScaler()
    to_b = [c for c in feats_b if not (c.startswith("season_") or c.startswith("is_") or c=="month")]
    to_a = [c for c in feats_a if not (c.startswith("season_") or c.startswith("is_") or c=="month")]
    Xb[to_b] = sb.fit_transform(Xb[to_b])
    Xa[to_a] = sa.fit_transform(Xa[to_a])

    params = {
        "max_depth":6,"eta":0.1,"subsample":0.8,"colsample_bytree":0.8,
        "min_child_weight":3,"alpha":0.1,"lambda":1.0,
        "objective":"reg:squarederror","seed":RANDOM_STATE
    }
    mb = xgb.train(params, xgb.DMatrix(Xb, label=yb), num_boost_round=200, verbose_eval=False)
    ma = xgb.train(params, xgb.DMatrix(Xa, label=ya), num_boost_round=200, verbose_eval=False)

    imp_b = mb.get_score(importance_type='gain')
    imp_a = ma.get_score(importance_type='gain')

    print("\nTop 10 Baseline Features:")
    for f, i in sorted(imp_b.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {f}: {i:.4f}")

    print("\nTop 10 Augmented Features:")
    for f, i in sorted(imp_a.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {f}: {i:.4f}")

if __name__ == "__main__":
    print("▶ Training burglary models…")
    train_and_save()
    print("▶ Evaluating…")
    evaluate()
    print("▶ Plotting…")
    plot_cv_performance()
    print("▶ Feature importance…")
    feature_importance_analysis()
    print("✅ All done!")


