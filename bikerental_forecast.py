#!/usr/bin/env python3
"""
Bike Sharing Forecasting - production-oriented script (25-12-2025)

What this script does
- Loads the hourly Bike Sharing dataset (hour.csv-like schema).
- Runs consistency checks and repairs missing hourly timestamps.
- Builds an interpretable feature set (lags, season/hour buckets, workingday interactions, weather buckets).
- Trains and evaluates 3 models (optional analysis mode):
    1) Seasonal naive baseline (lag_24)
    2) Ridge regression
    3) HistGradientBoostingRegressor (selected business model)
- Persists the selected model to disk (joblib).
- Produces a forward forecast for hours not yet present in the input data.

Usage examples
- Analysis + training + forecast:
    python bikerental_forecast.py --hourly_csv /path/to/hour.csv --analyse true --output_dir outputs

- Production run (load model if exists, else train):
    python bikerental_forecast.py --hourly_csv /path/to/hour.csv --analyse false --model_path outputs/model.joblib --output_dir outputs

Notes / assumptions for a "daily prediction service"
- This script forecasts the next H hours after the last observed timestamp.
- Future weather and holiday flags are usually not known from the historical log.
  Here we use a simple, explicit proxy:
    * weather for the forecast horizon is copied from the most recent available day by hour-of-day
      (fallback: last observed values).
    * holiday defaults to 0 unless you provide a holiday calendar hook.
  In a real service, you would replace these proxies with weather forecasts + a holiday calendar.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

try:
    import joblib
except ImportError as e:
    raise SystemExit("Missing dependency: joblib. Install via `pip install joblib`.") from e


# -----------------------------
# Config
# -----------------------------

TARGET_COLS = ["cnt", "casual", "registered"]
CAL_COLS = ["season", "yr", "mnth"]
DAY_FLAG_COLS = ["holiday", "weekday", "workingday"]
WEATHER_COLS = ["weathersit", "temp", "atemp", "hum", "windspeed"]

SEASON_MAP = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}


@dataclass(frozen=True)
class FillConfig:
    k_threshold: int = 2  # <=k missing hours => interpolate, >k => extraordinary event
    events: Optional[Dict[str, str]] = None
    human_override_days: Optional[List[str]] = None


DEFAULT_FILL_CONFIG = FillConfig(
    k_threshold=2,
    events={
        "2011-08-27": "Hurricane Irene",
        "2011-01-26": "Heavy snowfall",
        "2011-01-27": "Heavy snowfall",
        "2011-01-18": "Protests (Hu Jintao visit)",
        "2012-10-29": "Hurricane Sandy",
        "2012-10-30": "Hurricane Sandy",
    },
    human_override_days=["2011-02-22"],
)


# -----------------------------
# Logging
# -----------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# IO
# -----------------------------

def load_hourly(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "dteday" not in df.columns or "hr" not in df.columns:
        raise ValueError("hourly_csv must contain at least columns: dteday, hr")

    df["dteday"] = pd.to_datetime(df["dteday"])
    # ensure numeric where expected
    for c in TARGET_COLS + CAL_COLS + DAY_FLAG_COLS + WEATHER_COLS + ["hr", "instant"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # chronological sort
    df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return df


# -----------------------------
# Consistency checks
# -----------------------------

def assert_cnt_identity(df: pd.DataFrame) -> None:
    if all(c in df.columns for c in ["cnt", "casual", "registered"]):
        ok = (df["casual"] + df["registered"] == df["cnt"]).all()
        if not ok:
            bad = df.loc[(df["casual"] + df["registered"] != df["cnt"]), ["dteday", "hr", "casual", "registered", "cnt"]].head()
            raise ValueError(f"Identity violation: cnt != casual + registered. Example:\n{bad}")


def find_missing_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-day list of missing hours and counts."""
    dup = df.duplicated(subset=["dteday", "hr"]).sum()
    logging.info("Duplicate (dteday, hr) rows: %s", dup)

    missing_by_day = (
        df.groupby("dteday")["hr"]
        .apply(lambda x: sorted(set(range(24)) - set(x.dropna().astype(int))))
    )
    missing_by_day = missing_by_day[missing_by_day.apply(len) > 0]

    summary = pd.DataFrame(
        {"missing_hours": missing_by_day, "n_missing_hours": missing_by_day.apply(len)}
    ).sort_values("n_missing_hours", ascending=False)

    logging.info("Days with missing hours: %s", len(summary))
    if len(summary) > 0:
        logging.info("Missing-hour count distribution:\n%s", summary["n_missing_hours"].value_counts().sort_index().to_string())
    return summary


# -----------------------------
# Repair / filling
# -----------------------------

def hr_to_period(hr: int) -> str:
    if 0 <= hr <= 5:
        return "night"
    if 6 <= hr <= 11:
        return "morning"
    if 12 <= hr <= 17:
        return "afternoon"
    return "evening"


def _fill_day_flags(group: pd.DataFrame) -> pd.DataFrame:
    """Recompute weekday/workingday and set holiday by mode (or 0)."""
    d = group.name
    weekday = d.weekday()  # 0=Mon ... 6=Sun
    group["weekday"] = weekday

    # holiday: keep day-level mode if any exists else default 0
    if "holiday" in group.columns and group["holiday"].notna().any():
        holiday_val = int(group["holiday"].mode().iloc[0])
    else:
        holiday_val = 0
    group["holiday"] = holiday_val

    group["workingday"] = int((weekday < 5) and (holiday_val == 0))
    return group


def repair_missing_hours(
    df_hourly: pd.DataFrame,
    cfg: FillConfig = DEFAULT_FILL_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a full day x hour grid and fills missing rows.

    Rules
    - human_error days (<=k missing hours OR explicitly overridden):
        interpolate within-day for target + weather columns.
    - extraordinary_event days (>k missing hours OR in EVENTS):
        set missing target rows to 0
        fill calendar columns by within-day forward fill (constant per day)
        recompute weekday/holiday/workingday from date (+ mode of holiday if present)
        fill weather on missing rows by mean of nearest valid before/after in time
        then snap weathersit to {1,2,3,4}
    """
    df = df_hourly.copy()
    df["dteday"] = pd.to_datetime(df["dteday"])

    missing_summary = find_missing_hours(df)
    k = cfg.k_threshold

    small_gap_days = missing_summary[missing_summary["n_missing_hours"] <= k].index
    large_gap_days = missing_summary[missing_summary["n_missing_hours"] > k].index

    events = cfg.events or {}
    event_days = pd.to_datetime(list(events.keys())) if events else pd.to_datetime([])
    human_override_days = pd.to_datetime(cfg.human_override_days or [])

    # Apply overrides
    large_gap_days = large_gap_days.union(event_days).difference(human_override_days)
    small_gap_days = small_gap_days.union(human_override_days).difference(event_days)

    # Build full index
    full_days = pd.Index(df["dteday"].sort_values().unique())
    full_index = pd.MultiIndex.from_product([full_days, range(24)], names=["dteday", "hr"])

    df_full = (
        df.set_index(["dteday", "hr"])
        .reindex(full_index)
        .sort_index()
        .reset_index()
    )

    # Timestamp for convenience
    if "timestamp" not in df_full.columns:
        df_full["timestamp"] = df_full["dteday"] + pd.to_timedelta(df_full["hr"], unit="h")

    # Gap type labeling
    df_full["gap_type"] = "none"
    df_full.loc[df_full["dteday"].isin(small_gap_days), "gap_type"] = "human_error"
    df_full.loc[df_full["dteday"].isin(large_gap_days), "gap_type"] = "extraordinary_event"

    row_missing = df_full["cnt"].isna()
    mask_event_missing = (df_full["gap_type"] == "extraordinary_event") & row_missing

    # 1) HUMAN ERROR: interpolate within-day for targets + weather
    for col in TARGET_COLS + WEATHER_COLS:
        if col not in df_full.columns:
            continue
        df_full.loc[df_full["gap_type"] == "human_error", col] = (
            df_full.loc[df_full["gap_type"] == "human_error"]
            .groupby("dteday")[col]
            .transform(lambda s: s.interpolate(limit_direction="both"))
        )

    # 2) EXTRAORDINARY EVENTS
    # 2a) targets = 0 for missing rows only
    df_full.loc[mask_event_missing, TARGET_COLS] = 0.0

    # 2b) fill calendar columns by within-day ffill (constant per day)
    for col in CAL_COLS:
        if col in df_full.columns:
            df_full[col] = df_full.groupby("dteday")[col].ffill()

    # 2c) recompute day flags from date (+ holiday mode)
    df_full = df_full.groupby("dteday", group_keys=False).apply(_fill_day_flags)

    # 2d) weather for extraordinary-event missing rows: mean of nearest valid before/after in time
    df_full = df_full.sort_values(["dteday", "hr"]).reset_index(drop=True)
    for col in WEATHER_COLS:
        if col not in df_full.columns:
            continue
        prev_val = df_full[col].ffill()
        next_val = df_full[col].bfill()
        df_full.loc[mask_event_missing, col] = (prev_val[mask_event_missing] + next_val[mask_event_missing]) / 2

    # 2e) snap weathersit back to categorical {1,2,3,4}
    if "weathersit" in df_full.columns:
        df_full["weathersit"] = (
            df_full["weathersit"]
            .round()
            .clip(1, 4)
            .astype("Int64")
        )

    # Re-check identity when defined
    assert_cnt_identity(df_full.dropna(subset=["cnt", "casual", "registered"]))

    # Event mapping table for reporting
    event_table = pd.DataFrame(
        {"dteday": pd.to_datetime(list(events.keys())),
         "event": list(events.values())}
    ).sort_values("dteday") if events else pd.DataFrame(columns=["dteday", "event"])

    return df_full, event_table


# -----------------------------
# Feature engineering
# -----------------------------

def build_features(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Produces X, y and feature_names with:
    - lags: lag_24, lag_168
    - season one-hot (4) with reference dropped later
    - hour bucket one-hot (4) with reference dropped later
    - regime/interaction: workingday, holiday, wd_morning, wd_evening
    - weather buckets (atemp/hum/windspeed) + weathersit one-hot
    """
    df_feat = df_full.copy()
    df_feat["dteday"] = pd.to_datetime(df_feat["dteday"])

    # Ensure chronological sort before lagging
    df_feat = df_feat.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # Lags
    df_feat["lag_24"] = df_feat["cnt"].shift(24)
    df_feat["lag_168"] = df_feat["cnt"].shift(168)

    # Season dummies
    df_feat["season_name"] = df_feat["season"].map(SEASON_MAP)
    season_dummies = pd.get_dummies(df_feat["season_name"], prefix="season")

    # Hour period dummies
    df_feat["hr_period"] = df_feat["hr"].apply(lambda h: hr_to_period(int(h)))
    hr_dummies = pd.get_dummies(df_feat["hr_period"], prefix="hr")

    # Regime / interactions (IMPORTANT: leisure_day intentionally not used; redundant with workingday)
    df_feat["wd_morning"] = ((df_feat["workingday"] == 1) & (df_feat["hr_period"] == "morning")).astype(int)
    df_feat["wd_evening"] = ((df_feat["workingday"] == 1) & (df_feat["hr_period"] == "evening")).astype(int)

    behavior_features = ["wd_morning", "wd_evening", "holiday", "workingday"]

    # Interpretable weather buckets (tertiles)
    # Note: qcut can produce NaNs if too many ties; this dataset is fine. We later drop rows with NaNs.
    df_feat["atemp_bucket"] = pd.qcut(df_feat["atemp"], q=3, labels=["cold", "medium", "hot"])
    df_feat["hum_bucket"] = pd.qcut(df_feat["hum"], q=3, labels=["not_humid", "medium", "humid"])
    df_feat["windspeed_bucket"] = pd.qcut(df_feat["windspeed"], q=3, labels=["not_windy", "medium", "windy"])

    atemp_dummies = pd.get_dummies(df_feat["atemp_bucket"], prefix="atemp")
    hum_dummies = pd.get_dummies(df_feat["hum_bucket"], prefix="hum")
    wind_dummies = pd.get_dummies(df_feat["windspeed_bucket"], prefix="wind")

    # weathersit dummies
    weathersit_dummies = pd.get_dummies(df_feat["weathersit"].astype("Int64"), prefix="weathersit")

    # Assemble
    X = pd.concat(
        [
            df_feat[["lag_24", "lag_168"] + behavior_features],
            season_dummies,
            hr_dummies,
            atemp_dummies,
            hum_dummies,
            wind_dummies,
            weathersit_dummies,
        ],
        axis=1,
    )

    y = df_feat["cnt"]

    # Drop rows with missing due to lags/bucketing
    data = pd.concat([X, y], axis=1).dropna()
    X = data.drop(columns=["cnt"])
    y = data["cnt"]

    # Convert booleans to int
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Drop reference dummies to avoid dummy trap (for linear models)
    ref_cols = [
        "hr_night",
        "season_winter",
        "atemp_medium",
        "hum_medium",
        "wind_medium",
        "weathersit_1",  # clear baseline
    ]
    X = X.drop(columns=[c for c in ref_cols if c in X.columns], errors="ignore")

    feature_cols = list(X.columns)
    return X, y, feature_cols


# -----------------------------
# Modeling
# -----------------------------

def time_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    return X_train, X_test, y_train, y_test


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    ridge = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))]
    )
    ridge.fit(X_train, y_train)

    gbr = HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.05, max_iter=300, random_state=42
    )
    gbr.fit(X_train, y_train)

    return {"ridge": ridge, "gbr": gbr}


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    # Naive baseline uses lag_24
    if "lag_24" not in X_test.columns:
        raise ValueError("lag_24 must be present for the naive baseline.")
    y_pred_naive = X_test["lag_24"].to_numpy()
    mae_naive = mean_absolute_error(y_test, y_pred_naive)

    y_pred_ridge = models["ridge"].predict(X_test)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

    y_pred_gbr = models["gbr"].predict(X_test)
    mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

    res = pd.DataFrame(
        {
            "Model": ["Seasonal Naive (lag_24)", "Ridge Regression", "HistGradientBoostingRegressor"],
            "MAE": [mae_naive, mae_ridge, mae_gbr],
        }
    ).sort_values("MAE")

    return res


def save_model(model: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logging.info("Saved model to %s", path)


def load_model(path: str) -> object:
    model = joblib.load(path)
    logging.info("Loaded model from %s", path)
    return model


# -----------------------------
# Forecasting
# -----------------------------

def build_future_frame(
    df_full: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Create rows for future hours not yet present in df_full, with exogenous features
    approximated from recent patterns (explicit placeholder logic).
    """
    df = df_full.copy()
    df["timestamp"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp").reset_index(drop=True)

    last_ts = df["timestamp"].max()
    future_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon_hours, freq="H")

    future = pd.DataFrame({"timestamp": future_ts})
    future["dteday"] = future["timestamp"].dt.floor("D")
    future["hr"] = future["timestamp"].dt.hour

    # Fill calendar fields from date
    future["weekday"] = future["dteday"].dt.weekday
    # holiday unknown for future -> default 0 (hook for external calendar)
    future["holiday"] = 0
    future["workingday"] = ((future["weekday"] < 5) & (future["holiday"] == 0)).astype(int)

    # season/yr/mnth derived from date (year is encoded 0/1 relative to 2011)
    future["mnth"] = future["dteday"].dt.month
    future["yr"] = (future["dteday"].dt.year - df["dteday"].dt.year.min()).clip(lower=0, upper=1).astype(int)

    # crude season mapping by month (align with dataset semantics broadly)
    # 1:spring,2:summer,3:fall,4:winter
    m = future["mnth"]
    future["season"] = np.select(
        [m.isin([3,4,5]), m.isin([6,7,8]), m.isin([9,10,11]), m.isin([12,1,2])],
        [1,2,3,4],
        default=4
    ).astype(int)

    # Weather proxy: copy most recent day's weather by hour-of-day
    weather_cols = [c for c in WEATHER_COLS if c in df.columns]
    if weather_cols:
        last_day = df["dteday"].max()
        last_day_weather = df.loc[df["dteday"] == last_day, ["hr"] + weather_cols].dropna()
        if len(last_day_weather) == 0:
            # fallback: last observed values
            for c in weather_cols:
                future[c] = df[c].ffill().iloc[-1]
        else:
            future = future.merge(last_day_weather, on="hr", how="left", suffixes=("", "_lastday"))
            # any missing -> fallback to last observed
            for c in weather_cols:
                if c not in future.columns:
                    continue
                if future[c].isna().any():
                    future[c] = future[c].fillna(df[c].ffill().iloc[-1])
    # targets are unknown; keep NaN
    for c in TARGET_COLS:
        future[c] = np.nan

    return future


def make_recursive_forecast(
    model: object,
    df_full: pd.DataFrame,
    horizon_hours: int,
) -> pd.DataFrame:
    """
    Forecast next horizon_hours.
    Because lags depend on previous cnt values, we forecast recursively:
      - start from last observed history
      - for each future hour, compute features (including lag_24/lag_168 from history+preds)
      - predict cnt
    """
    history = df_full.copy()
    history["timestamp"] = history["dteday"] + pd.to_timedelta(history["hr"], unit="h")
    history = history.sort_values("timestamp").reset_index(drop=True)

    future = build_future_frame(history, horizon_hours)
    future = future.sort_values("timestamp").reset_index(drop=True)

    combined = pd.concat([history, future], ignore_index=True, sort=False)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # We'll fill predictions into combined['cnt'] for future timestamps
    # For safety, ensure required columns exist
    for col in ["cnt", "dteday", "hr", "season", "holiday", "workingday", "atemp", "hum", "windspeed", "weathersit"]:
        if col not in combined.columns:
            # some columns might be missing in non-UCI adaptations
            pass

    # Walk forward over future rows
    future_mask = combined["timestamp"].isin(future["timestamp"])
    future_idx = combined.index[future_mask].tolist()

    for i in future_idx:
        # Build features for the single row using the SAME feature builder logic,
        # but only for this row, relying on existing combined['cnt'] values for lags.
        # To reuse build_features without recomputing qcut on a single row (unstable),
        # we compute features for a rolling window containing enough history and take the last row.

        # Minimal window: need at least 168 hours of context
        start = max(0, i - 24 * 10)  # 10 days window is enough for qcut stability in practice
        window = combined.iloc[start:i+1].copy()

        # Important: weather buckets use qcut; on tiny samples qcut can fail.
        # To avoid that in production, we pre-fit bin edges from full history and apply fixed cuts.
        # Here we implement fixed cuts using quantiles from the full observed history (history only).
        # Compute once on first iteration.
        if i == future_idx[0]:
            # compute quantile edges from observed history (not including future)
            obs = history.dropna(subset=["atemp", "hum", "windspeed"])
            at_q = obs["atemp"].quantile([0.33, 0.66]).to_numpy()
            hu_q = obs["hum"].quantile([0.33, 0.66]).to_numpy()
            wi_q = obs["windspeed"].quantile([0.33, 0.66]).to_numpy()

            make_recursive_forecast._edges = (at_q, hu_q, wi_q)  # type: ignore[attr-defined]

        at_q, hu_q, wi_q = make_recursive_forecast._edges  # type: ignore[attr-defined]

        # Apply fixed buckets to window
        window["atemp_bucket"] = pd.cut(
            window["atemp"],
            bins=[-np.inf, at_q[0], at_q[1], np.inf],
            labels=["cold", "medium", "hot"],
            include_lowest=True,
        )
        window["hum_bucket"] = pd.cut(
            window["hum"],
            bins=[-np.inf, hu_q[0], hu_q[1], np.inf],
            labels=["not_humid", "medium", "humid"],
            include_lowest=True,
        )
        window["windspeed_bucket"] = pd.cut(
            window["windspeed"],
            bins=[-np.inf, wi_q[0], wi_q[1], np.inf],
            labels=["not_windy", "medium", "windy"],
            include_lowest=True,
        )

        # Build features for window (replicates build_features but avoids qcut)
        window = window.sort_values("timestamp").reset_index(drop=True)
        window["lag_24"] = window["cnt"].shift(24)
        window["lag_168"] = window["cnt"].shift(168)

        window["season_name"] = window["season"].map(SEASON_MAP)
        season_d = pd.get_dummies(window["season_name"], prefix="season")

        window["hr_period"] = window["hr"].apply(lambda h: hr_to_period(int(h)))
        hr_d = pd.get_dummies(window["hr_period"], prefix="hr")

        window["wd_morning"] = ((window["workingday"] == 1) & (window["hr_period"] == "morning")).astype(int)
        window["wd_evening"] = ((window["workingday"] == 1) & (window["hr_period"] == "evening")).astype(int)

        at_d = pd.get_dummies(window["atemp_bucket"], prefix="atemp")
        hu_d = pd.get_dummies(window["hum_bucket"], prefix="hum")
        wi_d = pd.get_dummies(window["windspeed_bucket"], prefix="wind")
        ws_d = pd.get_dummies(window["weathersit"].astype("Int64"), prefix="weathersit")

        Xw = pd.concat(
            [
                window[["lag_24", "lag_168", "wd_morning", "wd_evening", "holiday", "workingday"]],
                season_d,
                hr_d,
                at_d,
                hu_d,
                wi_d,
                ws_d,
            ],
            axis=1,
        )

        # align columns to model training columns (stored as attribute if available)
        model_cols = getattr(model, "feature_names_in_", None)
        if model_cols is None:
            raise ValueError("Model does not expose feature_names_in_. Use a sklearn model fitted on a DataFrame.")
        Xw = Xw.reindex(columns=list(model_cols), fill_value=0)

        # Drop reference dummy columns were already absent in model_cols, so reindex handles it.

        # Take last row feature vector
        x_last = Xw.iloc[[-1]]

        # If lags are missing early in horizon (shouldn't be), fallback to 0
        x_last = x_last.fillna(0)

        pred = float(model.predict(x_last)[0])

        # production guards
        pred = max(0.0, pred)          # no negative demand
        pred = round(pred)             # hourly count as integer

        combined.loc[i, "cnt"] = pred


        # Enforce identity for future (optional): split not predicted
        combined.loc[i, "casual"] = np.nan
        combined.loc[i, "registered"] = np.nan

    forecast = combined.loc[future_mask, ["timestamp", "dteday", "hr", "cnt"]].copy()
    return forecast


# -----------------------------
# Analysis outputs (plots, feature importance)
# -----------------------------

def save_analysis_artifacts(
    output_dir: str,
    df_full: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    # Prediction vs actual plot (first ~week of test)
    y_pred_naive = X_test["lag_24"].to_numpy()
    y_pred_ridge = models["ridge"].predict(X_test)
    y_pred_gbr = models["gbr"].predict(X_test)

    plot_n = min(7 * 24, len(y_test))
    x_axis = np.arange(plot_n)

    plt.figure()
    plt.plot(x_axis, y_test.iloc[:plot_n].to_numpy(), label="Actual")
    plt.plot(x_axis, y_pred_naive[:plot_n], label="Naive")
    plt.plot(x_axis, y_pred_ridge[:plot_n], label="Ridge")
    plt.plot(x_axis, y_pred_gbr[:plot_n], label="GBR")
    plt.xlabel("Test time index (hours)")
    plt.ylabel("cnt")
    plt.title("Predictions vs Actual (first ~week of test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()

    # Permutation feature importance for GBR
    perm = permutation_importance(
        models["gbr"], X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    feat_imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    feat_imp.to_csv(os.path.join(output_dir, "feature_importance_gbr.csv"), index=False)

    # Save top-20 importance bar plot
    top = feat_imp.head(20).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance (mean)")
    plt.title("Top-20 Feature Importance (GBR)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_gbr.png"), dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hourly_csv", required=True, help="Path to hour.csv (or equivalent hourly dataset)")
    p.add_argument("--output_dir", default="outputs", help="Output directory for artifacts and forecasts")
    p.add_argument("--model_path", default=None, help="Path to persist/load the selected model (joblib). Defaults to <output_dir>/model.joblib")
    p.add_argument("--analyse", default="true", help="Enable analysis outputs and model comparison (true/false)")
    p.add_argument("--horizon_hours", type=int, default=24, help="Forecast horizon in hours beyond last observed timestamp")
    p.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return p.parse_args()


def str2bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "t"}


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    analyse = str2bool(args.analyse)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = args.model_path or os.path.join(output_dir, "model.joblib")

    # 1) Load
    df_hourly = load_hourly(args.hourly_csv)
    logging.info("Loaded hourly rows: %s", len(df_hourly))

    # 2) High-signal consistency checks
    assert_cnt_identity(df_hourly)

    # 3) Repair missing hours
    df_full, event_table = repair_missing_hours(df_hourly, DEFAULT_FILL_CONFIG)
    df_full.to_csv(os.path.join(output_dir, "hourly_repaired.csv"), index=False)
    if len(event_table) > 0:
        event_table.to_csv(os.path.join(output_dir, "extraordinary_events.csv"), index=False)

    # 4) Feature engineering
    X, y, feature_cols = build_features(df_full)
    logging.info("Features: %s | Rows: %s", len(feature_cols), len(X))

    # 5) Train/test split for evaluation (analysis mode)
    X_train, X_test, y_train, y_test = time_split(X, y, train_frac=0.8)

    # 6) Train or load model
    model = None
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        models = train_models(X_train, y_train)
        model = models["gbr"]  # selected business model
        save_model(model, model_path)

    # If analysis is enabled, train all models (fresh) and create artifacts/metrics
    if analyse:
        models = train_models(X_train, y_train)
        results = evaluate_models(models, X_test, y_test)
        results.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        logging.info("Model comparison:\n%s", results.to_string(index=False))

        save_analysis_artifacts(output_dir, df_full, X, y, models, X_test, y_test)

        # Persist selected model from this run (GBR)
        save_model(models["gbr"], model_path)
        model = models["gbr"]

    # 7) Forecast next horizon_hours using recursive strategy
    forecast = make_recursive_forecast(model, df_full, horizon_hours=args.horizon_hours)
    forecast_path = os.path.join(output_dir, "hourly_forecast.csv")
    forecast["cnt"] = forecast["cnt"].round().clip(lower=0).astype("Int64")
    forecast.to_csv(forecast_path, index=False)

    logging.info("Saved forecast to %s", forecast_path)

    # 8) Also provide daily totals (common business view)
    daily_forecast = (
        forecast.groupby("dteday")["cnt"].sum().reset_index().rename(columns={"cnt": "cnt_daily_forecast"})
    )
    daily_path = os.path.join(output_dir, "daily_forecast.csv")
    daily_forecast.to_csv(daily_path, index=False)
    logging.info("Saved daily forecast to %s", daily_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
