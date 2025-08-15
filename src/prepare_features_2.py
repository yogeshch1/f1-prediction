# src/prepare_features.py
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("data/f1_results.csv")          # combined raw results (one row per driver/race)
OUTPUT_CSV = Path("data/f1_features_clean.csv")  # cleaned features (no post-race leakage)

def time_to_seconds(t):
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h*3600 + m*60 + s
        if len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m*60 + s
        return float(t)
    except:
        return np.nan

def prepare_features(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values(["date", "season", "round"]).reset_index(drop=True)

    # Convert common numeric fields (coerce errors -> NaN)
    for col in ["position", "grid", "laps", "points", "finish_time_millis", "fastest_lap_rank", "fastest_lap_number"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert fastest lap time string to seconds (if present)
    if "fastest_lap_time" in df.columns:
        df["fastest_lap_time_sec"] = df["fastest_lap_time"].apply(time_to_seconds)

    # Keep finishing position as numeric for label creation (but we'll not include it as a feature)
    df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce")

    # Compute rolling features (only use *past* races via shift())
    def add_driver_rolling_features(g):
        g = g.sort_values("date").copy()
        g["driver_avg_finish_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).mean()
        g["driver_wins_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).apply(lambda x: (x==1).sum(), raw=True)
        g["driver_podiums_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).apply(lambda x: (x<=3).sum(), raw=True)
        g["driver_avg_grid_last5"] = g["grid"].shift().rolling(5, min_periods=1).mean()
        g["driver_points_last5"] = g["points"].shift().rolling(5, min_periods=1).sum()
        return g

    df = df.groupby("driver_id", group_keys=False, sort=False).apply(add_driver_rolling_features)

    def add_constructor_rolling_features(g):
        g = g.sort_values("date").copy()
        g["constructor_avg_finish_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).mean()
        g["constructor_points_last5"] = g["points"].shift().rolling(5, min_periods=1).sum()
        return g

    df = df.groupby("constructor", group_keys=False, sort=False).apply(add_constructor_rolling_features)

    # Driver-track historical avg finish (use transform so index aligns)
    df["driver_track_avg_finish"] = df.groupby(["driver_id", "circuit"])["position_numeric"] \
        .transform(lambda x: x.shift().expanding().mean())

    # Label (we keep this so model has a target)
    df["is_winner"] = (df["position_numeric"] == 1).astype(int)

    # ------------------------
    # Remove leakage: do not include post-race info in features
    # ------------------------
    leakage_cols = {
        "position", "position_text", "points", "finish_time_millis",
        "fastest_lap_time", "fastest_lap_time_sec",
        "fastest_lap_rank", "fastest_lap_number", "driver_number",
        "position_numeric", "laps"
    }

    # Start with numeric columns as candidate features
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if "is_winner" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["is_winner"])

    # Remove meta columns if accidentally included in numeric_df
    meta_cols = ["season", "round", "date", "race_name", "driver_id", "driver_name", "constructor", "circuit"]
    for mc in meta_cols:
        if mc in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[mc])

    # Drop leakage numeric columns
    for c in list(numeric_df.columns):
        if c in leakage_cols:
            numeric_df = numeric_df.drop(columns=[c])

    # Impute medians for numeric features (baseline)
    for col in numeric_df.columns:
        numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

    # Reattach meta columns and label
    clean_features = pd.concat([df[meta_cols].reset_index(drop=True), numeric_df.reset_index(drop=True)], axis=1)
    clean_features["is_winner"] = df["is_winner"].values

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_features.to_csv(output_csv, index=False)
    print(f"Wrote cleaned features to {output_csv} (rows: {len(clean_features)}, cols: {len(clean_features.columns)})")

if __name__ == "__main__":
    prepare_features(INPUT_CSV, OUTPUT_CSV)
