import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = Path("data/f1_results.csv")
OUTPUT_CSV = Path("data/f1_features.csv")

def time_to_seconds(t):
    """Convert 'M:SS.mmm' or 'H:MM:SS.mmm' to seconds."""
    if pd.isna(t):
        return np.nan
    try:
        parts = str(t).split(":")
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h*3600 + m*60 + s
        elif len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m*60 + s
        return float(t)
    except:
        return np.nan

def prepare_features(input_csv, output_csv):
    # 1️⃣ Load CSV
    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values(["date", "season", "round"]).reset_index(drop=True)

    # 2️⃣ Numeric conversion
    numeric_cols = [
        "position", "grid", "laps", "points",
        "finish_time_millis", "fastest_lap_rank", "fastest_lap_number"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3️⃣ Time conversion
    if "fastest_lap_time" in df.columns:
        df["fastest_lap_time_sec"] = df["fastest_lap_time"].apply(time_to_seconds)

    df["position_numeric"] = df["position"]

    # 4️⃣ Driver rolling stats
    def add_driver_rolling_features(g):
        g = g.sort_values("date").copy()
        g["driver_avg_finish_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).mean()
        g["driver_wins_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).apply(lambda x: (x == 1).sum(), raw=True)
        g["driver_podiums_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).apply(lambda x: (x <= 3).sum(), raw=True)
        g["driver_avg_grid_last5"] = g["grid"].shift().rolling(5, min_periods=1).mean()
        g["driver_points_last5"] = g["points"].shift().rolling(5, min_periods=1).sum()
        return g
    df = df.groupby("driver_id", group_keys=False, sort=False).apply(add_driver_rolling_features)

    # 5️⃣ Constructor rolling stats
    def add_constructor_rolling_features(g):
        g = g.sort_values("date").copy()
        g["constructor_avg_finish_last5"] = g["position_numeric"].shift().rolling(5, min_periods=1).mean()
        g["constructor_points_last5"] = g["points"].shift().rolling(5, min_periods=1).sum()
        return g
    df = df.groupby("constructor", group_keys=False, sort=False).apply(add_constructor_rolling_features)

    # 6️⃣ Driver-track average finish (FIXED with transform)
    df["driver_track_avg_finish"] = df.groupby(["driver_id", "circuit"])["position_numeric"] \
        .transform(lambda x: x.shift().expanding().mean())

    # 7️⃣ Target
    df["is_winner"] = (df["position_numeric"] == 1).astype(int)

    # 8️⃣ Encode categoricals
    df["driver_code_num"] = df["driver_id"].astype("category").cat.codes
    df["constructor_code_num"] = df["constructor"].astype("category").cat.codes
    df["circuit_code_num"] = df["circuit"].astype("category").cat.codes

    # 9️⃣ Fill missing rolling stats
    rolling_cols = [
        "driver_avg_finish_last5", "driver_wins_last5", "driver_podiums_last5",
        "driver_avg_grid_last5", "driver_points_last5",
        "constructor_avg_finish_last5", "constructor_points_last5",
        "driver_track_avg_finish", "fastest_lap_time_sec"
    ]
    for col in rolling_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 🔟 Save only ML-relevant columns
    feature_cols = [
        "season", "round", "date", "race_name", "driver_id", "driver_name", "constructor",
        "grid", "laps", "points", "finish_time_millis", "fastest_lap_time_sec",
        "driver_code_num", "constructor_code_num", "circuit_code_num",
        "driver_avg_finish_last5", "driver_wins_last5", "driver_podiums_last5",
        "driver_avg_grid_last5", "driver_points_last5",
        "constructor_avg_finish_last5", "constructor_points_last5",
        "driver_track_avg_finish", "is_winner"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df[feature_cols].to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv} ({len(df)} rows)")

if __name__ == "__main__":
    prepare_features(INPUT_CSV, OUTPUT_CSV)